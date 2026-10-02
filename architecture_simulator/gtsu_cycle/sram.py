"""RTL-locked banked SRAM port and arbitration model.

The contract follows the useful part of Compiler_Codes' LBUF organization:
independent read/write ports on every bank, fixed client priority on a same-bank
collision, and a registered read response.  It models an SRAM interface, not a
foundry memory macro's PPA or analog timing.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable


@dataclass(frozen=True)
class SramConfig:
    banks: int = 16
    rows_per_bank: int = 16_384
    word_bytes: int = 16
    read_latency: int = 3
    clients: int = 2

    def validate(self) -> None:
        for name in ("banks", "rows_per_bank", "word_bytes", "read_latency", "clients"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.clients != 2:
            raise ValueError("the RTL-locked SRAM slice currently has exactly two clients")

    @property
    def capacity_bytes(self) -> int:
        return self.banks * self.rows_per_bank * self.word_bytes


@dataclass(frozen=True)
class SramRequest:
    client: int
    available_cycle: int
    is_write: bool
    address: int
    tag: int
    data: int = 0
    byte_enable: int | None = None


@dataclass(frozen=True)
class SramEvent:
    cycle: int
    event: str
    client: int
    tag: int
    bank: int
    row: int
    data: int

    def as_dict(self) -> dict[str, int | str]:
        return asdict(self)


@dataclass(frozen=True)
class SramResult:
    cycles: int
    events: tuple[SramEvent, ...]
    counters: dict[str, int]
    config: SramConfig

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": "0.1",
            "cycles": self.cycles,
            "events": [event.as_dict() for event in self.events],
            "counters": self.counters,
            "config": asdict(self.config),
            "fidelity": {
                "event_cycle_model": True,
                "rtl_contract": "gtsu_banked_sram_2client",
                "foundry_sram_macro": False,
                "full_model_cycle_accurate": False,
            },
        }


def decode_sram_address(config: SramConfig, address: int) -> tuple[int, int]:
    """Return ``(bank, row)`` for a linearly striped byte address."""
    config.validate()
    if address < 0 or address % config.word_bytes:
        raise ValueError(
            f"SRAM address must be non-negative and {config.word_bytes}-byte aligned"
        )
    word = address // config.word_bytes
    bank = word % config.banks
    row = word // config.banks
    if row >= config.rows_per_bank:
        raise ValueError(f"SRAM address 0x{address:x} exceeds configured capacity")
    return bank, row


def locked_sram_requests(config: SramConfig | None = None) -> tuple[SramRequest, ...]:
    """Conflict-heavy deterministic schedule shared with the RTL testbench."""
    config = config or SramConfig()
    w = config.word_bytes
    return (
        SramRequest(0, 0, True, 0 * w, 0, 0x11),
        SramRequest(0, 1, True, 1 * w, 1, 0x22),
        SramRequest(0, 2, True, config.banks * w, 2, 0x33),
        SramRequest(0, 3, True, 3 * w, 6, 0),
        SramRequest(0, 4, False, 0 * w, 10),
        SramRequest(0, 5, False, 1 * w, 11),
        SramRequest(0, 6, False, config.banks * w, 12),
        SramRequest(0, 10, False, 3 * w, 13),
        SramRequest(0, 11, False, 3 * w, 14),
        SramRequest(1, 0, True, 0 * w, 3, 0xAA),
        SramRequest(1, 1, True, 2 * w, 4, 0xBB),
        SramRequest(1, 4, False, 0 * w, 20),
        SramRequest(1, 5, False, 2 * w, 21),
        SramRequest(1, 10, True, 3 * w, 5, 0xCC),
    )


def run_sram_model(
    requests: Iterable[SramRequest] | None = None,
    config: SramConfig | None = None,
    *,
    max_cycles: int = 1_000,
) -> SramResult:
    """Run the two-client valid/ready interface one rising edge at a time.

    Each client presents requests in program order and holds a request until its
    corresponding port accepts it. Different banks proceed concurrently. On a
    same-bank collision client 0 has fixed priority, matching the locked RTL.
    """
    config = config or SramConfig()
    config.validate()
    source = tuple(requests if requests is not None else locked_sram_requests(config))
    streams: list[list[SramRequest]] = [[] for _ in range(config.clients)]
    word_mask = (1 << (8 * config.word_bytes)) - 1
    full_byte_enable = (1 << config.word_bytes) - 1
    for request in source:
        if request.client < 0 or request.client >= config.clients:
            raise ValueError(f"invalid SRAM client {request.client}")
        if request.available_cycle < 0:
            raise ValueError("available_cycle must be non-negative")
        decode_sram_address(config, request.address)
        streams[request.client].append(request)
    for stream in streams:
        if any(
            right.available_cycle < left.available_cycle
            for left, right in zip(stream, stream[1:])
        ):
            raise ValueError("per-client SRAM requests must be ordered by available_cycle")

    indices = [0] * config.clients
    memory: dict[tuple[int, int], int] = {}
    responses: list[tuple[int, SramRequest, int, int, int]] = []
    events: list[SramEvent] = []
    counters = {
        "read_accepts": 0,
        "write_accepts": 0,
        "read_completions": 0,
        "read_bank_conflicts": 0,
        "write_bank_conflicts": 0,
        "client_stall_cycles": 0,
    }

    for cycle in range(max_cycles):
        due = [entry for entry in responses if entry[0] == cycle]
        responses = [entry for entry in responses if entry[0] != cycle]
        for _, request, bank, row, data in sorted(due, key=lambda item: item[1].client):
            events.append(SramEvent(
                cycle, "READ_COMPLETE", request.client, request.tag, bank, row, data,
            ))
            counters["read_completions"] += 1

        presented: list[SramRequest | None] = []
        for client, stream in enumerate(streams):
            request = stream[indices[client]] if indices[client] < len(stream) else None
            if request is not None and request.available_cycle > cycle:
                request = None
            presented.append(request)

        accepted = [False] * config.clients
        for is_write, conflict_key in (
            (False, "read_bank_conflicts"),
            (True, "write_bank_conflicts"),
        ):
            contenders: dict[int, list[tuple[int, SramRequest, int]]] = {}
            for client, request in enumerate(presented):
                if request is None or request.is_write != is_write:
                    continue
                bank, row = decode_sram_address(config, request.address)
                contenders.setdefault(bank, []).append((client, request, row))
            for bank in sorted(contenders):
                bank_requests = sorted(contenders[bank], key=lambda item: item[0])
                client, request, row = bank_requests[0]
                accepted[client] = True
                if len(bank_requests) > 1:
                    counters[conflict_key] += len(bank_requests) - 1
                if is_write:
                    old = memory.get((bank, row), 0)
                    byte_enable = (
                        full_byte_enable if request.byte_enable is None else request.byte_enable
                    )
                    if byte_enable < 0 or byte_enable > full_byte_enable:
                        raise ValueError("byte_enable exceeds SRAM word width")
                    data = old
                    for byte_index in range(config.word_bytes):
                        if (byte_enable >> byte_index) & 1:
                            byte_mask = 0xFF << (8 * byte_index)
                            data = (data & ~byte_mask) | (request.data & byte_mask)
                    data &= word_mask
                    memory[(bank, row)] = data
                    events.append(SramEvent(
                        cycle, "WRITE_ACCEPT", client, request.tag, bank, row, data,
                    ))
                    counters["write_accepts"] += 1
                else:
                    data = memory.get((bank, row), 0)
                    events.append(SramEvent(
                        cycle, "READ_ACCEPT", client, request.tag, bank, row, 0,
                    ))
                    responses.append((
                        cycle + config.read_latency, request, bank, row, data,
                    ))
                    counters["read_accepts"] += 1

        stalled = sum(
            request is not None and not accepted[client]
            for client, request in enumerate(presented)
        )
        counters["client_stall_cycles"] += stalled
        for client, was_accepted in enumerate(accepted):
            if was_accepted:
                indices[client] += 1

        if all(indices[i] == len(streams[i]) for i in range(config.clients)) and not responses:
            return SramResult(cycle + 1, tuple(events), counters, config)

    raise TimeoutError(f"SRAM schedule did not finish in {max_cycles} cycles")
