"""Strict local DRAMsim3 adapter for GTSU memory request timing."""
from __future__ import annotations

import ctypes
from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
import re
from typing import Iterable

from .pointllm_lowering import DramBurst


_TCK_RE = re.compile(r"^\s*tCK\s*=\s*([0-9.]+)\s*$", re.MULTILINE)


@dataclass(frozen=True)
class DramSim3Paths:
    library: Path
    config: Path
    source_commit: str = "29817593b3389f1337235d63cac515024ab8fd6e"

    @classmethod
    def local_default(cls) -> "DramSim3Paths":
        return cls(
            library=Path(
                "/home/PointAcc/QuickFPS/build/dramsim3-latest/"
                "libquickfps_dramsim3_bridge.so"
            ),
            config=Path("/home/DRAMsim3/configs/DDR4_8Gb_x8_2400.ini"),
        )

    def validate(self) -> None:
        if not self.library.is_file():
            raise FileNotFoundError(f"DRAMsim3 bridge is unavailable: {self.library}")
        if not self.config.is_file():
            raise FileNotFoundError(f"DRAMsim3 config is unavailable: {self.config}")


@dataclass(frozen=True)
class DramRequest:
    tag: int
    address: int
    size: int
    stream: str
    submitted_cycle: int
    is_write: bool = False


@dataclass(frozen=True)
class DramCompletion:
    tag: int
    address: int
    size: int
    stream: str
    submitted_cycle: int
    completed_cycle: int
    is_write: bool

    @property
    def latency(self) -> int:
        return self.completed_cycle - self.submitted_cycle


@dataclass(frozen=True)
class DramEvent:
    cycle: int
    event: str
    tag: int
    address: int
    stream: str
    latency: int


@dataclass(frozen=True)
class DramTimingResult:
    accelerator_cycles: int
    dram_clock_ticks: int
    requests: int
    bytes_read: int
    latency_min: int
    latency_median: float
    latency_max: int
    events: tuple[DramEvent, ...]
    provenance: dict[str, object]

    def as_dict(self) -> dict[str, object]:
        return asdict(self) | {
            "events": [asdict(event) for event in self.events],
            "fidelity": {
                "backend": "official_dramsim3_via_local_c_abi",
                "fixed_latency_model": False,
                "controller_rtl_correlated": False,
                "full_model_cycle_accurate": False,
            },
        }


class ClockScaledDramSim3:
    """Expose DRAMsim3 callbacks in accelerator-cycle time."""

    def __init__(
        self,
        paths: DramSim3Paths,
        output_dir: str | Path,
        *,
        accelerator_clock_hz: int = 1_000_000_000,
        transaction_bytes: int = 64,
    ):
        paths.validate()
        if accelerator_clock_hz <= 0 or transaction_bytes <= 0:
            raise ValueError("clock and transaction size must be positive")
        self.paths = paths
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.transaction_bytes = transaction_bytes
        text = paths.config.read_text(encoding="utf-8", errors="replace")
        match = _TCK_RE.search(text)
        if not match:
            raise ValueError(f"DRAMsim3 config has no tCK: {paths.config}")
        self.dram_tck_ns = float(match.group(1))
        if self.dram_tck_ns <= 0:
            raise ValueError("DRAMsim3 tCK must be positive")
        self.accelerator_period_ns = 1.0e9 / accelerator_clock_hz
        self.accelerator_clock_hz = accelerator_clock_hz
        self._time_credit_ns = 0.0
        self.cycle = 0
        self.dram_clock_ticks = 0
        self._requests: dict[int, DramRequest] = {}
        self._lib = ctypes.CDLL(str(paths.library))
        self._configure_api()
        self._handle = self._lib.qfps_dramsim3_create(
            str(paths.config).encode(), str(self.output_dir).encode(),
        )
        if not self._handle:
            raise RuntimeError("local DRAMsim3 bridge failed to create a memory system")

    def _configure_api(self) -> None:
        self._lib.qfps_dramsim3_create.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
        self._lib.qfps_dramsim3_create.restype = ctypes.c_void_p
        self._lib.qfps_dramsim3_destroy.argtypes = [ctypes.c_void_p]
        self._lib.qfps_dramsim3_destroy.restype = None
        self._lib.qfps_dramsim3_will_accept.argtypes = [
            ctypes.c_void_p, ctypes.c_uint64, ctypes.c_int,
        ]
        self._lib.qfps_dramsim3_will_accept.restype = ctypes.c_int
        self._lib.qfps_dramsim3_add.argtypes = [
            ctypes.c_void_p, ctypes.c_uint64, ctypes.c_int, ctypes.c_uint64,
        ]
        self._lib.qfps_dramsim3_add.restype = ctypes.c_int
        self._lib.qfps_dramsim3_tick.argtypes = [ctypes.c_void_p]
        self._lib.qfps_dramsim3_tick.restype = None
        self._lib.qfps_dramsim3_poll.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_int),
        ]
        self._lib.qfps_dramsim3_poll.restype = ctypes.c_int

    def can_accept(self, address: int, is_write: bool = False) -> bool:
        self._validate_address(address)
        return bool(self._lib.qfps_dramsim3_will_accept(
            self._handle, address, int(is_write),
        ))

    def submit(self, request: DramRequest) -> bool:
        if request.tag in self._requests:
            raise ValueError(f"duplicate DRAM request tag {request.tag}")
        if request.submitted_cycle != self.cycle:
            raise ValueError(
                f"request cycle {request.submitted_cycle} does not match backend {self.cycle}"
            )
        if request.size != self.transaction_bytes:
            raise ValueError(
                f"DRAMsim3 request must be one {self.transaction_bytes}-byte transaction"
            )
        self._validate_address(request.address)
        accepted = bool(self._lib.qfps_dramsim3_add(
            self._handle, request.address, int(request.is_write), request.tag,
        ))
        if accepted:
            self._requests[request.tag] = request
        return accepted

    def tick(self) -> tuple[DramCompletion, ...]:
        self._time_credit_ns += self.accelerator_period_ns
        raw: list[tuple[int, int, bool]] = []
        while self._time_credit_ns + 1e-12 >= self.dram_tck_ns:
            self._lib.qfps_dramsim3_tick(self._handle)
            self._time_credit_ns -= self.dram_tck_ns
            self.dram_clock_ticks += 1
            while True:
                tag = ctypes.c_uint64()
                address = ctypes.c_uint64()
                is_write = ctypes.c_int()
                if not self._lib.qfps_dramsim3_poll(
                    self._handle, ctypes.byref(tag), ctypes.byref(address),
                    ctypes.byref(is_write),
                ):
                    break
                raw.append((int(tag.value), int(address.value), bool(is_write.value)))
        self.cycle += 1
        completions = []
        for tag, address, is_write in raw:
            request = self._requests.pop(tag, None)
            if request is None:
                raise RuntimeError(f"DRAMsim3 returned unknown request tag {tag}")
            if request.address != address or request.is_write != is_write:
                raise RuntimeError("DRAMsim3 callback does not match submitted request")
            completions.append(DramCompletion(
                tag, address, request.size, request.stream,
                request.submitted_cycle, self.cycle, is_write,
            ))
        return tuple(completions)

    def idle(self) -> bool:
        return not self._requests

    def close(self) -> None:
        if getattr(self, "_handle", None):
            self._lib.qfps_dramsim3_destroy(self._handle)
            self._handle = None

    def _validate_address(self, address: int) -> None:
        if address < 0 or address % self.transaction_bytes:
            raise ValueError(
                f"DRAM address must be non-negative and {self.transaction_bytes}-byte aligned"
            )

    def __enter__(self) -> "ClockScaledDramSim3":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def time_dram_bursts(
    bursts: Iterable[DramBurst],
    *,
    paths: DramSim3Paths,
    output_dir: str | Path,
    accelerator_clock_hz: int = 1_000_000_000,
    max_outstanding: int = 32,
    max_cycles: int = 1_000_000,
    stream_barriers: bool = False,
    reorder_window: int | None = None,
    max_submissions_per_cycle: int | None = None,
) -> DramTimingResult:
    if max_outstanding <= 0:
        raise ValueError("max_outstanding must be positive")
    if reorder_window is not None and reorder_window <= 0:
        raise ValueError("reorder_window must be positive when provided")
    if max_submissions_per_cycle is not None and max_submissions_per_cycle <= 0:
        raise ValueError("max_submissions_per_cycle must be positive when provided")
    pending = tuple(bursts)
    if not pending:
        raise ValueError("at least one DRAM burst is required")
    if len({burst.sequence for burst in pending}) != len(pending):
        raise ValueError("DRAM burst sequence tags must be unique")
    events: list[DramEvent] = []
    completions: list[DramCompletion] = []
    next_index = 0
    outstanding = 0
    active_stream = pending[0].stream
    retired_index = 0
    completed_indices: set[int] = set()
    with ClockScaledDramSim3(
        paths, output_dir, accelerator_clock_hz=accelerator_clock_hz,
        transaction_bytes=pending[0].size,
    ) as backend:
        for _ in range(max_cycles):
            submissions_this_cycle = 0
            while (
                next_index < len(pending) and outstanding < max_outstanding
                and (
                    max_submissions_per_cycle is None
                    or submissions_this_cycle < max_submissions_per_cycle
                )
            ):
                burst = pending[next_index]
                if stream_barriers and burst.stream != active_stream:
                    if outstanding:
                        break
                    active_stream = burst.stream
                    retired_index = next_index
                    completed_indices.clear()
                if reorder_window is not None and next_index - retired_index >= reorder_window:
                    break
                if burst.size != backend.transaction_bytes:
                    raise ValueError("all sampled DRAM bursts must have equal size")
                request = DramRequest(
                    tag=burst.sequence,
                    address=burst.address,
                    size=burst.size,
                    stream=burst.stream,
                    submitted_cycle=backend.cycle,
                )
                if not backend.can_accept(request.address) or not backend.submit(request):
                    break
                events.append(DramEvent(
                    backend.cycle, "SUBMIT", request.tag, request.address,
                    request.stream, 0,
                ))
                next_index += 1
                outstanding += 1
                submissions_this_cycle += 1
            for completion in backend.tick():
                completions.append(completion)
                outstanding -= 1
                if reorder_window is not None:
                    completed_indices.add(completion.tag)
                    while retired_index in completed_indices:
                        completed_indices.remove(retired_index)
                        retired_index += 1
                events.append(DramEvent(
                    completion.completed_cycle, "COMPLETE", completion.tag,
                    completion.address, completion.stream, completion.latency,
                ))
            if next_index == len(pending) and backend.idle():
                break
        else:
            raise TimeoutError(f"DRAMsim3 sample exceeded {max_cycles} accelerator cycles")
        accelerator_cycles = backend.cycle
        dram_ticks = backend.dram_clock_ticks

    if len(completions) != len(pending):
        raise AssertionError("DRAMsim3 completion count does not match submissions")
    latencies = sorted(completion.latency for completion in completions)
    middle = len(latencies) // 2
    median = (
        float(latencies[middle]) if len(latencies) % 2
        else (latencies[middle - 1] + latencies[middle]) / 2.0
    )
    return DramTimingResult(
        accelerator_cycles=accelerator_cycles,
        dram_clock_ticks=dram_ticks,
        requests=len(completions),
        bytes_read=sum(completion.size for completion in completions if not completion.is_write),
        latency_min=latencies[0],
        latency_median=median,
        latency_max=latencies[-1],
        events=tuple(events),
        provenance={
            "bridge": str(paths.library.resolve()),
            "bridge_sha256": _sha256(paths.library),
            "config": str(paths.config.resolve()),
            "config_sha256": _sha256(paths.config),
            "dramsim3_source_commit": paths.source_commit,
            "dram_tck_ns": _read_tck(paths.config),
            "accelerator_clock_hz": accelerator_clock_hz,
            "transaction_bytes": pending[0].size,
            "max_outstanding": max_outstanding,
            "stream_barriers": stream_barriers,
            "reorder_window": reorder_window,
            "max_submissions_per_cycle": max_submissions_per_cycle,
        },
    )


def _read_tck(path: Path) -> float:
    match = _TCK_RE.search(path.read_text(encoding="utf-8", errors="replace"))
    if not match:
        raise ValueError(f"DRAMsim3 config has no tCK: {path}")
    return float(match.group(1))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
