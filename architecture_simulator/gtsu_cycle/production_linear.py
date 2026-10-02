"""Integrated burst-level cycle model for production PointLLM W8 linears.

The model joins DRAM completion order, a finite reorder buffer, bank-striped
SRAM vector accesses, registered SRAM reads, W8 unpack, Split-K accumulation,
and output retirement.  The matching RTL controller consumes the same DMA
completion contract; arithmetic value parity remains covered by the separate
W8 GEMV datapath slice.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from typing import Iterable

from .dramsim3_backend import DramEvent, DramTimingResult
from .pointllm_lowering import DramBurst, LoweredLinear
from .sram import decode_sram_address


@dataclass(frozen=True)
class ProductionLinearConfig:
    rob_depth: int = 32
    sram_read_latency: int = 3
    unpack_latency: int = 1
    words_per_burst: int = 4
    weight_bank_offset_words: int = 8
    dma_ingress_per_cycle: int = 1
    max_cycles: int = 20_000_000

    def validate(self, lowered: LoweredLinear) -> None:
        for name in (
            "rob_depth", "sram_read_latency", "unpack_latency",
            "words_per_burst", "dma_ingress_per_cycle", "max_cycles",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.words_per_burst * lowered.sram_config.word_bytes != lowered.config.dram_burst_bytes:
            raise ValueError("SRAM vector width must equal one DRAM burst")
        if (
            self.weight_bank_offset_words * lowered.sram_config.word_bytes
            + self.rob_depth * lowered.config.dram_burst_bytes
        ) > next(
            region.allocated_bytes for region in lowered.regions
            if region.name == "weight_tile_w8"
        ):
            raise ValueError("ROB slots exceed the allocated weight SRAM tile")


@dataclass(frozen=True)
class RuntimeBurst:
    dram_tag: int
    kind: str
    linear_sequence: int
    address: int
    output_row: int
    split_index: int
    chunk_index: int
    chunks_in_split: int


@dataclass(frozen=True)
class DmaArrival:
    cycle: int
    burst: RuntimeBurst


@dataclass(frozen=True)
class LinearEvent:
    cycle: int
    event: str
    sequence: int
    output_row: int
    split_index: int
    chunk_index: int


@dataclass(frozen=True)
class ProductionLinearResult:
    cycles: int
    counters: dict[str, int]
    event_counts: dict[str, int]
    events: tuple[LinearEvent, ...]
    projection: str
    layer: int | None
    n: int
    k: int
    config: ProductionLinearConfig

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": "0.1",
            "projection": self.projection,
            "layer": self.layer,
            "shape": {"m": 1, "n": self.n, "k": self.k},
            "cycles": self.cycles,
            "counters": self.counters,
            "event_counts": self.event_counts,
            "config": asdict(self.config),
            "events": [asdict(event) for event in self.events],
            "fidelity": {
                "level": "integrated_event_model_pending_production_rtl_correlation",
                "dram_completion_timed": True,
                "sram_bank_conflicts": True,
                "functional_values": False,
                "full_model_cycle_accurate": False,
            },
        }


def build_runtime_bursts(lowered: LoweredLinear) -> tuple[RuntimeBurst, ...]:
    """Place scale prefetch before the weight stream and assign unique DRAM tags."""
    runtime: list[RuntimeBurst] = []
    for scale in lowered.iter_scale_bursts():
        runtime.append(RuntimeBurst(
            dram_tag=len(runtime), kind="scale", linear_sequence=-1,
            address=scale.address, output_row=-1, split_index=-1,
            chunk_index=-1, chunks_in_split=-1,
        ))
    for weight in lowered.iter_weight_bursts():
        runtime.append(RuntimeBurst(
            dram_tag=len(runtime), kind="weight", linear_sequence=weight.sequence,
            address=weight.address, output_row=weight.output_row,
            split_index=weight.split_index, chunk_index=weight.chunk_index,
            chunks_in_split=weight.chunks_in_split,
        ))
    return tuple(runtime)


def runtime_as_dram_bursts(runtime: Iterable[RuntimeBurst], burst_bytes: int = 64) -> tuple[DramBurst, ...]:
    return tuple(DramBurst(
        sequence=item.dram_tag,
        address=item.address,
        size=burst_bytes,
        stream=item.kind,
        tile_id=-1,
        output_row=item.output_row,
        split_index=item.split_index,
        chunk_index=item.chunk_index,
        chunks_in_split=item.chunks_in_split,
    ) for item in runtime)


def serialize_dram_completions(
    timing: DramTimingResult,
    runtime: Iterable[RuntimeBurst],
) -> tuple[DmaArrival, ...]:
    """Serialize callbacks onto a one-burst-per-cycle 512-bit DMA ingress."""
    by_tag = {item.dram_tag: item for item in runtime}
    completions = [event for event in timing.events if event.event == "COMPLETE"]
    next_cycle = 0
    arrivals = []
    for event in completions:
        burst = by_tag.get(event.tag)
        if burst is None:
            raise ValueError(f"DRAM completion has unknown tag {event.tag}")
        cycle = max(event.cycle, next_cycle)
        arrivals.append(DmaArrival(cycle, burst))
        next_cycle = cycle + 1
    if len(arrivals) != len(by_tag):
        raise ValueError("DRAM completion stream is incomplete")
    return tuple(arrivals)


def run_production_linear_model(
    lowered: LoweredLinear,
    arrivals: Iterable[DmaArrival],
    *,
    config: ProductionLinearConfig | None = None,
    trace_events: bool = False,
) -> ProductionLinearResult:
    config = config or ProductionLinearConfig()
    config.validate(lowered)
    arrivals = tuple(arrivals)
    expected_scale = lowered.scale_burst_count
    expected_weight = lowered.weight_burst_count
    if len(arrivals) != expected_scale + expected_weight:
        raise ValueError("arrival count does not match scale plus weight requests")

    rob: dict[int, RuntimeBurst] = {}
    read_due: dict[int, list[RuntimeBurst]] = {}
    unpack_due: dict[int, list[RuntimeBurst]] = {}
    arrival_index = 0
    next_weight = 0
    scale_received = 0
    outputs = 0
    read_blocked_until = 0
    counters: Counter[str] = Counter()
    event_counts: Counter[str] = Counter()
    events: list[LinearEvent] = []

    def record(cycle: int, name: str, burst: RuntimeBurst) -> None:
        event_counts[name] += 1
        if trace_events:
            events.append(LinearEvent(
                cycle, name, burst.linear_sequence, burst.output_row,
                burst.split_index, burst.chunk_index,
            ))

    for cycle in range(config.max_cycles):
        for burst in read_due.pop(cycle, []):
            record(cycle, "SRAM_READ_COMPLETE", burst)
            unpack_due.setdefault(cycle + config.unpack_latency, []).append(burst)

        for burst in unpack_due.pop(cycle, []):
            record(cycle, "COMPUTE", burst)
            counters["compute_bursts"] += 1
            if burst.chunk_index == burst.chunks_in_split - 1:
                counters["partial_sums"] += 1
                record(cycle, "PARTIAL", burst)
                if burst.split_index == lowered.config.split_k - 1:
                    outputs += 1
                    counters["outputs"] += 1
                    record(cycle, "OUTPUT", burst)

        ingress_used = 0
        while (
            arrival_index < len(arrivals)
            and arrivals[arrival_index].cycle <= cycle
            and ingress_used < config.dma_ingress_per_cycle
        ):
            burst = arrivals[arrival_index].burst
            if burst.kind == "weight":
                slot_occupied = any(
                    sequence % config.rob_depth == burst.linear_sequence % config.rob_depth
                    for sequence in rob
                )
                if len(rob) >= config.rob_depth or slot_occupied:
                    counters["dma_rob_backpressure_cycles"] += 1
                    if slot_occupied:
                        counters["dma_rob_slot_collision_cycles"] += 1
                    break
            arrival_index += 1
            ingress_used += 1
            counters["dma_ingress_bursts"] += 1
            counters["sram_write_vectors"] += 1
            record(cycle, "DMA_ACCEPT", burst)
            record(cycle, "SRAM_WRITE", burst)
            if burst.kind == "scale":
                scale_received += 1
                counters["scale_bursts"] += 1
            else:
                if burst.linear_sequence in rob:
                    raise AssertionError("duplicate weight sequence in ROB")
                rob[burst.linear_sequence] = burst
                counters["weight_bursts"] += 1
                counters["rob_max_occupancy"] = max(
                    counters["rob_max_occupancy"], len(rob),
                )

        can_issue = (
            scale_received == expected_scale
            and next_weight in rob
            and cycle >= read_blocked_until
        )
        if can_issue:
            burst = rob.pop(next_weight)
            conflict = _read_bank_conflict(lowered, burst, config)
            latency = config.sram_read_latency + int(conflict)
            read_due.setdefault(cycle + latency, []).append(burst)
            counters["sram_read_vectors"] += 2
            counters["sram_read_issue_cycles"] += 1 + int(conflict)
            if conflict:
                counters["sram_read_bank_conflicts"] += 1
                counters["sram_read_conflict_stall_cycles"] += 1
                read_blocked_until = cycle + 2
            record(cycle, "SRAM_READ_ISSUE", burst)
            next_weight += 1
        elif scale_received == expected_scale and next_weight < expected_weight:
            counters["ordered_weight_wait_cycles"] += 1

        if outputs == lowered.n:
            if next_weight != expected_weight or rob or read_due or unpack_due:
                raise AssertionError("outputs retired before the linear pipeline drained")
            counters["dram_requests"] = len(arrivals)
            counters["dram_payload_bytes"] = len(arrivals) * lowered.config.dram_burst_bytes
            counters["sram_write_bytes"] = counters["sram_write_vectors"] * lowered.config.dram_burst_bytes
            counters["sram_read_bytes"] = counters["sram_read_vectors"] * lowered.config.dram_burst_bytes
            return ProductionLinearResult(
                cycles=cycle + 1,
                counters=dict(counters),
                event_counts=dict(event_counts),
                events=tuple(events),
                projection=lowered.projection,
                layer=lowered.layer,
                n=lowered.n,
                k=lowered.k,
                config=config,
            )

    raise TimeoutError(f"production linear exceeded {config.max_cycles} cycles")


def _read_bank_conflict(
    lowered: LoweredLinear,
    burst: RuntimeBurst,
    config: ProductionLinearConfig,
) -> bool:
    weight_region = next(region for region in lowered.regions if region.name == "weight_tile_w8")
    activation_region = next(region for region in lowered.regions if region.name == "activation_a8")
    weight_address = (
        weight_region.base
        + config.weight_bank_offset_words * lowered.sram_config.word_bytes
        + (burst.linear_sequence % config.rob_depth) * lowered.config.dram_burst_bytes
    )
    row_offset = (burst.address - next(
        region.base for region in lowered.regions if region.name == "weight_w8"
    )) % lowered.k
    activation_address = activation_region.base + row_offset
    weight_banks = {
        decode_sram_address(
            lowered.sram_config,
            weight_address + word * lowered.sram_config.word_bytes,
        )[0]
        for word in range(config.words_per_burst)
    }
    activation_banks = {
        decode_sram_address(
            lowered.sram_config,
            activation_address + word * lowered.sram_config.word_bytes,
        )[0]
        for word in range(config.words_per_burst)
    }
    return bool(weight_banks & activation_banks)
