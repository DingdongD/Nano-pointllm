"""Cycle model for the RTL-locked W8 Split-K GEMV vertical slice."""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SplitKGemvConfig:
    n_outputs: int = 4
    split_k: int = 2
    k_chunks: int = 3
    lanes: int = 4
    source_latency: int = 3
    compressed_fifo_depth: int = 2
    decoded_fifo_depth: int = 2
    partial_fifo_depth: int = 2
    output_stall_mod: int = 5
    output_stall_phase: int = 0
    max_cycles: int = 10000

    @classmethod
    def load(cls, path: str | Path) -> "SplitKGemvConfig":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        payload.pop("schema_version", None)
        payload.pop("operator", None)
        config = cls(**payload)
        config.validate()
        return config

    def validate(self) -> None:
        positive = {
            "n_outputs": self.n_outputs,
            "split_k": self.split_k,
            "k_chunks": self.k_chunks,
            "lanes": self.lanes,
            "compressed_fifo_depth": self.compressed_fifo_depth,
            "decoded_fifo_depth": self.decoded_fifo_depth,
            "partial_fifo_depth": self.partial_fifo_depth,
            "max_cycles": self.max_cycles,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.source_latency < 0 or self.output_stall_mod < 0:
            raise ValueError("latency and stall modulus must be non-negative")
        if self.output_stall_mod and not 0 <= self.output_stall_phase < self.output_stall_mod:
            raise ValueError("output_stall_phase must be within output_stall_mod")

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class GemvBeat:
    n: int
    partition: int
    chunk: int
    weights: tuple[int, ...]
    activations: tuple[int, ...]


@dataclass(frozen=True)
class PartialSum:
    n: int
    partition: int
    value: int


@dataclass(frozen=True)
class OutputValue:
    n: int
    value: int


@dataclass(frozen=True)
class GemvEvent:
    cycle: int
    event: str
    n: int
    partition: int
    chunk: int
    value: int = 0

    def key(self) -> tuple[int, str, int, int, int, int]:
        return (
            self.cycle,
            self.event,
            self.n,
            self.partition,
            self.chunk,
            self.value,
        )


@dataclass(frozen=True)
class GemvResult:
    cycles: int
    events: tuple[GemvEvent, ...]
    outputs: tuple[OutputValue, ...]
    counters: dict[str, int]
    config: SplitKGemvConfig

    def as_dict(self) -> dict[str, Any]:
        return {
            "cycles": self.cycles,
            "outputs": [asdict(item) for item in self.outputs],
            "counters": self.counters,
            "config": self.config.as_dict(),
            "fidelity": {
                "level": "rtl_locked_event_model",
                "operator": "w8_splitk_gemv_vertical_slice",
                "full_model_cycle_accurate": False,
            },
        }


def build_beats(config: SplitKGemvConfig) -> tuple[GemvBeat, ...]:
    config.validate()
    beats = []
    for n_index in range(config.n_outputs):
        for partition in range(config.split_k):
            for chunk in range(config.k_chunks):
                weights = tuple(
                    _weight_value(config, n_index, partition, chunk, lane)
                    for lane in range(config.lanes)
                )
                activations = tuple(
                    _activation_value(config, partition, chunk, lane)
                    for lane in range(config.lanes)
                )
                beats.append(GemvBeat(
                    n=n_index,
                    partition=partition,
                    chunk=chunk,
                    weights=weights,
                    activations=activations,
                ))
    return tuple(beats)


def functional_outputs(config: SplitKGemvConfig) -> tuple[OutputValue, ...]:
    totals = [0 for _ in range(config.n_outputs)]
    for beat in build_beats(config):
        totals[beat.n] += _dot(beat)
    return tuple(OutputValue(n=index, value=value) for index, value in enumerate(totals))


def run_cycle_model(config: SplitKGemvConfig) -> GemvResult:
    """Advance the same edge-visible state implemented by the SystemVerilog slice."""
    config.validate()
    beats = build_beats(config)
    source_index = 0
    compressed: deque[GemvBeat] = deque()
    unpack_register: GemvBeat | None = None
    decoded: deque[GemvBeat] = deque()
    partials: deque[PartialSum] = deque()
    accumulator = 0
    reduction_accumulator = 0
    output_register: OutputValue | None = None
    outputs: list[OutputValue] = []
    events: list[GemvEvent] = []
    counters = {
        "input_beats": 0,
        "unpack_beats": 0,
        "compute_beats": 0,
        "partial_sums": 0,
        "outputs": 0,
        "source_backpressure_cycles": 0,
        "unpack_backpressure_cycles": 0,
        "compute_backpressure_cycles": 0,
        "reduction_backpressure_cycles": 0,
    }

    for cycle in range(config.max_cycles):
        output_ready = _output_ready(config, cycle)
        source_valid = cycle >= config.source_latency and source_index < len(beats)
        source_beat = beats[source_index] if source_valid else None
        input_ready = len(compressed) < config.compressed_fifo_depth
        input_accept = source_valid and input_ready

        decoded_input_ready = len(decoded) < config.decoded_fifo_depth
        unpack_ready = unpack_register is None or decoded_input_ready
        compressed_valid = bool(compressed)
        unpack_accept = compressed_valid and unpack_ready
        unpack_beat = compressed[0] if unpack_accept else None
        decoded_push = unpack_register is not None and decoded_input_ready

        partial_input_ready = len(partials) < config.partial_fifo_depth
        decoded_valid = bool(decoded)
        decoded_beat = decoded[0] if decoded_valid else None
        decoded_is_last = bool(decoded_beat and decoded_beat.chunk == config.k_chunks - 1)
        compute_ready = not decoded_is_last or partial_input_ready
        compute_accept = decoded_valid and compute_ready
        dot_value = _dot(decoded_beat) if decoded_beat else 0
        next_partial_value = (
            dot_value if decoded_beat and decoded_beat.chunk == 0
            else accumulator + dot_value
        )
        partial_push = compute_accept and decoded_is_last
        new_partial = (
            PartialSum(decoded_beat.n, decoded_beat.partition, next_partial_value)
            if partial_push and decoded_beat else None
        )

        output_accept = output_register is not None and output_ready
        reduction_ready = output_register is None or output_ready
        partial_valid = bool(partials)
        reduction_accept = partial_valid and reduction_ready
        reduction_partial = partials[0] if reduction_accept else None
        reduction_value = (
            reduction_partial.value
            if reduction_partial and reduction_partial.partition == 0
            else reduction_accumulator + (reduction_partial.value if reduction_partial else 0)
        )
        output_push = bool(
            reduction_partial and reduction_partial.partition == config.split_k - 1
        )

        if source_valid and not input_ready:
            counters["source_backpressure_cycles"] += 1
        if compressed_valid and not unpack_ready:
            counters["unpack_backpressure_cycles"] += 1
        if decoded_valid and not compute_ready:
            counters["compute_backpressure_cycles"] += 1
        if partial_valid and not reduction_ready:
            counters["reduction_backpressure_cycles"] += 1

        if input_accept and source_beat:
            counters["input_beats"] += 1
            events.append(_beat_event(cycle, "INPUT_ACCEPT", source_beat))
        if unpack_accept and unpack_beat:
            counters["unpack_beats"] += 1
            events.append(_beat_event(cycle, "UNPACK_ACCEPT", unpack_beat))
        if compute_accept and decoded_beat:
            counters["compute_beats"] += 1
            events.append(GemvEvent(
                cycle, "COMPUTE_ACCEPT", decoded_beat.n,
                decoded_beat.partition, decoded_beat.chunk, dot_value,
            ))
        if partial_push and new_partial:
            counters["partial_sums"] += 1
            events.append(GemvEvent(
                cycle, "PARTIAL_PUSH", new_partial.n,
                new_partial.partition, config.k_chunks - 1, new_partial.value,
            ))
        if output_accept and output_register:
            counters["outputs"] += 1
            outputs.append(output_register)
            events.append(GemvEvent(
                cycle, "OUTPUT_ACCEPT", output_register.n,
                config.split_k - 1, config.k_chunks - 1, output_register.value,
            ))

        if input_accept and source_beat:
            compressed.append(source_beat)
            source_index += 1
        if unpack_accept:
            compressed.popleft()

        if decoded_push and unpack_register:
            decoded.append(unpack_register)
        if compute_accept:
            decoded.popleft()

        if partial_push and new_partial:
            partials.append(new_partial)
        if reduction_accept:
            partials.popleft()

        if unpack_ready:
            unpack_register = unpack_beat
        if compute_accept:
            accumulator = next_partial_value

        if output_accept:
            output_register = None
        if reduction_accept and reduction_partial:
            reduction_accumulator = reduction_value
            if output_push:
                output_register = OutputValue(reduction_partial.n, reduction_value)

        if len(outputs) == config.n_outputs:
            result = GemvResult(
                cycles=cycle + 1,
                events=tuple(events),
                outputs=tuple(outputs),
                counters=counters,
                config=config,
            )
            expected = functional_outputs(config)
            if result.outputs != expected:
                raise AssertionError(f"cycle model output mismatch: {result.outputs} != {expected}")
            return result

    raise TimeoutError(f"Split-K GEMV did not finish in {config.max_cycles} cycles")


def _beat_event(cycle: int, event: str, beat: GemvBeat) -> GemvEvent:
    return GemvEvent(cycle, event, beat.n, beat.partition, beat.chunk, 0)


def _dot(beat: GemvBeat) -> int:
    return sum(weight * activation for weight, activation in zip(
        beat.weights, beat.activations, strict=True,
    ))


def _weight_value(
    config: SplitKGemvConfig,
    n_index: int,
    partition: int,
    chunk: int,
    lane: int,
) -> int:
    linear = (
        (((n_index * config.split_k) + partition) * config.k_chunks + chunk)
        * config.lanes + lane
    )
    return linear % 7 - 3


def _activation_value(
    config: SplitKGemvConfig,
    partition: int,
    chunk: int,
    lane: int,
) -> int:
    linear = ((partition * config.k_chunks + chunk) * config.lanes) + lane
    return linear % 5 - 2


def _output_ready(config: SplitKGemvConfig, cycle: int) -> bool:
    return (
        config.output_stall_mod == 0
        or cycle % config.output_stall_mod != config.output_stall_phase
    )
