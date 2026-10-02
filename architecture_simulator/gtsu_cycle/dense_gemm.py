"""Edge-state model for an output-stationary shared-Dot4 dense GEMM tile."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

from .shared_dot import dot4


@dataclass(frozen=True)
class DenseTileConfig:
    rows: int = 5
    columns: int = 7
    n_tile: int = 8
    k: int = 20
    source_stall_mod: int = 4
    source_stall_phase: int = 1
    output_stall_mod: int = 5
    output_stall_phase: int = 2
    max_cycles: int = 10000

    def validate(self) -> None:
        if min(self.rows, self.columns, self.n_tile, self.k, self.max_cycles) <= 0:
            raise ValueError("dense tile dimensions must be positive")
        if self.columns > self.n_tile:
            raise ValueError("one dense tile requires columns <= n_tile")
        if self.k % 4:
            raise ValueError("dense tile K must be divisible by four")

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class DenseBeat:
    tag: int
    first: bool
    last: bool
    column_mask: int
    a: tuple[int, int, int, int]
    b: tuple[tuple[int, int, int, int], ...]


@dataclass(frozen=True)
class DenseEvent:
    cycle: int
    event: str
    tag: int
    column: int
    value: int


@dataclass(frozen=True)
class DenseResult:
    config: DenseTileConfig
    cycles: int
    events: tuple[DenseEvent, ...]
    outputs: tuple[tuple[int, ...], ...]
    counters: dict[str, int]


def build_dense_beats(config: DenseTileConfig) -> tuple[DenseBeat, ...]:
    config.validate()
    a = [
        [_wrap_i8(row * 17 + k_index * 7 - 91) for k_index in range(config.k)]
        for row in range(config.rows)
    ]
    b = [
        [_wrap_i8(column * 13 - k_index * 5 + 47) for k_index in range(config.k)]
        for column in range(config.columns)
    ]
    chunks = config.k // 4
    mask = (1 << config.columns) - 1
    beats = []
    for row in range(config.rows):
        for chunk in range(chunks):
            start = chunk * 4
            columns = tuple(
                tuple(b[column][start:start + 4])
                if column < config.columns else (0, 0, 0, 0)
                for column in range(config.n_tile)
            )
            beats.append(DenseBeat(
                row, chunk == 0, chunk == chunks - 1, mask,
                tuple(a[row][start:start + 4]), columns,
            ))
    return tuple(beats)


def functional_outputs(
    config: DenseTileConfig, beats: tuple[DenseBeat, ...],
) -> tuple[tuple[int, ...], ...]:
    rows = [[0] * config.columns for _ in range(config.rows)]
    for beat in beats:
        for column in range(config.columns):
            rows[beat.tag][column] += dot4(beat.a, beat.b[column])
    return tuple(tuple(row) for row in rows)


def run_dense_tile_model(
    config: DenseTileConfig, beats: tuple[DenseBeat, ...] | None = None,
) -> DenseResult:
    config.validate()
    beats = beats or build_dense_beats(config)
    cycle = 0
    source_index = 0
    accumulator = [0] * config.n_tile
    output_register: tuple[int, int, tuple[int, ...]] | None = None
    events = []
    outputs = []
    counters = {
        "input_chunks": 0, "output_rows": 0, "output_values": 0,
        "source_backpressure_cycles": 0, "output_backpressure_cycles": 0,
    }
    while source_index < len(beats) or output_register is not None:
        if cycle >= config.max_cycles:
            raise RuntimeError("dense tile model exceeded max_cycles")
        source_valid = source_index < len(beats) and _ready(
            cycle, config.source_stall_mod, config.source_stall_phase,
        )
        output_ready = _ready(cycle, config.output_stall_mod, config.output_stall_phase)
        input_ready = output_register is None or output_ready
        accept_input = source_valid and input_ready
        accept_output = output_register is not None and output_ready
        if source_valid and not input_ready:
            counters["source_backpressure_cycles"] += 1
        if output_register is not None and not output_ready:
            counters["output_backpressure_cycles"] += 1

        incoming_output = None
        if accept_input:
            beat = beats[source_index]
            events.append(DenseEvent(cycle, "INPUT_ACCEPT", beat.tag, -1, 0))
            counters["input_chunks"] += 1
            source_index += 1
            for column in range(config.n_tile):
                value = dot4(beat.a, beat.b[column])
                accumulator[column] = value if beat.first else accumulator[column] + value
            if beat.last:
                incoming_output = (beat.tag, beat.column_mask, tuple(accumulator))
        if accept_output:
            assert output_register is not None
            tag, mask, values = output_register
            row = []
            for column, value in enumerate(values):
                if mask & (1 << column):
                    events.append(DenseEvent(cycle, "OUTPUT_ACCEPT", tag, column, value))
                    row.append(value)
                    counters["output_values"] += 1
            outputs.append(tuple(row))
            counters["output_rows"] += 1
        if incoming_output is not None:
            output_register = incoming_output
        elif accept_output:
            output_register = None
        cycle += 1
    expected = functional_outputs(config, beats)
    if tuple(outputs) != expected:
        raise AssertionError("dense tile functional and cycle outputs diverged")
    return DenseResult(config, cycle, tuple(events), tuple(outputs), counters)


def pointllm_dense_shape_mapping(n_tile: int = 64) -> dict[str, dict[str, int | str]]:
    shapes = {
        "local_encoder_1x1": (512 * 32, 256, 6),
        "point_transformer_qkv": (513, 1152, 384),
        "projector": (513, 4096, 2048),
        "llm_prefill_q": (768, 4096, 4096),
    }
    result = {}
    for name, (m, n, k) in shapes.items():
        padded_k = math.ceil(k / 4) * 4
        result[name] = {
            "m": m, "n": n, "k": k, "padded_k": padded_k,
            "n_tile": n_tile,
            "dot4_issue_cycles": m * math.ceil(n / n_tile) * (padded_k // 4),
            "status": "controller_issue_lower_bound_not_full_rtl_correlation",
        }
    return result


def _wrap_i8(value: int) -> int:
    return ((value + 128) % 256) - 128


def _ready(cycle: int, modulus: int, phase: int) -> bool:
    return modulus == 0 or cycle % modulus != phase
