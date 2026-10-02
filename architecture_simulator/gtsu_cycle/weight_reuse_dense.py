"""Cycle/value model for K-block-resident high-M Dense64 execution."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

from .shared_dot import dot4


@dataclass(frozen=True)
class WeightReuseDenseConfig:
    m: int = 5
    n: int = 70
    k: int = 24
    m_tile: int = 3
    k_block: int = 16
    n_tile: int = 64
    banks: int = 16
    memory_lanes: int = 4
    read_latency: int = 3
    fifo_depth: int = 8
    weight_stall_mod: int = 5
    weight_stall_phase: int = 2
    activation_stall_mod: int = 7
    activation_stall_phase: int = 3
    output_stall_mod: int = 11
    output_stall_phase: int = 4
    max_cycles: int = 5_000_000

    def validate(self) -> None:
        values = (
            self.m, self.n, self.k, self.m_tile, self.k_block, self.n_tile,
            self.banks, self.memory_lanes, self.read_latency, self.fifo_depth,
            self.max_cycles,
        )
        if min(values) <= 0:
            raise ValueError("weight-reuse Dense dimensions must be positive")
        if self.n_tile != 64 or self.banks != 16:
            raise ValueError("weight-reuse fabric is locked to 64 columns/16 banks")
        if self.k % 4 or self.k_block % 4:
            raise ValueError("K and K-block must be divisible by four")
        if self.memory_lanes not in (1, 2, 4):
            raise ValueError("memory_lanes must be one, two, or four")
        if self.fifo_depth <= self.read_latency:
            raise ValueError("response FIFO depth must exceed read latency")

    @property
    def n_tiles(self) -> int:
        return math.ceil(self.n / self.n_tile)

    @property
    def m_tiles(self) -> int:
        return math.ceil(self.m / self.m_tile)

    @property
    def chunks(self) -> int:
        return self.k // 4

    @property
    def block_chunks(self) -> int:
        return self.k_block // 4

    @property
    def k_blocks(self) -> int:
        return math.ceil(self.chunks / self.block_chunks)

    @property
    def weight_tiles(self) -> int:
        return self.m_tiles * self.n_tiles * self.k_blocks

    @property
    def weight_lines(self) -> int:
        return self.m_tiles * self.n_tiles * self.chunks * 4

    @property
    def activation_chunks(self) -> int:
        return self.m * self.n_tiles * self.chunks

    @property
    def output_tiles(self) -> int:
        return self.m * self.n_tiles

    def as_dict(self) -> dict[str, int]:
        return {
            **asdict(self), "n_tiles": self.n_tiles, "m_tiles": self.m_tiles,
            "chunks": self.chunks, "block_chunks": self.block_chunks,
            "k_blocks": self.k_blocks, "weight_tiles": self.weight_tiles,
            "weight_lines": self.weight_lines,
            "activation_chunks": self.activation_chunks,
            "output_tiles": self.output_tiles,
        }


@dataclass(frozen=True)
class WeightReuseEvent:
    cycle: int
    event: str
    tag: int
    value: int


@dataclass(frozen=True)
class WeightReuseDenseResult:
    cycles: int
    events: tuple[WeightReuseEvent, ...]
    outputs: tuple[tuple[int, ...], ...]
    counters: dict[str, int]


def run_weight_reuse_dense_model(
    config: WeightReuseDenseConfig, *, trace: bool | str = True,
    compute_values: bool = True,
) -> WeightReuseDenseResult:
    config.validate()
    state = "FILL"
    m_tile_index = n_tile_index = k_block_index = 0
    fill_line = local_row = chunk_in_block = 0
    all_tiles_complete = False
    read_due: list[tuple[int, tuple]] = []
    fifo: list[tuple] = []
    dense_accumulator = [0] * config.n_tile
    partial_register = None
    output_register = None
    partial_sums = [
        [0] * config.n_tile for _ in range(config.m_tile)
    ]
    events: list[WeightReuseEvent] = []
    outputs = []
    counters = {
        "weight_lines": 0, "wbuf_bank_writes": 0, "wbuf_load_tiles": 0,
        "activation_chunks": 0, "wbuf_read_issues": 0,
        "wbuf_responses": 0, "dot4_chunks": 0, "partial_tiles": 0,
        "output_tiles": 0, "output_values": 0,
        "weight_backpressure_cycles": 0,
        "activation_backpressure_cycles": 0,
        "output_backpressure_cycles": 0, "response_fifo_peak": 0,
    }

    def record(cycle: int, event: str, tag: int, value: int = 0) -> None:
        if trace is True or trace == "all" or (
            trace == "outputs" and event == "OUTPUT_ACCEPT"
        ):
            events.append(WeightReuseEvent(cycle, event, tag, value))

    for cycle in range(config.max_cycles):
        rows_this_tile = min(config.m_tile, config.m - m_tile_index * config.m_tile)
        chunks_this_block = min(
            config.block_chunks,
            config.chunks - k_block_index * config.block_chunks,
        )
        output_ready = _ready(
            cycle, config.output_stall_mod, config.output_stall_phase,
        )
        output_fire = output_register is not None and output_ready
        if output_register is not None and not output_ready:
            counters["output_backpressure_cycles"] += 1

        partial_final = (
            partial_register is not None
            and partial_register[2] == config.k_blocks - 1
        )
        partial_ready = not partial_final or output_register is None or output_ready
        partial_fire = partial_register is not None and partial_ready
        dense_ready = partial_register is None or partial_ready
        dot_fire = bool(fifo) and dense_ready

        due = [item for due_cycle, item in read_due if due_cycle == cycle]
        read_due = [(due_cycle, item) for due_cycle, item in read_due if due_cycle != cycle]
        if len(due) > 1:
            raise AssertionError("one WBUF response expected per cycle")
        response = due[0] if due else None
        if response is not None and len(fifo) >= config.fifo_depth and not dot_fire:
            raise AssertionError("weight-reuse response FIFO overflow")

        read_credit = len(read_due) + len(fifo) < config.fifo_depth
        activation_source_valid = state == "COMPUTE" and _ready(
            cycle, config.activation_stall_mod, config.activation_stall_phase,
        )
        activation_ready = state == "COMPUTE" and read_credit
        read_fire = activation_source_valid and activation_ready
        if state == "COMPUTE" and not activation_source_valid:
            counters["activation_backpressure_cycles"] += 1

        weight_source_valid = state == "FILL" and _ready(
            cycle, config.weight_stall_mod, config.weight_stall_phase,
        )
        weight_fire = weight_source_valid
        if state == "FILL" and not weight_source_valid:
            counters["weight_backpressure_cycles"] += 1

        if weight_fire:
            for lane in range(config.memory_lanes):
                record(cycle, "WEIGHT_LINE_ACCEPT", counters["weight_lines"] + lane)
        if read_fire:
            tag = _partial_tag(
                m_tile_index * config.m_tile + local_row,
                n_tile_index, k_block_index,
            )
            record(cycle, "WBUF_READ_ISSUE", tag)
        if response is not None:
            record(cycle, "WBUF_RESPONSE", response[0])
        if dot_fire:
            record(cycle, "DOT4_INPUT", fifo[0][0])
        if partial_fire:
            record(cycle, "PARTIAL_ACCEPT", partial_register[0])
        if output_fire:
            tag, mask, values = output_register
            row = tag // config.n_tiles
            base_column = (tag % config.n_tiles) * config.n_tile
            accepted = []
            for column, value in enumerate(values):
                if mask & (1 << column):
                    record(cycle, "OUTPUT_ACCEPT", row * config.n + base_column + column, value)
                    accepted.append(value)
            outputs.append(tuple(accepted))
            counters["output_tiles"] += 1
            counters["output_values"] += len(accepted)

        incoming_output = None
        if partial_fire:
            tag, row, block, mask, values = partial_register
            local = row - m_tile_index * config.m_tile
            combined = tuple(
                value if block == 0 else partial_sums[local][column] + value
                for column, value in enumerate(values)
            )
            if block == config.k_blocks - 1:
                incoming_output = (row * config.n_tiles + n_tile_index, mask, combined)
            else:
                partial_sums[local][:] = combined
            counters["partial_tiles"] += 1
            if local == rows_this_tile - 1:
                state, m_tile_index, n_tile_index, k_block_index, all_tiles_complete = (
                    _advance_tile(config, m_tile_index, n_tile_index, k_block_index)
                )
                fill_line = local_row = chunk_in_block = 0
        if incoming_output is not None:
            output_register = incoming_output
        elif output_fire:
            output_register = None

        incoming_partial = None
        if dot_fire:
            tag, activation, weights, mask, first, last, row, block = fifo[0]
            a = _unpack_i8(activation)
            if compute_values:
                for column in range(config.n_tile):
                    value = dot4(a, _unpack_i8(weights >> (column * 32)))
                    dense_accumulator[column] = (
                        value if first else dense_accumulator[column] + value
                    )
            if last:
                incoming_partial = (
                    tag, row, block, mask, tuple(dense_accumulator),
                )
            counters["dot4_chunks"] += 1
        if incoming_partial is not None:
            partial_register = incoming_partial
        elif partial_fire:
            partial_register = None

        if dot_fire:
            fifo.pop(0)
        if response is not None:
            fifo.append(response)
            counters["wbuf_responses"] += 1
            counters["response_fifo_peak"] = max(
                counters["response_fifo_peak"], len(fifo),
            )

        if read_fire:
            row = m_tile_index * config.m_tile + local_row
            global_chunk = k_block_index * config.block_chunks + chunk_in_block
            activation = _pack_i8(tuple(
                _activation(row, global_chunk * 4 + lane) for lane in range(4)
            ))
            weights = 0
            if compute_values:
                for column in range(config.n_tile):
                    global_column = n_tile_index * config.n_tile + column
                    values = tuple(
                        _weight(global_column, global_chunk * 4 + lane)
                        if global_column < config.n else 0
                        for lane in range(4)
                    )
                    weights |= _pack_i8(values) << (column * 32)
            mask = (1 << min(config.n_tile, config.n - n_tile_index * config.n_tile)) - 1
            tag = _partial_tag(row, n_tile_index, k_block_index)
            read_due.append((
                cycle + config.read_latency,
                (tag, activation, weights, mask,
                 chunk_in_block == 0, chunk_in_block == chunks_this_block - 1,
                 row, k_block_index),
            ))
            counters["activation_chunks"] += 1
            counters["wbuf_read_issues"] += 1
            if chunk_in_block == chunks_this_block - 1:
                chunk_in_block = 0
                if local_row == rows_this_tile - 1:
                    state = "WAIT"
                else:
                    local_row += 1
            else:
                chunk_in_block += 1

        if weight_fire:
            accepted = config.memory_lanes
            fill_line += accepted
            counters["weight_lines"] += accepted
            counters["wbuf_bank_writes"] += accepted * 4
            if fill_line == chunks_this_block * 4:
                counters["wbuf_load_tiles"] += 1
                fill_line = 0
                local_row = chunk_in_block = 0
                state = "COMPUTE"

        if (
            all_tiles_complete and not read_due and not fifo
            and partial_register is None and output_register is None
        ):
            return WeightReuseDenseResult(
                cycle + 1, tuple(events), tuple(outputs), counters,
            )
    raise TimeoutError("weight-reuse Dense model exceeded max_cycles")


def repeated_gemv_weight_bytes(config: WeightReuseDenseConfig) -> int:
    return config.m * config.n_tiles * config.chunks * 256


def weight_reuse_bytes(config: WeightReuseDenseConfig) -> int:
    return config.weight_lines * 64


def _advance_tile(config, mt, nt, kb):
    if kb + 1 < config.k_blocks:
        return "FILL", mt, nt, kb + 1, False
    if nt + 1 < config.n_tiles:
        return "FILL", mt, nt + 1, 0, False
    if mt + 1 < config.m_tiles:
        return "FILL", mt + 1, 0, 0, False
    return "WAIT", mt, nt, kb, True


def _partial_tag(row: int, n_tile: int, k_block: int) -> int:
    return (k_block << 24) | (n_tile << 16) | row


def _activation(row: int, k_index: int) -> int:
    return _wrap_i8(row * 17 + k_index * 7 - 33)


def _weight(column: int, k_index: int) -> int:
    return _wrap_i8(column * 13 - k_index * 5 + 29)


def _pack_i8(values) -> int:
    return sum((value & 0xFF) << (lane * 8) for lane, value in enumerate(values))


def _unpack_i8(value: int) -> tuple[int, int, int, int]:
    return tuple(
        _signed_i8((value >> (lane * 8)) & 0xFF) for lane in range(4)
    )


def _signed_i8(value: int) -> int:
    return value - 256 if value & 0x80 else value


def _wrap_i8(value: int) -> int:
    return ((value + 128) % 256) - 128


def _ready(cycle: int, modulus: int, phase: int) -> bool:
    return modulus == 0 or cycle % modulus != phase
