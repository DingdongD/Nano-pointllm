"""Cycle/value model for the 64-column ABUF/WBUF shared-Dot4 fabric."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

from .shared_dot import dot4


@dataclass(frozen=True)
class ProductionDenseFabricConfig:
    m: int = 2
    n: int = 70
    k: int = 16
    n_tile: int = 64
    banks: int = 16
    read_latency: int = 3
    fifo_depth: int = 8
    memory_lanes: int = 1
    source_stall_mod: int = 5
    source_stall_phase: int = 2
    output_stall_mod: int = 7
    output_stall_phase: int = 3
    max_cycles: int = 2_000_000

    def validate(self) -> None:
        if min(self.m, self.n, self.k, self.n_tile, self.banks,
               self.read_latency, self.fifo_depth, self.memory_lanes,
               self.max_cycles) <= 0:
            raise ValueError("fabric dimensions must be positive")
        if self.n_tile != 64 or self.banks != 16:
            raise ValueError("production fabric is locked to 64 columns and 16 banks")
        if self.k % 4:
            raise ValueError("K must be divisible by four")
        if self.fifo_depth <= self.read_latency:
            raise ValueError("response FIFO depth must exceed read latency")
        if self.memory_lanes not in (1, 2, 4):
            raise ValueError("memory_lanes must be one, two, or four")

    @property
    def n_tiles(self) -> int:
        return math.ceil(self.n / self.n_tile)

    @property
    def chunks(self) -> int:
        return self.k // 4

    @property
    def packets(self) -> int:
        return self.m * self.n_tiles * self.chunks

    @property
    def lines(self) -> int:
        return self.packets * 4

    @property
    def output_tiles(self) -> int:
        return self.m * self.n_tiles

    def as_dict(self) -> dict[str, int]:
        return {**asdict(self), "n_tiles": self.n_tiles,
                "chunks": self.chunks, "packets": self.packets,
                "lines": self.lines, "output_tiles": self.output_tiles}


@dataclass(frozen=True)
class FabricLine:
    line: int
    sequence: int
    quarter: int
    activation: int
    weights: int


@dataclass(frozen=True)
class FabricEvent:
    cycle: int
    event: str
    tag: int
    value: int


@dataclass(frozen=True)
class ProductionDenseFabricResult:
    cycles: int
    events: tuple[FabricEvent, ...]
    outputs: tuple[tuple[int, ...], ...]
    counters: dict[str, int]


def build_fabric_lines(
    config: ProductionDenseFabricConfig,
) -> tuple[FabricLine, ...]:
    config.validate()
    lines = []
    for sequence in range(config.packets):
        row, n_tile_index, chunk = decode_sequence(config, sequence)
        activation = _pack_i8(tuple(
            _wrap_i8(row * 17 + (chunk * 4 + lane) * 7 - 33)
            for lane in range(4)
        ))
        for quarter in range(4):
            packed = 0
            for local in range(16):
                local_column = quarter * 16 + local
                column = n_tile_index * config.n_tile + local_column
                values = tuple(
                    _wrap_i8(column * 13 - (chunk * 4 + lane) * 5 + 29)
                    if column < config.n else 0
                    for lane in range(4)
                )
                packed |= _pack_i8(values) << (local * 32)
            lines.append(FabricLine(
                len(lines), sequence, quarter, activation, packed,
            ))
    return tuple(lines)


def build_fabric_lines_from_arrays(
    config: ProductionDenseFabricConfig, activation, weight,
) -> tuple[FabricLine, ...]:
    """Pack caller-owned row-major INT8 tensors into physical ingress lines."""
    config.validate()
    if tuple(activation.shape) != (config.m, config.k):
        raise ValueError("activation shape does not match Dense64 config")
    if tuple(weight.shape) != (config.n, config.k):
        raise ValueError("weight shape does not match Dense64 config")
    lines = []
    for sequence in range(config.packets):
        row, n_tile_index, chunk = decode_sequence(config, sequence)
        start = chunk * 4
        activation_word = _pack_i8(tuple(
            int(value) for value in activation[row, start:start + 4]
        ))
        for quarter in range(4):
            packed = 0
            for local in range(16):
                column = n_tile_index * config.n_tile + quarter * 16 + local
                values = (
                    tuple(int(value) for value in weight[column, start:start + 4])
                    if column < config.n else (0, 0, 0, 0)
                )
                if any(value < -127 or value > 127 for value in values):
                    raise ValueError("physical INT8 payload is outside [-127,127]")
                packed |= _pack_i8(values) << (local * 32)
            lines.append(FabricLine(
                len(lines), sequence, quarter, activation_word, packed,
            ))
    return tuple(lines)


def decode_sequence(
    config: ProductionDenseFabricConfig, sequence: int,
) -> tuple[int, int, int]:
    tile_chunk = config.n_tiles * config.chunks
    row = sequence // tile_chunk
    remainder = sequence % tile_chunk
    return row, remainder // config.chunks, remainder % config.chunks


def run_production_dense_fabric_model(
    config: ProductionDenseFabricConfig,
    lines: tuple[FabricLine, ...] | None = None,
    *, trace: bool | str = True,
) -> ProductionDenseFabricResult:
    config.validate()
    lines = lines or build_fabric_lines(config)
    if len(lines) != config.lines:
        raise ValueError("line count does not match configured shape")
    slots = [None, None]
    line_index = expected_sequence = 0
    read_due: list[tuple[int, tuple[int, int, int, int, int]]] = []
    fifo: list[tuple[int, int, int, int, int]] = []
    accumulator = [0] * config.n_tile
    output_register: tuple[int, int, tuple[int, ...]] | None = None
    events: list[FabricEvent] = []
    outputs: list[tuple[int, ...]] = []
    counters = {
        "ingress_lines": 0, "wbuf_bank_writes": 0, "abuf_writes": 0,
        "wbuf_read_issues": 0, "wbuf_responses": 0, "dot4_chunks": 0,
        "output_tiles": 0, "output_values": 0,
        "fill_compute_overlap_cycles": 0, "source_backpressure_cycles": 0,
        "output_backpressure_cycles": 0, "response_fifo_peak": 0,
    }

    def record(cycle: int, event: str, tag: int, value: int) -> None:
        if trace is True or trace == "all" or (
            trace == "outputs" and event == "OUTPUT_ACCEPT"
        ):
            events.append(FabricEvent(cycle, event, tag, value))

    for cycle in range(config.max_cycles):
        output_ready = _ready(cycle, config.output_stall_mod, config.output_stall_phase)
        dense_ready = output_register is None or output_ready
        dot_fire = bool(fifo) and dense_ready
        output_fire = output_register is not None and output_ready
        if output_register is not None and not output_ready:
            counters["output_backpressure_cycles"] += 1

        due = [item for due_cycle, item in read_due if due_cycle == cycle]
        read_due = [(due_cycle, item) for due_cycle, item in read_due if due_cycle != cycle]
        if len(due) > 1:
            raise AssertionError("one WBUF vector response expected per cycle")
        response = due[0] if due else None
        if response is not None and len(fifo) >= config.fifo_depth and not dot_fire:
            raise AssertionError("WBUF response FIFO overflow")

        slot = slots[expected_sequence & 1] if expected_sequence < config.packets else None
        read_credit = len(read_due) + len(fifo) < config.fifo_depth
        read_fire = (
            slot is not None and slot[0] == expected_sequence
            and slot[1] == 4 and read_credit
        )

        source_valid = line_index < len(lines) and _ready(
            cycle, config.source_stall_mod, config.source_stall_phase,
        )
        group = lines[line_index:line_index + config.memory_lanes] if source_valid else ()
        line = group[0] if group else None
        fill_ready = False
        if line is not None:
            if len(group) != config.memory_lanes:
                raise AssertionError("incomplete final memory-lane group")
            if any(
                member.sequence != line.sequence
                or member.quarter != line.quarter + offset
                for offset, member in enumerate(group)
            ):
                raise AssertionError("memory lanes must carry consecutive quarters")
            current = slots[line.sequence & 1]
            fill_ready = current is None or (
                current[0] == line.sequence and current[1] == line.quarter
            )
        line_fire = source_valid and fill_ready
        if source_valid and not fill_ready:
            counters["source_backpressure_cycles"] += 1

        if line_fire:
            for member in group:
                record(cycle, "LINE_ACCEPT", member.line, member.quarter)
        if read_fire:
            record(cycle, "WBUF_READ_ISSUE", expected_sequence, 0)
        if response is not None:
            record(cycle, "WBUF_RESPONSE", response[0], 0)
        if dot_fire:
            record(cycle, "DOT4_INPUT", fifo[0][0], 0)
        if output_fire:
            assert output_register is not None
            tag, mask, values = output_register
            row_values = []
            base_column = (tag % config.n_tiles) * config.n_tile
            for column, value in enumerate(values):
                if mask & (1 << column):
                    record(cycle, "OUTPUT_ACCEPT", base_column + column, value)
                    row_values.append(value)
            outputs.append(tuple(row_values))
            counters["output_tiles"] += 1
            counters["output_values"] += len(row_values)

        incoming_output = None
        if dot_fire:
            sequence, activation, weights, mask, flags = fifo[0]
            first, last = flags & 1, (flags >> 1) & 1
            a = _unpack_i8(activation, 0)
            for column in range(config.n_tile):
                value = dot4(a, _unpack_i8(weights, column * 32))
                accumulator[column] = value if first else accumulator[column] + value
            if last:
                incoming_output = (sequence // config.chunks, mask, tuple(accumulator))
            counters["dot4_chunks"] += 1
        if incoming_output is not None:
            output_register = incoming_output
        elif output_fire:
            output_register = None

        if dot_fire:
            fifo.pop(0)
        if response is not None:
            fifo.append(response)
            counters["wbuf_responses"] += 1
            counters["response_fifo_peak"] = max(
                counters["response_fifo_peak"], len(fifo),
            )

        if read_fire:
            assert slot is not None
            sequence, _, activation, weights = slot
            row, n_tile_index, chunk = decode_sequence(config, sequence)
            valid_columns = min(
                config.n_tile, config.n - n_tile_index * config.n_tile,
            )
            mask = (1 << valid_columns) - 1
            flags = int(chunk == 0) | (int(chunk == config.chunks - 1) << 1)
            read_due.append((
                cycle + config.read_latency,
                (sequence, activation, weights, mask, flags),
            ))
            slots[expected_sequence & 1] = None
            expected_sequence += 1
            counters["wbuf_read_issues"] += 1

        if line_fire:
            assert line is not None
            slot_index = line.sequence & 1
            current = slots[slot_index]
            if current is None:
                current = [line.sequence, 0, line.activation, 0]
            else:
                current = list(current)
            if current[0] != line.sequence or current[1] != line.quarter:
                raise AssertionError("WBUF quarter protocol mismatch")
            for member in group:
                current[3] |= member.weights << (member.quarter * 512)
            current[1] += config.memory_lanes
            slots[slot_index] = tuple(current)
            if current[1] == 4:
                slots[slot_index] = (current[0], 4, current[2], current[3])
            line_index += config.memory_lanes
            counters["ingress_lines"] += config.memory_lanes
            counters["wbuf_bank_writes"] += 4 * config.memory_lanes
            if line.quarter == 0:
                counters["abuf_writes"] += 1
        if line_fire and read_fire:
            counters["fill_compute_overlap_cycles"] += 1

        if len(outputs) == config.output_tiles:
            if read_due or fifo or output_register is not None:
                raise AssertionError("fabric outputs completed before drain")
            return ProductionDenseFabricResult(
                cycle + 1, tuple(events), tuple(outputs), counters,
            )
    raise TimeoutError("production Dense fabric exceeded max_cycles")


def _pack_i8(values: tuple[int, int, int, int]) -> int:
    return sum((value & 0xFF) << (index * 8) for index, value in enumerate(values))


def _unpack_i8(value: int, shift: int) -> tuple[int, int, int, int]:
    return tuple(_signed_i8((value >> (shift + lane * 8)) & 0xFF) for lane in range(4))


def _signed_i8(value: int) -> int:
    return value - 256 if value & 0x80 else value


def _wrap_i8(value: int) -> int:
    return ((value + 128) % 256) - 128


def _ready(cycle: int, modulus: int, phase: int) -> bool:
    return modulus == 0 or cycle % modulus != phase
