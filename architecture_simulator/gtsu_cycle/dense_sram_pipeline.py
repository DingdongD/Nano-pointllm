"""Cycle and value golden for the physically composed Dense SRAM pipeline."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

from .requant import RequantVector, requantize_int32
from .shared_dot import dot4


@dataclass(frozen=True)
class DenseSramPipelineConfig:
    m: int = 2
    n: int = 5
    k: int = 16
    n_tile: int = 3
    k_block: int = 8
    read_latency: int = 3
    fifo_depth: int = 8
    requant_multiplier: int = 1
    requant_right_shift: int = 5
    source_stall_mod: int = 4
    source_stall_phase: int = 1
    output_stall_mod: int = 7
    output_stall_phase: int = 3
    max_cycles: int = 100_000

    def validate(self) -> None:
        if min(self.m, self.n, self.k, self.n_tile, self.k_block,
               self.read_latency, self.fifo_depth, self.max_cycles) <= 0:
            raise ValueError("pipeline dimensions must be positive")
        if self.n_tile > 3:
            raise ValueError("one-word physical slice supports at most three columns")
        if self.k % self.k_block or self.k_block % 4:
            raise ValueError("physical slice requires full four-element K blocks")
        if self.fifo_depth <= self.read_latency:
            raise ValueError("FIFO depth must exceed SRAM read latency")

    @property
    def n_tiles(self) -> int:
        return math.ceil(self.n / self.n_tile)

    @property
    def block_chunks(self) -> int:
        return self.k_block // 4

    @property
    def k_blocks(self) -> int:
        return self.k // self.k_block

    @property
    def total_blocks(self) -> int:
        return self.m * self.n_tiles * self.k_blocks

    @property
    def output_tiles(self) -> int:
        return self.m * self.n_tiles

    def as_dict(self) -> dict[str, int]:
        return {**asdict(self), "n_tiles": self.n_tiles,
                "k_blocks": self.k_blocks, "total_blocks": self.total_blocks}


@dataclass(frozen=True)
class PayloadBeat:
    sequence: int
    chunk: int
    data: int


@dataclass(frozen=True)
class DensePipelineEvent:
    cycle: int
    event: str
    tag: int
    value: int


@dataclass(frozen=True)
class DensePipelineOutput:
    tag: int
    mask: int
    packed_data: int


@dataclass(frozen=True)
class DenseSramPipelineResult:
    cycles: int
    events: tuple[DensePipelineEvent, ...]
    outputs: tuple[DensePipelineOutput, ...]
    counters: dict[str, int]


def build_payload_beats(config: DenseSramPipelineConfig) -> tuple[PayloadBeat, ...]:
    config.validate()
    beats = []
    for sequence in range(config.total_blocks):
        k_block = sequence % config.k_blocks
        tile_linear = sequence // config.k_blocks
        n_tile_index = tile_linear % config.n_tiles
        row = tile_linear // config.n_tiles
        for chunk in range(config.block_chunks):
            global_chunk = k_block * config.block_chunks + chunk
            start = global_chunk * 4
            a = tuple(_wrap_i8(row * 17 + (start + lane) * 7 - 33) for lane in range(4))
            columns = []
            for local_column in range(config.n_tile):
                column = n_tile_index * config.n_tile + local_column
                if column < config.n:
                    columns.append(tuple(
                        _wrap_i8(column * 13 - (start + lane) * 5 + 29)
                        for lane in range(4)
                    ))
                else:
                    columns.append((0, 0, 0, 0))
            packed = _pack_i8(a)
            for column, values in enumerate(columns):
                packed |= _pack_i8(values) << (32 * (column + 1))
            beats.append(PayloadBeat(sequence, chunk, packed))
    return tuple(beats)


def run_dense_sram_pipeline_model(
    config: DenseSramPipelineConfig,
    beats: tuple[PayloadBeat, ...] | None = None,
) -> DenseSramPipelineResult:
    config.validate()
    beats = beats or build_payload_beats(config)
    expected_payloads = config.total_blocks * config.block_chunks
    if len(beats) != expected_payloads:
        raise ValueError("payload count does not match configured blocks")

    cycle = 0
    payload_index = 0
    next_load_sequence = 0
    next_load_chunk = 0
    buffers: list[int | None] = [None, None]
    expected_sequence = 0
    row = n_tile_index = k_block_index = read_chunk = 0
    memory: dict[int, int] = {}
    sram_responses: list[tuple[int, int, int]] = []
    metadata: dict[int, tuple[int, int, int, int]] = {}
    issue_tag = 0
    outstanding = 0
    fifo: list[tuple[int, int, int, int, int]] = []
    accumulator = [0] * config.n_tile
    dense_output: tuple[int, int, tuple[int, ...]] | None = None
    requant_output: DensePipelineOutput | None = None
    events: list[DensePipelineEvent] = []
    outputs: list[DensePipelineOutput] = []
    counters = {
        "payload_writes": 0, "sram_read_issues": 0, "sram_responses": 0,
        "dot4_inputs": 0, "output_tiles": 0, "read_write_overlap_cycles": 0,
    }

    while len(outputs) < config.output_tiles:
        if cycle >= config.max_cycles:
            raise RuntimeError("dense SRAM pipeline exceeded max_cycles")
        output_ready = _ready(
            cycle, config.output_stall_mod, config.output_stall_phase,
        )
        requant_ready = requant_output is None or output_ready
        dense_ready = dense_output is None or requant_ready
        fifo_valid = bool(fifo)
        dot_fire = fifo_valid and dense_ready
        requant_fire = dense_output is not None and requant_ready
        output_fire = requant_output is not None and output_ready

        due = [item for item in sram_responses if item[0] == cycle]
        sram_responses = [item for item in sram_responses if item[0] != cycle]
        if len(due) > 1:
            raise AssertionError("one SRAM read port returned multiple responses")
        response = due[0] if due else None
        if response is not None and len(fifo) >= config.fifo_depth and not dot_fire:
            raise AssertionError("response FIFO overflow")

        read_valid = (
            expected_sequence < config.total_blocks
            and buffers[expected_sequence & 1] == expected_sequence
        )
        read_credit = outstanding + len(fifo) < config.fifo_depth
        read_fire = read_valid and read_credit

        source_valid = payload_index < len(beats) and _ready(
            cycle, config.source_stall_mod, config.source_stall_phase,
        )
        beat = beats[payload_index] if source_valid else None
        protocol_match = beat is not None and (
            beat.sequence == next_load_sequence and beat.chunk == next_load_chunk
        )
        load_ready = buffers[next_load_sequence & 1] is None
        payload_fire = bool(protocol_match and load_ready)

        if payload_fire:
            assert beat is not None
            events.append(DensePipelineEvent(cycle, "PAYLOAD_ACCEPT", beat.sequence, beat.chunk))
        if read_fire:
            events.append(DensePipelineEvent(cycle, "SRAM_READ_ISSUE", expected_sequence, read_chunk))
        if response is not None:
            _, response_tag, _ = response
            events.append(DensePipelineEvent(cycle, "SRAM_RESPONSE", metadata[response_tag][0], response_tag))
        if dot_fire:
            events.append(DensePipelineEvent(cycle, "DOT4_INPUT", fifo[0][0], 0))
        if output_fire:
            assert requant_output is not None
            events.append(DensePipelineEvent(
                cycle, "OUTPUT_ACCEPT", requant_output.tag,
                (requant_output.mask << (config.n_tile * 8)) | requant_output.packed_data,
            ))

        incoming_requant = None
        if requant_fire:
            assert dense_output is not None
            tag, mask, values = dense_output
            packed = 0
            for column, value in enumerate(values):
                quantized = requantize_int32(RequantVector(
                    tag, value, 0, config.requant_multiplier,
                    config.requant_right_shift,
                ))
                packed |= (quantized & 0xFF) << (column * 8)
            incoming_requant = DensePipelineOutput(tag, mask, packed)
        if output_fire:
            assert requant_output is not None
            outputs.append(requant_output)
            counters["output_tiles"] += 1
        if incoming_requant is not None:
            requant_output = incoming_requant
        elif output_fire:
            requant_output = None

        incoming_dense = None
        if dot_fire:
            sequence, mask, first, last, payload = fifo[0]
            a = _unpack_i8(payload, 0)
            for column in range(config.n_tile):
                b = _unpack_i8(payload, 32 * (column + 1))
                value = dot4(a, b)
                accumulator[column] = value if first else accumulator[column] + value
            if last:
                incoming_dense = (sequence, mask, tuple(accumulator))
            counters["dot4_inputs"] += 1
        if incoming_dense is not None:
            dense_output = incoming_dense
        elif requant_fire:
            dense_output = None

        if dot_fire:
            fifo.pop(0)
        if response is not None:
            _, response_tag, response_data = response
            sequence, mask, first, last = metadata[response_tag]
            fifo.append((sequence, mask, first, last, response_data))
            outstanding -= 1
            counters["sram_responses"] += 1

        if read_fire:
            block_chunks = config.block_chunks
            address = (expected_sequence & 1) * block_chunks + read_chunk
            data = memory.get(address, 0)
            mask_columns = min(config.n_tile, config.n - n_tile_index * config.n_tile)
            mask = (1 << mask_columns) - 1
            first = int(k_block_index == 0 and read_chunk == 0)
            last = int(
                k_block_index == config.k_blocks - 1
                and read_chunk == config.block_chunks - 1
            )
            metadata[issue_tag] = (expected_sequence, mask, first, last)
            sram_responses.append((cycle + config.read_latency, issue_tag, data))
            issue_tag = (issue_tag + 1) & 0xFF
            outstanding += 1
            counters["sram_read_issues"] += 1
            if read_chunk + 1 == config.block_chunks:
                buffers[expected_sequence & 1] = None
                expected_sequence += 1
                read_chunk = 0
                if k_block_index + 1 == config.k_blocks:
                    k_block_index = 0
                    if n_tile_index + 1 == config.n_tiles:
                        n_tile_index = 0
                        row += 1
                    else:
                        n_tile_index += 1
                else:
                    k_block_index += 1
            else:
                read_chunk += 1

        if payload_fire:
            assert beat is not None
            address = (next_load_sequence & 1) * config.block_chunks + next_load_chunk
            memory[address] = beat.data
            payload_index += 1
            counters["payload_writes"] += 1
            if next_load_chunk + 1 == config.block_chunks:
                buffers[next_load_sequence & 1] = next_load_sequence
                next_load_sequence += 1
                next_load_chunk = 0
            else:
                next_load_chunk += 1
        if payload_fire and read_fire:
            counters["read_write_overlap_cycles"] += 1
        cycle += 1

    if outstanding or fifo or dense_output or requant_output:
        raise AssertionError("pipeline reported all outputs before draining")
    return DenseSramPipelineResult(cycle, tuple(events), tuple(outputs), counters)


def _pack_i8(values: tuple[int, int, int, int]) -> int:
    result = 0
    for index, value in enumerate(values):
        result |= (value & 0xFF) << (index * 8)
    return result


def _unpack_i8(value: int, shift: int) -> tuple[int, int, int, int]:
    return tuple(
        _signed_i8((value >> (shift + index * 8)) & 0xFF)
        for index in range(4)
    )


def _signed_i8(value: int) -> int:
    return value - 256 if value & 0x80 else value


def _wrap_i8(value: int) -> int:
    return ((value + 128) % 256) - 128


def _ready(cycle: int, modulus: int, phase: int) -> bool:
    return modulus == 0 or cycle % modulus != phase
