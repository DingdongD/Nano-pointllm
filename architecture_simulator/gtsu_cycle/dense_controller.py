"""Edge-state model for the Dense GEMM M/N/K ping-pong controller."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class DenseControllerConfig:
    m: int = 3
    n: int = 70
    k: int = 132
    n_tile: int = 64
    k_block: int = 64
    source_stall_mod: int = 4
    source_stall_phase: int = 1
    compute_stall_mod: int = 5
    compute_stall_phase: int = 2
    max_cycles: int = 100_000

    def validate(self) -> None:
        if min(self.m, self.n, self.k, self.n_tile, self.k_block, self.max_cycles) <= 0:
            raise ValueError("dense controller dimensions must be positive")
        if self.k % 4 or self.k_block % 4:
            raise ValueError("K and K block must be divisible by four")
        if self.n_tile > 64:
            raise ValueError("locked column mask supports at most 64 columns")

    @property
    def n_tiles(self) -> int:
        return math.ceil(self.n / self.n_tile)

    @property
    def total_chunks(self) -> int:
        return self.k // 4

    @property
    def block_chunks(self) -> int:
        return self.k_block // 4

    @property
    def k_blocks(self) -> int:
        return math.ceil(self.total_chunks / self.block_chunks)

    @property
    def total_blocks(self) -> int:
        return self.m * self.n_tiles * self.k_blocks

    def as_dict(self) -> dict[str, int]:
        return {**asdict(self), "n_tiles": self.n_tiles, "k_blocks": self.k_blocks,
                "total_blocks": self.total_blocks}


@dataclass(frozen=True)
class DenseControllerEvent:
    cycle: int
    event: str
    sequence: int
    row: int
    n_tile_index: int
    k_block_index: int
    chunk: int
    column_mask: int
    first: int
    last: int


@dataclass(frozen=True)
class DenseControllerResult:
    cycles: int
    events: tuple[DenseControllerEvent, ...]
    counters: dict[str, int]


def run_dense_controller_model(config: DenseControllerConfig) -> DenseControllerResult:
    config.validate()
    cycle = 0
    load_sequence = 0
    expected_sequence = 0
    buffers: list[int | None] = [None, None]
    row = n_tile_index = k_block_index = chunk = 0
    events: list[DenseControllerEvent] = []
    counters = {"loads": 0, "read_issues": 0, "blocks_released": 0,
                "load_backpressure_cycles": 0, "compute_wait_cycles": 0,
                "downstream_stall_cycles": 0, "ping_pong_overlap_cycles": 0}
    while expected_sequence < config.total_blocks:
        if cycle >= config.max_cycles:
            raise RuntimeError("dense controller exceeded max_cycles")
        load_valid = load_sequence < config.total_blocks and _ready(
            cycle, config.source_stall_mod, config.source_stall_phase,
        )
        load_slot = load_sequence & 1
        load_ready = buffers[load_slot] is None
        compute_slot = expected_sequence & 1
        compute_valid = buffers[compute_slot] == expected_sequence
        compute_ready = _ready(cycle, config.compute_stall_mod, config.compute_stall_phase)

        if load_valid and not load_ready:
            counters["load_backpressure_cycles"] += 1
        if not compute_valid:
            counters["compute_wait_cycles"] += 1
        elif not compute_ready:
            counters["downstream_stall_cycles"] += 1
        if compute_valid and any(
            item is not None and item != expected_sequence for item in buffers
        ):
            counters["ping_pong_overlap_cycles"] += 1

        if load_valid and load_ready:
            buffers[load_slot] = load_sequence
            events.append(_event(
                cycle, "LOAD_ACCEPT", load_sequence, config, 0, 0, 0, 0,
            ))
            counters["loads"] += 1
            load_sequence += 1

        if compute_valid and compute_ready:
            block_chunks = min(
                config.block_chunks,
                config.total_chunks - k_block_index * config.block_chunks,
            )
            first = int(k_block_index == 0 and chunk == 0)
            last = int(k_block_index == config.k_blocks - 1 and chunk == block_chunks - 1)
            events.append(_event(
                cycle, "READ_ISSUE", expected_sequence, config,
                row, n_tile_index, k_block_index, chunk, first, last,
            ))
            counters["read_issues"] += 1
            if chunk + 1 == block_chunks:
                buffers[compute_slot] = None
                events.append(_event(
                    cycle, "BLOCK_RELEASE", expected_sequence, config,
                    row, n_tile_index, k_block_index, chunk, first, last,
                ))
                counters["blocks_released"] += 1
                expected_sequence += 1
                chunk = 0
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
                chunk += 1
        cycle += 1
    return DenseControllerResult(cycle, tuple(events), counters)


def _event(
    cycle: int, event: str, sequence: int, config: DenseControllerConfig,
    row: int, n_tile_index: int, k_block_index: int, chunk: int,
    first: int = 0, last: int = 0,
) -> DenseControllerEvent:
    valid_columns = min(config.n_tile, config.n - n_tile_index * config.n_tile)
    mask = 0 if event == "LOAD_ACCEPT" else (1 << valid_columns) - 1
    return DenseControllerEvent(
        cycle, event, sequence, row, n_tile_index, k_block_index, chunk,
        mask, first, last,
    )


def _ready(cycle: int, modulus: int, phase: int) -> bool:
    return modulus == 0 or cycle % modulus != phase
