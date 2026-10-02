"""Cycle/value model for Dense64 ACC32 to parallel A8 and tiled BF16 exits."""
from __future__ import annotations

from dataclasses import asdict, dataclass

from .bf16_dequant import Bf16DequantVector, dequantize_int32_to_bf16
from .requant import RequantVector, requantize_int32


@dataclass(frozen=True)
class DenseDualOutputConfig:
    bf16_lanes: int = 16
    source_stall_mod: int = 5
    source_stall_phase: int = 2
    output_stall_mod: int = 7
    output_stall_phase: int = 3
    max_cycles: int = 10_000

    def validate(self) -> None:
        if self.bf16_lanes not in (8, 16, 32, 64):
            raise ValueError("BF16 lanes must be 8, 16, 32, or 64")
        if self.max_cycles <= 0:
            raise ValueError("max_cycles must be positive")

    @property
    def groups(self) -> int:
        return 64 // self.bf16_lanes

    def as_dict(self) -> dict[str, int]:
        return {**asdict(self), "groups": self.groups, "a8_lanes": 64}


@dataclass(frozen=True)
class DenseOutputTile:
    tag: int
    mask: int
    accumulators: tuple[int, ...]
    biases: tuple[int, ...]
    multipliers: tuple[int, ...]
    right_shifts: tuple[int, ...]
    activation_scale_fp16: int
    weight_scales_fp16: tuple[int, ...]


@dataclass(frozen=True)
class DenseDualOutputEvent:
    cycle: int
    event: str
    tag: int
    value: int


@dataclass(frozen=True)
class DenseDualOutputResult:
    cycles: int
    events: tuple[DenseDualOutputEvent, ...]
    counters: dict[str, int]


def locked_dense_output_tiles(count: int = 3) -> tuple[DenseOutputTile, ...]:
    tiles = []
    for tile in range(count):
        accumulators = tuple(
            ((tile * 65537 + column * 7919 + 12345) % 2_000_001) - 1_000_000
            for column in range(64)
        )
        tiles.append(DenseOutputTile(
            tag=tile,
            mask=(1 << (64 if tile + 1 < count else 59)) - 1,
            accumulators=accumulators,
            biases=tuple((column % 7) - 3 for column in range(64)),
            multipliers=tuple(0x40000000 + column * 1024 for column in range(64)),
            right_shifts=tuple(31 + column % 3 for column in range(64)),
            activation_scale_fp16=0x226F,
            weight_scales_fp16=tuple(0x2000 + column * 3 for column in range(64)),
        ))
    return tuple(tiles)


def run_dense_dual_output_model(
    config: DenseDualOutputConfig,
    tiles: tuple[DenseOutputTile, ...] | None = None,
) -> DenseDualOutputResult:
    config.validate()
    tiles = tiles or locked_dense_output_tiles()
    source_index = 0
    active = None
    group_issue = 0
    responses: list[tuple[int, int]] = []
    output_register = None
    events = []
    counters = {
        "input_tiles": 0, "a8_values": 0, "bf16_groups": 0,
        "bf16_values": 0, "output_tiles": 0,
        "source_backpressure_cycles": 0, "output_backpressure_cycles": 0,
    }
    for cycle in range(config.max_cycles):
        output_ready = _ready(
            cycle, config.output_stall_mod, config.output_stall_phase,
        )
        output_fire = output_register is not None and output_ready
        if output_register is not None and not output_ready:
            counters["output_backpressure_cycles"] += 1
        source_valid = source_index < len(tiles) and _ready(
            cycle, config.source_stall_mod, config.source_stall_phase,
        )
        input_ready = active is None and output_register is None
        input_fire = source_valid and input_ready
        if source_valid and not input_ready:
            counters["source_backpressure_cycles"] += 1

        due_groups = [group for due, group in responses if due == cycle]
        responses = [(due, group) for due, group in responses if due != cycle]
        if len(due_groups) > 1:
            raise AssertionError("one BF16 group may retire per cycle")

        issue_active = active
        issue_group = group_issue
        if input_fire:
            tile = tiles[source_index]
            events.append(DenseDualOutputEvent(cycle, "INPUT_ACCEPT", tile.tag, 0))
            active = tile
            group_issue = 0
            source_index += 1
            counters["input_tiles"] += 1
        if issue_active is not None and issue_group < config.groups:
            events.append(DenseDualOutputEvent(
                cycle, "BF16_GROUP_ISSUE", issue_active.tag, issue_group,
            ))
            responses.append((cycle + 1, issue_group))
            group_issue += 1
            counters["bf16_groups"] += 1

        if due_groups:
            assert active is not None
            group = due_groups[0]
            start = group * config.bf16_lanes
            for column in range(start, start + config.bf16_lanes):
                value = dequantize_int32_to_bf16(Bf16DequantVector(
                    column, active.accumulators[column],
                    active.activation_scale_fp16,
                    active.weight_scales_fp16[column],
                ))
                events.append(DenseDualOutputEvent(
                    cycle, "BF16_VALUE", active.tag * 64 + column, value,
                ))
                counters["bf16_values"] += 1
            if group + 1 == config.groups:
                a8 = tuple(requantize_int32(RequantVector(
                    column, active.accumulators[column], active.biases[column],
                    active.multipliers[column], active.right_shifts[column],
                )) for column in range(64))
                bf16 = tuple(dequantize_int32_to_bf16(Bf16DequantVector(
                    column, active.accumulators[column],
                    active.activation_scale_fp16,
                    active.weight_scales_fp16[column],
                )) for column in range(64))
                output_register = (active, a8, bf16)
                counters["a8_values"] += 64
                active = None

        if output_fire:
            tile, a8, bf16 = output_register
            for column in range(64):
                if tile.mask & (1 << column):
                    events.append(DenseDualOutputEvent(
                        cycle, "A8_OUTPUT", tile.tag * 64 + column, a8[column],
                    ))
                    events.append(DenseDualOutputEvent(
                        cycle, "BF16_OUTPUT", tile.tag * 64 + column, bf16[column],
                    ))
            counters["output_tiles"] += 1
            output_register = None

        if source_index == len(tiles) and active is None and not responses \
                and output_register is None:
            return DenseDualOutputResult(cycle + 1, tuple(events), counters)
    raise TimeoutError("Dense dual-output model exceeded max_cycles")


def _ready(cycle: int, modulus: int, phase: int) -> bool:
    return modulus == 0 or cycle % modulus != phase
