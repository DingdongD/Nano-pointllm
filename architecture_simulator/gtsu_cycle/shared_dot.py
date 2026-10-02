"""Bit-exact edge model for the shared INT8 SIMD4 feature/geometry fabric."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class SharedDotConfig:
    pe_count: int = 8
    norm_width: int = 16
    distance_width: int = 18
    tag_width: int = 16
    beats: int = 9
    source_stall_mod: int = 4
    source_stall_phase: int = 1
    output_stall_mod: int = 5
    output_stall_phase: int = 2
    max_cycles: int = 10_000

    def validate(self) -> None:
        if min(self.pe_count, self.norm_width, self.distance_width,
               self.tag_width, self.beats, self.max_cycles) <= 0:
            raise ValueError("shared-dot dimensions and max_cycles must be positive")
        if self.norm_width < 16:
            raise ValueError("norm_width must cover three signed INT8 squares")
        if self.distance_width < 18:
            raise ValueError("distance_width must cover signed INT8 3-D distance")
        for name, modulus, phase in (
            ("source", self.source_stall_mod, self.source_stall_phase),
            ("output", self.output_stall_mod, self.output_stall_phase),
        ):
            if modulus < 0 or phase < 0 or (modulus and phase >= modulus):
                raise ValueError(f"invalid {name} stall modulus/phase")

    def as_dict(self) -> dict[str, int]:
        return asdict(self)

    @classmethod
    def load(cls, path: str | Path) -> "SharedDotConfig":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        payload.pop("schema_version", None)
        payload.pop("operator", None)
        return cls(**payload)


@dataclass(frozen=True)
class SharedDotLane:
    a: tuple[int, int, int, int]
    b: tuple[int, int, int, int]
    a_norm: int
    b_norm: int


@dataclass(frozen=True)
class SharedDotBeat:
    tag: int
    geometry: bool
    lanes: tuple[SharedDotLane, ...]


@dataclass(frozen=True)
class SharedDotEvent:
    cycle: int
    event: str
    tag: int
    lane: int
    dot: int
    distance: int


@dataclass(frozen=True)
class SharedDotOutput:
    tag: int
    lane: int
    geometry: bool
    dot: int
    distance: int


@dataclass(frozen=True)
class SharedDotResult:
    config: SharedDotConfig
    cycles: int
    events: tuple[SharedDotEvent, ...]
    outputs: tuple[SharedDotOutput, ...]
    counters: dict[str, int]


def dot4(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    return sum(left * right for left, right in zip(a, b, strict=True))


def build_shared_dot_beats(config: SharedDotConfig) -> tuple[SharedDotBeat, ...]:
    config.validate()
    result = []
    for tag in range(config.beats):
        geometry = tag % 2 == 1
        lanes = []
        for lane in range(config.pe_count):
            if geometry:
                a = (_i8(tag * 7 - 61), _i8(lane * 9 - 57), _i8(tag - lane * 3), 0)
                b = (_i8(lane * 5 - 63), _i8(tag * 11 - 49), _i8(lane - tag * 4), 0)
                a_norm = dot4(a, a)
                b_norm = dot4(b, b)
            else:
                a = tuple(_i8(tag * 13 + lane * 7 + index * 19 - 101) for index in range(4))
                b = tuple(_i8(tag * 5 - lane * 11 + index * 23 - 89) for index in range(4))
                a_norm = 0
                b_norm = 0
            lanes.append(SharedDotLane(a, b, a_norm, b_norm))
        result.append(SharedDotBeat(tag, geometry, tuple(lanes)))
    return tuple(result)


def functional_outputs(beats: tuple[SharedDotBeat, ...]) -> tuple[SharedDotOutput, ...]:
    outputs = []
    for beat in beats:
        for lane_index, lane in enumerate(beat.lanes):
            dot = dot4(lane.a, lane.b)
            distance = lane.a_norm + lane.b_norm - 2 * dot if beat.geometry else 0
            outputs.append(SharedDotOutput(
                beat.tag, lane_index, beat.geometry, dot, distance,
            ))
    return tuple(outputs)


def run_shared_dot_model(
    config: SharedDotConfig,
    beats: tuple[SharedDotBeat, ...] | None = None,
) -> SharedDotResult:
    config.validate()
    beats = beats or build_shared_dot_beats(config)
    if len(beats) != config.beats:
        raise ValueError("beat count must equal config.beats")
    if any(len(beat.lanes) != config.pe_count for beat in beats):
        raise ValueError("every beat must contain config.pe_count lanes")

    cycle = 0
    source_index = 0
    output_register: SharedDotBeat | None = None
    events: list[SharedDotEvent] = []
    outputs: list[SharedDotOutput] = []
    counters = {
        "input_beats": 0,
        "output_lanes": 0,
        "feature_lanes": 0,
        "geometry_lanes": 0,
        "source_backpressure_cycles": 0,
        "output_backpressure_cycles": 0,
    }
    while source_index < len(beats) or output_register is not None:
        if cycle >= config.max_cycles:
            raise RuntimeError("shared-dot model exceeded max_cycles")
        source_valid = source_index < len(beats) and _ready(
            cycle, config.source_stall_mod, config.source_stall_phase,
        )
        output_ready = _ready(
            cycle, config.output_stall_mod, config.output_stall_phase,
        )
        input_ready = output_register is None or output_ready
        accept_input = source_valid and input_ready
        accept_output = output_register is not None and output_ready
        if source_valid and not input_ready:
            counters["source_backpressure_cycles"] += 1
        if output_register is not None and not output_ready:
            counters["output_backpressure_cycles"] += 1

        incoming = beats[source_index] if accept_input else None
        if incoming is not None:
            events.append(SharedDotEvent(cycle, "INPUT_ACCEPT", incoming.tag, -1, 0, 0))
            counters["input_beats"] += 1
            source_index += 1
        if accept_output:
            assert output_register is not None
            for lane_index, lane in enumerate(output_register.lanes):
                dot = dot4(lane.a, lane.b)
                distance = (
                    lane.a_norm + lane.b_norm - 2 * dot
                    if output_register.geometry else 0
                )
                events.append(SharedDotEvent(
                    cycle, "OUTPUT_ACCEPT", output_register.tag,
                    lane_index, dot, distance,
                ))
                outputs.append(SharedDotOutput(
                    output_register.tag, lane_index, output_register.geometry,
                    dot, distance,
                ))
                counters["output_lanes"] += 1
                key = "geometry_lanes" if output_register.geometry else "feature_lanes"
                counters[key] += 1

        if incoming is not None:
            output_register = incoming
        elif accept_output:
            output_register = None
        cycle += 1

    if tuple(outputs) != functional_outputs(beats):
        raise AssertionError("shared-dot functional and cycle outputs diverged")
    return SharedDotResult(config, cycle, tuple(events), tuple(outputs), counters)


def pointllm_shared_geometry_mapping(
    pe_count: int, *, points: int = 8192, centers: int = 512,
) -> dict[str, int | str]:
    if min(pe_count, points, centers) <= 0:
        raise ValueError("mapping dimensions must be positive")
    evaluations = points * centers
    return {
        "points": points,
        "centers": centers,
        "distance_evaluations": evaluations,
        "distance_dot_cycles": (evaluations + pe_count - 1) // pe_count,
        "point_norm_precompute_cycles": (points + pe_count - 1) // pe_count,
        "point_norm_sram_bytes": points * 2,
        "arithmetic_contract": "cached_norms_plus_shared_int8_dot4",
        "int16_fallback_partial_products_per_dot": 4,
        "int16_fallback_distance_dot_cycles": 4 * ((evaluations + pe_count - 1) // pe_count),
        "int16_fallback_status": "arithmetic_mapping_only_not_scheduled_or_rtl_correlated",
        "status": "arithmetic_issue_lower_bound_only",
    }


def signed_int16_product_via_int8(left: int, right: int) -> int:
    """Exact four-product decomposition used to bound an INT16 fallback."""
    if not -(1 << 15) <= left < (1 << 15):
        raise ValueError("left operand does not fit signed INT16")
    if not -(1 << 15) <= right < (1 << 15):
        raise ValueError("right operand does not fit signed INT16")
    left_low = (left & 0xFF) - 128
    right_low = (right & 0xFF) - 128
    left_high = left >> 8
    right_high = right >> 8
    # x = low_signed + 128 + 256 * high_signed. All four variable
    # products are signed INT8 multiplies; shifts/additions apply afterward.
    low_low = left_low * right_low
    low_high = left_low * right_high
    high_low = left_high * right_low
    high_high = left_high * right_high
    return (
        low_low
        + 128 * (left_low + right_low)
        + 128 * 128
        + 256 * (low_high + high_low)
        + 256 * 128 * (left_high + right_high)
        + 65536 * high_high
    )


def _i8(value: int) -> int:
    return ((value + 128) % 256) - 128


def _ready(cycle: int, modulus: int, phase: int) -> bool:
    return modulus == 0 or cycle % modulus != phase
