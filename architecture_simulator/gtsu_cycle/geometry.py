"""Functional and edge-state model for the RTL-locked geometry distance tile."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path


@dataclass(frozen=True)
class GeometryConfig:
    lanes: int = 8
    coordinate_width: int = 16
    distance_width: int = 36
    tag_width: int = 16
    tiles: int = 7
    source_stall_mod: int = 4
    source_stall_phase: int = 1
    output_stall_mod: int = 5
    output_stall_phase: int = 2
    max_cycles: int = 10_000

    def validate(self) -> None:
        if min(self.lanes, self.coordinate_width, self.distance_width,
               self.tag_width, self.tiles, self.max_cycles) <= 0:
            raise ValueError("geometry dimensions and max_cycles must be positive")
        required_distance_width = 2 * self.coordinate_width + 4
        if self.distance_width < required_distance_width:
            raise ValueError(
                f"distance_width must be at least {required_distance_width} "
                "for an overflow-free three-coordinate squared distance"
            )
        for name, modulus, phase in (
            ("source", self.source_stall_mod, self.source_stall_phase),
            ("output", self.output_stall_mod, self.output_stall_phase),
        ):
            if modulus < 0 or phase < 0:
                raise ValueError(f"{name} stall modulus/phase must be nonnegative")
            if modulus and phase >= modulus:
                raise ValueError(f"{name}_stall_phase must be smaller than its modulus")

    def as_dict(self) -> dict[str, int]:
        return asdict(self)

    @classmethod
    def load(cls, path: str | Path) -> "GeometryConfig":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        payload.pop("schema_version", None)
        payload.pop("operator", None)
        return cls(**payload)


@dataclass(frozen=True)
class GeometryBeat:
    tag: int
    query: tuple[int, int, int]
    points: tuple[tuple[int, int, int], ...]


@dataclass(frozen=True)
class GeometryEvent:
    cycle: int
    event: str
    tag: int
    lane: int
    value: int


@dataclass(frozen=True)
class GeometryOutput:
    tag: int
    lane: int
    distance: int


@dataclass(frozen=True)
class GeometryResult:
    config: GeometryConfig
    cycles: int
    events: tuple[GeometryEvent, ...]
    outputs: tuple[GeometryOutput, ...]
    counters: dict[str, int]


def squared_distance(
    query: tuple[int, int, int], point: tuple[int, int, int],
) -> int:
    return sum((left - right) ** 2 for left, right in zip(query, point, strict=True))


def build_geometry_beats(config: GeometryConfig) -> tuple[GeometryBeat, ...]:
    """Build deterministic signed-coordinate traffic for RTL regression."""
    config.validate()
    coordinate_limit = (1 << (config.coordinate_width - 1)) - 1
    beats = []
    for tag in range(config.tiles):
        query = (tag * 7 - 19, 23 - tag * 5, tag * 3 - 11)
        points = tuple(
            (
                lane * 11 - tag * 2 - 31,
                tag * 13 - lane * 7 + 17,
                lane * lane - tag * 9 - 5,
            )
            for lane in range(config.lanes)
        )
        values = (*query, *(value for point in points for value in point))
        if any(abs(value) > coordinate_limit for value in values):
            raise ValueError("generated coordinate exceeds configured signed width")
        beats.append(GeometryBeat(tag=tag, query=query, points=points))
    return tuple(beats)


def functional_outputs(
    beats: tuple[GeometryBeat, ...],
) -> tuple[GeometryOutput, ...]:
    return tuple(
        GeometryOutput(beat.tag, lane, squared_distance(beat.query, point))
        for beat in beats
        for lane, point in enumerate(beat.points)
    )


def pointllm_geometry_mapping(
    lanes: int, *, points: int = 8192, centers: int = 512, neighbors: int = 32,
) -> dict[str, int | str]:
    """Return exact work counts, not a complete FPS/KNN cycle prediction."""
    if min(lanes, points, centers, neighbors) <= 0:
        raise ValueError("PointLLM geometry dimensions must be positive")
    tiles_per_center = math.ceil(points / lanes)
    distance_evaluations = points * centers
    return {
        "points": points,
        "centers": centers,
        "neighbors": neighbors,
        "lanes": lanes,
        "distance_evaluations_per_operator": distance_evaluations,
        "tiles_per_center": tiles_per_center,
        "distance_tiles_per_operator": tiles_per_center * centers,
        "point_coordinate_sram_bytes_int16": points * 3 * 2,
        "fps_min_state_packed_bytes": math.ceil(points * 36 / 8),
        "status": "distance_issue_lower_bound_only",
    }


def run_geometry_model(
    config: GeometryConfig,
    beats: tuple[GeometryBeat, ...] | None = None,
) -> GeometryResult:
    """Advance the one-entry elastic RTL output register edge by edge."""
    config.validate()
    beats = beats or build_geometry_beats(config)
    if len(beats) != config.tiles:
        raise ValueError("beat count must equal config.tiles")

    cycle = 0
    source_index = 0
    output_register: GeometryBeat | None = None
    events: list[GeometryEvent] = []
    outputs: list[GeometryOutput] = []
    counters = {
        "input_tiles": 0,
        "output_points": 0,
        "source_backpressure_cycles": 0,
        "output_backpressure_cycles": 0,
    }
    while source_index < len(beats) or output_register is not None:
        if cycle >= config.max_cycles:
            raise RuntimeError("geometry cycle model exceeded max_cycles")
        source_valid = (
            source_index < len(beats)
            and _periodic_ready(cycle, config.source_stall_mod,
                                config.source_stall_phase)
        )
        output_ready = _periodic_ready(
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
            events.append(GeometryEvent(cycle, "INPUT_ACCEPT", incoming.tag, -1, 0))
            counters["input_tiles"] += 1
            source_index += 1
        if accept_output:
            assert output_register is not None
            for lane, point in enumerate(output_register.points):
                value = squared_distance(output_register.query, point)
                events.append(GeometryEvent(
                    cycle, "OUTPUT_ACCEPT", output_register.tag, lane, value,
                ))
                outputs.append(GeometryOutput(output_register.tag, lane, value))
                counters["output_points"] += 1

        if incoming is not None:
            output_register = incoming
        elif accept_output:
            output_register = None
        cycle += 1

    expected = functional_outputs(beats)
    if tuple(outputs) != expected:
        raise AssertionError("geometry functional and cycle outputs diverged")
    return GeometryResult(
        config=config,
        cycles=cycle,
        events=tuple(events),
        outputs=tuple(outputs),
        counters=counters,
    )


def _periodic_ready(cycle: int, modulus: int, phase: int) -> bool:
    return modulus == 0 or cycle % modulus != phase
