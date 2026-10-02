"""Exact cycle model for fused FPS min-state and per-center KNN selection."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class FusedFpsKnnConfig:
    points: int = 16
    centers: int = 5
    neighbors: int = 4
    lanes: int = 4
    distance_width: int = 18
    source_stall_mod: int = 5
    source_stall_phase: int = 2
    output_stall_mod: int = 7
    output_stall_phase: int = 3
    max_cycles: int = 100_000

    def validate(self) -> None:
        values = (self.points, self.centers, self.neighbors, self.lanes,
                  self.distance_width, self.max_cycles)
        if min(values) <= 0:
            raise ValueError("FPS/KNN dimensions must be positive")
        if self.centers > self.points or self.neighbors > self.points:
            raise ValueError("centers and neighbors cannot exceed points")
        if self.distance_width < 18:
            raise ValueError("distance_width must cover INT8 3-D squared distance")
        for name, modulus, phase in (
            ("source", self.source_stall_mod, self.source_stall_phase),
            ("output", self.output_stall_mod, self.output_stall_phase),
        ):
            if modulus < 0 or phase < 0 or (modulus and phase >= modulus):
                raise ValueError(f"invalid {name} stall modulus/phase")

    @property
    def tiles(self) -> int:
        return math.ceil(self.points / self.lanes)

    def as_dict(self) -> dict[str, int]:
        return {**asdict(self), "tiles": self.tiles}


@dataclass(frozen=True)
class FusedRoundOutput:
    round_index: int
    center: int
    next_center: int
    neighbor_indices: tuple[int, ...]
    neighbor_distances: tuple[int, ...]


@dataclass(frozen=True)
class FusedFpsKnnEvent:
    cycle: int
    event: str
    round_index: int
    item: int
    value: int


@dataclass(frozen=True)
class FusedFpsKnnResult:
    cycles: int
    events: tuple[FusedFpsKnnEvent, ...]
    outputs: tuple[FusedRoundOutput, ...]
    counters: dict[str, int]


def point_coordinate(index: int) -> tuple[int, int, int]:
    """Deterministic INT8 cloud with one duplicate point to lock tie semantics."""
    if index == 1:
        index = 0
    return (
        (index * 17 + 3) % 31 - 15,
        (index * index * 5 + 7) % 29 - 14,
        (index * 11 + 5) % 23 - 11,
    )


def point_distance(left: int, right: int) -> int:
    a = point_coordinate(left)
    b = point_coordinate(right)
    return sum((x - y) ** 2 for x, y in zip(a, b, strict=True))


def golden_fps_knn(config: FusedFpsKnnConfig) -> tuple[FusedRoundOutput, ...]:
    config.validate()
    nearest = [None] * config.points
    center = 0
    outputs = []
    for round_index in range(config.centers):
        distances = [point_distance(center, point) for point in range(config.points)]
        nearest = [
            distance if old is None else min(old, distance)
            for old, distance in zip(nearest, distances, strict=True)
        ]
        next_center = min(range(config.points), key=lambda i: (-nearest[i], i))
        neighbors = sorted(range(config.points), key=lambda i: (distances[i], i))[
            :config.neighbors
        ]
        outputs.append(FusedRoundOutput(
            round_index, center, next_center, tuple(neighbors),
            tuple(distances[index] for index in neighbors),
        ))
        center = next_center
    return tuple(outputs)


def distance_beats(config: FusedFpsKnnConfig) -> tuple[tuple[int, ...], ...]:
    """Flatten round-major, tile-major distance beats from the golden centers."""
    beats = []
    for output in golden_fps_knn(config):
        for tile in range(config.tiles):
            beats.append(tuple(
                point_distance(output.center, tile * config.lanes + lane)
                if tile * config.lanes + lane < config.points else 0
                for lane in range(config.lanes)
            ))
    return tuple(beats)


def run_fused_fps_knn_model(config: FusedFpsKnnConfig) -> FusedFpsKnnResult:
    config.validate()
    nearest = [None] * config.points
    center = 0
    round_index = tile = cycle = 0
    top_candidates: list[tuple[int, int]] = []
    farthest = (0, 0)
    output_register: FusedRoundOutput | None = None
    events = []
    outputs = []
    counters = {
        "input_beats": 0, "distance_values": 0, "round_outputs": 0,
        "min_state_writes": 0, "source_backpressure_cycles": 0,
        "output_backpressure_cycles": 0,
    }
    while round_index < config.centers or output_register is not None:
        if cycle >= config.max_cycles:
            raise TimeoutError("fused FPS/KNN model exceeded max_cycles")
        output_ready = _ready(cycle, config.output_stall_mod, config.output_stall_phase)
        output_fire = output_register is not None and output_ready
        if output_register is not None and not output_ready:
            counters["output_backpressure_cycles"] += 1
        source_valid = counters["input_beats"] < config.centers * config.tiles and _ready(
            cycle, config.source_stall_mod, config.source_stall_phase,
        )
        input_ready = output_register is None
        input_fire = source_valid and input_ready
        if source_valid and not input_ready:
            counters["source_backpressure_cycles"] += 1

        if output_fire:
            assert output_register is not None
            output = output_register
            events.append(FusedFpsKnnEvent(
                cycle, "ROUND_OUTPUT", output.round_index,
                output.center, output.next_center,
            ))
            for slot, (index, distance) in enumerate(zip(
                output.neighbor_indices, output.neighbor_distances, strict=True,
            )):
                events.append(FusedFpsKnnEvent(
                    cycle, "NEIGHBOR", output.round_index, slot,
                    (index << config.distance_width) | distance,
                ))
            outputs.append(output)
            counters["round_outputs"] += 1
            center = output.next_center
            round_index += 1
            tile = 0
            top_candidates = []
            farthest = (0, 0)
            output_register = None

        if input_fire:
            events.append(FusedFpsKnnEvent(
                cycle, "INPUT_BEAT", round_index, tile, 0,
            ))
            for lane in range(config.lanes):
                point = tile * config.lanes + lane
                if point >= config.points:
                    continue
                distance = point_distance(center, point)
                updated = distance if nearest[point] is None else min(
                    nearest[point], distance,
                )
                nearest[point] = updated
                counters["distance_values"] += 1
                counters["min_state_writes"] += 1
                if updated > farthest[0] or (
                    updated == farthest[0] and point < farthest[1]
                ):
                    farthest = (updated, point)
                top_candidates.append((distance, point))
            counters["input_beats"] += 1
            if tile + 1 == config.tiles:
                ordered = sorted(top_candidates)[:config.neighbors]
                output_register = FusedRoundOutput(
                    round_index, center, farthest[1],
                    tuple(index for _, index in ordered),
                    tuple(distance for distance, _ in ordered),
                )
            else:
                tile += 1
        cycle += 1

    result = FusedFpsKnnResult(cycle, tuple(events), tuple(outputs), counters)
    if result.outputs != golden_fps_knn(config):
        raise AssertionError("cycle model diverged from FPS/KNN functional golden")
    return result


def _ready(cycle: int, modulus: int, phase: int) -> bool:
    return modulus == 0 or cycle % modulus != phase
