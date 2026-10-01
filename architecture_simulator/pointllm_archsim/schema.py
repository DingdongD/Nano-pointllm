"""Stable JSON contracts shared by the analytical model and future RTL/C-models."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any


SUPPORTED_PHASES = (
    "geometry",
    "point_encoder",
    "projector",
    "llm_prefill",
    "llm_decode",
)
SUPPORTED_KINDS = ("fps", "knn", "linear", "attention", "vector")
SUPPORTED_PRECISIONS = (4, 8, 16)


@dataclass(frozen=True)
class TensorFabricConfig:
    rows: int = 32
    cols: int = 32
    split_k: int = 32
    decode_split_k_threshold: int = 8
    reduction_stage_cycles: int = 1
    pipeline_fill_cycles: int = 2
    precision_macs_per_cycle: dict[int, int] = field(
        default_factory=lambda: {4: 4, 8: 2, 16: 1}
    )

    def validate(self) -> None:
        _positive(self, "rows", "cols", "split_k", "decode_split_k_threshold")
        _positive(self, "reduction_stage_cycles", "pipeline_fill_cycles")
        for bits in SUPPORTED_PRECISIONS:
            if int(self.precision_macs_per_cycle.get(bits, 0)) <= 0:
                raise ValueError(f"missing positive packing factor for W{bits}")


@dataclass(frozen=True)
class GeometryConfig:
    distance_lanes: int = 256
    distance_pipeline_cycles: int = 1
    reduction_stage_cycles: int = 1
    feedback_cycles: int = 2
    topk_lanes: int = 128
    topk_compare_cycles: int = 1

    def validate(self) -> None:
        _positive(
            self,
            "distance_lanes",
            "distance_pipeline_cycles",
            "reduction_stage_cycles",
            "feedback_cycles",
            "topk_lanes",
            "topk_compare_cycles",
        )


@dataclass(frozen=True)
class MemoryConfig:
    hbm_bandwidth_gbps: float = 819.2
    hbm_efficiency: float = 0.80
    sram_bytes: int = 32 * 1024 * 1024
    sram_banks: int = 32
    sram_bytes_per_bank_cycle: int = 64
    sram_efficiency: float = 0.85
    weight_buffer_bytes: int = 8 * 1024 * 1024
    weight_tile_bytes: int = 256 * 1024
    weight_unpack_bytes_per_cycle: int = 256
    weight_scale_group: int = 128
    weight_scale_bytes: int = 2
    phase_partitions: dict[str, dict[str, float]] = field(
        default_factory=lambda: {
            "geometry": {"point": 0.50, "tensor": 0.30, "weight": 0.00, "kv": 0.00},
            "point_encoder": {"point": 0.10, "tensor": 0.65, "weight": 0.20, "kv": 0.00},
            "projector": {"point": 0.05, "tensor": 0.65, "weight": 0.25, "kv": 0.00},
            "llm_prefill": {"point": 0.00, "tensor": 0.55, "weight": 0.25, "kv": 0.15},
            "llm_decode": {"point": 0.00, "tensor": 0.10, "weight": 0.65, "kv": 0.20},
        }
    )

    def validate(self) -> None:
        _positive(
            self,
            "hbm_bandwidth_gbps",
            "sram_bytes",
            "sram_banks",
            "sram_bytes_per_bank_cycle",
            "weight_buffer_bytes",
            "weight_tile_bytes",
            "weight_unpack_bytes_per_cycle",
            "weight_scale_group",
            "weight_scale_bytes",
        )
        _fraction(self.hbm_efficiency, "hbm_efficiency")
        _fraction(self.sram_efficiency, "sram_efficiency")
        for phase in SUPPORTED_PHASES:
            if phase not in self.phase_partitions:
                raise ValueError(f"missing SRAM partition for phase {phase!r}")
            values = self.phase_partitions[phase]
            if any(value < 0 for value in values.values()):
                raise ValueError(f"negative SRAM partition in phase {phase!r}")
            if sum(values.values()) > 1.0 + 1e-9:
                raise ValueError(f"SRAM partitions exceed capacity in phase {phase!r}")

    @property
    def sram_bytes_per_cycle(self) -> float:
        return self.sram_banks * self.sram_bytes_per_bank_cycle * self.sram_efficiency

    def partition_bytes(self, phase: str, role: str) -> int:
        return int(self.sram_bytes * self.phase_partitions[phase].get(role, 0.0))


@dataclass(frozen=True)
class AttentionConfig:
    fused_online_softmax: bool = True
    parallelize_heads: bool = True
    softmax_lanes: int = 128
    softmax_passes: float = 1.5
    paged_kv_efficiency: float = 0.85

    def validate(self) -> None:
        _positive(self, "softmax_lanes", "softmax_passes")
        _fraction(self.paged_kv_efficiency, "paged_kv_efficiency")


@dataclass(frozen=True)
class VectorConfig:
    lanes: int = 128
    operations_per_lane_cycle: int = 1

    def validate(self) -> None:
        _positive(self, "lanes", "operations_per_lane_cycle")


@dataclass(frozen=True)
class CharacterizationConfig:
    source: str = "uncharacterized analytical model"
    mac_energy_pj: dict[int, float] = field(default_factory=dict)
    hbm_energy_pj_per_byte: float | None = None
    sram_energy_pj_per_byte: float | None = None
    block_area_um2: dict[str, float] = field(default_factory=dict)

    @property
    def energy_enabled(self) -> bool:
        return bool(
            self.mac_energy_pj
            and self.hbm_energy_pj_per_byte is not None
            and self.sram_energy_pj_per_byte is not None
        )


@dataclass(frozen=True)
class HardwareConfig:
    name: str = "gtsu_baseline"
    schema_version: str = "0.1"
    clock_hz: int = 1_000_000_000
    phase_reconfiguration_cycles: int = 64
    tensor: TensorFabricConfig = field(default_factory=TensorFabricConfig)
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    attention: AttentionConfig = field(default_factory=AttentionConfig)
    vector: VectorConfig = field(default_factory=VectorConfig)
    characterization: CharacterizationConfig = field(default_factory=CharacterizationConfig)
    calibration_scales: dict[str, float] = field(default_factory=dict)

    def validate(self) -> None:
        _positive(self, "clock_hz", "phase_reconfiguration_cycles")
        self.tensor.validate()
        self.geometry.validate()
        self.memory.validate()
        self.attention.validate()
        self.vector.validate()
        if any(value <= 0 for value in self.calibration_scales.values()):
            raise ValueError("calibration scales must be positive")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "HardwareConfig":
        config = cls(
            name=str(value.get("name", "gtsu_baseline")),
            schema_version=str(value.get("schema_version", "0.1")),
            clock_hz=int(value.get("clock_hz", 1_000_000_000)),
            phase_reconfiguration_cycles=int(value.get("phase_reconfiguration_cycles", 64)),
            tensor=_nested(TensorFabricConfig, value.get("tensor", {}), "precision_macs_per_cycle"),
            geometry=_nested(GeometryConfig, value.get("geometry", {})),
            memory=_nested(MemoryConfig, value.get("memory", {})),
            attention=_nested(AttentionConfig, value.get("attention", {})),
            vector=_nested(VectorConfig, value.get("vector", {})),
            characterization=_nested(
                CharacterizationConfig,
                value.get("characterization", {}),
                "mac_energy_pj",
            ),
            calibration_scales={
                str(key): float(item)
                for key, item in value.get("calibration_scales", {}).items()
            },
        )
        config.validate()
        return config

    @classmethod
    def load(cls, path: str | Path) -> "HardwareConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def to_dict(self) -> dict[str, Any]:
        return _stringify_int_keys(asdict(self))


@dataclass(frozen=True)
class Operation:
    name: str
    phase: str
    kind: str
    repeat: int = 1
    batch: int = 1
    m: int = 0
    n: int = 0
    k: int = 0
    q_len: int = 0
    kv_len: int = 0
    heads: int = 0
    hidden: int = 0
    elements: int = 0
    weight_bits: int = 16
    activation_bits: int = 16
    attributes: dict[str, Any] = field(default_factory=dict)
    reference: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.name:
            raise ValueError("operation name cannot be empty")
        if self.phase not in SUPPORTED_PHASES:
            raise ValueError(f"unsupported phase {self.phase!r}")
        if self.kind not in SUPPORTED_KINDS:
            raise ValueError(f"unsupported operation kind {self.kind!r}")
        if self.repeat <= 0 or self.batch <= 0:
            raise ValueError("repeat and batch must be positive")
        if self.weight_bits not in SUPPORTED_PRECISIONS:
            raise ValueError(f"unsupported weight precision W{self.weight_bits}")
        if self.activation_bits not in SUPPORTED_PRECISIONS:
            raise ValueError(f"unsupported activation precision A{self.activation_bits}")
        if self.kind == "linear" and min(self.m, self.n, self.k) <= 0:
            raise ValueError(f"linear operation {self.name!r} needs positive M/N/K")
        if self.kind == "attention" and min(
            self.q_len, self.kv_len, self.heads, self.hidden
        ) <= 0:
            raise ValueError(f"attention operation {self.name!r} has invalid shape")
        if self.kind == "vector" and self.elements <= 0:
            raise ValueError(f"vector operation {self.name!r} needs elements")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Operation":
        operation = cls(**value)
        operation.validate()
        return operation


@dataclass(frozen=True)
class Workload:
    name: str
    operations: tuple[Operation, ...]
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = "0.1"

    def validate(self) -> None:
        if not self.operations:
            raise ValueError("workload has no operations")
        for operation in self.operations:
            operation.validate()

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Workload":
        workload = cls(
            name=str(value["name"]),
            operations=tuple(Operation.from_dict(item) for item in value["operations"]),
            metadata=dict(value.get("metadata", {})),
            schema_version=str(value.get("schema_version", "0.1")),
        )
        workload.validate()
        return workload

    @classmethod
    def load(cls, path: str | Path) -> "Workload":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")


@dataclass(frozen=True)
class OperationResult:
    name: str
    phase: str
    kind: str
    cycles: int
    compute_cycles: int
    memory_cycles: int
    sram_cycles: int
    overhead_cycles: int
    macs: int
    flops: int
    hbm_read_bytes: int
    hbm_write_bytes: int
    sram_bytes: int
    utilization: float
    bottleneck: str
    dataflow: str
    energy_pj: float | None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PhaseResult:
    phase: str
    cycles: int
    seconds: float
    hbm_bytes: int
    sram_bytes: int
    macs: int
    energy_pj: float | None
    operation_count: int


@dataclass(frozen=True)
class SimulationResult:
    config: dict[str, Any]
    workload: dict[str, Any]
    cycles: int
    seconds: float
    throughput_tokens_per_second: float | None
    phase_results: tuple[PhaseResult, ...]
    operation_results: tuple[OperationResult, ...]
    counters: dict[str, int | float]
    energy_pj: float | None
    area_um2: float | None
    methodology: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")


def _positive(obj: Any, *names: str) -> None:
    for name in names:
        if getattr(obj, name) <= 0:
            raise ValueError(f"{name} must be positive")


def _fraction(value: float, name: str) -> None:
    if not 0 < value <= 1:
        raise ValueError(f"{name} must be in (0, 1]")


def _nested(cls, value: dict[str, Any], int_key_field: str | None = None):
    copied = dict(value)
    if int_key_field and int_key_field in copied:
        copied[int_key_field] = {
            int(key): item for key, item in copied[int_key_field].items()
        }
    return cls(**copied)


def _stringify_int_keys(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _stringify_int_keys(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_stringify_int_keys(item) for item in value]
    return value
