"""Independent shape and conservation checks for the analytical estimator."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
import struct
from typing import Any

from .engines import ceil_div, packed_bytes
from .schema import HardwareConfig, SimulationResult, Workload


@dataclass(frozen=True)
class ValidationCheck:
    name: str
    passed: bool
    expected: Any
    actual: Any
    source: str


def validate_model_files(
    checkpoint_config: str | Path,
    pointbert_config: str | Path,
) -> list[ValidationCheck]:
    checkpoint_path = Path(checkpoint_config)
    pointbert_path = Path(pointbert_config)
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    pointbert_text = pointbert_path.read_text(encoding="utf-8")
    expected_checkpoint = {
        "hidden_size": 4096,
        "intermediate_size": 11008,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "vocab_size": 32003,
        "point_backbone_config_name": "PointTransformer_8192point_2layer",
    }
    expected_pointbert = {
        "trans_dim": 384,
        "depth": 12,
        "num_heads": 6,
        "group_size": 32,
        "num_group": 512,
        "encoder_dims": 256,
        "point_dims": 3,
        "projection_hidden_layer": 2,
        "npoints": 8192,
    }
    checks = [
        ValidationCheck(
            name=f"checkpoint.{key}",
            passed=checkpoint.get(key) == expected,
            expected=expected,
            actual=checkpoint.get(key),
            source=str(checkpoint_path),
        )
        for key, expected in expected_checkpoint.items()
    ]
    for key, expected in expected_pointbert.items():
        actual = _yaml_integer(pointbert_text, key)
        checks.append(ValidationCheck(
            name=f"pointbert.{key}",
            passed=actual == expected,
            expected=expected,
            actual=actual,
            source=str(pointbert_path),
        ))
    effective_dims = 6 if bool(checkpoint.get("use_color", False)) else 3
    checks.append(ValidationCheck(
        name="pointbert.effective_point_dims",
        passed=effective_dims == 6,
        expected=6,
        actual=effective_dims,
        source="checkpoint.use_color overrides PointBERT point_dims",
    ))
    index_path = checkpoint_path.parent / "model.safetensors.index.json"
    if index_path.exists():
        shapes = _safetensors_shapes(checkpoint_path.parent, index_path)
        expected_shapes = {
            "model.point_backbone.reduce_dim.weight": (384, 256),
            "model.point_backbone.pos_embed.0.weight": (128, 3),
            "model.point_backbone.pos_embed.2.weight": (384, 128),
            "model.point_proj.0.weight": (1024, 384),
            "model.point_proj.2.weight": (2048, 1024),
            "model.point_proj.4.weight": (4096, 2048),
            "model.layers.0.self_attn.q_proj.weight": (4096, 4096),
            "model.layers.0.mlp.gate_proj.weight": (11008, 4096),
            "model.layers.0.mlp.down_proj.weight": (4096, 11008),
            "lm_head.weight": (32003, 4096),
        }
        for key, expected in expected_shapes.items():
            actual = shapes.get(key)
            checks.append(ValidationCheck(
                name=f"checkpoint_tensor.{key}",
                passed=actual == expected,
                expected=expected,
                actual=actual,
                source=str(index_path),
            ))
    return checks


def validate_workload_contract(workload: Workload) -> list[ValidationCheck]:
    operations = {operation.name: operation for operation in workload.operations}
    batch = int(workload.metadata["batch_size"])
    input_tokens = int(workload.metadata["input_tokens"])
    expected_shapes = {
        "point_reduce_dim_256_384": (batch * 512, 384, 256),
        "point_position_fc_3_128": (batch * 512, 128, 3),
        "point_position_fc_128_384": (batch * 512, 384, 128),
        "point_transformer_qkv": (batch * 513, 1152, 384),
        "projector_fc0": (batch * 513, 1024, 384),
        "projector_fc1": (batch * 513, 2048, 1024),
        "projector_fc2": (batch * 513, 4096, 2048),
        "prefill_qkv": (batch * input_tokens, 12288, 4096),
        "decode_qkv": (batch, 12288, 4096),
        "decode_gate_up": (batch, 11008, 4096),
        "decode_down": (batch, 4096, 11008),
        "decode_lm_head": (batch, 32003, 4096),
    }
    checks = []
    for name, expected in expected_shapes.items():
        operation = operations.get(name)
        actual = None if operation is None else (operation.m, operation.n, operation.k)
        checks.append(ValidationCheck(
            name=f"workload.{name}.mnk",
            passed=actual == expected,
            expected=expected,
            actual=actual,
            source="PointLLM-v1.2 model contract",
        ))
    required = {
        "point_transformer_norm_residual",
        "point_transformer_gelu",
        "projector_gelu",
        "prefill_norm_rope_silu_residual",
        "decode_norm_rope_silu_residual",
        "decode_greedy_sampling",
    }
    actual_required = required.intersection(operations)
    checks.append(ValidationCheck(
        name="workload.required_vector_stages",
        passed=actual_required == required,
        expected=sorted(required),
        actual=sorted(actual_required),
        source="PointBERT and LLaMA forward graphs",
    ))
    return checks


def validate_conservation(
    config: HardwareConfig,
    workload: Workload,
    result: SimulationResult,
) -> list[ValidationCheck]:
    checks: list[ValidationCheck] = []
    operation_cycles = sum(item.cycles for item in result.operation_results)
    reconfiguration_cycles = int(result.counters.get("phase_reconfiguration_cycles", 0))
    checks.append(_check(
        "cycles.operation_plus_reconfiguration",
        operation_cycles + reconfiguration_cycles,
        result.cycles,
    ))
    checks.append(_check(
        "cycles.phase_sum",
        sum(item.cycles for item in result.phase_results),
        result.cycles,
    ))
    checks.append(_check(
        "traffic.hbm_sum",
        sum(item.hbm_read_bytes + item.hbm_write_bytes for item in result.operation_results),
        int(result.counters.get("hbm_read_bytes", 0))
        + int(result.counters.get("hbm_write_bytes", 0)),
    ))

    for operation, estimate in zip(workload.operations, result.operation_results):
        if operation.name != estimate.name:
            checks.append(_check(
                f"operation_order.{operation.name}", operation.name, estimate.name
            ))
            continue
        if operation.kind == "linear":
            expected_macs = operation.m * operation.n * operation.k * operation.repeat
            checks.append(_check(f"macs.{operation.name}", expected_macs, estimate.macs))
            weight_elements = operation.n * operation.k * operation.repeat
            checks.append(_check(
                f"weight_payload.{operation.name}",
                packed_bytes(weight_elements, operation.weight_bits),
                int(estimate.details["weight_payload_bytes"]),
            ))
            packing = config.tensor.precision_macs_per_cycle[operation.weight_bits]
        elif operation.kind == "attention":
            expected_macs = (
                2 * operation.repeat * operation.batch * operation.q_len
                * operation.kv_len * operation.hidden
            )
            checks.append(_check(f"macs.{operation.name}", expected_macs, estimate.macs))
            packing = config.tensor.precision_macs_per_cycle[operation.activation_bits]
        else:
            continue
        lower_bound = ceil_div(
            estimate.macs,
            config.tensor.rows * config.tensor.cols * packing,
        )
        checks.append(ValidationCheck(
            name=f"compute_lower_bound.{operation.name}",
            passed=estimate.compute_cycles >= lower_bound,
            expected=f">={lower_bound}",
            actual=estimate.compute_cycles,
            source="MAC conservation / configured peak throughput",
        ))
    return checks


def validation_report(
    config: HardwareConfig,
    workload: Workload,
    result: SimulationResult,
    *,
    checkpoint_config: str | Path,
    pointbert_config: str | Path,
) -> dict[str, Any]:
    model_checks = validate_model_files(checkpoint_config, pointbert_config)
    workload_checks = validate_workload_contract(workload)
    conservation_checks = validate_conservation(config, workload, result)
    checks = model_checks + workload_checks + conservation_checks
    passed = sum(check.passed for check in checks)
    invariant_status = passed == len(checks)
    shape_status = all(check.passed for check in model_checks + workload_checks)
    conservation_status = all(check.passed for check in conservation_checks)
    return {
        "schema_version": "0.1",
        "status": "analytical_invariants_passed" if invariant_status else "failed",
        "checks_passed": passed,
        "checks_total": len(checks),
        "maturity": {
            "model_shape_contract": shape_status,
            "analytical_conservation": conservation_status,
            "event_level_cycle_model": False,
            "rtl_implementation": False,
            "rtl_cycle_correlation": False,
            "area_energy_characterized": False,
            "publishable_performance_claim": False,
        },
        "checks": [asdict(check) for check in checks],
    }


def _yaml_integer(text: str, key: str) -> int | None:
    match = re.search(rf"(?:^|\s){re.escape(key)}\s*:\s*([0-9]+)", text)
    return int(match.group(1)) if match else None


def _safetensors_shapes(root: Path, index_path: Path) -> dict[str, tuple[int, ...]]:
    index = json.loads(index_path.read_text(encoding="utf-8"))
    required = set(index.get("weight_map", {}))
    shapes: dict[str, tuple[int, ...]] = {}
    for shard in sorted(set(index.get("weight_map", {}).values())):
        shard_path = root / shard
        with shard_path.open("rb") as handle:
            header_size = struct.unpack("<Q", handle.read(8))[0]
            header = json.loads(handle.read(header_size).decode("utf-8"))
        for key, value in header.items():
            if key in required and key != "__metadata__":
                shapes[key] = tuple(int(item) for item in value["shape"])
    return shapes


def _check(name: str, expected: Any, actual: Any) -> ValidationCheck:
    return ValidationCheck(
        name=name,
        passed=expected == actual,
        expected=expected,
        actual=actual,
        source="analytical conservation invariant",
    )
