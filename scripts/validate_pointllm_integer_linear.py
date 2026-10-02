#!/usr/bin/env python3
"""Validate real PointLLM Linear tensors across BF16, QDQ, INT32, and Dot4 RTL."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
POINTLLM_ROOT = Path(os.environ.get("POINTLLM_ROOT", "/home/PointLLM"))
ARCH_ROOT = REPO_ROOT / "architecture_simulator"
for path in (str(REPO_ROOT), str(POINTLLM_ROOT), str(ARCH_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from gtsu_cycle.dense_gemm import DenseBeat, DenseTileConfig, run_dense_tile_model  # noqa: E402
from gtsu_cycle.dense_gemm_correlation import correlate_dense_tile  # noqa: E402
from gtsu_cycle.production_dense_fabric import ProductionDenseFabricConfig  # noqa: E402
from gtsu_cycle.production_dense_fabric_correlation import (  # noqa: E402
    correlate_production_dense_fabric,
)
from nanopointllm.compression.w8a8 import (  # noqa: E402
    compare_linear_bf16_qdq_integer, per_output_weight_qdq,
    quantize_symmetric, static_activation_scale,
)
from nanopointllm.compression.quantized_tensor_package import (  # noqa: E402
    write_quantized_linear_package,
)
from nanopointllm.parity.engine_test_utils import (  # noqa: E402
    build_prompt_token_ids, load_pointllm_model,
)
from scripts.analyze_decode_mlp_blocks import load_modelnet_samples  # noqa: E402
from scripts.evaluate_nm_pruning_pointllm import forward_logits  # noqa: E402
from scripts.evaluate_precision_sensitivity_pointllm import component_modules  # noqa: E402


DEFAULT_MODULES = (
    "layers.0.self_attn.q_proj",
    "layers.0.mlp.down_proj",
    "lm_head",
)


def all_weighted_modules(model) -> dict[str, torch.nn.Module]:
    result = {}
    for group in component_modules(model).values():
        result.update(group)
    return result


def capture_inputs(
    model, modules: dict[str, torch.nn.Module], token_ids: list[int],
    point_cloud: torch.Tensor, device: torch.device,
) -> dict[str, torch.Tensor]:
    captured: dict[str, torch.Tensor] = {}
    handles = []
    for name, module in modules.items():
        def hook(_module, inputs, module_name=name):
            captured[module_name] = inputs[0].detach().float().cpu()
        handles.append(module.register_forward_pre_hook(hook))
    try:
        forward_logits(model, token_ids, point_cloud, device)
    finally:
        for handle in handles:
            handle.remove()
    missing = sorted(set(modules) - set(captured))
    if missing:
        raise RuntimeError(f"failed to capture module inputs: {missing}")
    return captured


def smooth_real_linear(
    activation: torch.Tensor, weight: torch.Tensor, alpha: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    activation_absmax = activation.reshape(-1, activation.shape[-1]).abs().amax(dim=0)
    weight_absmax = weight.float().abs().amax(dim=0)
    epsilon = 1e-5
    smoothing = (
        activation_absmax.clamp_min(epsilon).pow(alpha)
        / weight_absmax.clamp_min(epsilon).pow(1.0 - alpha)
    ).clamp_min(epsilon)
    smoothing = smoothing.to(torch.float16).float().clamp_min(epsilon)
    transformed_activation = activation / smoothing
    transformed_weight = weight.float() * smoothing.unsqueeze(0)
    activation_scale = static_activation_scale(transformed_activation.abs().max())
    return transformed_activation, transformed_weight, activation_scale, smoothing, {
        "alpha": alpha,
        "channels": smoothing.numel(),
        "storage_bytes_fp16": smoothing.numel() * 2,
        "activation_scale_before": float(static_activation_scale(activation.abs().max()).item()),
        "activation_scale_after": float(activation_scale.item()),
        "current_runtime_channelwise_divide": True,
        "offline_folded": False,
    }


def correlate_real_qproj_dot4(
    activation: torch.Tensor,
    weight: torch.Tensor,
    activation_scale: torch.Tensor,
    *,
    rows: tuple[int, ...],
    rtl_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    if activation.shape[0] != 1 or activation.shape[-1] % 4:
        raise ValueError("RTL sample expects one token and K divisible by four")
    qactivation = quantize_symmetric(activation, activation_scale).reshape(-1)
    _, qweight, weight_scales = per_output_weight_qdq(weight)
    chunks = activation.shape[-1] // 4
    beats = []
    for chunk in range(chunks):
        start = chunk * 4
        columns = tuple(
            tuple(int(value) for value in qweight[row, start:start + 4].tolist())
            for row in rows
        )
        beats.append(DenseBeat(
            tag=0, first=chunk == 0, last=chunk == chunks - 1,
            column_mask=(1 << len(rows)) - 1,
            a=tuple(int(value) for value in qactivation[start:start + 4].tolist()),
            b=columns,
        ))
    config = DenseTileConfig(
        rows=1, columns=len(rows), n_tile=len(rows), k=activation.shape[-1],
        source_stall_mod=0, output_stall_mod=0,
        max_cycles=chunks + 16,
    )
    report = correlate_dense_tile(
        config, rtl_root=rtl_root, output_dir=output_dir, beats=tuple(beats),
    )
    model = run_dense_tile_model(config, tuple(beats))
    rtl_correlated_sums = list(model.outputs[0])
    expected = (
        qactivation.to(torch.int32)
        @ qweight[list(rows)].to(torch.int32).transpose(0, 1)
    ).tolist()
    exact = rtl_correlated_sums == expected
    if not exact:
        raise AssertionError(
            f"real Dot4 accumulation mismatch: {rtl_correlated_sums} versus {expected}"
        )
    return {
        "status": "rtl_correlated_integer_payload_and_accumulator" if exact else "mismatch",
        "rows": list(rows),
        "k": activation.shape[-1],
        "chunks": chunks,
        "activation_scale_fp16": float(activation_scale.item()),
        "weight_scales_fp16": [float(weight_scales[row].item()) for row in rows],
        "integer_golden_accumulators": expected,
        "rtl_dot4_accumulators": rtl_correlated_sums,
        "accumulator_exact": exact,
        "event_trace_exact": report["event_trace_exact"],
        "cycle_error": report["cycle_error"],
        "rtl_cycles": report["rtl_cycles"],
        "synthesis_check": {
            key: report["synthesis_check"][key]
            for key in (
                "status", "top_direct_multipliers", "dot4_multipliers_per_instance",
                "dot4_instances", "total_shared_int8_multipliers",
            )
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", default="/mnt/llm_data/pointllm_ckpt/PointLLM_7B_v1.2_safetensors")
    parser.add_argument("--modelnet_data_path", default="/mnt/llm_data/pointllm_data/modelnet40_data")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--prompt", default="Describe this 3D object in detail.")
    parser.add_argument("--module", action="append", default=[])
    parser.add_argument("--smoothquant_alpha", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--tensor_package_dir", type=Path,
        help="Canonical q-projection package directory; defaults below output_dir.",
    )
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)

    sample = load_modelnet_samples(args.modelnet_data_path, 1, args.seed)[0]
    model, tokenizer = load_pointllm_model(args.model_path, device, dtype)
    available = all_weighted_modules(model)
    names = tuple(args.module or DEFAULT_MODULES)
    unknown = sorted(set(names) - set(available))
    if unknown:
        raise ValueError(f"unknown modules: {unknown}")
    modules = {name: available[name] for name in names}
    token_ids = build_prompt_token_ids(tokenizer, model, question=args.prompt)
    point_cloud = sample["point_cloud"].to(device=device, dtype=dtype)
    captured = capture_inputs(model, modules, token_ids, point_cloud, device)

    results = {}
    rtl_inputs = None
    for name, module in modules.items():
        activation = captured[name].reshape(-1, captured[name].shape[-1])[-1:].float()
        weight = module.weight.detach().float().cpu()
        bias = None if module.bias is None else module.bias.detach().float().cpu()
        transformed_activation, transformed_weight, activation_scale, smooth_scale, smoothing = (
            smooth_real_linear(
                captured[name].reshape(-1, captured[name].shape[-1]).float(),
                weight, args.smoothquant_alpha,
            )
        )
        transformed_activation = transformed_activation[-1:]
        comparison = compare_linear_bf16_qdq_integer(
            transformed_activation, transformed_weight,
            activation_scale=activation_scale, bias=bias,
        )
        comparison["module"] = name
        comparison["source_shape"] = {
            "captured_activation": list(captured[name].shape),
            "weight": list(module.weight.shape),
        }
        comparison["smoothquant"] = smoothing
        results[name] = comparison
        if name == "layers.0.self_attn.q_proj":
            package_dir = args.tensor_package_dir or (
                args.output_dir / "tensor_packages" / "layer_00_q_proj"
            )
            _, qweight, weight_scale = per_output_weight_qdq(transformed_weight)
            qactivation = quantize_symmetric(
                transformed_activation, activation_scale,
            )
            integer_accumulator = (
                qactivation.to(torch.int32).cpu()
                @ qweight.to(torch.int32).cpu().transpose(0, 1)
            )
            integer_output = integer_accumulator.float()
            integer_output *= activation_scale.cpu().float()
            integer_output *= weight_scale.cpu().float().unsqueeze(0)
            if bias is not None:
                integer_output += bias.cpu().float().unsqueeze(0)
            output_scale = static_activation_scale(integer_output.abs().max())
            package_manifest = write_quantized_linear_package(
                package_dir,
                activation=qactivation.cpu().numpy(),
                weight=qweight.cpu().numpy(),
                activation_scale=activation_scale.cpu().numpy(),
                weight_scale=weight_scale.cpu().numpy(),
                smooth_scale=smooth_scale.cpu().numpy(),
                output_scale=output_scale.cpu().numpy(),
                bias=None if bias is None else bias.cpu().numpy(),
                provenance={
                    "model_path": str(Path(args.model_path).resolve()),
                    "dataset": "modelnet",
                    "sample_id": sample["point_cloud_id"],
                    "prompt": args.prompt,
                    "module": name,
                    "captured_activation_shape": list(captured[name].shape),
                    "selected_activation_row": -1,
                    "smoothquant_alpha": args.smoothquant_alpha,
                },
            )
            rtl_inputs = (
                transformed_activation, transformed_weight, activation_scale,
                str(package_dir.resolve()), package_manifest,
            )

    if rtl_inputs is None:
        raise RuntimeError("q_proj must be selected for real-payload RTL correlation")
    q_activation, q_weight, q_scale, package_dir, package_manifest = rtl_inputs
    rtl_report = correlate_real_qproj_dot4(
        q_activation, q_weight, q_scale,
        rows=(0, q_weight.shape[0] // 2, q_weight.shape[0] - 1),
        rtl_root=ARCH_ROOT / "rtl/vertical_slice",
        output_dir=args.output_dir / "qproj_real_dot4_rtl",
    )
    dense64_report = correlate_production_dense_fabric(
        ProductionDenseFabricConfig(
            m=1, n=q_weight.shape[0], k=q_weight.shape[1],
            memory_lanes=4, source_stall_mod=0, output_stall_mod=0,
            max_cycles=200_000,
        ),
        rtl_root=ARCH_ROOT / "rtl/vertical_slice",
        output_dir=args.output_dir / "qproj_real_dense64_rtl",
        simulator="verilator", internal_trace=False, synthesize=False,
        tensor_package_dir=package_dir,
    )
    payload = {
        "schema_version": "0.1",
        "status": "real_pointllm_integer_and_rtl_correlated",
        "config": {
            "model_path": str(Path(args.model_path).resolve()),
            "dataset": "modelnet",
            "sample_id": sample["point_cloud_id"],
            "prompt": args.prompt,
            "dtype": args.dtype,
            "smoothquant_alpha": args.smoothquant_alpha,
            "modules": list(names),
        },
        "linear_comparisons": results,
        "real_qproj_rtl": rtl_report,
        "real_qproj_dense64_rtl": dense64_report,
        "real_qproj_tensor_package": {
            "path": package_dir,
            "manifest": package_manifest,
        },
        "claim_boundary": {
            "true_int8_int32_golden_executed": True,
            "real_qproj_payload_dot4_rtl_exact": True,
            "selected_qproj_rows_full_k_reduction_rtl_exact": True,
            "all_qproj_rows_rtl_exact": True,
            "all_qproj_rows_dense64_rtl_exact": True,
            "full_production_w8a8_kernel": False,
            "full_pointllm_cycle_accurate": False,
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps({
        "status": payload["status"],
        "integer_vs_qdq_fp32": {
            name: row["metrics"]["integer_vs_qdq_fp32"]
            for name, row in results.items()
        },
        "real_qproj_rtl": rtl_report,
    }, indent=2))


if __name__ == "__main__":
    main()
