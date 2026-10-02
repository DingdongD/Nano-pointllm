#!/usr/bin/env python3
"""Evaluate sequential W8A8 QDQ across the selected PointLLM pipeline."""
from __future__ import annotations

import argparse
from collections import defaultdict
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
for path in (str(REPO_ROOT), str(POINTLLM_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from nanopointllm.compression.w8a8 import (  # noqa: E402
    CONTRACT, ActivationAbsmaxCollector, SequentialActivationQDQ,
    apply_smoothquant_transform, apply_weight_qdq,
)
from nanopointllm.parity.engine_test_utils import (  # noqa: E402
    build_prompt_token_ids, load_pointllm_model,
)
from scripts.analyze_decode_mlp_blocks import (  # noqa: E402
    load_modelnet_samples, load_objaverse_samples,
)
from scripts.evaluate_nm_pruning_pointllm import (  # noqa: E402
    DEFAULT_PROMPTS, distribution_metrics, forward_logits, generate,
)
from scripts.evaluate_precision_sensitivity_pointllm import component_modules  # noqa: E402


def selected_modules(model, components: list[str]) -> dict[str, torch.nn.Module]:
    available = component_modules(model)
    unknown = sorted(set(components) - set(available))
    if unknown:
        raise ValueError(f"unknown components: {unknown}")
    modules = {}
    for component in components:
        for name, module in available[component].items():
            if name in modules and modules[name] is not module:
                raise RuntimeError(f"module-name collision: {name}")
            modules[name] = module
    return modules


def load_samples(args: argparse.Namespace) -> list[dict[str, Any]]:
    count = args.num_calibration_samples + args.num_eval_samples
    if args.dataset == "modelnet":
        return load_modelnet_samples(args.modelnet_data_path, count, args.seed)
    return load_objaverse_samples(
        args.objaverse_data_path, args.objaverse_anno_path, count, args.seed,
    )


def normalize_sample(sample: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": sample["point_cloud_id"],
        "point_cloud": sample["point_cloud"],
        "metadata": sample.get("metadata", {}),
    }


def calibration_forward(
    model, tokenizer, sample, prompts, max_new_tokens, device,
) -> None:
    point_cloud = sample["point_cloud"].to(device=device, dtype=model.dtype)
    for prompt in prompts:
        prompt_ids = build_prompt_token_ids(tokenizer, model, question=prompt)
        generated = generate(model, prompt_ids, point_cloud, max_new_tokens, device)
        teacher_ids = prompt_ids + generated[:-1]
        forward_logits(model, teacher_ids, point_cloud, device)


def build_reference_cases(
    model, tokenizer, samples, prompts, max_new_tokens, device,
) -> list[dict[str, Any]]:
    cases = []
    for sample in samples:
        point_cloud = sample["point_cloud"].to(device=device, dtype=model.dtype)
        for prompt_index, prompt in enumerate(prompts):
            prompt_ids = build_prompt_token_ids(tokenizer, model, question=prompt)
            generated = generate(model, prompt_ids, point_cloud, max_new_tokens, device)
            teacher_ids = prompt_ids + generated[:-1]
            logits = forward_logits(model, teacher_ids, point_cloud, device)
            start = len(prompt_ids) - 1
            cases.append({
                "sample_id": sample["id"],
                "prompt_id": f"prompt-{prompt_index}",
                "prompt": prompt,
                "metadata": sample["metadata"],
                "point_cloud": point_cloud,
                "prompt_ids": prompt_ids,
                "teacher_ids": teacher_ids,
                "reference_generated": generated,
                "reference_logits": logits[0, start:start + len(generated)].float().cpu(),
            })
            del logits
    return cases


def evaluate(model, tokenizer, cases, max_new_tokens, device) -> tuple[list[dict[str, Any]], dict[str, float]]:
    rows = []
    for case in cases:
        logits = forward_logits(model, case["teacher_ids"], case["point_cloud"], device)
        start = len(case["prompt_ids"]) - 1
        length = len(case["reference_generated"])
        candidate = logits[0, start:start + length].float().cpu()
        generated = generate(
            model, case["prompt_ids"], case["point_cloud"], max_new_tokens, device,
        )
        metrics = distribution_metrics(case["reference_logits"], candidate)
        metrics.update({
            "sample_id": case["sample_id"],
            "prompt_id": case["prompt_id"],
            "prompt": case["prompt"],
            "reference_generated": case["reference_generated"],
            "candidate_generated": generated,
            "free_generation_token_match": sum(
                left == right for left, right in zip(case["reference_generated"], generated)
            ) / max(1, length),
            "free_generation_exact_match": generated == case["reference_generated"],
            "candidate_text": tokenizer.decode(generated, skip_special_tokens=True),
        })
        rows.append(metrics)
        del logits
    keys = (
        "kl_mean", "kl_max", "top1_agreement", "logit_rmse",
        "free_generation_token_match", "free_generation_exact_match",
    )
    return rows, {
        key: float(np.mean([row[key] for row in rows])) for key in keys
    }


def aggregate_weight_report(reports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    by_granularity = defaultdict(lambda: {"parameters": 0, "modules": 0, "rmse_sum": 0.0})
    for report in reports.values():
        row = by_granularity[report["granularity"]]
        row["parameters"] += report["parameters"]
        row["modules"] += 1
        row["rmse_sum"] += report["rmse"] * report["parameters"]
    return {
        key: {
            "parameters": value["parameters"],
            "modules": value["modules"],
            "parameter_weighted_rmse": value["rmse_sum"] / value["parameters"],
        }
        for key, value in by_granularity.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", default="/mnt/llm_data/pointllm_ckpt/PointLLM_7B_v1.2_safetensors")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--dataset", choices=("modelnet", "objaverse"), default="modelnet")
    parser.add_argument("--modelnet_data_path", default="/mnt/llm_data/pointllm_data/modelnet40_data")
    parser.add_argument("--objaverse_data_path", default="/mnt/llm_data/pointllm_data/objaverse_660k_splits")
    parser.add_argument("--objaverse_anno_path", default="/mnt/llm_data/pointllm_data/anno_data/PointLLM_brief_description_val_200_GT.json")
    parser.add_argument("--num_calibration_samples", type=int, default=2)
    parser.add_argument("--num_eval_samples", type=int, default=2)
    parser.add_argument("--prompt", action="append", default=[])
    parser.add_argument("--max_new_tokens", type=int, default=8)
    parser.add_argument("--components", default="pointbert,projector,decoder_qkv,decoder_o,decoder_mlp,lm_head")
    parser.add_argument("--weight_granularity", choices=("per_output", "group"), default="per_output")
    parser.add_argument("--group_size", type=int, default=128)
    parser.add_argument(
        "--activation_mode",
        choices=("static", "smoothquant_static", "dynamic_per_token"),
        default="static",
    )
    parser.add_argument("--smoothquant_alpha", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    if min(args.num_calibration_samples, args.num_eval_samples, args.max_new_tokens) <= 0:
        raise ValueError("sample counts and max_new_tokens must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    prompts = args.prompt or list(DEFAULT_PROMPTS)
    components = [item for item in args.components.split(",") if item]

    raw_samples = [normalize_sample(sample) for sample in load_samples(args)]
    calibration_samples = raw_samples[:args.num_calibration_samples]
    eval_samples = raw_samples[args.num_calibration_samples:]
    if {item["id"] for item in calibration_samples} & {item["id"] for item in eval_samples}:
        raise RuntimeError("calibration and evaluation samples overlap")

    model, tokenizer = load_pointllm_model(args.model_path, device, dtype)
    modules = selected_modules(model, components)
    with ActivationAbsmaxCollector(modules) as collector:
        for sample in calibration_samples:
            calibration_forward(
                model, tokenizer, sample, prompts, args.max_new_tokens, device,
            )
        cases = build_reference_cases(
            model, tokenizer, eval_samples, prompts, args.max_new_tokens, device,
        )
    input_smoothing_scales = None
    if args.activation_mode == "smoothquant_static":
        input_smoothing_scales, static_scales = apply_smoothquant_transform(
            modules, collector.channel_absmax, alpha=args.smoothquant_alpha,
        )
    else:
        static_scales = collector.static_scales()
    weight_report = apply_weight_qdq(
        modules, granularity=args.weight_granularity, group_size=args.group_size,
    )
    with SequentialActivationQDQ(
        modules,
        mode="static" if args.activation_mode == "smoothquant_static" else args.activation_mode,
        static_scales=static_scales if args.activation_mode != "dynamic_per_token" else None,
        input_smoothing_scales=input_smoothing_scales,
    ):
        rows, aggregate = evaluate(
            model, tokenizer, cases, args.max_new_tokens, device,
        )

    payload = {
        "schema_version": "0.1",
        "status": "sequential_w8a8_qdq_fidelity_only",
        "config": {
            "model_path": str(Path(args.model_path).resolve()),
            "device": str(device),
            "dtype": args.dtype,
            "dataset": args.dataset,
            "calibration_sample_ids": [item["id"] for item in calibration_samples],
            "evaluation_sample_ids": [item["id"] for item in eval_samples],
            "prompts": prompts,
            "max_new_tokens": args.max_new_tokens,
            "components": components,
            "module_count": len(modules),
            "weight_granularity": args.weight_granularity,
            "group_size": args.group_size if args.weight_granularity == "group" else None,
            "activation_mode": args.activation_mode,
            "smoothquant_alpha": (
                args.smoothquant_alpha if args.activation_mode == "smoothquant_static" else None
            ),
        },
        "numeric_contract": CONTRACT.as_dict(),
        "aggregate": aggregate,
        "cases": rows,
        "weight_summary": aggregate_weight_report(weight_report),
        "module_weight_reports": weight_report,
        "activation_calibration": {
            name: {
                "absmax": collector.tensor_absmax[name],
                "static_scale_fp16": float(scale.item()),
                "samples": collector.samples[name],
            }
            for name, scale in static_scales.items()
        },
        "guardrails": [
            "All selected weights are quantized simultaneously and activation QDQ propagates sequentially.",
            "Calibration and evaluation point-cloud IDs are disjoint.",
            "QDQ executes dense BF16/FP16 operators and establishes no runtime speedup.",
            "Norm, softmax, nonlinear activation, residual, and bias remain BF16/FP16.",
            "Production W8A8 RTL remains disabled until integer-kernel and larger semantic gates pass.",
        ],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps({"status": payload["status"], "aggregate": aggregate}, indent=2))


if __name__ == "__main__":
    main()
