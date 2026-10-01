#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
POINTLLM_ROOT = Path(os.environ.get("POINTLLM_ROOT", "/home/PointLLM"))
for path in (str(REPO_ROOT), str(POINTLLM_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from nanopointllm.compression.nm_pruning import decoder_linear_modules, module_category  # noqa: E402
from nanopointllm.compression.weight_quant import symmetric_quantize_dequantize  # noqa: E402
from nanopointllm.parity.engine_test_utils import build_prompt_token_ids, load_pointllm_model  # noqa: E402
from scripts.evaluate_nm_pruning_pointllm import (  # noqa: E402
    DEFAULT_PROMPTS,
    distribution_metrics,
    forward_logits,
    generate,
    load_modelnet_samples,
)


def weighted_modules(module: nn.Module, prefix: str) -> dict[str, nn.Module]:
    result = {}
    for name, child in module.named_modules():
        if isinstance(child, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            result[f"{prefix}.{name}" if name else prefix] = child
    return result


def component_modules(model) -> dict[str, dict[str, nn.Module]]:
    inner = model.get_model()
    decoder = decoder_linear_modules(model)
    grouped_decoder = defaultdict(dict)
    for name, module in decoder.items():
        grouped_decoder[module_category(name)][name] = module
    return {
        "pointbert": weighted_modules(inner.point_backbone, "point_backbone"),
        "projector": weighted_modules(inner.point_proj, "point_proj"),
        "decoder_qkv": grouped_decoder["decoder_qkv"],
        "decoder_o": grouped_decoder["decoder_o"],
        "decoder_mlp": grouped_decoder["decoder_mlp"],
        "lm_head": {"lm_head": model.lm_head},
    }


@torch.no_grad()
def quantize_component(
    modules: dict[str, nn.Module], bits: int
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    originals = {}
    reports = {}
    for name, module in modules.items():
        originals[name] = module.weight.detach().cpu().clone()
        dequantized, report = symmetric_quantize_dequantize(module.weight, bits=bits)
        module.weight.copy_(dequantized)
        reports[name] = {**report, "parameters": module.weight.numel()}
    return originals, reports


@torch.no_grad()
def restore_component(
    modules: dict[str, nn.Module], originals: dict[str, torch.Tensor]
) -> None:
    for name, module in modules.items():
        module.weight.copy_(originals[name].to(device=module.weight.device, dtype=module.weight.dtype))


def aggregate_weight_error(reports: dict[str, Any]) -> dict[str, float | int]:
    total = sum(report["parameters"] for report in reports.values())
    return {
        "parameters": total,
        "modules": len(reports),
        "relative_rmse_parameter_weighted": sum(
            report["relative_rmse"] * report["parameters"] for report in reports.values()
        ) / total,
        "max_abs_error": max(report["max_abs_error"] for report in reports.values()),
    }


def evaluate_cases(model, tokenizer, cases, device, max_new_tokens) -> tuple[list[dict[str, Any]], dict[str, float]]:
    rows = []
    for case in cases:
        logits = forward_logits(model, case["teacher_ids"], case["point_cloud"], device)
        start = len(case["prompt_ids"]) - 1
        candidate = logits[0, start:start + len(case["reference_generated"])].float().cpu()
        candidate_generated = generate(
            model, case["prompt_ids"], case["point_cloud"], max_new_tokens, device
        )
        metrics = distribution_metrics(case["reference_logits"], candidate)
        metrics.update({
            "sample_id": case["sample_id"],
            "prompt_id": case["prompt_id"],
            "free_generation_token_match": sum(
                left == right for left, right in zip(case["reference_generated"], candidate_generated)
            ) / len(case["reference_generated"]),
            "free_generation_exact_match": candidate_generated == case["reference_generated"],
            "candidate_text": tokenizer.decode(candidate_generated, skip_special_tokens=True),
        })
        rows.append(metrics)
    keys = (
        "kl_mean",
        "kl_max",
        "top1_agreement",
        "logit_rmse",
        "free_generation_token_match",
        "free_generation_exact_match",
    )
    return rows, {key: float(np.mean([row[key] for row in rows])) for key in keys}


def make_plot(results: list[dict[str, Any]], output_dir: Path) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    components = list(dict.fromkeys(row["component"] for row in results))
    x = np.arange(len(components))
    width = 0.35
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for offset, bits, color in ((-width / 2, 8, "#33658a"), (width / 2, 4, "#c44536")):
        rows = {row["component"]: row for row in results if row["bits"] == bits}
        axes[0].bar(x + offset, [rows[c]["aggregate"]["kl_mean"] for c in components], width, label=f"W{bits}", color=color)
        axes[1].bar(x + offset, [rows[c]["aggregate"]["top1_agreement"] for c in components], width, label=f"W{bits}", color=color)
        axes[2].bar(x + offset, [rows[c]["aggregate"]["free_generation_token_match"] for c in components], width, label=f"W{bits}", color=color)
    axes[0].set_title("Teacher-Forced KL")
    axes[1].set_title("Top-1 Agreement")
    axes[2].set_title("Free-Generation Token Match")
    axes[1].set_ylim(0, 1.02)
    axes[2].set_ylim(0, 1.02)
    for axis in axes:
        axis.set_xticks(x, labels=components, rotation=30, ha="right")
        axis.grid(axis="y", alpha=0.2)
        axis.legend(frameon=False)
    fig.tight_layout()
    path = output_dir / "precision_sensitivity.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="PointLLM component-wise W8/W4 sensitivity")
    parser.add_argument("--model_path", default="/mnt/llm_data/pointllm_ckpt/PointLLM_7B_v1.2_safetensors")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--modelnet_data_path", default="/mnt/llm_data/pointllm_data/modelnet40_data")
    parser.add_argument("--num_samples", type=int, default=2)
    parser.add_argument("--prompt", action="append", default=[])
    parser.add_argument("--max_new_tokens", type=int, default=8)
    parser.add_argument("--components", default="pointbert,projector,decoder_qkv,decoder_o,decoder_mlp,lm_head")
    parser.add_argument("--bits", default="8,4")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    components = [item for item in args.components.split(",") if item]
    bit_widths = [int(item) for item in args.bits.split(",") if item]

    samples = load_modelnet_samples(args.modelnet_data_path, args.num_samples, args.seed)
    prompts = args.prompt or list(DEFAULT_PROMPTS)
    model, tokenizer = load_pointllm_model(args.model_path, device, dtype)
    available = component_modules(model)
    unknown = sorted(set(components) - set(available))
    if unknown:
        raise ValueError(f"unknown components: {unknown}")

    cases = []
    for sample in samples:
        point_cloud = sample["point_cloud"].to(device=device, dtype=dtype)
        for prompt_index, prompt in enumerate(prompts):
            prompt_ids = build_prompt_token_ids(tokenizer, model, question=prompt)
            generated = generate(model, prompt_ids, point_cloud, args.max_new_tokens, device)
            teacher_ids = prompt_ids + generated[:-1]
            logits = forward_logits(model, teacher_ids, point_cloud, device)
            start = len(prompt_ids) - 1
            cases.append({
                "sample_id": sample["id"],
                "prompt_id": f"prompt-{prompt_index}",
                "point_cloud": point_cloud,
                "prompt_ids": prompt_ids,
                "teacher_ids": teacher_ids,
                "reference_generated": generated,
                "reference_logits": logits[0, start:start + len(generated)].float().cpu(),
            })
            del logits

    results = []
    for component in components:
        modules = available[component]
        for bits in bit_widths:
            originals, module_reports = quantize_component(modules, bits)
            try:
                rows, aggregate = evaluate_cases(
                    model, tokenizer, cases, device, args.max_new_tokens
                )
            finally:
                restore_component(modules, originals)
            results.append({
                "component": component,
                "bits": bits,
                "weight_error": aggregate_weight_error(module_reports),
                "aggregate": aggregate,
                "cases": rows,
            })
            print(component, bits, json.dumps(aggregate))

    payload = {
        "config": {
            "model_path": str(Path(args.model_path).resolve()),
            "device": str(device),
            "dtype": args.dtype,
            "quantization": "per-output-channel symmetric round-to-nearest simulation",
            "components": components,
            "bits": bit_widths,
            "num_samples": len(samples),
            "num_prompts": len(prompts),
            "max_new_tokens": args.max_new_tokens,
        },
        "results": results,
        "guardrails": [
            "Weights are dequantized back to BF16/FP16 for dense execution; this measures sensitivity, not speed.",
            "Only one component is quantized at a time and original weights are restored before the next experiment.",
            "Activation quantization is not included.",
        ],
    }
    payload["plot"] = make_plot(results, output_dir)
    with (output_dir / "summary.json").open("w") as handle:
        json.dump(payload, handle, indent=2)


if __name__ == "__main__":
    main()
