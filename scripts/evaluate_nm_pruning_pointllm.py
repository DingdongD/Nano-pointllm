#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
POINTLLM_ROOT = Path(os.environ.get("POINTLLM_ROOT", "/home/PointLLM"))
for path in (str(REPO_ROOT), str(POINTLLM_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from nanopointllm.compression.nm_pruning import (  # noqa: E402
    InputSecondMomentCollector,
    apply_nm_pruning,
    decoder_linear_modules,
)
from nanopointllm.parity.engine_test_utils import (  # noqa: E402
    build_prompt_token_ids,
    load_pointllm_model,
)
from nanopointllm.parity.hf_manual_greedy import manual_greedy_decode  # noqa: E402


DEFAULT_PROMPTS = (
    "What is this?",
    "Describe this 3D object in detail.",
)


def parse_pattern(value: str) -> tuple[int, int]:
    try:
        prune_n, prune_m = (int(part) for part in value.split(":"))
    except Exception as exc:
        raise argparse.ArgumentTypeError("pattern must look like 2:4 or 4:8") from exc
    if prune_n <= 0 or prune_m <= prune_n:
        raise argparse.ArgumentTypeError("pattern requires 0 < N < M")
    return prune_n, prune_m


def load_modelnet_samples(data_path: str, count: int, seed: int) -> list[dict[str, Any]]:
    import yaml
    from pointllm.data import ModelNet

    config_path = POINTLLM_ROOT / "pointllm/data/modelnet_config/ModelNet40.yaml"
    with config_path.open() as handle:
        config = yaml.safe_load(handle)
    config["DATA_PATH"] = data_path
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
            yaml.safe_dump(config, handle)
            temp_path = handle.name
        dataset = ModelNet(temp_path, split="test", subset_nums=-1, use_color=True)
        indices = random.Random(seed).sample(range(len(dataset)), count)
        return [
            {
                "id": f"modelnet-test-{index}",
                "point_cloud": dataset[index]["point_clouds"],
                "label": dataset[index]["label_names"],
            }
            for index in indices
        ]
    finally:
        if temp_path:
            Path(temp_path).unlink(missing_ok=True)


@torch.inference_mode()
def forward_logits(
    model,
    token_ids: list[int],
    point_cloud: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    output = model(
        input_ids=ids,
        attention_mask=torch.ones_like(ids),
        point_clouds=point_cloud.unsqueeze(0),
        use_cache=False,
        return_dict=True,
    )
    return output.logits


@torch.inference_mode()
def generate(
    model,
    prompt_ids: list[int],
    point_cloud: torch.Tensor,
    max_new_tokens: int,
    device: torch.device,
) -> list[int]:
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    output = manual_greedy_decode(
        model,
        ids,
        torch.ones_like(ids),
        max_new_tokens,
        point_clouds=point_cloud.unsqueeze(0),
    )
    return output[0, len(prompt_ids):].tolist()


def distribution_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    ref = reference.float()
    cand = candidate.float()
    ref_logp = F.log_softmax(ref, dim=-1)
    cand_logp = F.log_softmax(cand, dim=-1)
    ref_prob = ref_logp.exp()
    kl = (ref_prob * (ref_logp - cand_logp)).sum(dim=-1)
    ref_top2 = torch.topk(ref, 2, dim=-1).values
    cand_top2 = torch.topk(cand, 2, dim=-1).values
    return {
        "kl_mean": float(kl.mean().item()),
        "kl_max": float(kl.max().item()),
        "top1_agreement": float((ref.argmax(-1) == cand.argmax(-1)).float().mean().item()),
        "reference_margin_mean": float((ref_top2[:, 0] - ref_top2[:, 1]).mean().item()),
        "candidate_margin_mean": float((cand_top2[:, 0] - cand_top2[:, 1]).mean().item()),
        "logit_rmse": float(torch.sqrt(torch.mean((ref - cand).square())).item()),
    }


def aggregate_module_reports(reports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    grouped = defaultdict(lambda: {"pruned": 0, "total": 0, "modules": 0})
    for report in reports.values():
        row = grouped[report["category"]]
        row["pruned"] += report["pruned"]
        row["total"] += report["total"]
        row["modules"] += 1
    return {
        category: {
            **values,
            "sparsity": values["pruned"] / values["total"],
        }
        for category, values in grouped.items()
    }


def plot_results(payload: dict[str, Any], output_dir: Path) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = payload["evaluation"]
    labels = [row["sample_id"] + "/" + row["prompt_id"] for row in rows]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    axes[0].bar(x, [row["kl_mean"] for row in rows], color="#c44536")
    axes[0].set_title("Teacher-Forced KL")
    axes[1].bar(x, [row["top1_agreement"] for row in rows], color="#33658a")
    axes[1].set_title("Top-1 Agreement")
    axes[1].set_ylim(0, 1.02)
    axes[2].bar(x, [row["free_generation_token_match"] for row in rows], color="#0b6e4f")
    axes[2].set_title("Free-Generation Token Match")
    axes[2].set_ylim(0, 1.02)
    for axis in axes:
        axis.set_xticks(x, labels=labels, rotation=35, ha="right")
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle(f"{payload['config']['method']} {payload['config']['pattern']} PointLLM")
    fig.tight_layout()
    path = output_dir / "quality_summary.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate static N:M pruning on PointLLM")
    parser.add_argument("--model_path", default="/mnt/llm_data/pointllm_ckpt/PointLLM_7B_v1.2_safetensors")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--method", choices=("wanda", "magnitude"), default="wanda")
    parser.add_argument("--pattern", type=parse_pattern, default=parse_pattern("2:4"))
    parser.add_argument("--modelnet_data_path", default="/mnt/llm_data/pointllm_data/modelnet40_data")
    parser.add_argument("--num_samples", type=int, default=2)
    parser.add_argument("--prompt", action="append", default=[])
    parser.add_argument("--max_new_tokens", type=int, default=8)
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
    prune_n, prune_m = args.pattern

    samples = load_modelnet_samples(args.modelnet_data_path, args.num_samples, args.seed)
    prompts = args.prompt or list(DEFAULT_PROMPTS)
    model, tokenizer = load_pointllm_model(args.model_path, device, dtype)
    modules = decoder_linear_modules(model)
    cases = []
    for sample in samples:
        point_cloud = sample["point_cloud"].to(device=device, dtype=dtype)
        for prompt_index, prompt in enumerate(prompts):
            prompt_ids = build_prompt_token_ids(tokenizer, model, question=prompt)
            generated = generate(model, prompt_ids, point_cloud, args.max_new_tokens, device)
            teacher_ids = prompt_ids + generated[:-1]
            logits = forward_logits(model, teacher_ids, point_cloud, device)
            start = len(prompt_ids) - 1
            reference = logits[0, start:start + len(generated)].float().cpu()
            cases.append({
                "sample_id": sample["id"],
                "prompt_id": f"prompt-{prompt_index}",
                "prompt": prompt,
                "label": sample["label"],
                "point_cloud": point_cloud,
                "prompt_ids": prompt_ids,
                "teacher_ids": teacher_ids,
                "reference_logits": reference,
                "reference_generated": generated,
            })
            del logits

    moments = None
    if args.method == "wanda":
        with InputSecondMomentCollector(modules) as collector:
            for case in cases:
                forward_logits(model, case["teacher_ids"], case["point_cloud"], device)
        moments = collector.second_moments()
    reports = apply_nm_pruning(
        modules,
        prune_n=prune_n,
        prune_m=prune_m,
        input_second_moments=moments,
    )

    evaluations = []
    for case in cases:
        logits = forward_logits(model, case["teacher_ids"], case["point_cloud"], device)
        start = len(case["prompt_ids"]) - 1
        candidate = logits[0, start:start + len(case["reference_generated"])].float().cpu()
        candidate_generated = generate(
            model,
            case["prompt_ids"],
            case["point_cloud"],
            args.max_new_tokens,
            device,
        )
        metrics = distribution_metrics(case["reference_logits"], candidate)
        metrics.update({
            "sample_id": case["sample_id"],
            "prompt_id": case["prompt_id"],
            "prompt": case["prompt"],
            "label": case["label"],
            "reference_generated_ids": case["reference_generated"],
            "candidate_generated_ids": candidate_generated,
            "reference_text": tokenizer.decode(case["reference_generated"], skip_special_tokens=True),
            "candidate_text": tokenizer.decode(candidate_generated, skip_special_tokens=True),
            "free_generation_token_match": sum(
                left == right for left, right in zip(case["reference_generated"], candidate_generated)
            ) / len(case["reference_generated"]),
            "free_generation_exact_match": candidate_generated == case["reference_generated"],
        })
        evaluations.append(metrics)

    payload = {
        "config": {
            "model_path": str(Path(args.model_path).resolve()),
            "device": str(device),
            "dtype": args.dtype,
            "method": args.method,
            "pattern": f"{prune_n}:{prune_m}",
            "num_samples": len(samples),
            "num_prompts": len(prompts),
            "max_new_tokens": args.max_new_tokens,
            "calibration_tokens": int(next(iter(collector.samples.values()))) if args.method == "wanda" else 0,
        },
        "module_sparsity": aggregate_module_reports(reports),
        "modules": reports,
        "evaluation": evaluations,
        "aggregate": {
            key: float(np.mean([row[key] for row in evaluations]))
            for key in (
                "kl_mean",
                "kl_max",
                "top1_agreement",
                "logit_rmse",
                "free_generation_token_match",
                "free_generation_exact_match",
            )
        },
        "guardrails": [
            "Masks are applied in memory; the source checkpoint is never modified.",
            "N:M is enforced independently in every output row along contiguous input columns.",
            "Dense PyTorch kernels do not realize sparse speedup; this experiment measures quality only.",
        ],
    }
    payload["plot"] = plot_results(payload, output_dir)
    with (output_dir / "summary.json").open("w") as handle:
        json.dump(payload, handle, indent=2)
    print(json.dumps(payload["aggregate"], indent=2))


if __name__ == "__main__":
    main()
