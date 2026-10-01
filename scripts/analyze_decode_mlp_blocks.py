#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import random
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
POINTLLM_ROOT = Path(os.environ.get("POINTLLM_ROOT", "/home/PointLLM"))
for path in (str(REPO_ROOT), str(POINTLLM_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from nanopointllm.analysis.mlp_block_trace import (  # noqa: E402
    MLPBlockTraceCollector,
    analyze_mlp_block_trace,
)
from nanopointllm.engine.llm_engine import PointLLMLLMEngine  # noqa: E402
from nanopointllm.parity.engine_test_utils import (  # noqa: E402
    build_prompt_token_ids,
    load_pointllm_model,
)
from nanopointllm.sampling_params import SamplingParams  # noqa: E402


DEFAULT_PROMPTS = (
    "What is this?",
    "Describe this 3D object in detail.",
    "Which object category best describes this shape?",
    "What are the main geometric and functional features of this object?",
)


def parse_csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def parse_csv_floats(value: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in value.split(",") if item.strip())


def normalize_point_cloud(array: np.ndarray, *, use_color: bool = True) -> torch.Tensor:
    points = np.asarray(array, dtype=np.float32).copy()
    xyz = points[:, :3]
    xyz -= xyz.mean(axis=0, keepdims=True)
    radius = np.sqrt((xyz * xyz).sum(axis=1)).max()
    if radius > 0:
        xyz /= radius
    points[:, :3] = xyz
    if use_color:
        if points.shape[1] == 3:
            points = np.concatenate([points, np.zeros_like(points)], axis=1)
        else:
            points = points[:, :6]
    else:
        points = points[:, :3]
    return torch.from_numpy(points)


def load_modelnet_samples(
    data_path: str,
    num_samples: int,
    seed: int,
) -> list[dict[str, Any]]:
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
        dataset = ModelNet(
            config_path=temp_path,
            split="test",
            subset_nums=-1,
            use_color=True,
        )
        rng = random.Random(seed)
        indices = rng.sample(range(len(dataset)), min(num_samples, len(dataset)))
        samples = []
        for index in indices:
            item = dataset[index]
            samples.append({
                "point_cloud_id": f"modelnet-test-{index}",
                "point_cloud": item["point_clouds"],
                "metadata": {"label": item["label_names"], "dataset_index": index},
            })
        return samples
    finally:
        if temp_path is not None:
            Path(temp_path).unlink(missing_ok=True)


def load_objaverse_samples(
    data_path: str,
    annotation_path: str,
    num_samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    root = Path(data_path)
    if not any(root.glob("*_8192.npy")):
        root = root / "8192_npy"
    with Path(annotation_path).open() as handle:
        annotations = json.load(handle)
    rng = random.Random(seed)
    rng.shuffle(annotations)
    samples = []
    for item in annotations:
        object_id = str(item["object_id"])
        point_path = root / f"{object_id}_8192.npy"
        if not point_path.exists():
            continue
        ground_truth = next(
            (
                turn.get("value", "").strip()
                for turn in item.get("conversations", [])
                if turn.get("from") == "gpt" and turn.get("value", "").strip()
            ),
            None,
        )
        samples.append({
            "point_cloud_id": object_id,
            "point_cloud": normalize_point_cloud(np.load(point_path), use_color=True),
            "metadata": {"ground_truth": ground_truth, "point_path": str(point_path)},
        })
        if len(samples) >= num_samples:
            break
    if len(samples) < num_samples:
        raise RuntimeError(f"found only {len(samples)} usable Objaverse samples")
    return samples


def load_npy_samples(paths: list[str]) -> list[dict[str, Any]]:
    return [
        {
            "point_cloud_id": Path(path).stem.removesuffix("_8192"),
            "point_cloud": normalize_point_cloud(np.load(path), use_color=True),
            "metadata": {"point_path": str(Path(path).resolve())},
        }
        for path in paths
    ]


def load_samples(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.dataset == "modelnet":
        return load_modelnet_samples(args.modelnet_data_path, args.num_point_clouds, args.seed)
    if args.dataset == "objaverse":
        return load_objaverse_samples(
            args.objaverse_data_path,
            args.objaverse_anno_path,
            args.num_point_clouds,
            args.seed,
        )
    if args.dataset == "npy":
        if not args.point_cloud_path:
            raise ValueError("--point_cloud_path is required for --dataset npy")
        return load_npy_samples(args.point_cloud_path)
    generator = torch.Generator().manual_seed(args.seed)
    return [
        {
            "point_cloud_id": f"synthetic-{index}",
            "point_cloud": torch.randn(8192, 6, generator=generator),
            "metadata": {"synthetic": True},
        }
        for index in range(args.num_point_clouds)
    ]


def write_jsonl_gz(path: Path, rows: list[dict[str, Any]]) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def write_curves_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_analysis(analysis: dict[str, Any], output_dir: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {32: "#0b6e4f", 64: "#c44536", 128: "#33658a", 256: "#e09f3e"}
    curves = analysis["curves"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    for block_size in analysis["block_sizes"]:
        rows = [row for row in curves if row["block_size"] == block_size]
        x = [row["retained_neuron_ratio_mean"] for row in rows]
        color = colors.get(block_size)
        axes[0, 0].plot(
            x,
            [row["post_gate_contribution_retained_mean"] for row in rows],
            marker="o",
            color=color,
            label=f"{block_size} neurons",
        )
        axes[0, 0].plot(
            x,
            [row["previous_token_contribution_retained_mean"] for row in rows],
            linestyle="--",
            color=color,
            alpha=0.85,
        )
        axes[0, 1].plot(
            x,
            [row["adjacent_token_jaccard_mean"] for row in rows],
            marker="o",
            color=color,
        )
        axes[0, 1].plot(
            x,
            [row["previous_token_recall_mean"] for row in rows],
            linestyle="--",
            color=color,
            alpha=0.85,
        )
        axes[1, 0].plot(
            x,
            [row["same_point_cross_prompt_jaccard_mean"] for row in rows],
            marker="o",
            color=color,
        )
        axes[1, 1].plot(
            x,
            [row["post_gate_weight_byte_reduction_mean"] for row in rows],
            color=color,
            linestyle=":",
        )
        axes[1, 1].plot(
            x,
            [row["oracle_pre_gate_weight_byte_reduction_mean"] for row in rows],
            color=color,
        )
    axes[0, 0].set_title("Retained Contribution vs Block Ratio")
    axes[0, 0].set_ylabel("Contribution Retained")
    axes[0, 0].legend(frameon=False)
    axes[0, 0].text(0.02, 0.05, "solid: current/oracle\ndashed: previous token", transform=axes[0, 0].transAxes)
    axes[0, 1].set_title("Temporal Support Stability")
    axes[0, 1].set_ylabel("Overlap")
    axes[0, 1].text(0.02, 0.05, "solid: Jaccard\ndashed: recall", transform=axes[0, 1].transAxes)
    axes[1, 0].set_title("Same Point Cloud, Cross-Prompt Overlap")
    axes[1, 0].set_ylabel("Jaccard")
    axes[1, 1].set_title("Theoretical MLP Weight-Byte Reduction")
    axes[1, 1].set_ylabel("Reduction")
    axes[1, 1].text(0.02, 0.8, "solid: oracle/previous\ndotted: post-gate", transform=axes[1, 1].transAxes)
    for axis in axes.flat:
        axis.set_xlabel("Retained Neuron Ratio")
        axis.set_ylim(0.0, 1.02)
        axis.grid(alpha=0.2)
    fig.suptitle("PointLLM Decode MLP Block Liveness", fontsize=16)
    fig.tight_layout()
    overview_path = output_dir / "mlp_block_trace_overview.png"
    fig.savefig(overview_path, dpi=180)
    plt.close(fig)

    chosen_block = min(analysis["block_sizes"], key=lambda value: abs(value - 64))
    chosen_ratio = min(analysis["block_ratios"], key=lambda value: abs(value - 0.1))
    layer_rows = [
        row for row in analysis["layer_curves"]
        if row["block_size"] == chosen_block
        and row["requested_block_ratio"] == chosen_ratio
    ]
    layers = [row["layer"] for row in layer_rows]
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    axes[0].plot(layers, [row["post_gate_contribution_retained_mean"] for row in layer_rows], label="current/oracle", color="#0b6e4f")
    axes[0].plot(layers, [row["previous_token_contribution_retained_mean"] for row in layer_rows], label="previous token", color="#c44536")
    axes[0].set_ylabel("Contribution Retained")
    axes[0].legend(frameon=False)
    axes[1].plot(layers, [row["adjacent_token_jaccard_mean"] for row in layer_rows], label="adjacent-token Jaccard", color="#33658a")
    axes[1].plot(layers, [row["same_point_cross_prompt_jaccard_mean"] for row in layer_rows], label="cross-prompt Jaccard", color="#e09f3e")
    axes[1].set_ylabel("Overlap")
    axes[1].set_xlabel("Decoder Layer")
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.set_ylim(0.0, 1.02)
        axis.grid(alpha=0.2)
    fig.suptitle(f"Layer Diagnostics: block={chosen_block}, budget={chosen_ratio:.0%}")
    fig.tight_layout()
    layer_path = output_dir / "mlp_block_trace_by_layer.png"
    fig.savefig(layer_path, dpi=180)
    plt.close(fig)
    return [str(overview_path), str(layer_path)]


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.max_new_tokens < 2:
        raise ValueError("max_new_tokens must be at least 2 to observe a decode pass")
    os.environ["NANOPOINTLLM_ENABLE_CUDA_GRAPH"] = "0"
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)

    samples = load_samples(args)
    model, tokenizer = load_pointllm_model(args.model_path, device, dtype)
    engine = PointLLMLLMEngine(
        hf_model=model,
        eos_token_id=tokenizer.eos_token_id,
        max_num_seqs=1,
        max_num_batched_tokens=args.max_num_batched_tokens,
        num_kvcache_blocks=args.num_kvcache_blocks,
        kvcache_block_size=args.kvcache_block_size,
    )
    runner = engine.runner.lightweight_runner
    if runner is None:
        raise RuntimeError("MLP tracing requires LightweightLlamaRunner")
    num_layers = len(runner.layers)
    if args.require_layers and num_layers != args.require_layers:
        raise RuntimeError(f"model has {num_layers} decoder layers, expected {args.require_layers}")
    storage_dtype = dtype if args.storage_dtype == "auto" else getattr(torch, args.storage_dtype)
    collector = MLPBlockTraceCollector(
        expected_layers=num_layers,
        storage_dtype=storage_dtype,
    )
    runner.set_mlp_trace_observer(collector)

    prompts = args.prompt or list(DEFAULT_PROMPTS)
    sample_outputs = []
    try:
        for point_index, sample in enumerate(samples):
            point_cloud = sample["point_cloud"].to(device=device, dtype=dtype)
            point_cloud_id = str(sample["point_cloud_id"])
            cache_key = f"mlp-trace:{point_cloud_id}"
            for prompt_index, prompt in enumerate(prompts):
                prompt_id = f"prompt-{prompt_index}"
                sample_id = f"point-{point_index}:{prompt_id}"
                token_ids = build_prompt_token_ids(tokenizer, model, question=prompt)
                seq = engine.add_request(
                    token_ids=token_ids,
                    point_clouds=point_cloud,
                    point_cloud_cache_key=cache_key,
                    sampling_params=SamplingParams(
                        max_tokens=args.max_new_tokens,
                        ignore_eos=True,
                        do_sample=False,
                        temperature=0.0,
                    ),
                )
                engine.step()
                while not seq.is_finished:
                    input_generated_index = seq.num_completion_tokens - 1
                    decode_step = input_generated_index
                    collector.begin_step([{
                        "sample_id": sample_id,
                        "point_cloud_id": point_cloud_id,
                        "point_cloud_index": point_index,
                        "prompt_id": prompt_id,
                        "prompt": prompt,
                        "decode_step": decode_step,
                        "input_token_id": seq.last_token,
                        "input_generated_token_index": input_generated_index,
                        "output_generated_token_index": input_generated_index + 1,
                    }])
                    engine.step()
                    collector.end_step([seq.last_token])
                generated = seq.token_ids[seq.num_prompt_tokens:]
                sample_outputs.append({
                    "sample_id": sample_id,
                    "point_cloud_id": point_cloud_id,
                    "prompt_id": prompt_id,
                    "prompt": prompt,
                    "prompt_tokens": len(token_ids),
                    "generated_token_ids": generated,
                    "generated_text": tokenizer.decode(generated, skip_special_tokens=True),
                    "point_metadata": sample["metadata"],
                })
                print(
                    f"traced {sample_id}: {len(generated)} generated tokens, "
                    f"{max(len(generated) - 1, 0)} decode passes"
                )
    finally:
        runner.set_mlp_trace_observer(None)

    raw_path = output_dir / "mlp_swiglu_activations.pt"
    raw_payload = collector.payload()
    torch.save(raw_payload, raw_path)
    analysis = analyze_mlp_block_trace(
        raw_payload,
        block_sizes=args.block_sizes,
        block_ratios=args.block_ratios,
    )
    details_path = output_dir / "block_trace_details.jsonl.gz"
    cross_path = output_dir / "cross_prompt_overlap.jsonl.gz"
    curves_path = output_dir / "retained_contribution_curves.csv"
    layer_curves_path = output_dir / "layer_curves.csv"
    step_curves_path = output_dir / "step_curves.csv"
    write_jsonl_gz(details_path, analysis.pop("details"))
    write_jsonl_gz(cross_path, analysis.pop("cross_prompt_overlap"))
    write_curves_csv(curves_path, analysis["curves"])
    write_curves_csv(layer_curves_path, analysis["layer_curves"])
    write_curves_csv(step_curves_path, analysis["step_curves"])
    plots = plot_analysis(analysis, output_dir)
    summary = {
        "config": {
            "model_path": str(Path(args.model_path).resolve()),
            "device": str(device),
            "dtype": args.dtype,
            "storage_dtype": str(storage_dtype).removeprefix("torch."),
            "dataset": args.dataset,
            "num_point_clouds": len(samples),
            "num_prompts_per_point_cloud": len(prompts),
            "max_new_tokens": args.max_new_tokens,
            "decode_passes_per_request": args.max_new_tokens - 1,
            "num_decoder_layers": num_layers,
            "block_sizes": list(args.block_sizes),
            "block_ratios": list(args.block_ratios),
            "seed": args.seed,
        },
        "interpretation_guardrails": [
            "The first generated token comes from prefill and has no decode-pass MLP trace.",
            "Post-gate sparsity can skip only down-projection weights.",
            "Oracle pre-gate uses current-token support and is a non-causal upper bound.",
            "Previous-token sparsity is causal; its quality is measured on current-token contribution.",
            "Weight-byte reductions exclude activation, index, cache, and metadata traffic.",
        ],
        "analysis": analysis,
        "samples": sample_outputs,
        "artifacts": {
            "raw_activations": str(raw_path),
            "details_jsonl_gz": str(details_path),
            "cross_prompt_jsonl_gz": str(cross_path),
            "curves_csv": str(curves_path),
            "layer_curves_csv": str(layer_curves_path),
            "step_curves_csv": str(step_curves_path),
            "plots": plots,
        },
    }
    summary_path = output_dir / "summary.json"
    with summary_path.open("w") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(f"wrote {summary_path}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Trace PointLLM decode MLP block liveness")
    parser.add_argument("--model_path", default="/mnt/llm_data/pointllm_ckpt/PointLLM_7B_v1.2_safetensors")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--storage_dtype", choices=("auto", "float16", "bfloat16", "float32"), default="auto")
    parser.add_argument("--dataset", choices=("modelnet", "objaverse", "npy", "synthetic"), default="modelnet")
    parser.add_argument("--modelnet_data_path", default="/mnt/llm_data/pointllm_data/modelnet40_data")
    parser.add_argument("--objaverse_data_path", default="/mnt/llm_data/pointllm_data/objaverse_660k_splits")
    parser.add_argument("--objaverse_anno_path", default="/mnt/llm_data/pointllm_data/anno_data/PointLLM_brief_description_val_200_GT.json")
    parser.add_argument("--point_cloud_path", action="append", default=[])
    parser.add_argument("--num_point_clouds", type=int, default=1)
    parser.add_argument("--prompt", action="append", default=[])
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--block_sizes", type=parse_csv_ints, default=parse_csv_ints("32,64,128,256"))
    parser.add_argument("--block_ratios", type=parse_csv_floats, default=parse_csv_floats("0.05,0.1,0.2,0.3,0.5,0.75,1.0"))
    parser.add_argument("--num_kvcache_blocks", type=int, default=256)
    parser.add_argument("--kvcache_block_size", type=int, default=16)
    parser.add_argument("--max_num_batched_tokens", type=int, default=4096)
    parser.add_argument("--require_layers", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", default="results/mlp_block_trace/modelnet_smoke")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
