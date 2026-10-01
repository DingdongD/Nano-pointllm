#!/usr/bin/env python3
"""Capture and summarize stage-isolated PointLLM decode metrics with Nsight Compute."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


STAGE_RANGES = {
    "qkv": "pointllm_qkv",
    "attention": "pointllm_attention",
    "o_proj": "pointllm_o_proj",
    "mlp": "pointllm_mlp",
    "lm_head": "pointllm_lm_head",
}
NCU_METRICS = (
    "gpu__time_duration.sum",
    "sm__cycles_active.avg.pct_of_peak_sustained_elapsed",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16"))
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--input_length", type=int, default=768)
    parser.add_argument("--warmup_decode_steps", type=int, default=10)
    parser.add_argument("--profile_steps", type=int, default=1)
    parser.add_argument("--stages", default=",".join(STAGE_RANGES))
    parser.add_argument("--peak_hbm_gbps", type=float, default=1555.0)
    parser.add_argument("--peak_compute_tflops", type=float, default=312.0)
    parser.add_argument("--min_dram_throughput_pct", type=float, default=50.0)
    parser.add_argument("--ncu", default="/usr/local/cuda/bin/ncu")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--no_plot", action="store_true")
    return parser.parse_args()


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def plot_stage_summary(summaries: dict, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stages = [stage for stage in STAGE_RANGES if stage in summaries]
    durations = [summaries[stage]["total_profiled_kernel_time_ms"] for stage in stages]
    dram = [summaries[stage]["duration_weighted_dram_throughput_pct"] for stage in stages]
    sm = [summaries[stage]["duration_weighted_sm_throughput_pct"] or 0.0 for stage in stages]
    intensity = [
        summaries[stage]["bottleneck"]["arithmetic_intensity_flops_per_byte"]
        for stage in stages
    ]
    ridge = next(
        (
            summaries[stage]["bottleneck"]["ridge_point_flops_per_byte"]
            for stage in stages
            if summaries[stage]["bottleneck"]["ridge_point_flops_per_byte"] is not None
        ),
        None,
    )
    colors = [
        "#c94f3d"
        if summaries[stage]["bottleneck"]["strict_weight_bound_confirmed"]
        else "#397a8f"
        for stage in stages
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    axes[0].bar(stages, durations, color=colors)
    axes[0].set_title("Profiled CUDA time")
    axes[0].set_ylabel("ms")

    x = list(range(len(stages)))
    width = 0.38
    axes[1].bar([value - width / 2 for value in x], dram, width, label="DRAM", color="#c9822b")
    axes[1].bar([value + width / 2 for value in x], sm, width, label="SM", color="#397a8f")
    axes[1].set_xticks(x, stages)
    axes[1].set_ylim(0, 100)
    axes[1].set_title("Duration-weighted throughput")
    axes[1].set_ylabel("% of peak")
    axes[1].legend(frameon=False)

    axes[2].bar(stages, intensity, color=colors)
    if ridge is not None:
        axes[2].axhline(ridge, color="#222222", linestyle="--", label=f"ridge={ridge:.1f}")
        axes[2].legend(frameon=False)
    axes[2].set_yscale("log")
    axes[2].set_title("Arithmetic intensity")
    axes[2].set_ylabel("FLOP / byte (log)")
    for axis in axes:
        axis.tick_params(axis="x", rotation=25)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("PointLLM decode stage bottlenecks")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    stages = [stage.strip() for stage in args.stages.split(",") if stage.strip()]
    unknown = sorted(set(stages) - set(STAGE_RANGES))
    if unknown:
        raise SystemExit(f"unknown stages: {unknown}")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[1]
    summaries = {}

    for stage in stages:
        csv_path = output_dir / f"{stage}.csv"
        manifest_path = output_dir / f"{stage}_manifest.json"
        summary_path = output_dir / f"{stage}_summary.json"
        profile_command = [
            args.python,
            str(repo / "scripts/profile_pointllm_ncu.py"),
            "--model_path", args.model_path,
            "--device", args.device,
            "--dtype", args.dtype,
            "--batch_size", str(args.batch_size),
            "--input_length", str(args.input_length),
            "--warmup_decode_steps", str(args.warmup_decode_steps),
            "--profile_steps", str(args.profile_steps),
            "--manifest_output", str(manifest_path),
            "--require_idle_gpu",
        ]
        ncu_command = [
            args.ncu,
            "--target-processes", "all",
            "--nvtx",
            "--nvtx-include", f"{STAGE_RANGES[stage]}/",
            "--metrics", ",".join(NCU_METRICS),
            "--csv",
            "--log-file", str(csv_path),
            *profile_command,
        ]
        run(ncu_command)
        run([
            args.python,
            str(repo / "scripts/summarize_ncu_pointllm.py"),
            "--input_csv", str(csv_path),
            "--output_json", str(summary_path),
            "--stage", stage,
            "--manifest", str(manifest_path),
            "--peak_hbm_gbps", str(args.peak_hbm_gbps),
            "--peak_compute_tflops", str(args.peak_compute_tflops),
            "--min_dram_throughput_pct", str(args.min_dram_throughput_pct),
        ])
        summaries[stage] = json.loads(summary_path.read_text(encoding="utf-8"))

    linear_stages = ("qkv", "o_proj", "mlp", "lm_head")
    has_all_linear_stages = all(stage in summaries for stage in linear_stages)
    plot_path = output_dir / "stage_bottleneck.png"
    if not args.no_plot:
        plot_stage_summary(summaries, plot_path)
    result = {
        "config": vars(args),
        "stage_summaries": summaries,
        "strict_weight_bound_stages": [
            stage
            for stage, summary in summaries.items()
            if summary["bottleneck"]["strict_weight_bound_confirmed"]
        ],
        "all_linear_stages_strictly_weight_bound": (
            all(
                summaries[stage]["bottleneck"]["strict_weight_bound_confirmed"]
                for stage in linear_stages
            )
            if has_all_linear_stages
            else None
        ),
        "attention_note": (
            "Attention has no decoder weights in this accounting; classify it from paged K/V "
            "traffic rather than calling it weight-bound."
        ),
        "plot": str(plot_path) if not args.no_plot else None,
    }
    aggregate_path = output_dir / "stage_bottleneck_summary.json"
    aggregate_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
