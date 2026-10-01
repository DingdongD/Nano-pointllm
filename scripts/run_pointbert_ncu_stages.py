#!/usr/bin/env python3
"""Capture B1/B4 PointBERT stages with NCU and render a unified roofline report."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
import time

from nanopointllm.gpu_telemetry import query_gpu_state
from nanopointllm.analysis.pointbert_roofline import (
    read_ncu_csv,
    rows_for_range,
    summarize_stage,
)


STAGE_RANGES = {
    "fps": "pointbert_fps",
    "knn": "pointbert_knn",
    "local_encoder": "pointbert_local_encoder",
    "pt_qkv": "pointbert_pt_qkv",
    "pt_attention": "pointbert_pt_attention",
    "pt_o_proj": "pointbert_pt_o_proj",
    "pt_mlp": "pointbert_pt_mlp",
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
    parser.add_argument("--batch_sizes", default="1,4")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--profile_iterations", type=int, default=1)
    parser.add_argument("--peak_hbm_gbps", type=float, default=1555.0)
    parser.add_argument("--peak_compute_tflops", type=float, default=312.0)
    parser.add_argument("--ncu", default="/usr/local/cuda/bin/ncu")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--replay_mode", default="kernel", choices=("application", "kernel"))
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty NCU stage CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)


def plot_report(results: dict, output: Path, peak_hbm_gbps: float, peak_compute_tflops: float) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    batches = sorted(results, key=int)
    stages = list(STAGE_RANGES)
    labels = [stage.replace("pt_", "PT ").replace("_", " ") for stage in stages]
    colors = ("#cf5c36", "#286f6c", "#d69f33", "#385f93")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    x = np.arange(len(stages))
    width = 0.34

    for index, batch in enumerate(batches):
        summaries = results[batch]["stage_summaries"]
        offset = (index - (len(batches) - 1) / 2) * width
        axes[0, 0].bar(
            x + offset,
            [summaries[stage]["total_profiled_kernel_time_ms"] for stage in stages],
            width,
            label=f"B={batch}",
            color=colors[index],
        )
        axes[0, 1].bar(
            x + offset,
            [summaries[stage]["measured_dram_bytes"] / 1e6 for stage in stages],
            width,
            label=f"B={batch}",
            color=colors[index],
        )
    axes[0, 0].set_title("Profiled CUDA kernel time")
    axes[0, 0].set_ylabel("ms")
    axes[0, 1].set_title("Measured DRAM traffic")
    axes[0, 1].set_ylabel("MB")

    marker_by_batch = {batch: marker for batch, marker in zip(batches, ("o", "s", "^", "D"))}
    for batch in batches:
        summaries = results[batch]["stage_summaries"]
        axes[1, 0].plot(
            x,
            [summaries[stage]["duration_weighted_dram_throughput_pct"] for stage in stages],
            marker=marker_by_batch[batch],
            label=f"DRAM B={batch}",
        )
        axes[1, 0].plot(
            x,
            [summaries[stage]["duration_weighted_sm_throughput_pct"] for stage in stages],
            marker=marker_by_batch[batch],
            linestyle="--",
            label=f"SM B={batch}",
        )
    axes[1, 0].set_title("Duration-weighted throughput")
    axes[1, 0].set_ylabel("% of peak")
    axes[1, 0].set_ylim(0, 100)

    all_ai = [
        results[batch]["stage_summaries"][stage]["measured_ai_flops_per_byte"]
        for batch in batches for stage in stages
    ]
    valid_ai = [value for value in all_ai if value and value > 0]
    minimum_ai = max(min(valid_ai) / 3, 1e-3)
    maximum_ai = max(max(valid_ai) * 3, peak_compute_tflops * 1000 / peak_hbm_gbps * 2)
    ai_grid = np.logspace(np.log10(minimum_ai), np.log10(maximum_ai), 300)
    roofline = np.minimum(peak_compute_tflops, ai_grid * peak_hbm_gbps / 1000)
    axes[1, 1].plot(ai_grid, roofline, color="#222222", linewidth=2, label="A100 roofline")
    for batch_index, batch in enumerate(batches):
        summaries = results[batch]["stage_summaries"]
        for stage_index, stage in enumerate(stages):
            summary = summaries[stage]
            ai = summary["measured_ai_flops_per_byte"]
            achieved = summary["achieved_tflops_from_estimated_work"]
            if ai and achieved:
                axes[1, 1].scatter(
                    ai,
                    achieved,
                    marker=marker_by_batch[batch],
                    color=plt.cm.tab10(stage_index),
                    s=55,
                )
                axes[1, 1].annotate(
                    f"{stage.replace('pt_', '')} B{batch}",
                    (ai, achieved),
                    xytext=(4, 4),
                    textcoords="offset points",
                    fontsize=7,
                )
    axes[1, 1].set_xscale("log")
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_title("PointBERT measured-byte roofline")
    axes[1, 1].set_xlabel("Estimated FLOP / measured DRAM byte")
    axes[1, 1].set_ylabel("Estimated achieved TFLOP/s")
    axes[1, 1].legend(frameon=False)

    for axis in (axes[0, 0], axes[0, 1], axes[1, 0]):
        axis.set_xticks(x, labels, rotation=25, ha="right")
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False)
    axes[1, 1].spines[["top", "right"]].set_visible(False)
    fig.suptitle("PointLLM PointBERT hardware-counter breakdown", fontsize=16)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    batch_sizes = [int(value) for value in args.batch_sizes.split(",") if value.strip()]
    if not batch_sizes or any(value <= 0 for value in batch_sizes):
        raise SystemExit("batch_sizes must contain positive integers")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[1]
    results = {}
    device_index = int(args.device.split(":", 1)[1]) if ":" in args.device else 0

    for batch_size in batch_sizes:
        batch_dir = output_dir / f"b{batch_size}"
        batch_dir.mkdir(parents=True, exist_ok=True)
        all_csv = batch_dir / "all_stages.csv"
        manifest_path = batch_dir / "manifest.json"
        precheck_path = batch_dir / "gpu_precheck.json"
        if results:
            time.sleep(2.0)
        uuid = subprocess.check_output(
            [
                "nvidia-smi", "--id", str(device_index),
                "--query-gpu=uuid", "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip()
        gpu_precheck = query_gpu_state(uuid)
        idle = (
            gpu_precheck["gpu_utilization_pct"] <= 5.0
            and gpu_precheck["memory_used_mb"] <= 1024.0
        )
        if not idle:
            raise RuntimeError(
                "target GPU is not idle before NCU attach: "
                f"util={gpu_precheck['gpu_utilization_pct']:.0f}%, "
                f"memory={gpu_precheck['memory_used_mb']:.0f}MiB"
            )
        precheck_path.write_text(json.dumps(gpu_precheck, indent=2), encoding="utf-8")
        profile_command = [
            args.python,
            str(repo / "scripts/profile_pointbert_ncu.py"),
            "--model_path", args.model_path,
            "--device", args.device,
            "--dtype", args.dtype,
            "--batch_size", str(batch_size),
            "--warmup", str(args.warmup),
            "--profile_iterations", str(args.profile_iterations),
            "--manifest_output", str(manifest_path),
            "--gpu_precheck_json", str(precheck_path),
            "--require_idle_gpu",
        ]
        command = [
            args.ncu,
            "--target-processes", "all",
            "--replay-mode", args.replay_mode,
            "--nvtx",
        ]
        for nvtx_range in STAGE_RANGES.values():
            command.extend(("--nvtx-include", f"{nvtx_range}/"))
        command.extend((
            "--metrics", ",".join(NCU_METRICS),
            "--csv",
            "--log-file", str(all_csv),
            *profile_command,
        ))
        print("+", " ".join(command), flush=True)
        subprocess.run(command, check=True)

        rows = read_ncu_csv(all_csv)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        summaries = {}
        for stage, nvtx_range in STAGE_RANGES.items():
            stage_rows = rows_for_range(rows, nvtx_range)
            if not stage_rows:
                raise RuntimeError(f"NCU captured no kernels for {stage!r} at B={batch_size}")
            write_rows(batch_dir / f"{stage}.csv", stage_rows)
            summaries[stage] = summarize_stage(
                stage_rows,
                manifest["stages"][stage],
                peak_hbm_gbps=args.peak_hbm_gbps,
                peak_compute_tflops=args.peak_compute_tflops,
            )
            (batch_dir / f"{stage}_summary.json").write_text(
                json.dumps(summaries[stage], indent=2), encoding="utf-8"
            )
        results[str(batch_size)] = {
            "manifest": manifest,
            "stage_summaries": summaries,
        }

    report = {
        "config": vars(args),
        "batches": results,
        "methodology_note": (
            "NCU measured bytes/time/throughput are hardware counters. FLOPs are an explicit operation model; "
            "FPS/KNN comparisons and indexing plus normalization/nonlinear/reduction work are not counted."
        ),
    }
    summary_path = output_dir / "pointbert_roofline_summary.json"
    summary_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    plot_report(
        results,
        output_dir / "pointbert_roofline.png",
        args.peak_hbm_gbps,
        args.peak_compute_tflops,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
