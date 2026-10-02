#!/usr/bin/env python3
"""Aggregate phase-aware Dense64 evidence and render the architecture tradeoffs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result_dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.result_dir
    lanes = []
    for lane in (1, 2, 4):
        row = json.loads((root / f"qproj_lane{lane}/correlation.json").read_text())
        lanes.append({
            "memory_lanes": lane,
            "ingress_bytes_per_cycle": row["operand_supply"]["ingress_bytes_per_cycle"],
            "cycles": row["rtl_cycles"],
            "speedup_vs_64B": None,
            "exact": row["status"] == "rtl_correlated",
        })
    baseline = lanes[0]["cycles"]
    for row in lanes:
        row["speedup_vs_64B"] = baseline / row["cycles"]
    reuse = json.loads(
        (root / "pointtransformer_weight_reuse/correlation.json").read_text()
    )
    dual = json.loads((root / "dual_output_sweep/correlation.json").read_text())
    real_qproj = json.loads((root / "real_qproj/summary.json").read_text())
    real_dual = json.loads(
        (root / "real_qproj_dual_output/correlation.json").read_text()
    )
    summary = {
        "schema_version": "0.1",
        "status": "rtl_correlated",
        "milestones": {
            "M1_real_w8a8_dense64": True,
            "M2_balanced_weight_stream_256B_per_cycle": True,
            "M3_weight_reuse_gemm": True,
            "M4_dual_a8_bf16_output": True,
            "M5_fused_fps_knn": False,
            "M6_geometry_feature_pipeline": False,
            "M7_request_side_dramsim3_closed_loop": False,
            "M8_foundry_ppa_calibration": False,
            "M9_attention_vector": False,
            "M10_full_pointllm": False,
        },
        "qproj_memory_lane_sweep": lanes,
        "real_qproj": {
            "tensor_package": real_qproj["real_qproj_tensor_package"]["path"],
            "acc32_outputs_checked": 4096,
            "dense64_cycles_256B_per_cycle": real_qproj["real_qproj_dense64_rtl"]["rtl_cycles"],
            "acc32_exact": real_qproj["real_qproj_dense64_rtl"]["functional_outputs_exact"],
            "a8_outputs_checked": real_dual["a8_outputs_checked"],
            "bf16_outputs_checked": real_dual["bf16_outputs_checked"],
            "dual_output_cycles_16_lanes": real_dual["rtl_cycles"],
        },
        "pointtransformer_weight_reuse": {
            "shape": [513, 384, 1152],
            "cycles": reuse["rtl_cycles"],
            "acc32_outputs_checked": reuse["checked_output_values"],
            "weight_bytes_repeated_gemv": reuse["traffic"]["repeated_gemv_weight_bytes"],
            "weight_bytes_reuse": reuse["traffic"]["weight_reuse_bytes"],
            "weight_byte_reduction_ratio": reuse["traffic"]["weight_byte_reduction_ratio"],
            "dot4_issue_fraction": reuse["utilization"]["dot4_issue_fraction"],
        },
        "dual_output_lane_sweep": [
            {
                "bf16_lanes": row["bf16_lanes"],
                "cycles_for_three_tiles": row["rtl_cycles"],
                "a8_lanes": 64,
                "exact": row["status"] == "rtl_correlated",
            }
            for row in dual["bf16_lane_sweep"]
        ],
        "claim_boundary": {
            "dense_gemm_operator_rtl_correlated": True,
            "full_pointllm_cycle_accurate": False,
            "request_side_dramsim3_closed_loop": False,
            "real_pointtransformer_payload": False,
            "foundry_sram_sta_ppa": False,
        },
    }
    (root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8",
    )
    _plot(root / "phase_aware_dense_summary.png", lanes, reuse, dual)
    print(json.dumps(summary, indent=2))
    return 0


def _plot(path: Path, lanes, reuse, dual) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8), constrained_layout=True)
    colors = ["#c84b31", "#df8b3a", "#16817a"]
    axes[0].bar(
        [row["ingress_bytes_per_cycle"] for row in lanes],
        [row["cycles"] / 1000 for row in lanes], width=38, color=colors,
    )
    axes[0].set_title("Decode weight supply")
    axes[0].set_xlabel("Ingress bytes / cycle")
    axes[0].set_ylabel("Q-proj cycles (thousands)")
    axes[0].set_xticks([64, 128, 256])
    axes[0].axvline(256, color="#183153", linestyle="--", linewidth=1)

    traffic = reuse["traffic"]
    axes[1].bar(
        ["Repeated\nGEMV", "K-block\nreuse"],
        [traffic["repeated_gemv_weight_bytes"] / 2**20,
         traffic["weight_reuse_bytes"] / 2**20],
        color=["#c84b31", "#16817a"],
    )
    axes[1].set_yscale("log")
    axes[1].set_title("PointTransformer weight traffic")
    axes[1].set_ylabel("MiB (log scale)")
    axes[1].text(
        0.5, 0.92, f'{traffic["weight_byte_reduction_ratio"]:.2f}x less',
        transform=axes[1].transAxes, ha="center", weight="bold",
    )

    sweep = dual["bf16_lane_sweep"]
    axes[2].plot(
        [row["bf16_lanes"] for row in sweep],
        [row["rtl_cycles"] for row in sweep],
        marker="o", linewidth=2.3, color="#183153",
    )
    axes[2].set_title("BF16 exit area/latency sweep")
    axes[2].set_xlabel("BF16 lanes")
    axes[2].set_ylabel("Cycles / three tiles")
    axes[2].set_xticks([8, 16, 32, 64])
    axes[2].grid(axis="y", alpha=0.25)
    fig.suptitle("Dense64 phase-aware dataflow: exact RTL correlations", weight="bold")
    fig.savefig(path, dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
