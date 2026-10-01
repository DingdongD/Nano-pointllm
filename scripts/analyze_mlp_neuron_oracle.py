#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path
from typing import Any

import torch

from nanopointllm.analysis.mlp_block_trace import load_mlp_block_trace
from nanopointllm.analysis.neuron_oracle import analyze_neuron_oracle


def parse_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(",") if item)


def parse_floats(value: str) -> tuple[float, ...]:
    return tuple(float(item) for item in value.split(",") if item)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_plots(result: dict[str, Any], output_dir: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary = result["summary"]
    sizes = [row["group_size"] for row in summary]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    axes[0].plot(sizes, [row["gini_mean"] for row in summary], marker="o", label="Gini")
    axes[0].plot(sizes, [row["normalized_entropy_mean"] for row in summary], marker="s", label="Normalized entropy")
    axes[0].plot(sizes, [row["effective_support_ratio_mean"] for row in summary], marker="^", label="Effective support ratio")
    axes[0].set_xscale("log", base=2)
    axes[0].set_xticks(sizes, labels=[str(size) for size in sizes])
    axes[0].set_title("Contribution Concentration")
    axes[0].legend(frameon=False)

    budgets = result["budget_ratios"]
    for row in summary:
        axes[1].plot(
            budgets,
            [row[f"retention_at_{int(round(b * 100)):02d}pct_mean"] for b in budgets],
            marker="o",
            label=f"B={row['group_size']}",
        )
    axes[1].plot([0, 1], [0, 1], color="black", linestyle=":", label="uniform")
    axes[1].set_title("Fine-Grained Oracle")
    axes[1].set_xlabel("Retained neuron ratio")
    axes[1].set_ylabel("Contribution retained")
    axes[1].legend(frameon=False)

    targets = result["contribution_targets"]
    for row in summary:
        axes[2].plot(
            targets,
            [row[f"neuron_ratio_for_{int(round(t * 100)):02d}pct_contribution_mean"] for t in targets],
            marker="o",
            label=f"B={row['group_size']}",
        )
    axes[2].set_title("Support Needed for Contribution")
    axes[2].set_xlabel("Contribution target")
    axes[2].set_ylabel("Required neuron ratio")
    for axis in axes:
        axis.set_ylim(0.0, 1.02)
        axis.grid(alpha=0.2)
    fig.tight_layout()
    overview = output_dir / "neuron_oracle_overview.png"
    fig.savefig(overview, dpi=180)
    plt.close(fig)

    neuron_layers = [row for row in result["by_layer"] if row["group_size"] == 1]
    fig, ax = plt.subplots(figsize=(12, 5))
    layers = [row["layer"] for row in neuron_layers]
    ax.plot(layers, [row["retention_at_10pct_mean"] for row in neuron_layers], label="retention @ 10%", color="#0b6e4f")
    ax.plot(layers, [row["neuron_ratio_for_90pct_contribution_mean"] for row in neuron_layers], label="ratio for 90% contribution", color="#c44536")
    ax.plot(layers, [row["effective_support_ratio_mean"] for row in neuron_layers], label="effective support ratio", color="#33658a")
    ax.set_xlabel("Decoder layer")
    ax.set_ylabel("Ratio")
    ax.set_ylim(0.0, 1.02)
    ax.set_title("Neuron-Level Oracle by Layer")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    layer_plot = output_dir / "neuron_oracle_by_layer.png"
    fig.savefig(layer_plot, dpi=180)
    plt.close(fig)
    return [str(overview), str(layer_plot)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze fine-grained MLP neuron contribution oracle")
    parser.add_argument("--trace", required=True)
    parser.add_argument("--group_sizes", type=parse_ints, default=parse_ints("1,4,8,16,32"))
    parser.add_argument("--budget_ratios", type=parse_floats, default=parse_floats("0.01,0.02,0.05,0.1,0.2,0.3,0.5,0.75,1.0"))
    parser.add_argument("--contribution_targets", type=parse_floats, default=parse_floats("0.5,0.8,0.9,0.95,0.99"))
    parser.add_argument("--chunk_size", type=int, default=64)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = load_mlp_block_trace(args.trace)
    result = analyze_neuron_oracle(
        payload,
        group_sizes=args.group_sizes,
        budget_ratios=args.budget_ratios,
        contribution_targets=args.contribution_targets,
        chunk_size=args.chunk_size,
    )
    details = result.pop("details")
    with gzip.open(output_dir / "neuron_oracle_details.jsonl.gz", "wt") as handle:
        for row in details:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    write_csv(output_dir / "neuron_oracle_summary.csv", result["summary"])
    write_csv(output_dir / "neuron_oracle_by_layer.csv", result["by_layer"])
    write_csv(output_dir / "neuron_oracle_by_step.csv", result["by_step"])
    result["artifacts"] = make_plots(result, output_dir)
    result["source_trace"] = str(Path(args.trace).resolve())
    with (output_dir / "summary.json").open("w") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result["decision"], indent=2))


if __name__ == "__main__":
    main()
