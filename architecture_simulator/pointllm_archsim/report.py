"""CSV and optional plot output for simulation and DSE."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from .schema import SimulationResult


def write_rows_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output = Path(path)
    fields = sorted({key for row in rows for key in row})
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def plot_simulation(result: SimulationResult, path: str | Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    phases = list(result.phase_results)
    names = [item.phase.replace("_", " ") for item in phases]
    latency = [item.seconds * 1e3 for item in phases]
    hbm = [item.hbm_bytes / 1e9 for item in phases]
    operations = sorted(result.operation_results, key=lambda item: item.cycles, reverse=True)[:12]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    axes[0].bar(names, latency, color="#d0643b")
    axes[0].set_ylabel("Modeled latency (ms)")
    axes[0].set_title("Phase latency")
    axes[1].bar(names, hbm, color="#27736f")
    axes[1].set_ylabel("HBM traffic (GB)")
    axes[1].set_title("Phase traffic")
    axes[2].barh(
        [item.name for item in reversed(operations)],
        [item.cycles / result.config["clock_hz"] * 1e3 for item in reversed(operations)],
        color="#d3a13b",
    )
    axes[2].set_xlabel("Modeled latency (ms)")
    axes[2].set_title("Top operations")
    for axis in axes[:2]:
        axis.tick_params(axis="x", rotation=25)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("PointLLM phase-adaptive architecture model")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_sweep(rows: list[dict[str, Any]], pareto: list[dict[str, Any]], path: str | Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pareto_names = {item["config"] for item in pareto}
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    dominated = [row for row in rows if row["config"] not in pareto_names]
    frontier = [row for row in rows if row["config"] in pareto_names]
    panels = (
        ("tensor_pe_count", "latency_ms"),
        ("hbm_bandwidth_gbps", "llm_decode_ms"),
        ("weight_unpack_bytes_per_cycle", "llm_decode_ms"),
    )
    for axis, (x_key, y_key) in zip(axes, panels):
        axis.scatter(
            [row[x_key] for row in dominated],
            [row[y_key] for row in dominated],
            color="#88a9a6",
            alpha=0.35,
            label="dominated",
        )
        axis.scatter(
            [row[x_key] for row in frontier],
            [row[y_key] for row in frontier],
            color="#d0643b",
            edgecolor="#7e321b",
            linewidth=0.5,
            alpha=0.95,
            label="Pareto",
        )
    axes[0].set_xlabel("Tensor PE count")
    axes[0].set_ylabel("End-to-end modeled latency (ms)")
    axes[0].set_title("Compute-resource sweep")
    axes[1].set_xlabel("HBM bandwidth (GB/s)")
    axes[1].set_ylabel("Decode modeled latency (ms)")
    axes[1].set_title("Weight-streaming sweep")
    axes[2].set_xlabel("Weight unpack throughput (B/cycle)")
    axes[2].set_ylabel("Decode modeled latency (ms)")
    axes[2].set_title("Precision unpack sweep")
    axes[0].legend(frameon=False)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(alpha=0.2)
    fig.savefig(path, dpi=180)
    plt.close(fig)
