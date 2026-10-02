#!/usr/bin/env python3
"""Plot comparable sequential-W8A8 summary JSON files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", action="append", required=True,
        help="LABEL=path/to/summary.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for item in args.input:
        if "=" not in item:
            raise ValueError("--input must use LABEL=PATH")
        label, raw_path = item.split("=", 1)
        payload = json.loads(Path(raw_path).read_text(encoding="utf-8"))
        rows.append((label, payload["aggregate"]))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    labels = [row[0] for row in rows]
    x = np.arange(len(rows))
    colors = ["#2364aa", "#f28f3b", "#2a9d6f", "#c44536"][:len(rows)]
    metrics = (
        ("kl_mean", "Teacher-Forced KL Mean", None),
        ("kl_max", "Teacher-Forced KL Max", None),
        ("logit_rmse", "Logit RMSE", None),
        ("free_generation_token_match", "Greedy Token Match", (0, 1.05)),
    )
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    for axis, (key, title, limits) in zip(axes.flat, metrics, strict=True):
        values = [row[1][key] for row in rows]
        bars = axis.bar(x, values, color=colors, width=0.68)
        axis.set_title(title)
        axis.set_xticks(x, labels=labels, rotation=18, ha="right")
        axis.grid(axis="y", alpha=0.2)
        if limits:
            axis.set_ylim(*limits)
        for bar, value in zip(bars, values, strict=True):
            axis.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{value:.4f}", ha="center", va="bottom", fontsize=9,
            )
    fig.suptitle("Sequential PointLLM W8A8 QDQ Smoke (1 Held-Out Request Each)")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
