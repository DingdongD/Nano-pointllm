#!/usr/bin/env python3
"""Run exact shared-Dot4 dense microtile RTL correlation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gtsu_cycle.dense_gemm import DenseTileConfig
from gtsu_cycle.dense_gemm_correlation import correlate_dense_tile


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rtl_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=5)
    parser.add_argument("--columns", type=int, default=61)
    parser.add_argument("--n_tile", type=int, default=64)
    parser.add_argument("--k", type=int, default=384)
    args = parser.parse_args()
    config = DenseTileConfig(
        rows=args.rows, columns=args.columns, n_tile=args.n_tile, k=args.k,
    )
    report = correlate_dense_tile(
        config, rtl_root=args.rtl_root, output_dir=args.output_dir,
    )
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "rtl_correlated" else 1


if __name__ == "__main__":
    raise SystemExit(main())
