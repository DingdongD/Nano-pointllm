#!/usr/bin/env python3
"""Run Dense GEMM M/N/K ping-pong-controller RTL correlation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gtsu_cycle.dense_controller import DenseControllerConfig
from gtsu_cycle.dense_controller_correlation import correlate_dense_controller


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rtl_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--m", type=int, default=3)
    parser.add_argument("--n", type=int, default=70)
    parser.add_argument("--k", type=int, default=132)
    parser.add_argument("--n_tile", type=int, default=64)
    parser.add_argument("--k_block", type=int, default=64)
    args = parser.parse_args()
    config = DenseControllerConfig(
        m=args.m, n=args.n, k=args.k, n_tile=args.n_tile, k_block=args.k_block,
    )
    report = correlate_dense_controller(
        config, rtl_root=args.rtl_root, output_dir=args.output_dir,
    )
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "rtl_correlated" else 1


if __name__ == "__main__":
    raise SystemExit(main())
