#!/usr/bin/env python3
"""Run edge or full q-projection Dense64 ABUF/WBUF RTL correlation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gtsu_cycle.production_dense_fabric import ProductionDenseFabricConfig
from gtsu_cycle.production_dense_fabric_correlation import (
    correlate_production_dense_fabric,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rtl_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--shape", choices=("edge", "q_proj"), default="edge")
    parser.add_argument("--memory_lanes", type=int, choices=(1, 2, 4), default=1)
    args = parser.parse_args()
    if args.shape == "q_proj":
        config = ProductionDenseFabricConfig(
            m=1, n=4096, k=4096, source_stall_mod=0,
            output_stall_mod=0, memory_lanes=args.memory_lanes,
            max_cycles=1_000_000,
        )
        simulator, internal_trace, synthesize = "verilator", False, False
    else:
        config = ProductionDenseFabricConfig(memory_lanes=args.memory_lanes)
        simulator, internal_trace, synthesize = "iverilog", True, True
    report = correlate_production_dense_fabric(
        config, rtl_root=args.rtl_root, output_dir=args.output_dir,
        simulator=simulator, internal_trace=internal_trace,
        synthesize=synthesize,
    )
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "rtl_correlated" else 1


if __name__ == "__main__":
    raise SystemExit(main())
