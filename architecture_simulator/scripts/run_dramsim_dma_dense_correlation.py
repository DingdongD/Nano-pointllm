#!/usr/bin/env python3
"""Run real DRAMsim3 completion-to-Dense RTL correlation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gtsu_cycle.dense_sram_pipeline import DenseSramPipelineConfig
from gtsu_cycle.dram_dma_correlation import correlate_dense_dramsim_dma
from gtsu_cycle.dramsim3_backend import DramSim3Paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rtl_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--dramsim_output_dir", type=Path)
    parser.add_argument("--rob_depth", type=int, default=4)
    args = parser.parse_args()
    dramsim_output = args.dramsim_output_dir or args.output_dir / "dramsim3"
    report = correlate_dense_dramsim_dma(
        DenseSramPipelineConfig(), rtl_root=args.rtl_root,
        dramsim_paths=DramSim3Paths.local_default(),
        dramsim_output_dir=dramsim_output, output_dir=args.output_dir,
        rob_depth=args.rob_depth,
    )
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "rtl_correlated" else 1


if __name__ == "__main__":
    raise SystemExit(main())
