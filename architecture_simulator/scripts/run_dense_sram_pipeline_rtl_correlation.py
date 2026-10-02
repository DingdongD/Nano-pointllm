#!/usr/bin/env python3
"""Run exact physical Dense SRAM/Dot4/requant RTL correlation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gtsu_cycle.dense_sram_pipeline import DenseSramPipelineConfig
from gtsu_cycle.dense_sram_pipeline_correlation import correlate_dense_sram_pipeline


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rtl_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    report = correlate_dense_sram_pipeline(
        DenseSramPipelineConfig(), rtl_root=args.rtl_root,
        output_dir=args.output_dir,
    )
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "rtl_correlated" else 1


if __name__ == "__main__":
    raise SystemExit(main())
