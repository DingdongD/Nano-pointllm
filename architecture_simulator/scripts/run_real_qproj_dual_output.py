#!/usr/bin/env python3
"""Run real layer-0 q-projection A8/BF16 output RTL correlation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gtsu_cycle.real_qproj_dual_output_correlation import correlate_real_qproj_dual_output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tensor_package_dir", type=Path, required=True)
    parser.add_argument("--rtl_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--bf16_lanes", type=int, choices=(8, 16, 32, 64), default=16)
    args = parser.parse_args()
    report = correlate_real_qproj_dual_output(
        tensor_package_dir=args.tensor_package_dir, rtl_root=args.rtl_root,
        output_dir=args.output_dir, bf16_lanes=args.bf16_lanes,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
