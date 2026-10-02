#!/usr/bin/env python3
"""Run the full `[513,384]x[384,1152]` Dense64 reuse lock."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gtsu_cycle.weight_reuse_production_correlation import (
    correlate_pointtransformer_weight_reuse,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rtl_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    report = correlate_pointtransformer_weight_reuse(
        rtl_root=args.rtl_root, output_dir=args.output_dir,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
