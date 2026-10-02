#!/usr/bin/env python3
"""Emit the strict RTL/cycle coverage matrix and optionally enforce operators."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gtsu_cycle.coverage import (
    UnsupportedCycleAccurateOperator,
    coverage_report,
    require_rtl_correlated,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require", default="", help="comma-separated operator contracts")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = coverage_report()
    if args.require:
        try:
            require_rtl_correlated(
                item.strip() for item in args.require.split(",") if item.strip()
            )
        except UnsupportedCycleAccurateOperator as error:
            parser.error(str(error))
    text = json.dumps(payload, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
