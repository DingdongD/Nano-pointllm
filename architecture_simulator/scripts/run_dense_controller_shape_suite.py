#!/usr/bin/env python3
"""Correlate representative controller traces for PointLLM dense K classes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gtsu_cycle.dense_controller import DenseControllerConfig
from gtsu_cycle.dense_controller_correlation import correlate_dense_controller


CASES = {
    "local_encoder_k8": DenseControllerConfig(m=2, n=70, k=8, k_block=8),
    "point_transformer_k384": DenseControllerConfig(m=2, n=70, k=384),
    "projector_k2048": DenseControllerConfig(m=1, n=70, k=2048),
    "llm_prefill_k4096": DenseControllerConfig(m=1, n=70, k=4096),
    "edge_k132": DenseControllerConfig(m=3, n=70, k=132),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rtl_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reports = {}
    for name, config in CASES.items():
        reports[name] = correlate_dense_controller(
            config, rtl_root=args.rtl_root, output_dir=args.output_dir / name,
        )
    payload = {
        "schema_version": "0.1",
        "status": "rtl_correlated" if all(
            report["status"] == "rtl_correlated" for report in reports.values()
        ) else "mismatch",
        "cases": {
            name: {
                "config": report["config"],
                "cycles": report["rtl_cycles"],
                "cycle_error": report["cycle_error"],
                "event_trace_exact": report["event_trace_exact"],
                "counter_trace_exact": report["counter_trace_exact"],
            }
            for name, report in reports.items()
        },
        "claim_boundary": {
            "representative_k_shape_controller_correlated": True,
            "representative_rows_and_edge_n_tile_only": True,
            "full_production_shape_trace_correlated": False,
            "physical_banked_sram_integrated": False,
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(payload, indent=2))
    return 0 if payload["status"] == "rtl_correlated" else 1


if __name__ == "__main__":
    raise SystemExit(main())
