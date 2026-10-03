#!/usr/bin/env python3
"""Run geometry SRAM, AXI splitter, and DRAMsim3 request-side closure."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gtsu_cycle.geometry_dram import (
    correlate_axi_burst_splitter, time_geometry_preload,
)
from gtsu_cycle.geometry_sram_correlation import correlate_geometry_sram
from gtsu_cycle.geometry_dma_sram_correlation import correlate_geometry_dma_sram


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rtl_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--max_outstanding", type=int, default=32)
    args = parser.parse_args()
    sram = correlate_geometry_sram(
        rtl_root=args.rtl_root, output_dir=args.output_dir / "sram",
    )
    axi = correlate_axi_burst_splitter(
        rtl_root=args.rtl_root, output_dir=args.output_dir / "axi_splitter",
    )
    _timing, dram = time_geometry_preload(
        output_dir=args.output_dir / "dramsim3",
        max_outstanding=args.max_outstanding,
    )
    dma_sram = correlate_geometry_dma_sram(
        rtl_root=args.rtl_root,
        dramsim_output_dir=args.output_dir / "dma_sram" / "dramsim3",
        output_dir=args.output_dir / "dma_sram", rob_depth=8,
    )
    summary = {
        "schema_version": "0.1",
        "status": "memory_contract_correlated",
        "sram_status": sram["status"],
        "axi_splitter_status": axi["status"],
        "dramsim3_status": dram["status"],
        "dma_sram_status": dma_sram["status"],
        "geometry_sram_bytes": sram["production_mapping"][
            "total_geometry_sram_bytes"
        ],
        "dram_preload_bytes": dram["timing"]["bytes_read"],
        "dram_preload_cycles": dram["timing"]["accelerator_cycles"],
        "claim_boundary": {
            "sram_behavior_correlated": True,
            "axi_request_splitter_correlated": True,
            "dramsim3_request_side_closed_loop": True,
            "payload_dma_to_sram_rtl_composed": True,
            "payload_dma_to_sram_scope": "128-point reduced RTL lock",
            "fp32_arithmetic_and_selector_composed": False,
            "foundry_sram_sta_ppa": False,
            "full_pointllm_cycle_accurate": False,
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
