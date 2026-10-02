#!/usr/bin/env python3
"""Generate SRAM RTL, PointLLM lowering, and DRAMsim3 evidence artifacts."""
from __future__ import annotations

import argparse
import csv
from itertools import islice
import json
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gtsu_cycle.coverage import coverage_report  # noqa: E402
from gtsu_cycle.dramsim3_backend import (  # noqa: E402
    DramSim3Paths,
    time_dram_bursts,
)
from gtsu_cycle.pointllm_lowering import (  # noqa: E402
    lower_decoder_linear,
    lower_decoder_manifest,
)
from gtsu_cycle.sram_correlation import correlate_banked_sram  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("/mnt/llm_data/pointllm_ckpt/PointLLM_7B_v1.2_safetensors"),
    )
    parser.add_argument("--projection", default="q_proj")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--token", type=int, default=0)
    parser.add_argument("--sample-bursts", type=int, default=64)
    parser.add_argument("--max-outstanding", type=int, default=32)
    parser.add_argument("--accelerator-clock-hz", type=int, default=1_000_000_000)
    parser.add_argument("--dramsim3-lib", type=Path)
    parser.add_argument("--dramsim3-config", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.sample_bursts <= 0:
        raise SystemExit("--sample-bursts must be positive")
    defaults = DramSim3Paths.local_default()
    paths = DramSim3Paths(
        library=args.dramsim3_lib or defaults.library,
        config=args.dramsim3_config or defaults.config,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sram = correlate_banked_sram(
        rtl_root=ROOT / "rtl/vertical_slice",
        output_dir=args.output_dir,
    )
    lowered = lower_decoder_linear(
        args.checkpoint, args.projection,
        layer=args.layer, token=args.token,
    )
    lowering = lower_decoder_manifest(
        args.checkpoint, layer=args.layer, token=args.token,
    )
    (args.output_dir / "pointllm_lowering.json").write_text(
        json.dumps(lowering, indent=2) + "\n", encoding="utf-8",
    )

    bursts = tuple(islice(lowered.iter_weight_bursts(), args.sample_bursts))
    dram = time_dram_bursts(
        bursts,
        paths=paths,
        output_dir=args.output_dir / "dramsim3_stats",
        accelerator_clock_hz=args.accelerator_clock_hz,
        max_outstanding=args.max_outstanding,
    )
    dram_payload = dram.as_dict()
    (args.output_dir / "dramsim3_sample.json").write_text(
        json.dumps(dram_payload, indent=2) + "\n", encoding="utf-8",
    )
    with (args.output_dir / "dramsim3_trace.csv").open(
        "w", encoding="utf-8", newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("cycle", "event", "tag", "address", "stream", "latency"),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(dram_payload["events"])

    synthesis = _yosys_check()
    coverage = coverage_report()
    summary = {
        "schema_version": "0.1",
        "status": "memory_vertical_slices_validated",
        "sram": sram,
        "pointllm_lowering": {
            "projection": args.projection,
            "layer": args.layer,
            "shape": {"m": lowered.m, "n": lowered.n, "k": lowered.k},
            "source_dtype": lowered.source.dtype,
            "target_weight_payload_bytes": lowered.target_weight_payload_bytes,
            "weight_burst_count": lowered.weight_burst_count,
            "tile_count": len(lowered.tiles),
            "validation": lowered.as_dict(sample_tiles=0, sample_bursts=0)["validation"],
        },
        "dramsim3_sample": {
            key: value for key, value in dram_payload.items() if key != "events"
        },
        "yosys": synthesis,
        "coverage": coverage,
        "claim_boundary": {
            "sram_port_cycle_rtl_correlated": True,
            "pointllm_shape_address_byte_exact": True,
            "sampled_dram_timing_real_dramsim3": True,
            "compute_memory_integrated_cycle_rtl_correlated": False,
            "full_pointllm_cycle_accurate": False,
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8",
    )
    (args.output_dir / "coverage.json").write_text(
        json.dumps(coverage, indent=2) + "\n", encoding="utf-8",
    )
    (args.output_dir / "README.md").write_text(
        _readme(summary, args), encoding="utf-8",
    )
    print(json.dumps(summary["claim_boundary"], indent=2))


def _yosys_check() -> dict[str, object]:
    yosys = shutil.which("yosys")
    if not yosys:
        return {"status": "unavailable"}
    rtl = ROOT / "rtl/vertical_slice/gtsu_banked_sram_2client.sv"
    command = (
        f"read_verilog -sv {rtl}; "
        "hierarchy -check -top gtsu_banked_sram_2client; "
        "proc; memory; opt; check -assert"
    )
    result = subprocess.run(
        [yosys, "-q", "-p", command], check=False,
        capture_output=True, text=True, timeout=60,
    )
    if result.returncode:
        raise RuntimeError(f"SRAM Yosys check failed:\n{result.stdout}\n{result.stderr}")
    return {
        "status": "passed",
        "configuration": "compact default logic smoke; production geometry is RTL-simulated",
        "warnings": [line for line in result.stderr.splitlines() if line.strip()],
    }


def _readme(summary: dict[str, object], args: argparse.Namespace) -> str:
    sram = summary["sram"]
    dram = summary["dramsim3_sample"]
    lowering = summary["pointllm_lowering"]
    return f"""# GTSU Memory And PointLLM Lowering Evidence

This artifact uses the local Compiler_Codes LBUF contract as the SRAM interface
reference and the pinned local DRAMsim3 bridge for sampled DDR timing.

## Verified

- SRAM Python/RTL exact correlation: `{sram['cycle_model_cycles']}` cycles,
  `{sram['events']}` events, cycle error `{sram['cycle_error']}`.
- PointLLM `{args.projection}` source shape: `{lowering['shape']}`, source
  `{lowering['source_dtype']}`, target W8 bytes `{lowering['target_weight_payload_bytes']}`.
- DRAMsim3 sample: `{dram['requests']}` x 64-byte requests,
  `{dram['accelerator_cycles']}` accelerator cycles, latency min/median/max
  `{dram['latency_min']}/{dram['latency_median']}/{dram['latency_max']}` cycles.

## Claim Boundary

The SRAM port contract is RTL-correlated, PointLLM shape/address/byte lowering is
exact, and sampled memory requests use real DRAMsim3 timing. Compute and memory
have not yet been integrated into one RTL-correlated production linear pipeline,
so full PointLLM cycle accuracy remains false and no end-to-end speedup is claimed.
"""


if __name__ == "__main__":
    main()
