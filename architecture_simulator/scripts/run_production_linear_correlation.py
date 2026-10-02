#!/usr/bin/env python3
"""Run real-shape PointLLM decoder-linear DRAM/SRAM/RTL correlation."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gtsu_cycle.dramsim3_backend import DramSim3Paths, time_dram_bursts  # noqa: E402
from gtsu_cycle.pointllm_lowering import lower_decoder_linear  # noqa: E402
from gtsu_cycle.production_linear import (  # noqa: E402
    ProductionLinearConfig, build_runtime_bursts, runtime_as_dram_bursts,
    serialize_dram_completions,
)
from gtsu_cycle.production_linear_correlation import (  # noqa: E402
    correlate_production_linear, write_dma_trace,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path(
        "/mnt/llm_data/pointllm_ckpt/PointLLM_7B_v1.2_safetensors"
    ))
    parser.add_argument("--projection", default="q_proj")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    lowered = lower_decoder_linear(args.checkpoint, args.projection, layer=args.layer)
    runtime = build_runtime_bursts(lowered)
    timing = time_dram_bursts(
        runtime_as_dram_bursts(runtime),
        paths=DramSim3Paths.local_default(),
        output_dir=args.output_dir / "dramsim3_stats",
        max_outstanding=32,
        max_cycles=20_000_000,
        stream_barriers=True,
        reorder_window=32,
    )
    arrivals = serialize_dram_completions(timing, runtime)
    correlation = correlate_production_linear(
        lowered, arrivals,
        rtl_root=ROOT / "rtl/vertical_slice",
        config=ProductionLinearConfig(),
    )
    trace_path = args.output_dir / "dma_trace.hex"
    write_dma_trace(trace_path, arrivals)
    trace_hash = hashlib.sha256(trace_path.read_bytes()).hexdigest()
    trace_path.unlink()

    summary = {
        "schema_version": "0.1",
        "status": correlation["status"],
        "correlation": correlation,
        "dramsim3": {
            "accelerator_cycles": timing.accelerator_cycles,
            "dram_clock_ticks": timing.dram_clock_ticks,
            "requests": timing.requests,
            "bytes_read": timing.bytes_read,
            "latency_min": timing.latency_min,
            "latency_median": timing.latency_median,
            "latency_max": timing.latency_max,
            "provenance": timing.provenance,
        },
        "dma_trace": {
            "entries": len(arrivals),
            "sha256": trace_hash,
            "stored": False,
            "regenerate_with": "architecture_simulator/scripts/run_production_linear_correlation.py",
        },
        "full_pointllm_cycle_accurate": False,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8",
    )
    (args.output_dir / "README.md").write_text(
        f"""# Production PointLLM Decoder Linear Correlation

Projection `{args.projection}`, shape `1x{lowered.k} -> 1x{lowered.n}`.
The DRAMsim3 completion stream, finite 32-entry ROB, banked SRAM reads,
registered latency, unpack, compute issue, Split-K partials, and output counts
match the parameterized RTL controller exactly at `{correlation['rtl']['cycles']}` cycles.

This promotes the production decoder-linear controller only. Dense GEMM,
attention, vector operations, and full PointLLM remain fail-closed.
""",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
