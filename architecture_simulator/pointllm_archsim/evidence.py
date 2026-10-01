"""Import measured Nano-pointllm summaries without fitting undocumented constants."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def extract_ncu_evidence(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root)
    pointbert_path = root / "docs/results/pointbert_ncu_roofline_2026-10-01/pointbert_roofline_summary.json"
    decoder_dir = root / "docs/results/ncu_weight_bound_2026-09-30"
    pointbert = json.loads(pointbert_path.read_text(encoding="utf-8"))
    decoder = {
        batch: json.loads((decoder_dir / f"b{batch}_summary.json").read_text(encoding="utf-8"))
        for batch in (1, 8)
    }
    return {
        "schema_version": "0.1",
        "pointbert": {
            batch: {
                stage: {
                    "time_ms": values["total_profiled_kernel_time_ms"],
                    "dram_pct": values["duration_weighted_dram_throughput_pct"],
                    "sm_pct": values["duration_weighted_sm_throughput_pct"],
                    "dram_bytes": values["measured_dram_bytes"],
                    "measured_ai": values["measured_ai_flops_per_byte"],
                }
                for stage, values in data["stage_summaries"].items()
            }
            for batch, data in pointbert["batches"].items()
        },
        "decoder": {
            str(batch): {
                stage: {
                    "time_ms": values["total_profiled_kernel_time_ms"],
                    "dram_pct": values["duration_weighted_dram_throughput_pct"],
                    "sm_pct": values["duration_weighted_sm_throughput_pct"],
                    "classification": values["bottleneck"]["classification"],
                }
                for stage, values in data["stage_summaries"].items()
            }
            for batch, data in decoder.items()
        },
        "usage": (
            "Measured GPU data is a workload/bottleneck reference. Do not directly fit ASIC cycles "
            "to A100 latency; calibration_scales require an RTL/C-model correlation target."
        ),
    }
