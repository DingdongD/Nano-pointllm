#!/usr/bin/env python3
"""Generate deployed PointLLM geometry traces and correlate the production RTL selector."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gtsu_cycle.fused_fps_knn_real import (
    correlate_real_pointllm_fps_knn, generate_pointllm_geometry_trace,
)
from gtsu_cycle.geometry_semantics import load_modelnet_clouds, load_objaverse_clouds


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("modelnet", "objaverse"), required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--rtl_root", type=Path, required=True)
    parser.add_argument("--trace_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    count = args.sample + 1
    clouds = load_modelnet_clouds(args.data, count) if args.dataset == "modelnet" \
        else load_objaverse_clouds(args.data, count)
    cloud_id, points = clouds[args.sample]
    trace_dir = args.trace_root / args.dataset / cloud_id
    extension_source = (
        Path(__file__).resolve().parents[1] / "csrc" / "pointllm_geometry_trace.cu"
    )
    metadata = generate_pointllm_geometry_trace(
        points, trace_dir=trace_dir, extension_source=extension_source,
    )
    report = correlate_real_pointllm_fps_knn(
        rtl_root=args.rtl_root, trace_dir=trace_dir, output_dir=args.output_dir,
    )
    print(json.dumps({
        "dataset": args.dataset,
        "cloud_id": cloud_id,
        "point_sha256": metadata["point_sha256"],
        "status": report["status"],
        "rtl_cycles": report["rtl_cycles"],
        "mismatches": report["mismatches"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
