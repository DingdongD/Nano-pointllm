#!/usr/bin/env python3
"""Validate hardware geometry semantics against deployed PointLLM operators."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gtsu_cycle.geometry_semantics import (
    evaluate_pointllm_geometry, load_modelnet_clouds, load_objaverse_clouds,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modelnet", type=Path, required=True)
    parser.add_argument("--objaverse", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    clouds = {
        "modelnet": load_modelnet_clouds(args.modelnet, args.samples),
        "objaverse": load_objaverse_clouds(args.objaverse, args.samples),
    }
    datasets = {}
    for dataset, rows in clouds.items():
        datasets[dataset] = [
            evaluate_pointllm_geometry(cloud_id, points)
            for cloud_id, points in rows
        ]
    report = {
        "schema_version": "0.1",
        "status": "semantic_audit_complete",
        "oracle": {
            "fps": "pointnet2_ops.furthest_point_sample",
            "knn": "PointLLM square_distance plus torch.topk set",
            "fps_candidate": "FP32 direct-distance CUDA semantic schedule",
        },
        "datasets": datasets,
        "summary": {
            dataset: {
                "samples": len(rows),
                "fps_all_exact": all(row["deployed_fps_center_exact"] for row in rows),
                "fps_agreement_mean": sum(
                    row["deployed_fps_center_agreement"] for row in rows
                ) / len(rows),
                "single_distance_knn_recall_mean": sum(
                    row["single_direct_distance_knn_recall_mean"] for row in rows
                ) / len(rows),
                "dual_distance_required": any(
                    row["dual_distance_semantics_required"] for row in rows
                ),
            }
            for dataset, rows in datasets.items()
        },
        "claim_boundary": {
            "fp32_fps_semantics_validated": all(
                row["deployed_fps_center_exact"]
                for rows in datasets.values() for row in rows
            ),
            "single_numeric_distance_preserves_knn_sets": not any(
                row["dual_distance_semantics_required"]
                for rows in datasets.values() for row in rows
            ),
            "dual_fp32_distance_interface_required": any(
                row["dual_distance_semantics_required"]
                for rows in datasets.values() for row in rows
            ),
            "rtl_production_correlated": False,
            "full_pointllm_cycle_accurate": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
