"""Compiled production-shape RTL correlation for the FPS/KNN selector."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import tempfile
from typing import Any


def correlate_production_fps_knn(
    *, rtl_root: str | Path, output_dir: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(rtl_root)
    source = root / "gtsu_fused_fps_knn_controller.sv"
    driver = root / "verilator_fused_fps_knn_driver.cpp"
    with tempfile.TemporaryDirectory(prefix="gtsu_fps_knn_prod_") as name:
        build = Path(name) / "obj_dir"
        command = [
            "verilator", "--cc", "--exe", "-Wno-fatal",
            "--top-module", "gtsu_fused_fps_knn_controller",
            "--Mdir", str(build), "-GPOINTS=8192", "-GCENTERS=512",
            "-GK=32", "-GLANES=64", "-GDIST_WIDTH=32",
            "-GFPS_REDUCTION_LANES=512", str(source.resolve()),
            str(driver.resolve()),
        ]
        compiled = subprocess.run(command, capture_output=True, text=True)
        if compiled.returncode:
            raise RuntimeError(f"production FPS/KNN compile failed:\n{compiled.stderr}")
        built = subprocess.run([
            "make", "-C", str(build),
            "-f", "Vgtsu_fused_fps_knn_controller.mk", "-j2",
        ], capture_output=True, text=True)
        if built.returncode:
            raise RuntimeError(
                f"production FPS/KNN build failed:\n{built.stdout}\n{built.stderr}"
            )
        simulated = subprocess.run([
            str(build / "Vgtsu_fused_fps_knn_controller"), "100000",
        ], capture_output=True, text=True)
        if simulated.returncode:
            raise RuntimeError(
                f"production FPS/KNN simulation failed:\n"
                f"{simulated.stdout}\n{simulated.stderr}"
            )
    summary = _parse_summary(simulated.stdout)
    exact = (
        summary["input_beats"] == 512 * 128
        and summary["output_rounds"] == 512
        and summary["neighbor_values"] == 512 * 32
        and summary["mismatches"] == 0
    )
    report = {
        "schema_version": "0.1",
        "status": "rtl_correlated" if exact else "mismatch",
        "operator": "production_fused_fps_knn_selector",
        "config": {
            "points": 8192, "centers": 512, "neighbors": 32,
            "physical_lanes": 64, "logical_fps_reduction_lanes": 512,
            "distance_width": 32, "dual_distance_stream": True,
        },
        **summary,
        "all_center_and_neighbor_values_exact": exact,
        "claim_boundary": {
            "production_selector_shape_correlated": exact,
            "synthetic_dual_distance_values": True,
            "fp32_fma_frontend_rtl_correlated": False,
            "real_payload_rtl_correlated": False,
            "point_norm_coordinate_sram_integrated": False,
            "fps_coverage_enabled": False,
            "knn_topk_coverage_enabled": False,
            "full_pointllm_cycle_accurate": False,
        },
    }
    if output_dir is not None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "verilator_stdout.log").write_text(
            simulated.stdout, encoding="utf-8",
        )
        (destination / "correlation.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8",
        )
    if not exact:
        raise AssertionError(f"production FPS/KNN mismatch: {report}")
    return report


def _parse_summary(text: str) -> dict[str, int | str]:
    for line in text.splitlines():
        fields = line.split()
        if fields[:1] == ["SUMMARY"] and len(fields) == 8:
            return {
                "rtl_cycles": int(fields[1]),
                "input_beats": int(fields[2]),
                "output_rounds": int(fields[3]),
                "neighbor_values": int(fields[4]),
                "mismatches": int(fields[5]),
                "center_hash_fnv1a64": fields[6],
                "neighbor_hash_fnv1a64": fields[7],
            }
    raise RuntimeError(f"production FPS/KNN summary missing:\n{text[-4000:]}")
