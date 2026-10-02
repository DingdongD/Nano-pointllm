"""Real PointLLM payload correlation for the production FPS/KNN selector."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
from typing import Any

import numpy as np


def generate_pointllm_geometry_trace(
    points: np.ndarray, *, trace_dir: str | Path, extension_source: str | Path,
    centers: int = 512, neighbors: int = 32,
) -> dict[str, Any]:
    import torch
    from pointnet2_ops import pointnet2_utils
    from torch.utils.cpp_extension import load

    destination = Path(trace_dir)
    destination.mkdir(parents=True, exist_ok=True)
    build_directory = destination / "cuda_build"
    build_directory.mkdir(parents=True, exist_ok=True)
    xyz = torch.from_numpy(np.ascontiguousarray(points[:, :3])).cuda().float().contiguous()
    center_indices = pointnet2_utils.furthest_point_sample(
        xyz.unsqueeze(0), centers,
    )[0].to(dtype=torch.int64).contiguous()
    extension = load(
        name="pointllm_geometry_trace_cuda",
        sources=[str(Path(extension_source).resolve())],
        extra_cuda_cflags=["-O3"],
        build_directory=str(build_directory),
        verbose=False,
    )
    fps_distances = extension.fps_distances(xyz, center_indices)
    fps_valid = extension.fps_valid(xyz)

    center_xyz = xyz[center_indices]
    knn_distances = -2.0 * torch.matmul(center_xyz, xyz.t())
    knn_distances += torch.sum(center_xyz ** 2, dim=-1).unsqueeze(1)
    knn_distances += torch.sum(xyz ** 2, dim=-1).unsqueeze(0)
    knn_reference = torch.topk(
        knn_distances, neighbors, dim=-1, largest=False, sorted=False,
    ).indices

    fps_bits = _float_bits(fps_distances)
    knn_bits = _float_bits(knn_distances)
    center_array = center_indices.cpu().numpy().astype("<u4", copy=False)
    knn_array = knn_reference.cpu().numpy().astype("<u4", copy=False)
    valid_array = fps_valid.cpu().numpy().astype("u1", copy=False)
    paths = {
        "fps": destination / "fps_distance_bits.bin",
        "knn": destination / "knn_distance_bits.bin",
        "centers": destination / "center_indices.bin",
        "knn_reference": destination / "knn_reference_indices.bin",
        "fps_valid": destination / "fps_valid_mask.bin",
    }
    for key, array in (
        ("fps", fps_bits), ("knn", knn_bits), ("centers", center_array),
        ("knn_reference", knn_array),
        ("fps_valid", valid_array),
    ):
        array.tofile(paths[key])

    metadata = {
        "points": int(xyz.shape[0]),
        "centers": centers,
        "neighbors": neighbors,
        "point_sha256": hashlib.sha256(
            np.ascontiguousarray(points[:, :3], dtype="<f4").tobytes()
        ).hexdigest(),
        "files": {
            key: {
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for key, path in paths.items()
        },
        "knn_negative_values": int((knn_distances < 0).sum().item()),
        "knn_minimum": float(knn_distances.min().item()),
        "fps_cuda_contract": "mul.rn(dy,dy); fma.rn(dx,dx,dy2); fma.rn(dz,dz,dxy2)",
    }
    (destination / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8",
    )
    return metadata


def correlate_real_pointllm_fps_knn(
    *, rtl_root: str | Path, trace_dir: str | Path,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(rtl_root)
    trace = Path(trace_dir)
    metadata = json.loads((trace / "metadata.json").read_text(encoding="utf-8"))
    if (metadata["points"], metadata["centers"], metadata["neighbors"]) != (
        8192, 512, 32,
    ):
        raise ValueError("real-payload RTL lock requires PointLLM 8192x512xTop32")
    _verify_trace_hashes(metadata)
    with tempfile.TemporaryDirectory(prefix="gtsu_fps_knn_real_") as name:
        build = Path(name) / "obj_dir"
        command = [
            "verilator", "--cc", "--exe", "-Wno-fatal",
            "--top-module", "gtsu_fused_fps_knn_controller",
            "--Mdir", str(build), "-GPOINTS=8192", "-GCENTERS=512",
            "-GK=32", "-GLANES=64", "-GDIST_WIDTH=32", "-GFP32_ORDER=1",
            "-GFPS_REDUCTION_LANES=512",
            str((root / "gtsu_fused_fps_knn_controller.sv").resolve()),
            str((root / "verilator_fused_fps_knn_real_driver.cpp").resolve()),
        ]
        compiled = subprocess.run(command, capture_output=True, text=True)
        if compiled.returncode:
            raise RuntimeError(f"real FPS/KNN compile failed:\n{compiled.stderr}")
        built = subprocess.run([
            "make", "-C", str(build),
            "-f", "Vgtsu_fused_fps_knn_controller.mk", "-j2",
        ], capture_output=True, text=True)
        if built.returncode:
            raise RuntimeError(f"real FPS/KNN build failed:\n{built.stdout}\n{built.stderr}")
        simulated = subprocess.run([
            str(build / "Vgtsu_fused_fps_knn_controller"),
            str((trace / "fps_distance_bits.bin").resolve()),
            str((trace / "knn_distance_bits.bin").resolve()),
            str((trace / "center_indices.bin").resolve()),
            str((trace / "knn_reference_indices.bin").resolve()),
            str((trace / "fps_valid_mask.bin").resolve()),
            "100000",
        ], capture_output=True, text=True)
        if simulated.returncode:
            raise RuntimeError(
                f"real FPS/KNN simulation failed:\n{simulated.stdout}\n{simulated.stderr}"
            )
    summary = _parse_summary(simulated.stdout)
    exact = summary["mismatches"] == 0 and summary["rounds"] == 512
    report = {
        "schema_version": "0.1",
        "status": "rtl_correlated" if exact else "mismatch",
        "operator": "production_fused_fps_knn_selector_real_payload",
        "trace": metadata,
        **summary,
        "claim_boundary": {
            "production_selector_real_payload_correlated": exact,
            "deployed_fps_center_sequence_exact": summary["center_mismatches"] == 0,
            "pointllm_knn_top32_set_exact": summary["knn_set_mismatches"] == 0,
            "fp32_numeric_order_handles_negative_roundoff": exact,
            "fp32_fma_frontend_rtl_correlated": False,
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
        raise AssertionError(f"real PointLLM FPS/KNN mismatch: {report}")
    return report


def _float_bits(value) -> np.ndarray:
    array = value.detach().cpu().contiguous().numpy().astype("<f4", copy=False)
    return array.view("<u4")


def _verify_trace_hashes(metadata: dict[str, Any]) -> None:
    for row in metadata["files"].values():
        path = Path(row["path"])
        if path.stat().st_size != row["bytes"]:
            raise ValueError(f"trace size mismatch: {path}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError(f"trace hash mismatch: {path}")


def _parse_summary(text: str) -> dict[str, int | str]:
    for line in text.splitlines():
        fields = line.split()
        if fields[:1] == ["SUMMARY"] and len(fields) == 13:
            return {
                "rtl_cycles": int(fields[1]),
                "input_beats": int(fields[2]),
                "rounds": int(fields[3]),
                "neighbor_values": int(fields[4]),
                "mismatches": int(fields[5]),
                "center_mismatches": int(fields[6]),
                "next_center_mismatches": int(fields[7]),
                "knn_set_mismatches": int(fields[8]),
                "neighbor_index_mismatches": int(fields[9]),
                "neighbor_distance_mismatches": int(fields[10]),
                "center_hash_fnv1a64": fields[11],
                "neighbor_hash_fnv1a64": fields[12],
            }
    raise RuntimeError(f"real FPS/KNN summary missing:\n{text[-4000:]}")
