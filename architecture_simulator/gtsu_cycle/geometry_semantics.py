"""Semantic oracles for PointLLM FPS/KNN and precision sensitivity."""
from __future__ import annotations

import hashlib
from pathlib import Path
import pickle
from typing import Iterable

import numpy as np


def load_modelnet_clouds(path: str | Path, count: int) -> list[tuple[str, np.ndarray]]:
    with Path(path).open("rb") as handle:
        points, _labels = pickle.load(handle)
    return [
        (f"modelnet_test_{index}", np.ascontiguousarray(points[index][:, :3], dtype=np.float32))
        for index in range(min(count, len(points)))
    ]


def load_objaverse_clouds(path: str | Path, count: int) -> list[tuple[str, np.ndarray]]:
    files = sorted(Path(path).glob("*_8192.npy"))[:count]
    return [
        (file.stem.removesuffix("_8192"),
         np.ascontiguousarray(np.load(file)[:, :3], dtype=np.float32))
        for file in files
    ]


def evaluate_pointllm_geometry(
    cloud_id: str, points: np.ndarray, *, centers: int = 512,
    neighbors: int = 32, device: str = "cuda",
) -> dict[str, object]:
    import torch
    from pointnet2_ops import pointnet2_utils

    xyz = torch.from_numpy(np.ascontiguousarray(points[:, :3])).to(
        device=device, dtype=torch.float32,
    ).contiguous()
    reference_centers = pointnet2_utils.furthest_point_sample(
        xyz.unsqueeze(0), centers,
    )[0]
    candidate_centers = cuda_semantic_fps(xyz, centers)
    center_equal = reference_centers == candidate_centers

    center_xyz = xyz[reference_centers]
    reference_distance = -2.0 * torch.matmul(center_xyz, xyz.t())
    reference_distance += torch.sum(center_xyz ** 2, dim=-1).unsqueeze(1)
    reference_distance += torch.sum(xyz ** 2, dim=-1).unsqueeze(0)
    candidate_distance = direct_squared_distance(center_xyz, xyz)
    reference_knn = torch.topk(
        reference_distance, neighbors, dim=-1, largest=False, sorted=False,
    ).indices.cpu().numpy()
    candidate_knn = torch.topk(
        candidate_distance, neighbors, dim=-1, largest=False, sorted=False,
    ).indices.cpu().numpy()
    recalls = _set_recalls(reference_knn, candidate_knn)

    precision = []
    fp32_centers = reference_centers.cpu().numpy()
    for bits in (8, 10, 12, 14, 16):
        quantized = symmetric_coordinate_quantize(points[:, :3], bits)
        quant_centers = numpy_fps(quantized, centers, skip_origin=True)
        quant_knn = numpy_knn(quantized, fp32_centers, neighbors)
        precision.append({
            "coordinate_bits": bits,
            "center_index_agreement": float(np.mean(quant_centers == fp32_centers)),
            "center_exact_prefix": _exact_prefix(fp32_centers, quant_centers),
            "knn_set_recall_mean": float(_set_recalls(reference_knn, quant_knn).mean()),
            "knn_set_exact_rate": float((_set_recalls(reference_knn, quant_knn) == 1).mean()),
        })

    return {
        "cloud_id": cloud_id,
        "shape": list(points.shape),
        "xyz_sha256": hashlib.sha256(
            np.ascontiguousarray(points[:, :3], dtype="<f4").tobytes()
        ).hexdigest(),
        "deployed_fps_center_exact": bool(torch.all(center_equal).item()),
        "deployed_fps_center_agreement": float(center_equal.float().mean().item()),
        "deployed_fps_first_divergence": _first_false(center_equal.cpu().numpy()),
        "single_direct_distance_knn_recall_mean": float(recalls.mean()),
        "single_direct_distance_knn_recall_min": float(recalls.min()),
        "single_direct_distance_knn_set_exact_rate": float((recalls == 1).mean()),
        "dual_distance_semantics_required": bool(np.any(recalls < 1)),
        "precision_sweep": precision,
    }


def cuda_semantic_fps(points, centers: int):
    """Torch expression matching pointnet2_ops sampling_gpu.cu semantics."""
    import torch

    nearest = torch.full(
        (points.shape[0],), 1e10, dtype=torch.float32, device=points.device,
    )
    valid = torch.sum(points * points, dim=-1) > 1e-3
    selected = torch.zeros(centers, dtype=torch.long, device=points.device)
    current = 0
    for round_index in range(1, centers):
        center = points[current]
        dx = points[:, 0] - center[0]
        dy = points[:, 1] - center[1]
        dz = points[:, 2] - center[2]
        distance = dx * dx + dy * dy + dz * dz
        nearest = torch.where(valid, torch.minimum(nearest, distance), nearest)
        score = torch.where(valid, nearest, -torch.ones_like(nearest))
        block_size = min(512, 1 << (points.shape[0].bit_length() - 1))
        padded = ((points.shape[0] + block_size - 1) // block_size) * block_size
        if padded != points.shape[0]:
            score = torch.cat((score, torch.full(
                (padded - points.shape[0],), -1.0,
                dtype=score.dtype, device=score.device,
            )))
        lane_values, lane_rows = score.reshape(-1, block_size).max(dim=0)
        lane = int(torch.argmax(lane_values))
        current = int(lane_rows[lane]) * block_size + lane
        selected[round_index] = current
    return selected


def direct_squared_distance(centers, points):
    dx = points[None, :, 0] - centers[:, None, 0]
    dy = points[None, :, 1] - centers[:, None, 1]
    dz = points[None, :, 2] - centers[:, None, 2]
    return dx * dx + dy * dy + dz * dz


def symmetric_coordinate_quantize(points: np.ndarray, bits: int) -> np.ndarray:
    limit = (1 << (bits - 1)) - 1
    scale = float(np.max(np.abs(points))) / limit
    if not scale:
        return np.zeros_like(points, dtype=np.int64)
    return np.clip(np.rint(points / scale), -limit, limit).astype(np.int64)


def numpy_fps(points: np.ndarray, centers: int, *, skip_origin: bool) -> np.ndarray:
    nearest = np.full(points.shape[0], np.inf)
    valid = np.sum(points.astype(np.float64) ** 2, axis=-1) > 0 if skip_origin else np.ones(
        points.shape[0], dtype=bool,
    )
    result = np.zeros(centers, dtype=np.int64)
    current = 0
    for round_index in range(1, centers):
        delta = points - points[current]
        distance = np.sum(delta * delta, axis=-1)
        nearest[valid] = np.minimum(nearest[valid], distance[valid])
        current = int(np.argmax(np.where(valid, nearest, -1)))
        result[round_index] = current
    return result


def numpy_knn(points: np.ndarray, centers: Iterable[int], neighbors: int) -> np.ndarray:
    indices = np.arange(points.shape[0])
    rows = []
    for center in centers:
        delta = points - points[int(center)]
        distance = np.sum(delta * delta, axis=-1)
        rows.append(np.lexsort((indices, distance))[:neighbors])
    return np.asarray(rows)


def _set_recalls(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    return np.asarray([
        len(set(left.tolist()) & set(right.tolist())) / len(left)
        for left, right in zip(reference, candidate, strict=True)
    ])


def _exact_prefix(reference: np.ndarray, candidate: np.ndarray) -> int:
    mismatch = np.flatnonzero(reference != candidate)
    return int(mismatch[0]) if mismatch.size else int(reference.size)


def _first_false(values: np.ndarray) -> int | None:
    mismatch = np.flatnonzero(~values)
    return int(mismatch[0]) if mismatch.size else None
