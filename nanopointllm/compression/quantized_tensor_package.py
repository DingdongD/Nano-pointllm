"""Checksummed tensor package shared by QDQ, integer golden, and RTL flows."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class QuantizedLinearTensors:
    activation: np.ndarray
    weight: np.ndarray
    activation_scale: np.ndarray
    weight_scale: np.ndarray
    smooth_scale: np.ndarray
    output_scale: np.ndarray | None
    bias: np.ndarray | None
    metadata: dict[str, Any]

    @property
    def m(self) -> int:
        return int(self.activation.shape[0])

    @property
    def n(self) -> int:
        return int(self.weight.shape[0])

    @property
    def k(self) -> int:
        return int(self.weight.shape[1])

    def integer_golden(self) -> np.ndarray:
        return (
            self.activation.astype(np.int32)
            @ self.weight.astype(np.int32).T
        )


def write_quantized_linear_package(
    directory: str | Path,
    *,
    activation: np.ndarray,
    weight: np.ndarray,
    activation_scale: np.ndarray | float,
    weight_scale: np.ndarray,
    smooth_scale: np.ndarray,
    output_scale: np.ndarray | float | None = None,
    bias: np.ndarray | None = None,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write one canonical row-major package and return its manifest."""
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    activation = _canonical(activation, np.dtype("i1"), "activation", dimensions=2)
    weight = _canonical(weight, np.dtype("i1"), "weight", dimensions=2)
    if activation.shape[1] != weight.shape[1]:
        raise ValueError("activation and weight K dimensions differ")
    activation_scale = _canonical(
        np.asarray(activation_scale).reshape(-1), np.dtype("<f2"),
        "activation_scale", dimensions=1,
    )
    if activation_scale.shape != (1,):
        raise ValueError("activation_scale must contain exactly one FP16 value")
    weight_scale = _canonical(
        weight_scale, np.dtype("<f2"), "weight_scale", dimensions=1,
    )
    smooth_scale = _canonical(
        smooth_scale, np.dtype("<f2"), "smooth_scale", dimensions=1,
    )
    if weight_scale.shape != (weight.shape[0],):
        raise ValueError("weight_scale must contain one value per output")
    if smooth_scale.shape != (weight.shape[1],):
        raise ValueError("smooth_scale must contain one value per input channel")
    if np.any(activation_scale <= 0) or np.any(weight_scale <= 0):
        raise ValueError("quantization scales must be positive")
    if np.any(smooth_scale <= 0):
        raise ValueError("SmoothQuant scales must be positive")
    tensors: dict[str, tuple[str, np.ndarray]] = {
        "activation": ("activation.int8", activation),
        "weight": ("weight.int8", weight),
        "activation_scale": ("activation_scale.fp16", activation_scale),
        "weight_scale": ("weight_scale.fp16", weight_scale),
        "smooth_scale": ("smooth_scale.fp16", smooth_scale),
    }
    if output_scale is not None:
        output_scale_array = _canonical(
            np.asarray(output_scale).reshape(-1), np.dtype("<f2"),
            "output_scale", dimensions=1,
        )
        if output_scale_array.shape != (1,) or np.any(output_scale_array <= 0):
            raise ValueError("output_scale must contain one positive FP16 value")
        tensors["output_scale"] = ("output_scale.fp16", output_scale_array)
    if bias is not None:
        bias_array = _canonical(bias, np.dtype("<f2"), "bias", dimensions=1)
        if bias_array.shape != (weight.shape[0],):
            raise ValueError("bias must contain one value per output")
        tensors["bias"] = ("bias.fp16", bias_array)

    entries = {}
    for name, (filename, array) in tensors.items():
        path = root / filename
        payload = array.tobytes(order="C")
        path.write_bytes(payload)
        entries[name] = {
            "file": filename,
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "order": "C",
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "package_type": "smoothquant_static_a8_per_output_w8_linear",
        "shape": {
            "m": int(activation.shape[0]),
            "n": int(weight.shape[0]),
            "k": int(weight.shape[1]),
        },
        "numerical_contract": {
            "activation": "symmetric_per_tensor_int8[-127,127]",
            "weight": "symmetric_per_output_int8[-127,127]",
            "scale_storage": "IEEE754_binary16_little_endian",
            "accumulator": "signed_int32",
            "rounding": "round_to_nearest_ties_to_even",
            "layout": "row_major",
        },
        "tensors": entries,
        "provenance": provenance or {},
    }
    manifest_payload = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    (root / "metadata.json").write_text(manifest_payload, encoding="utf-8")
    return manifest


def load_quantized_linear_package(
    directory: str | Path,
) -> QuantizedLinearTensors:
    """Load and fully validate a quantized package before exposing arrays."""
    root = Path(directory)
    manifest = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported quantized tensor package schema")
    required = {
        "activation", "weight", "activation_scale", "weight_scale", "smooth_scale",
    }
    entries = manifest.get("tensors", {})
    missing = sorted(required - set(entries))
    if missing:
        raise ValueError(f"quantized tensor package is missing {missing}")
    arrays = {
        name: _load_entry(root, name, entry)
        for name, entry in entries.items()
    }
    shape = manifest.get("shape", {})
    expected = (int(shape["m"]), int(shape["n"]), int(shape["k"]))
    if arrays["activation"].shape != (expected[0], expected[2]):
        raise ValueError("activation shape does not match package shape")
    if arrays["weight"].shape != (expected[1], expected[2]):
        raise ValueError("weight shape does not match package shape")
    if arrays["activation_scale"].shape != (1,):
        raise ValueError("activation_scale shape must be [1]")
    if arrays["weight_scale"].shape != (expected[1],):
        raise ValueError("weight_scale shape does not match N")
    if arrays["smooth_scale"].shape != (expected[2],):
        raise ValueError("smooth_scale shape does not match K")
    bias = arrays.get("bias")
    output_scale = arrays.get("output_scale")
    if output_scale is not None and output_scale.shape != (1,):
        raise ValueError("output_scale shape must be [1]")
    if bias is not None and bias.shape != (expected[1],):
        raise ValueError("bias shape does not match N")
    return QuantizedLinearTensors(
        activation=arrays["activation"], weight=arrays["weight"],
        activation_scale=arrays["activation_scale"],
        weight_scale=arrays["weight_scale"],
        smooth_scale=arrays["smooth_scale"], output_scale=output_scale,
        bias=bias, metadata=manifest,
    )


def _canonical(
    values: np.ndarray, dtype: np.dtype, name: str, *, dimensions: int,
) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != dimensions:
        raise ValueError(f"{name} must have {dimensions} dimensions")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return np.ascontiguousarray(array.astype(dtype, copy=False))


def _load_entry(root: Path, name: str, entry: dict[str, Any]) -> np.ndarray:
    path = root / entry["file"]
    payload = path.read_bytes()
    if len(payload) != int(entry["bytes"]):
        raise ValueError(f"{name} byte count mismatch")
    digest = hashlib.sha256(payload).hexdigest()
    if digest != entry["sha256"]:
        raise ValueError(f"{name} SHA256 mismatch")
    dtype = np.dtype(entry["dtype"])
    shape = tuple(int(value) for value in entry["shape"])
    expected_bytes = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
    if expected_bytes != len(payload):
        raise ValueError(f"{name} metadata size mismatch")
    return np.frombuffer(payload, dtype=dtype).reshape(shape).copy()
