import json

import numpy as np
import pytest

from nanopointllm.compression.quantized_tensor_package import (
    load_quantized_linear_package, write_quantized_linear_package,
)


def _write_package(path):
    activation = np.array([[1, -2, 3, -4], [5, 6, -7, 8]], dtype=np.int8)
    weight = np.array([
        [2, 3, 4, 5], [-1, -2, 1, 2], [7, 0, -3, 1],
    ], dtype=np.int8)
    write_quantized_linear_package(
        path, activation=activation, weight=weight,
        activation_scale=np.array([0.125], dtype=np.float16),
        weight_scale=np.array([0.25, 0.5, 1.0], dtype=np.float16),
        smooth_scale=np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float16),
        output_scale=np.array([0.75], dtype=np.float16),
        provenance={"module": "layers.0.self_attn.q_proj"},
    )
    return activation, weight


def test_quantized_tensor_package_round_trip_and_integer_golden(tmp_path):
    activation, weight = _write_package(tmp_path)
    package = load_quantized_linear_package(tmp_path)
    np.testing.assert_array_equal(package.activation, activation)
    np.testing.assert_array_equal(package.weight, weight)
    np.testing.assert_array_equal(
        package.integer_golden(),
        activation.astype(np.int32) @ weight.astype(np.int32).T,
    )
    metadata = json.loads((tmp_path / "metadata.json").read_text())
    assert metadata["shape"] == {"m": 2, "n": 3, "k": 4}
    assert len(metadata["tensors"]["weight"]["sha256"]) == 64
    assert float(package.output_scale[0]) == 0.75


def test_quantized_tensor_package_rejects_corruption(tmp_path):
    _write_package(tmp_path)
    weight_path = tmp_path / "weight.int8"
    payload = bytearray(weight_path.read_bytes())
    payload[0] ^= 1
    weight_path.write_bytes(payload)
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        load_quantized_linear_package(tmp_path)


def test_quantized_tensor_package_rejects_incompatible_shapes(tmp_path):
    with pytest.raises(ValueError, match="K dimensions differ"):
        write_quantized_linear_package(
            tmp_path,
            activation=np.ones((1, 4), dtype=np.int8),
            weight=np.ones((3, 8), dtype=np.int8),
            activation_scale=1.0,
            weight_scale=np.ones(3), smooth_scale=np.ones(8),
        )
