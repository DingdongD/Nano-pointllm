import pytest
import torch

from nanopointllm.compression.weight_quant import symmetric_quantize_dequantize


def test_symmetric_quantization_preserves_shape_and_row_extrema():
    weight = torch.tensor([[0.0, 1.0, -1.0], [0.0, 2.0, -2.0]])
    output, metrics = symmetric_quantize_dequantize(weight, bits=4)
    assert output.shape == weight.shape
    torch.testing.assert_close(output[:, -1], weight[:, -1])
    assert metrics["relative_rmse"] >= 0.0


def test_more_bits_reduce_error():
    torch.manual_seed(0)
    weight = torch.randn(8, 16)
    _, int4 = symmetric_quantize_dequantize(weight, bits=4)
    _, int8 = symmetric_quantize_dequantize(weight, bits=8)
    assert int8["rmse"] < int4["rmse"]


def test_quantization_rejects_vector_weights():
    with pytest.raises(ValueError, match="at least two"):
        symmetric_quantize_dequantize(torch.ones(4), bits=4)
