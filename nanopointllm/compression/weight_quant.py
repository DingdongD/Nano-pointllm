from __future__ import annotations

import torch


def symmetric_quantize_dequantize(
    weight: torch.Tensor,
    *,
    bits: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    if bits < 2 or bits > 16:
        raise ValueError("bits must be between 2 and 16")
    if weight.dim() < 2:
        raise ValueError("weight must have at least two dimensions")
    original_shape = weight.shape
    rows = weight.detach().float().reshape(weight.shape[0], -1)
    qmax = (1 << (bits - 1)) - 1
    scale = rows.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / qmax
    quantized = torch.round(rows / scale).clamp(-qmax, qmax)
    dequantized = (quantized * scale).reshape(original_shape).to(weight.dtype)
    error = dequantized.float() - weight.detach().float()
    denominator = torch.sqrt(torch.mean(weight.detach().float().square())).clamp_min(1e-12)
    return dequantized, {
        "rmse": float(torch.sqrt(torch.mean(error.square())).item()),
        "relative_rmse": float((torch.sqrt(torch.mean(error.square())) / denominator).item()),
        "max_abs_error": float(error.abs().max().item()),
    }
