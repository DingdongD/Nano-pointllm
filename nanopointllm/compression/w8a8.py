"""Explicit W8A8 numerical contracts and sequential activation QDQ hooks."""
from __future__ import annotations

from collections import defaultdict
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn


@dataclass(frozen=True)
class W8A8Contract:
    weight_bits: int = 8
    activation_bits: int = 8
    symmetric_min: int = -127
    symmetric_max: int = 127
    rounding: str = "round_to_nearest_ties_to_even"
    scale_storage: str = "float16"
    accumulator: str = "signed_int32"
    bias: str = "floating_point_after_dequantization"
    nonlinear_norm_residual: str = "bf16_or_fp16"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


CONTRACT = W8A8Contract()


def quantize_symmetric(
    values: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """Quantize with the locked [-127, 127], ties-to-even contract."""
    if torch.any(scale <= 0):
        raise ValueError("quantization scale must be positive")
    quantized = torch.round(values.float() / scale.float()).clamp(
        CONTRACT.symmetric_min, CONTRACT.symmetric_max,
    )
    return quantized.to(torch.int8)


def scale_from_absmax(absmax: torch.Tensor) -> torch.Tensor:
    """Compute a positive FP16-stored scale and return it in FP32 arithmetic."""
    minimum = torch.finfo(torch.float16).tiny
    raw = absmax.float().clamp_min(minimum) / CONTRACT.symmetric_max
    return raw.to(torch.float16).float().clamp_min(minimum)


def per_output_weight_qdq(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if weight.dim() < 2:
        raise ValueError("weight must have at least two dimensions")
    rows = weight.detach().float().reshape(weight.shape[0], -1)
    scales = scale_from_absmax(rows.abs().amax(dim=1))
    qweight = quantize_symmetric(rows, scales[:, None])
    dequantized = (qweight.float() * scales[:, None]).reshape_as(weight)
    return dequantized.to(weight.dtype), qweight.reshape_as(weight), scales


def groupwise_weight_qdq(
    weight: torch.Tensor,
    *,
    group_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if weight.dim() < 2:
        raise ValueError("groupwise W8 expects at least two weight dimensions")
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    original_shape = weight.shape
    rows = weight.shape[0]
    columns = weight.numel() // rows
    flattened = weight.detach().reshape(rows, columns)
    groups = (columns + group_size - 1) // group_size
    qweight = torch.empty_like(flattened, dtype=torch.int8)
    dequantized = torch.empty_like(flattened)
    scales = torch.empty((rows, groups), device=weight.device, dtype=torch.float32)
    for group in range(groups):
        start = group * group_size
        end = min(columns, start + group_size)
        values = flattened[:, start:end].float()
        scale = scale_from_absmax(values.abs().amax(dim=1))
        quantized = quantize_symmetric(values, scale[:, None])
        qweight[:, start:end] = quantized
        dequantized[:, start:end] = (quantized.float() * scale[:, None]).to(weight.dtype)
        scales[:, group] = scale
    return dequantized.reshape(original_shape), qweight.reshape(original_shape), scales


def static_activation_scale(absmax: float | torch.Tensor) -> torch.Tensor:
    value = torch.as_tensor(absmax, dtype=torch.float32)
    if value.numel() != 1:
        raise ValueError("static activation scale expects one tensor-wide absmax")
    return scale_from_absmax(value.reshape(()))


def static_activation_qdq(values: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    quantized = quantize_symmetric(values, scale.to(values.device))
    return (quantized.float() * scale.float().to(values.device)).to(values.dtype)


def dynamic_per_token_activation_qdq(
    values: torch.Tensor,
    *,
    axis: int = -1,
) -> tuple[torch.Tensor, torch.Tensor]:
    if values.dim() == 0:
        raise ValueError("dynamic per-token quantization expects a tensor")
    scales = scale_from_absmax(values.detach().float().abs().amax(dim=axis, keepdim=True))
    quantized = quantize_symmetric(values, scales)
    return (quantized.float() * scales).to(values.dtype), scales


def integer_linear_per_output_reference(
    values: torch.Tensor,
    weight: torch.Tensor,
    *,
    activation_scale: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """CPU golden for INT8 x INT8 -> INT32 and per-output dequantization."""
    if values.shape[-1] != weight.shape[1] or weight.dim() != 2:
        raise ValueError("linear input/weight shapes do not match")
    qactivation = quantize_symmetric(values, activation_scale)
    _, qweight, weight_scales = per_output_weight_qdq(weight)
    flat = qactivation.reshape(-1, qactivation.shape[-1]).cpu().to(torch.int32)
    qweight_cpu = qweight.cpu().to(torch.int32)
    accumulator = flat @ qweight_cpu.transpose(0, 1)
    output = accumulator.float()
    output *= activation_scale.detach().float().cpu()
    output *= weight_scales.detach().float().cpu().unsqueeze(0)
    if bias is not None:
        output += bias.detach().float().cpu().unsqueeze(0)
    output = output.reshape(*values.shape[:-1], weight.shape[0])
    return output, accumulator.reshape(*values.shape[:-1], weight.shape[0])


class ActivationAbsmaxCollector(AbstractContextManager):
    """Collect tensor and input-channel ranges without changing model outputs."""

    def __init__(self, modules: dict[str, nn.Module]) -> None:
        self.modules = modules
        self.tensor_absmax: dict[str, float] = defaultdict(float)
        self.channel_absmax: dict[str, torch.Tensor] = {}
        self.samples: dict[str, int] = defaultdict(int)
        self._handles = []

    def __enter__(self) -> "ActivationAbsmaxCollector":
        for name, module in self.modules.items():
            self._handles.append(module.register_forward_pre_hook(self._hook(name)))
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _hook(self, name: str):
        def collect(module, inputs) -> None:
            if not inputs or not isinstance(inputs[0], torch.Tensor):
                raise TypeError(f"{name} does not have a tensor first input")
            values = inputs[0].detach().float()
            self.tensor_absmax[name] = max(
                self.tensor_absmax[name], float(values.abs().max().item()),
            )
            if values.dim() >= 1:
                if isinstance(module, (nn.Conv1d, nn.Conv2d)):
                    reduction_dims = tuple(index for index in range(values.dim()) if index != 1)
                    flattened = values.abs().amax(dim=reduction_dims).cpu()
                else:
                    flattened = values.reshape(-1, values.shape[-1]).abs().amax(dim=0).cpu()
                previous = self.channel_absmax.get(name)
                self.channel_absmax[name] = (
                    flattened if previous is None else torch.maximum(previous, flattened)
                )
                channels = values.shape[1] if isinstance(module, (nn.Conv1d, nn.Conv2d)) else values.shape[-1]
                self.samples[name] += values.numel() // channels
        return collect

    def static_scales(self) -> dict[str, torch.Tensor]:
        missing = sorted(set(self.modules) - set(self.tensor_absmax))
        if missing:
            raise RuntimeError(f"no activation calibration for modules: {missing[:3]}")
        return {
            name: static_activation_scale(absmax)
            for name, absmax in self.tensor_absmax.items()
        }


class SequentialActivationQDQ(AbstractContextManager):
    """Inject activation QDQ at every selected module input in execution order."""

    def __init__(
        self,
        modules: dict[str, nn.Module],
        *,
        mode: str,
        static_scales: dict[str, torch.Tensor] | None = None,
        input_smoothing_scales: dict[str, torch.Tensor] | None = None,
    ) -> None:
        if mode not in ("static", "dynamic_per_token"):
            raise ValueError("activation mode must be static or dynamic_per_token")
        if mode == "static" and static_scales is None:
            raise ValueError("static activation mode requires calibrated scales")
        self.modules = modules
        self.mode = mode
        self.static_scales = static_scales or {}
        self.input_smoothing_scales = input_smoothing_scales or {}
        self._handles = []

    def __enter__(self) -> "SequentialActivationQDQ":
        for name, module in self.modules.items():
            self._handles.append(module.register_forward_pre_hook(self._hook(name)))
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _hook(self, name: str):
        def quantize(_module, inputs):
            values = inputs[0]
            smoothing = self.input_smoothing_scales.get(name)
            if smoothing is not None:
                smoothing = smoothing.to(device=values.device, dtype=values.dtype)
                if isinstance(_module, (nn.Conv1d, nn.Conv2d)):
                    shape = [1] * values.dim()
                    shape[1] = smoothing.numel()
                    values = values / smoothing.reshape(shape)
                else:
                    values = values / smoothing
            if self.mode == "static":
                replacement = static_activation_qdq(values, self.static_scales[name])
            else:
                axis = 1 if isinstance(_module, (nn.Conv1d, nn.Conv2d)) else -1
                replacement, _ = dynamic_per_token_activation_qdq(values, axis=axis)
            return (replacement, *inputs[1:])
        return quantize


@torch.no_grad()
def apply_smoothquant_transform(
    modules: dict[str, nn.Module],
    channel_absmax: dict[str, torch.Tensor],
    *,
    alpha: float = 0.5,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Apply exact x/s, W*s smoothing and derive post-transform A8 scales."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("SmoothQuant alpha must be in [0, 1]")
    missing = sorted(set(modules) - set(channel_absmax))
    if missing:
        raise RuntimeError(f"no channel calibration for modules: {missing[:3]}")
    input_scales = {}
    activation_scales = {}
    epsilon = 1e-5
    for name, module in modules.items():
        weight = module.weight.detach().float()
        if isinstance(module, (nn.Conv1d, nn.Conv2d)):
            weight_absmax = weight.abs().amax(
                dim=tuple(index for index in range(weight.dim()) if index != 1),
            )
            broadcast_shape = [1] * weight.dim()
            broadcast_shape[1] = weight.shape[1]
        else:
            rows = weight.reshape(weight.shape[0], -1)
            weight_absmax = rows.abs().amax(dim=0)
            broadcast_shape = [1, weight.shape[1]]
        activation_absmax = channel_absmax[name].float().to(weight.device)
        if activation_absmax.numel() != weight_absmax.numel():
            raise ValueError(
                f"activation/weight channel mismatch for {name}: "
                f"{activation_absmax.numel()} versus {weight_absmax.numel()}"
            )
        smoothing = (
            activation_absmax.clamp_min(epsilon).pow(alpha)
            / weight_absmax.clamp_min(epsilon).pow(1.0 - alpha)
        ).clamp_min(epsilon)
        smoothing = smoothing.to(torch.float16).float().clamp_min(epsilon)
        module.weight.mul_(smoothing.to(module.weight.dtype).reshape(broadcast_shape))
        post_absmax = (activation_absmax / smoothing).amax().cpu()
        input_scales[name] = smoothing.cpu()
        activation_scales[name] = static_activation_scale(post_absmax)
    return input_scales, activation_scales


@torch.no_grad()
def apply_weight_qdq(
    modules: dict[str, nn.Module],
    *,
    granularity: str,
    group_size: int = 128,
) -> dict[str, dict[str, Any]]:
    if granularity not in ("per_output", "group"):
        raise ValueError("weight granularity must be per_output or group")
    reports = {}
    for name, module in modules.items():
        if not hasattr(module, "weight"):
            raise TypeError(f"{name} has no weight")
        weight = module.weight
        if granularity == "per_output":
            dequantized, quantized, scales = per_output_weight_qdq(weight)
        else:
            dequantized, quantized, scales = groupwise_weight_qdq(
                weight, group_size=group_size,
            )
        error = dequantized.float() - weight.detach().float()
        module.weight.copy_(dequantized)
        reports[name] = {
            "shape": list(weight.shape),
            "parameters": weight.numel(),
            "granularity": granularity,
            "group_size": group_size if granularity == "group" else None,
            "scale_shape": list(scales.shape),
            "quantized_min": int(quantized.min().item()),
            "quantized_max": int(quantized.max().item()),
            "rmse": float(error.square().mean().sqrt().item()),
        }
    return reports
