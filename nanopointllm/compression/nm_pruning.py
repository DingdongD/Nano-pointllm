from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import torch
import torch.nn as nn


def decoder_linear_modules(model: nn.Module) -> dict[str, nn.Linear]:
    backbone = model.get_model() if hasattr(model, "get_model") else model.model
    modules = {}
    for layer_index, layer in enumerate(backbone.layers):
        for name, module in layer.named_modules():
            if isinstance(module, nn.Linear):
                modules[f"layers.{layer_index}.{name}"] = module
    return modules


def module_category(name: str) -> str:
    if any(part in name for part in ("q_proj", "k_proj", "v_proj")):
        return "decoder_qkv"
    if "o_proj" in name:
        return "decoder_o"
    if any(part in name for part in ("gate_proj", "up_proj", "down_proj")):
        return "decoder_mlp"
    return "decoder_other"


class InputSecondMomentCollector:
    def __init__(self, modules: dict[str, nn.Linear]) -> None:
        self.modules = modules
        self.sum_squares: dict[str, torch.Tensor] = {}
        self.samples: dict[str, int] = defaultdict(int)
        self._handles = []

    def __enter__(self) -> "InputSecondMomentCollector":
        for name, module in self.modules.items():
            self._handles.append(module.register_forward_hook(self._hook(name)))
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _hook(self, name: str):
        def collect(_module, inputs, _output) -> None:
            values = inputs[0].detach().reshape(-1, inputs[0].shape[-1]).float()
            batch_sum = values.square().sum(dim=0)
            if name not in self.sum_squares:
                self.sum_squares[name] = batch_sum
            else:
                self.sum_squares[name].add_(batch_sum)
            self.samples[name] += values.shape[0]
        return collect

    def second_moments(self) -> dict[str, torch.Tensor]:
        missing = sorted(set(self.modules) - set(self.sum_squares))
        if missing:
            raise RuntimeError(f"no calibration activations for modules: {missing[:3]}")
        return {
            name: values / self.samples[name]
            for name, values in self.sum_squares.items()
        }


def nm_mask(
    weight: torch.Tensor,
    *,
    prune_n: int,
    prune_m: int,
    input_second_moment: torch.Tensor | None = None,
) -> torch.Tensor:
    if prune_n <= 0 or prune_m <= prune_n:
        raise ValueError("N:M requires 0 < N < M")
    if weight.dim() != 2:
        raise ValueError("N:M pruning expects a matrix")
    if weight.shape[1] % prune_m != 0:
        raise ValueError(
            f"input dimension {weight.shape[1]} is not divisible by M={prune_m}"
        )
    metric = weight.detach().float().abs()
    if input_second_moment is not None:
        if input_second_moment.shape != (weight.shape[1],):
            raise ValueError("input second moment width does not match weight")
        metric = metric * input_second_moment.float().clamp_min(0).sqrt().unsqueeze(0)
    grouped = metric.view(weight.shape[0], -1, prune_m)
    indices = torch.topk(grouped, prune_n, dim=-1, largest=False).indices
    grouped_mask = torch.zeros_like(grouped, dtype=torch.bool)
    grouped_mask.scatter_(dim=-1, index=indices, value=True)
    return grouped_mask.view_as(weight)


def validate_nm_mask(mask: torch.Tensor, *, prune_n: int, prune_m: int) -> bool:
    if mask.dim() != 2 or mask.shape[1] % prune_m != 0:
        return False
    grouped = mask.view(mask.shape[0], -1, prune_m)
    return bool(torch.all(grouped.sum(dim=-1) == prune_n).item())


@torch.no_grad()
def apply_nm_pruning(
    modules: dict[str, nn.Linear],
    *,
    prune_n: int,
    prune_m: int,
    input_second_moments: dict[str, torch.Tensor] | None = None,
) -> dict[str, dict[str, float | int | str]]:
    reports = {}
    for name, module in modules.items():
        moment = None if input_second_moments is None else input_second_moments[name]
        mask = nm_mask(
            module.weight,
            prune_n=prune_n,
            prune_m=prune_m,
            input_second_moment=moment,
        )
        if not validate_nm_mask(mask, prune_n=prune_n, prune_m=prune_m):
            raise RuntimeError(f"invalid {prune_n}:{prune_m} mask for {name}")
        module.weight.masked_fill_(mask, 0)
        reports[name] = {
            "category": module_category(name),
            "rows": module.weight.shape[0],
            "columns": module.weight.shape[1],
            "pruned": int(mask.sum().item()),
            "total": mask.numel(),
            "sparsity": float(mask.float().mean().item()),
        }
    return reports
