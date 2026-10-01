"""Profiling helpers that are inert outside explicit Nsight runs."""
from __future__ import annotations

from contextlib import nullcontext
import os

import torch


NVTX_STAGE_ENV = "NANOPOINTLLM_NVTX_STAGES"
_nvtx_enabled = os.environ.get(NVTX_STAGE_ENV, "0") == "1"
_nvtx_layer_filter: int | None = None
_nvtx_current_layer: int | None = None
_LAYER_SCOPED_STAGES = {
    "pointllm_qkv",
    "pointllm_attention",
    "pointllm_o_proj",
    "pointllm_mlp",
}


def set_nvtx_stages(enabled: bool) -> None:
    global _nvtx_enabled
    _nvtx_enabled = bool(enabled)


def set_nvtx_layer_filter(layer_index: int | None) -> None:
    global _nvtx_layer_filter
    _nvtx_layer_filter = layer_index


class _NvtxLayer:
    def __init__(self, layer_index: int) -> None:
        self.layer_index = layer_index
        self.previous: int | None = None

    def __enter__(self):
        global _nvtx_current_layer
        self.previous = _nvtx_current_layer
        _nvtx_current_layer = self.layer_index
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        global _nvtx_current_layer
        _nvtx_current_layer = self.previous


class _NvtxRange:
    def __init__(self, name: str) -> None:
        self.name = name

    def __enter__(self):
        torch.cuda.nvtx.range_push(self.name)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        torch.cuda.nvtx.range_pop()


def nvtx_stage(name: str, tensor: torch.Tensor | None = None):
    """Return an NVTX context only when the profiler explicitly enables it."""
    layer_matches = (
        _nvtx_layer_filter is None
        or name not in _LAYER_SCOPED_STAGES
        or _nvtx_current_layer == _nvtx_layer_filter
    )
    enabled = (
        _nvtx_enabled
        and layer_matches
        and torch.cuda.is_available()
        and (tensor is None or tensor.is_cuda)
    )
    return _NvtxRange(name) if enabled else nullcontext()


def nvtx_decoder_layer(layer_index: int):
    if not _nvtx_enabled or _nvtx_layer_filter is None:
        return nullcontext()
    return _NvtxLayer(layer_index)
