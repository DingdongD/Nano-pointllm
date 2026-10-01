"""Profiling helpers that are inert outside explicit Nsight runs."""
from __future__ import annotations

from contextlib import nullcontext
import os

import torch


NVTX_STAGE_ENV = "NANOPOINTLLM_NVTX_STAGES"
_nvtx_enabled = os.environ.get(NVTX_STAGE_ENV, "0") == "1"


def set_nvtx_stages(enabled: bool) -> None:
    global _nvtx_enabled
    _nvtx_enabled = bool(enabled)


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
    enabled = _nvtx_enabled and torch.cuda.is_available() and (tensor is None or tensor.is_cuda)
    return _NvtxRange(name) if enabled else nullcontext()
