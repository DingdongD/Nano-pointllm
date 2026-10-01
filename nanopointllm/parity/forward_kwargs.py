"""仅传入 `model.forward` 接受的参数（标准 Llama 无 `point_clouds`）。"""
from __future__ import annotations

import inspect
from typing import Any

import torch.nn as nn


def filter_forward_kwargs(model: nn.Module, **kwargs: Any) -> dict[str, Any]:
    params = inspect.signature(model.forward).parameters
    return {k: v for k, v in kwargs.items() if k in params}
