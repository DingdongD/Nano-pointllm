"""
Decode 边界类型（与 HF `past_key_values` 约定对齐，供 M2+ 自定义引擎使用）。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch


@dataclass
class PrefillOutput:
    """HF `forward(..., use_cache=True)` 首步输出抽象。"""

    # 常为末位 `(B, 1, V)`；greedy 时请先压成 `(B, V)` 再 argmax，避免得到 `(B, 1, 1)`。
    logits: torch.Tensor
    past_key_values: Any
    prompt_len: int


@dataclass
class DecodeStepInput:
    """单步 decode 输入（不含点云；点云仅在 prefill 侧）。"""

    input_ids: torch.Tensor  # (B, 1)
    attention_mask: torch.Tensor
    past_key_values: Any
