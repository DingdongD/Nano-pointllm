"""与 greedy parity 相关的注意力实现切换（便于 SDPA/Flash 与 eager 数值对齐）。"""
from __future__ import annotations

import torch.nn as nn


def set_eager_attention_if_supported(model: nn.Module) -> None:
    """
    将 HF 模型的注意力设为 eager，减轻 SDPA/Flash 在不同序列形状（整段 prefill vs 单步 decode）
    下 bf16 累加顺序差异导致的 greedy 分叉。

    对无该 API 的旧版 transformers 为 no-op。
    """
    if hasattr(model, "set_attn_implementation"):
        model.set_attn_implementation("eager")
