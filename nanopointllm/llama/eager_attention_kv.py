"""
Decode 步 **eager 缩放点积注意力**（与 HF `eager_attention_forward` 同公式），含 GQA 的 `repeat_kv`。

供单测对齐与后续 Triton/CUDA 替换参考；完整 decode 仍经各层 `LlamaAttention` 时由 HF 内部调用同名逻辑。
"""
from __future__ import annotations

from typing import Any, Optional, Protocol

import torch
import torch.nn as nn


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """`(B, n_kv, S, D) -> (B, n_kv*n_rep, S, D)`，与 HF `repeat_kv` 一致。"""
    if n_rep == 1:
        return hidden_states
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class _EagerAttnModuleLike(Protocol):
    num_key_value_groups: int
    training: bool


def eager_attention_forward_ref(
    module: _EagerAttnModuleLike,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    与 ``transformers.models.llama.modeling_llama.eager_attention_forward`` 对齐（含 mask 切片）。

    ``query`` / ``key`` / ``value`` 形状为 ``(B, n_heads, Q, D)`` / ``(B, n_kv, Q, D)`` / ``(B, n_kv, Q, D)``。
    """
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights
