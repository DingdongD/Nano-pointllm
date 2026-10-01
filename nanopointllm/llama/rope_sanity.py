"""
RoPE 与 HuggingFace `LlamaRotaryEmbedding` + `apply_rotary_pos_emb` 对齐自检（decode 步 offset 场景）。
"""
from __future__ import annotations

import torch
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, apply_rotary_pos_emb


@torch.inference_mode()
def verify_decode_step_rope(
    *,
    device: torch.device,
    num_heads: int,
    head_dim: int,
    rope_theta: float,
    max_position_embeddings: int,
    kv_seq_len: int,
    dtype: torch.dtype = torch.float32,
) -> None:
    """
    模拟「已有 KV 长度 `kv_seq_len - 1`、当前再 decode 1 个 token」时 RoPE 的 cos/sin 切片与输出形状。

    `offset` 使用 `kv_seq_len - 1`，与 HF LlamaAttention 中 `past_key_values_length` 在单步 decode 时一致。
    """
    if kv_seq_len < 2:
        raise ValueError("kv_seq_len must be >= 2 for this sanity check")

    emb = LlamaRotaryEmbedding(
        head_dim,
        max_position_embeddings=max_position_embeddings,
        base=rope_theta,
        device=device,
    )
    x_dummy = torch.zeros(1, num_heads, 1, head_dim, device=device, dtype=dtype)
    cos, sin = emb(x_dummy, seq_len=kv_seq_len)
    assert cos.shape[-2] >= kv_seq_len and sin.shape[-2] >= kv_seq_len

    q = torch.randn(1, num_heads, 1, head_dim, device=device, dtype=dtype)
    k = torch.randn(1, num_heads, 1, head_dim, device=device, dtype=dtype)
    offset = kv_seq_len - 1
    qo, ko = apply_rotary_pos_emb(q, k, cos, sin, offset=offset)
    if qo.shape != q.shape or ko.shape != k.shape:
        raise RuntimeError(f"RoPE output shape mismatch: q {qo.shape} vs {q.shape}")

    if not torch.isfinite(qo).all() or not torch.isfinite(ko).all():
        raise RuntimeError("RoPE produced non-finite values")
