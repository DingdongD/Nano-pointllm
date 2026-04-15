# tests/test_paged_attention.py
"""
Unit tests for PagedLlamaAttention.
Uses a minimal fake LlamaAttention to avoid loading real weights.
"""
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from unittest.mock import MagicMock

from nanopointllm.engine.forward_context import (
    ForwardContext, set_forward_context, clear_forward_context,
)


def _make_fake_hf_attn(B: int, S: int, H: int = 4, D: int = 8):
    hidden = H * D
    cfg = MagicMock()
    cfg.num_attention_heads = H
    cfg.num_key_value_heads = H
    cfg.hidden_size = hidden

    attn = MagicMock()
    attn.config = cfg
    attn.layer_idx = 0
    attn.head_dim = D
    attn.num_heads = H
    attn.num_kv_heads = H
    attn.num_key_value_groups = 1
    attn.scaling = D ** -0.5

    torch.manual_seed(42)
    q_proj = nn.Linear(hidden, hidden, bias=False)
    k_proj = nn.Linear(hidden, hidden, bias=False)
    v_proj = nn.Linear(hidden, hidden, bias=False)
    o_proj = nn.Linear(hidden, hidden, bias=False)
    attn.q_proj = q_proj
    attn.k_proj = k_proj
    attn.v_proj = v_proj
    attn.o_proj = o_proj

    def _apply_rotary(q, k, cos, sin, unsqueeze_dim=1):
        return q, k  # no-op RoPE for testing

    attn.rotary_fn = _apply_rotary
    return attn


def _make_caches(H: int, D: int, num_blocks: int = 8, block_size: int = 4):
    k_cache = torch.zeros(num_blocks, block_size, H, D)
    v_cache = torch.zeros(num_blocks, block_size, H, D)
    return k_cache, v_cache


def test_prefill_parity_vs_dense_sdpa():
    """PagedLlamaAttention prefill (varlen) must match per-seq causal sdpa."""
    from nanopointllm.llama.paged_attention import PagedLlamaAttention

    # Two prefill sequences each of length S=6, using varlen API
    # hidden_states is [1, total_tokens, hidden] = [1, 12, hidden]
    num_seqs, S, H, D = 2, 6, 4, 8
    hidden = H * D
    total_tokens = num_seqs * S
    hf_attn = _make_fake_hf_attn(num_seqs, S, H, D)
    k_cache, v_cache = _make_caches(H, D, num_blocks=8, block_size=4)

    paged = PagedLlamaAttention.from_hf(hf_attn, k_cache, v_cache)

    torch.manual_seed(0)
    # Flat [1, 12, hidden]: seq0 tokens then seq1 tokens
    hidden_states = torch.randn(1, total_tokens, hidden)
    cos = torch.ones(1, total_tokens, D)
    sin = torch.zeros(1, total_tokens, D)

    # slot_mapping: seq0 → slots 0..5 (block 0: off 0-3, block 1: off 0-1)
    #              seq1 → slots 8..13 (block 2: off 0-3, block 3: off 0-1)
    slots = torch.cat([
        torch.arange(6, dtype=torch.int32),
        torch.arange(8, 14, dtype=torch.int32),
    ])
    # block_tables[i] = physical block indices for seq i
    # seq0: blocks 0,1 (slots 0-3 in blk0, slots 4-5 in blk1)
    # seq1: blocks 2,3 (slots 8-11 in blk2, slots 12-13 in blk3)
    block_tables = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)
    context_lens = torch.tensor([S, S], dtype=torch.int32)

    ctx = ForwardContext(
        slot_mapping=slots,
        block_tables=block_tables,
        context_lens=context_lens,
        seq_lens=[S, S],
    )
    set_forward_context(ctx)

    try:
        out_paged, _ = paged(hidden_states, (cos, sin), attention_mask=None)
    finally:
        clear_forward_context()

    # Reference: per-seq causal sdpa on each seq independently
    ref_outs = []
    with torch.no_grad():
        for i in range(num_seqs):
            hs_i = hidden_states[0, i*S:(i+1)*S, :].unsqueeze(0)  # [1, S, hidden]
            q_i = hf_attn.q_proj(hs_i).view(1, S, H, D).transpose(1, 2)
            k_i = hf_attn.k_proj(hs_i).view(1, S, H, D).transpose(1, 2)
            v_i = hf_attn.v_proj(hs_i).view(1, S, H, D).transpose(1, 2)
            oi = F.scaled_dot_product_attention(q_i, k_i, v_i, is_causal=True, scale=D**-0.5)
            ref_outs.append(hf_attn.o_proj(oi.transpose(1, 2).reshape(1, S, H * D)))
    # Concatenate into [1, total_tokens, hidden]
    ref_out = torch.cat([r.squeeze(0) for r in ref_outs], dim=0).unsqueeze(0)

    assert out_paged.shape == ref_out.shape, f"shape mismatch: {out_paged.shape} vs {ref_out.shape}"
    assert torch.allclose(out_paged, ref_out, atol=1e-5), \
        f"max diff = {(out_paged - ref_out).abs().max()}"


def test_decode_parity_vs_full_context_sdpa():
    """PagedLlamaAttention decode must match full-context sdpa when cache pre-filled."""
    from nanopointllm.llama.paged_attention import PagedLlamaAttention, store_kvcache

    B, S_ctx, H, D = 1, 4, 4, 8
    hidden = H * D
    hf_attn = _make_fake_hf_attn(B, S_ctx, H, D)
    k_cache, v_cache = _make_caches(H, D)

    paged = PagedLlamaAttention.from_hf(hf_attn, k_cache, v_cache)

    torch.manual_seed(1)
    ctx_hidden = torch.randn(B, S_ctx, hidden)

    with torch.no_grad():
        k_ctx = hf_attn.k_proj(ctx_hidden).view(B, S_ctx, H, D).transpose(1, 2)  # [B,H,S,D]
        v_ctx = hf_attn.v_proj(ctx_hidden).view(B, S_ctx, H, D).transpose(1, 2)

    # Pre-fill cache at slots 0..3 (block 0, block_size=4)
    ctx_slots = torch.arange(S_ctx, dtype=torch.int32)
    store_kvcache(
        k_ctx.transpose(1, 2).reshape(B * S_ctx, H, D).contiguous(),
        v_ctx.transpose(1, 2).reshape(B * S_ctx, H, D).contiguous(),
        k_cache, v_cache, ctx_slots,
    )

    # Decode step: new token at slot 4 (blk 1, off 0)
    torch.manual_seed(2)
    new_hidden = torch.randn(B, 1, hidden)
    cos_new = torch.ones(1, 1, D)
    sin_new = torch.zeros(1, 1, D)

    decode_slots  = torch.tensor([4], dtype=torch.int32)
    block_tables  = torch.tensor([[0, 1]], dtype=torch.int32)   # blk0 and blk1
    context_lens  = torch.tensor([S_ctx + 1], dtype=torch.int32)  # 5 tokens total

    decode_ctx = ForwardContext(
        slot_mapping=decode_slots,
        block_tables=block_tables,
        context_lens=context_lens,
        seq_lens=[1] * B,
    )
    set_forward_context(decode_ctx)
    try:
        out_paged, _ = paged(new_hidden, (cos_new, sin_new), attention_mask=None)
    finally:
        clear_forward_context()

    # Reference: full-context sdpa (ctx + new token)
    with torch.no_grad():
        q_new = hf_attn.q_proj(new_hidden).view(B, 1, H, D).transpose(1, 2)
        k_new = hf_attn.k_proj(new_hidden).view(B, 1, H, D).transpose(1, 2)
        v_new = hf_attn.v_proj(new_hidden).view(B, 1, H, D).transpose(1, 2)
        k_full = torch.cat([k_ctx, k_new], dim=2)  # [B,H,S+1,D]
        v_full = torch.cat([v_ctx, v_new], dim=2)
        ref_attn = F.scaled_dot_product_attention(q_new, k_full, v_full, scale=D**-0.5)
        ref_out = hf_attn.o_proj(ref_attn.transpose(1, 2).reshape(B, 1, H * D))

    assert out_paged.shape == ref_out.shape
    assert torch.allclose(out_paged, ref_out, atol=1e-5), \
        f"max diff = {(out_paged - ref_out).abs().max()}"
