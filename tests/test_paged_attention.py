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


def _make_fake_hf_attn(B: int, S: int, H: int = 4, D: int = 8, Hkv: int | None = None):
    Hkv = H if Hkv is None else Hkv
    hidden = H * D
    cfg = MagicMock()
    cfg.num_attention_heads = H
    cfg.num_key_value_heads = Hkv
    cfg.hidden_size = hidden

    attn = MagicMock()
    attn.config = cfg
    attn.layer_idx = 0
    attn.head_dim = D
    attn.num_heads = H
    attn.num_kv_heads = Hkv
    attn.num_key_value_groups = H // Hkv
    attn.scaling = D ** -0.5

    torch.manual_seed(42)
    q_proj = nn.Linear(hidden, hidden, bias=False)
    k_proj = nn.Linear(hidden, Hkv * D, bias=False)
    v_proj = nn.Linear(hidden, Hkv * D, bias=False)
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


def test_kernel_config_defaults_and_env_overrides(monkeypatch):
    from nanopointllm.llama.paged_attention import _decode_block_n, _varlen_blocks

    assert _decode_block_n(128) == 64
    assert _varlen_blocks(128) == (16, 64)
    assert _varlen_blocks(64) == (16, 64)

    monkeypatch.setenv("NANOPOINTLLM_DECODE_BLOCK_N", "32")
    monkeypatch.setenv("NANOPOINTLLM_VARLEN_BLOCK_M", "16")
    monkeypatch.setenv("NANOPOINTLLM_VARLEN_BLOCK_N", "32")
    assert _decode_block_n(128) == 32
    assert _varlen_blocks(128) == (16, 32)

    monkeypatch.setenv("NANOPOINTLLM_VARLEN_BLOCK_M", "8")
    monkeypatch.setenv("NANOPOINTLLM_VARLEN_BLOCK_N", "16")
    assert _varlen_blocks(128) == (16, 64)


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


def test_decode_qkv_input_staging_matches_reference_projection():
    from nanopointllm.llama.paged_attention import _project_qkv_decode_staged

    B, S, H, Hkv, D = 2, 3, 4, 2, 8
    hidden = H * D
    hf_attn = _make_fake_hf_attn(B, S, H, D, Hkv=Hkv)
    torch.manual_seed(7)
    hidden_states = torch.randn(1, S, hidden)

    q, k, v = _project_qkv_decode_staged(
        hidden_states,
        hf_attn.q_proj,
        hf_attn.k_proj,
        hf_attn.v_proj,
        num_heads=H,
        num_kv_heads=Hkv,
        head_dim=D,
    )
    q_ref = hf_attn.q_proj(hidden_states).view(1, S, H, D)[0]
    k_ref = hf_attn.k_proj(hidden_states).view(1, S, Hkv, D)[0]
    v_ref = hf_attn.v_proj(hidden_states).view(1, S, Hkv, D)[0]

    torch.testing.assert_close(q, q_ref)
    torch.testing.assert_close(k, k_ref)
    torch.testing.assert_close(v, v_ref)


def test_decode_packed_qkv_matches_reference_projection():
    from nanopointllm.llama.paged_attention import _project_qkv_decode_packed

    B, S, H, Hkv, D = 2, 3, 4, 2, 8
    hidden = H * D
    hf_attn = _make_fake_hf_attn(B, S, H, D, Hkv=Hkv)
    packed_weight = torch.cat(
        [hf_attn.q_proj.weight, hf_attn.k_proj.weight, hf_attn.v_proj.weight],
        dim=0,
    )
    torch.manual_seed(9)
    hidden_states = torch.randn(1, S, hidden)

    q, k, v = _project_qkv_decode_packed(
        hidden_states,
        packed_weight,
        None,
        num_heads=H,
        num_kv_heads=Hkv,
        head_dim=D,
    )
    q_ref = hf_attn.q_proj(hidden_states).view(1, S, H, D)[0]
    k_ref = hf_attn.k_proj(hidden_states).view(1, S, Hkv, D)[0]
    v_ref = hf_attn.v_proj(hidden_states).view(1, S, Hkv, D)[0]

    torch.testing.assert_close(q, q_ref)
    torch.testing.assert_close(k, k_ref)
    torch.testing.assert_close(v, v_ref)


def test_decode_output_staged_matches_o_proj():
    from nanopointllm.llama.paged_attention import _project_output_decode_staged

    B, S, H, D = 2, 3, 4, 8
    hidden = H * D
    hf_attn = _make_fake_hf_attn(B, S, H, D)
    torch.manual_seed(10)
    hidden_states = torch.randn(1, S, hidden)

    out = _project_output_decode_staged(hidden_states, hf_attn.o_proj)
    ref = hf_attn.o_proj(hidden_states)
    torch.testing.assert_close(out, ref)


def test_fused_rmsnorm_packed_qkv_matches_reference_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for fused RMSNorm + packed QKV parity test")

    from nanopointllm.llama.paged_attention import _rmsnorm_packed_qkv_decode_triton

    B, S, H, Hkv, D = 2, 3, 4, 2, 8
    hidden = H * D
    hf_attn = _make_fake_hf_attn(B, S, H, D, Hkv=Hkv)
    device = torch.device("cuda")
    hf_attn.q_proj = hf_attn.q_proj.to(device)
    hf_attn.k_proj = hf_attn.k_proj.to(device)
    hf_attn.v_proj = hf_attn.v_proj.to(device)
    norm_weight = torch.randn(hidden, device=device)
    packed_weight = torch.cat(
        [hf_attn.q_proj.weight, hf_attn.k_proj.weight, hf_attn.v_proj.weight],
        dim=0,
    ).contiguous()
    torch.manual_seed(11)
    hidden_states = torch.randn(1, S, hidden, device=device, dtype=torch.float16)

    q, k, v = _rmsnorm_packed_qkv_decode_triton(
        hidden_states,
        norm_weight,
        packed_weight,
        None,
        eps=1.0e-6,
        num_heads=H,
        num_kv_heads=Hkv,
        head_dim=D,
    )

    x = hidden_states.float()
    x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1.0e-6)
    x = (norm_weight * x).to(hf_attn.q_proj.weight.dtype)
    q_ref = hf_attn.q_proj(x).view(1, S, H, D)[0]
    k_ref = hf_attn.k_proj(x).view(1, S, Hkv, D)[0]
    v_ref = hf_attn.v_proj(x).view(1, S, Hkv, D)[0]

    torch.testing.assert_close(q, q_ref.to(q.dtype), atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(k, k_ref.to(k.dtype), atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(v, v_ref.to(v.dtype), atol=2e-2, rtol=2e-2)


def test_rmsnorm_staged_packed_qkv_matches_reference_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for RMSNorm staged packed QKV parity test")

    from nanopointllm.llama.paged_attention import _rmsnorm_staged_packed_qkv_decode

    B, S, H, Hkv, D = 2, 3, 4, 2, 8
    hidden = H * D
    hf_attn = _make_fake_hf_attn(B, S, H, D, Hkv=Hkv)
    device = torch.device("cuda")
    hf_attn.q_proj = hf_attn.q_proj.to(device)
    hf_attn.k_proj = hf_attn.k_proj.to(device)
    hf_attn.v_proj = hf_attn.v_proj.to(device)
    norm_weight = torch.randn(hidden, device=device)
    packed_weight = torch.cat(
        [hf_attn.q_proj.weight, hf_attn.k_proj.weight, hf_attn.v_proj.weight],
        dim=0,
    ).contiguous()
    torch.manual_seed(13)
    hidden_states = torch.randn(1, S, hidden, device=device, dtype=torch.float16)

    q, k, v = _rmsnorm_staged_packed_qkv_decode(
        hidden_states,
        norm_weight,
        packed_weight,
        None,
        eps=1.0e-6,
        num_heads=H,
        num_kv_heads=Hkv,
        head_dim=D,
    )

    x = hidden_states.float()
    x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1.0e-6)
    x = (norm_weight * x).to(hf_attn.q_proj.weight.dtype)
    q_ref = hf_attn.q_proj(x).view(1, S, H, D)[0]
    k_ref = hf_attn.k_proj(x).view(1, S, Hkv, D)[0]
    v_ref = hf_attn.v_proj(x).view(1, S, Hkv, D)[0]

    torch.testing.assert_close(q, q_ref.to(q.dtype), atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(k, k_ref.to(k.dtype), atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(v, v_ref.to(v.dtype), atol=2e-2, rtol=2e-2)


def test_partial_prefix_prefill_parity_vs_masked_sdpa():
    """Suffix prefill after prefix-cache hit must attend to prefix plus causal suffix."""
    from nanopointllm.llama.paged_attention import PagedLlamaAttention, store_kvcache

    B, prefix_len, q_len, H, Hkv, D = 1, 4, 2, 4, 2, 8
    hidden = H * D
    hf_attn = _make_fake_hf_attn(B, prefix_len + q_len, H, D, Hkv=Hkv)
    k_cache, v_cache = _make_caches(Hkv, D)

    paged = PagedLlamaAttention.from_hf(hf_attn, k_cache, v_cache)

    torch.manual_seed(3)
    prefix_hidden = torch.randn(B, prefix_len, hidden)
    new_hidden = torch.randn(B, q_len, hidden)

    with torch.no_grad():
        k_prefix = hf_attn.k_proj(prefix_hidden).view(B, prefix_len, Hkv, D).transpose(1, 2)
        v_prefix = hf_attn.v_proj(prefix_hidden).view(B, prefix_len, Hkv, D).transpose(1, 2)

    store_kvcache(
        k_prefix.transpose(1, 2).reshape(prefix_len, Hkv, D).contiguous(),
        v_prefix.transpose(1, 2).reshape(prefix_len, Hkv, D).contiguous(),
        k_cache,
        v_cache,
        torch.arange(prefix_len, dtype=torch.int32),
    )

    slots = torch.tensor([4, 5], dtype=torch.int32)
    block_tables = torch.tensor([[0, 1]], dtype=torch.int32)
    context_lens = torch.tensor([prefix_len + q_len], dtype=torch.int32)
    cos = torch.ones(1, q_len, D)
    sin = torch.zeros(1, q_len, D)

    set_forward_context(ForwardContext(
        slot_mapping=slots,
        block_tables=block_tables,
        context_lens=context_lens,
        seq_lens=[q_len],
    ))
    try:
        out_paged, _ = paged(new_hidden, (cos, sin), attention_mask=None)
    finally:
        clear_forward_context()

    with torch.no_grad():
        q_new = hf_attn.q_proj(new_hidden).view(B, q_len, H, D).transpose(1, 2)
        k_new = hf_attn.k_proj(new_hidden).view(B, q_len, Hkv, D).transpose(1, 2)
        v_new = hf_attn.v_proj(new_hidden).view(B, q_len, Hkv, D).transpose(1, 2)
        k_full = torch.cat([k_prefix, k_new], dim=2).repeat_interleave(H // Hkv, 1)
        v_full = torch.cat([v_prefix, v_new], dim=2).repeat_interleave(H // Hkv, 1)
        mask = torch.full((q_len, prefix_len + q_len), torch.finfo(q_new.dtype).min)
        mask[0, :prefix_len + 1] = 0
        mask[1, :prefix_len + 2] = 0
        ref_attn = F.scaled_dot_product_attention(
            q_new,
            k_full,
            v_full,
            attn_mask=mask,
            scale=D ** -0.5,
        )
        ref_out = hf_attn.o_proj(ref_attn.transpose(1, 2).reshape(B, q_len, H * D))

    assert out_paged.shape == ref_out.shape
    assert torch.allclose(out_paged, ref_out, atol=1e-5), \
        f"max diff = {(out_paged - ref_out).abs().max()}"


def test_mixed_prefill_decode_uses_decode_fast_path_parity():
    """Mixed prefill+decode batch must match per-sequence SDPA reference."""
    from nanopointllm.llama.paged_attention import PagedLlamaAttention, store_kvcache

    H, Hkv, D = 4, 2, 8
    hidden = H * D
    block_size = 4
    hf_attn = _make_fake_hf_attn(B=2, S=3, H=H, D=D, Hkv=Hkv)
    k_cache, v_cache = _make_caches(Hkv, D, num_blocks=4, block_size=block_size)
    paged = PagedLlamaAttention.from_hf(hf_attn, k_cache, v_cache)

    torch.manual_seed(7)
    prefill_hidden = torch.randn(1, 2, hidden)
    decode_prefix_hidden = torch.randn(1, 4, hidden)
    decode_new_hidden = torch.randn(1, 1, hidden)
    hidden_states = torch.cat([prefill_hidden, decode_new_hidden], dim=1)

    with torch.no_grad():
        k_prefix = hf_attn.k_proj(decode_prefix_hidden).view(1, 4, Hkv, D).transpose(1, 2)
        v_prefix = hf_attn.v_proj(decode_prefix_hidden).view(1, 4, Hkv, D).transpose(1, 2)
    store_kvcache(
        k_prefix.transpose(1, 2).reshape(4, Hkv, D).contiguous(),
        v_prefix.transpose(1, 2).reshape(4, Hkv, D).contiguous(),
        k_cache,
        v_cache,
        torch.arange(4, 8, dtype=torch.int32),
    )

    ctx = ForwardContext(
        slot_mapping=torch.tensor([0, 1, 8], dtype=torch.int32),
        block_tables=torch.tensor([[0, -1], [1, 2]], dtype=torch.int32),
        context_lens=torch.tensor([2, 5], dtype=torch.int32),
        seq_lens=[2, 1],
    )
    cos = torch.ones(1, 3, D)
    sin = torch.zeros(1, 3, D)
    set_forward_context(ctx)
    try:
        out_paged, _ = paged(hidden_states, (cos, sin), attention_mask=None)
    finally:
        clear_forward_context()

    with torch.no_grad():
        q0 = hf_attn.q_proj(prefill_hidden).view(1, 2, H, D).transpose(1, 2)
        k0 = hf_attn.k_proj(prefill_hidden).view(1, 2, Hkv, D).transpose(1, 2)
        v0 = hf_attn.v_proj(prefill_hidden).view(1, 2, Hkv, D).transpose(1, 2)
        k0 = k0.repeat_interleave(H // Hkv, 1)
        v0 = v0.repeat_interleave(H // Hkv, 1)
        ref0 = F.scaled_dot_product_attention(q0, k0, v0, is_causal=True, scale=D ** -0.5)
        ref0 = hf_attn.o_proj(ref0.transpose(1, 2).reshape(1, 2, H * D))

        q1 = hf_attn.q_proj(decode_new_hidden).view(1, 1, H, D).transpose(1, 2)
        k1 = hf_attn.k_proj(decode_new_hidden).view(1, 1, Hkv, D).transpose(1, 2)
        v1 = hf_attn.v_proj(decode_new_hidden).view(1, 1, Hkv, D).transpose(1, 2)
        k_full = torch.cat([k_prefix, k1], dim=2).repeat_interleave(H // Hkv, 1)
        v_full = torch.cat([v_prefix, v1], dim=2).repeat_interleave(H // Hkv, 1)
        ref1 = F.scaled_dot_product_attention(q1, k_full, v_full, scale=D ** -0.5)
        ref1 = hf_attn.o_proj(ref1.transpose(1, 2).reshape(1, 1, H * D))

        ref = torch.cat([ref0, ref1], dim=1)

    assert out_paged.shape == ref.shape
    assert torch.allclose(out_paged, ref, atol=1e-5), \
        f"max diff = {(out_paged - ref).abs().max()}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton kernel")
def test_decode_only_triton_kernel_parity_gqa_cuda():
    """Decode-only Triton paged attention matches direct SDPA for GQA."""
    from nanopointllm.llama.paged_attention import paged_decode_attention

    device = torch.device("cuda")
    B, H, Hkv, D = 3, 4, 2, 16
    block_size = 4
    ctx_lens = torch.tensor([5, 7, 3], dtype=torch.int32, device=device)
    block_tables = torch.tensor(
        [
            [0, 1],
            [2, 3],
            [4, 5],
        ],
        dtype=torch.int32,
        device=device,
    )
    k_cache = torch.zeros(6, block_size, Hkv, D, device=device)
    v_cache = torch.zeros_like(k_cache)

    torch.manual_seed(11)
    q = torch.randn(B, H, D, device=device)
    ref_outs = []
    for b, ctx_len in enumerate(ctx_lens.tolist()):
        k_tokens = torch.randn(ctx_len, Hkv, D, device=device)
        v_tokens = torch.randn(ctx_len, Hkv, D, device=device)
        for pos in range(ctx_len):
            phys_blk = int(block_tables[b, pos // block_size])
            k_cache[phys_blk, pos % block_size] = k_tokens[pos]
            v_cache[phys_blk, pos % block_size] = v_tokens[pos]

        qi = q[b].unsqueeze(0).unsqueeze(2)  # [1, H, 1, D]
        ki = k_tokens.permute(1, 0, 2).unsqueeze(0).repeat_interleave(H // Hkv, 1)
        vi = v_tokens.permute(1, 0, 2).unsqueeze(0).repeat_interleave(H // Hkv, 1)
        oi = F.scaled_dot_product_attention(qi, ki, vi, scale=D ** -0.5)
        ref_outs.append(oi.squeeze(0).squeeze(1))
    ref = torch.stack(ref_outs, dim=0)

    out = paged_decode_attention(
        q,
        k_cache,
        v_cache,
        block_tables,
        ctx_lens,
        scale=D ** -0.5,
        num_kv_groups=H // Hkv,
    )

    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton kernel")
def test_varlen_prefill_triton_kernel_parity_gqa_cuda():
    """Varlen paged prefill Triton kernel matches direct masked SDPA."""
    from nanopointllm.llama.paged_attention import paged_varlen_attention

    device = torch.device("cuda")
    H, Hkv, D = 4, 2, 16
    block_size = 4
    q_lens = torch.tensor([5, 3], dtype=torch.int32, device=device)
    ctx_lens = torch.tensor([5, 7], dtype=torch.int32, device=device)
    q_offsets = torch.tensor([0, 5], dtype=torch.int32, device=device)
    total_q = int(q_lens.sum().item())
    block_tables = torch.tensor(
        [
            [0, 1],
            [2, 3],
        ],
        dtype=torch.int32,
        device=device,
    )
    k_cache = torch.zeros(4, block_size, Hkv, D, device=device)
    v_cache = torch.zeros_like(k_cache)

    torch.manual_seed(13)
    q = torch.randn(total_q, H, D, device=device)
    ref = torch.empty_like(q)

    for b in range(2):
        q_len = int(q_lens[b])
        ctx_len = int(ctx_lens[b])
        prefix_len = ctx_len - q_len
        k_tokens = torch.randn(ctx_len, Hkv, D, device=device)
        v_tokens = torch.randn(ctx_len, Hkv, D, device=device)
        for pos in range(ctx_len):
            phys_blk = int(block_tables[b, pos // block_size])
            k_cache[phys_blk, pos % block_size] = k_tokens[pos]
            v_cache[phys_blk, pos % block_size] = v_tokens[pos]

        start = int(q_offsets[b])
        qi = q[start:start + q_len].permute(1, 0, 2).unsqueeze(0)
        ki = k_tokens.permute(1, 0, 2).unsqueeze(0).repeat_interleave(H // Hkv, 1)
        vi = v_tokens.permute(1, 0, 2).unsqueeze(0).repeat_interleave(H // Hkv, 1)
        mask = torch.full((q_len, ctx_len), torch.finfo(q.dtype).min, device=device)
        for row in range(q_len):
            mask[row, :prefix_len + row + 1] = 0
        oi = F.scaled_dot_product_attention(qi, ki, vi, attn_mask=mask, scale=D ** -0.5)
        ref[start:start + q_len] = oi.squeeze(0).permute(1, 0, 2)

    out = paged_varlen_attention(
        q,
        k_cache,
        v_cache,
        block_tables,
        ctx_lens,
        q_lens,
        q_offsets,
        scale=D ** -0.5,
        num_kv_groups=H // Hkv,
    )

    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)
