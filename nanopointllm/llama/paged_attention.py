"""
PagedLlamaAttention + Triton store_kvcache kernel.

store_kvcache writes K/V tokens into the physical KV pool at given slots.
PagedLlamaAttention replaces HF LlamaAttention; reads routing from ForwardContext.
"""
from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

from nanopointllm.engine.forward_context import get_forward_context
from nanopointllm.profiling import nvtx_stage


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value >= minimum else default


def _decode_block_n(head_dim: int) -> int:
    default = 64 if head_dim >= 128 else 128
    return _env_int("NANOPOINTLLM_DECODE_BLOCK_N", default, minimum=16)


def _decode_num_warps() -> int:
    return _env_int("NANOPOINTLLM_DECODE_NUM_WARPS", 4, minimum=1)


def _decode_num_stages() -> int:
    return _env_int("NANOPOINTLLM_DECODE_NUM_STAGES", 2, minimum=1)


def _decode_grouped_variant() -> str:
    variant = os.environ.get("NANOPOINTLLM_DECODE_GROUPED_VARIANT", "g4").strip().lower()
    if variant in {"g4", "generic"}:
        return variant
    return "g4"


def _varlen_blocks(head_dim: int) -> tuple[int, int]:
    if head_dim >= 128:
        default_m, default_n = 16, 64
    elif head_dim >= 64:
        default_m, default_n = 16, 64
    else:
        default_m, default_n = 16, 128
    return (
        _env_int("NANOPOINTLLM_VARLEN_BLOCK_M", default_m, minimum=16),
        _env_int("NANOPOINTLLM_VARLEN_BLOCK_N", default_n, minimum=32),
    )


def _linear_weight_and_bias(proj: nn.Module) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    return proj.weight, getattr(proj, "bias", None)


def _project_qkv_decode_staged(
    hidden_states: torch.Tensor,
    q_proj: nn.Module,
    k_proj: nn.Module,
    v_proj: nn.Module,
    *,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project decode-only hidden states through a shared contiguous [B, hidden] staging view.

    Returns:
      q_heads: [B, H, D]
      k_heads: [B, Hkv, D]
      v_heads: [B, Hkv, D]
    """
    staged = hidden_states[0].contiguous()
    q_weight, q_bias = _linear_weight_and_bias(q_proj)
    k_weight, k_bias = _linear_weight_and_bias(k_proj)
    v_weight, v_bias = _linear_weight_and_bias(v_proj)
    q = F.linear(staged, q_weight, q_bias).view(-1, num_heads, head_dim)
    k = F.linear(staged, k_weight, k_bias).view(-1, num_kv_heads, head_dim)
    v = F.linear(staged, v_weight, v_bias).view(-1, num_kv_heads, head_dim)
    return q, k, v


def _project_qkv_decode_packed(
    hidden_states: torch.Tensor,
    packed_weight: torch.Tensor,
    packed_bias: Optional[torch.Tensor],
    *,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    staged = hidden_states[0].contiguous()
    q_size = num_heads * head_dim
    kv_size = num_kv_heads * head_dim
    qkv = F.linear(staged, packed_weight, packed_bias)
    q, k, v = torch.split(qkv, (q_size, kv_size, kv_size), dim=-1)
    return (
        q.view(-1, num_heads, head_dim),
        k.view(-1, num_kv_heads, head_dim),
        v.view(-1, num_kv_heads, head_dim),
    )


def _project_output_decode_staged(
    hidden_states: torch.Tensor,
    o_proj: nn.Module,
) -> torch.Tensor:
    staged = hidden_states[0].contiguous()
    out = F.linear(staged, o_proj.weight, getattr(o_proj, "bias", None))
    return out.unsqueeze(0)


@triton.jit
def _rmsnorm_only_kernel(
    x_ptr,
    weight_ptr,
    out_ptr,
    stride_row: tl.constexpr,
    hidden_size: tl.constexpr,
    block_h: tl.constexpr,
    eps: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, block_h)
    mask = offs < hidden_size
    x = tl.load(x_ptr + row * stride_row + offs, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / hidden_size
    y = x * tl.rsqrt(var + eps) * weight
    tl.store(out_ptr + row * stride_row + offs, y, mask=mask)


def _rmsnorm_decode_triton(hidden_states: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    rows, hidden_size = hidden_states.shape
    out = torch.empty_like(hidden_states)
    block_h = 256 if hidden_size >= 256 else triton.next_power_of_2(hidden_size)
    _rmsnorm_only_kernel[(rows,)](
        hidden_states,
        weight,
        out,
        hidden_states.stride(0),
        hidden_size=hidden_size,
        block_h=block_h,
        eps=float(eps),
    )
    return out


@triton.jit
def _rmsnorm_packed_qkv_kernel(
    x_ptr,
    norm_weight_ptr,
    packed_weight_ptr,
    packed_bias_ptr,
    out_ptr,
    stride_x_row: tl.constexpr,
    stride_x_col: tl.constexpr,
    stride_w_out: tl.constexpr,
    stride_w_col: tl.constexpr,
    stride_out_row: tl.constexpr,
    stride_out_col: tl.constexpr,
    hidden_size: tl.constexpr,
    out_size: tl.constexpr,
    block_h: tl.constexpr,
    block_o: tl.constexpr,
    eps: tl.constexpr,
    has_bias: tl.constexpr,
):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    offs_h = tl.arange(0, block_h)
    offs_o = tile * block_o + tl.arange(0, block_o)
    o_mask = offs_o < out_size

    sumsq = tl.zeros((), dtype=tl.float32)
    for start_h in tl.range(0, hidden_size, block_h):
        h = start_h + offs_h
        h_mask = h < hidden_size
        x = tl.load(
            x_ptr + row * stride_x_row + h * stride_x_col,
            mask=h_mask,
            other=0.0,
        ).to(tl.float32)
        sumsq += tl.sum(x * x, axis=0)

    inv_rms = tl.rsqrt(sumsq / hidden_size + eps)
    acc = tl.zeros((block_o,), dtype=tl.float32)

    for start_h in tl.range(0, hidden_size, block_h):
        h = start_h + offs_h
        h_mask = h < hidden_size
        x = tl.load(
            x_ptr + row * stride_x_row + h * stride_x_col,
            mask=h_mask,
            other=0.0,
        ).to(tl.float32)
        norm_weight = tl.load(
            norm_weight_ptr + h,
            mask=h_mask,
            other=0.0,
        ).to(tl.float32)
        x_norm = x * inv_rms * norm_weight
        w = tl.load(
            packed_weight_ptr
            + offs_o[:, None] * stride_w_out
            + h[None, :] * stride_w_col,
            mask=o_mask[:, None] & h_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        acc += tl.sum(w * x_norm[None, :], axis=1)

    if has_bias:
        bias = tl.load(packed_bias_ptr + offs_o, mask=o_mask, other=0.0).to(tl.float32)
        acc += bias

    tl.store(
        out_ptr + row * stride_out_row + offs_o * stride_out_col,
        acc,
        mask=o_mask,
    )


def _rmsnorm_packed_qkv_decode_triton(
    hidden_states: torch.Tensor,
    norm_weight: torch.Tensor,
    packed_weight: torch.Tensor,
    packed_bias: Optional[torch.Tensor],
    *,
    eps: float,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    staged = hidden_states[0].contiguous()
    rows, hidden_size = staged.shape
    out_size = packed_weight.shape[0]
    out = torch.empty((rows, out_size), device=staged.device, dtype=staged.dtype)
    block_h = 256 if hidden_size >= 256 else triton.next_power_of_2(hidden_size)
    block_o = 128 if out_size >= 128 else triton.next_power_of_2(out_size)
    _rmsnorm_packed_qkv_kernel[(rows, triton.cdiv(out_size, block_o))](
        staged,
        norm_weight,
        packed_weight,
        packed_bias if packed_bias is not None else out,
        out,
        staged.stride(0),
        staged.stride(1),
        packed_weight.stride(0),
        packed_weight.stride(1),
        out.stride(0),
        out.stride(1),
        hidden_size=hidden_size,
        out_size=out_size,
        block_h=block_h,
        block_o=block_o,
        eps=float(eps),
        has_bias=packed_bias is not None,
    )
    q_size = num_heads * head_dim
    kv_size = num_kv_heads * head_dim
    q, k, v = torch.split(out, (q_size, kv_size, kv_size), dim=-1)
    return (
        q.view(-1, num_heads, head_dim),
        k.view(-1, num_kv_heads, head_dim),
        v.view(-1, num_kv_heads, head_dim),
    )


def _rmsnorm_staged_packed_qkv_decode(
    hidden_states: torch.Tensor,
    norm_weight: torch.Tensor,
    packed_weight: torch.Tensor,
    packed_bias: Optional[torch.Tensor],
    *,
    eps: float,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    staged = _rmsnorm_decode_triton(hidden_states[0].contiguous(), norm_weight, eps).contiguous()
    if staged.dtype != packed_weight.dtype:
        staged = staged.to(packed_weight.dtype)
    qkv = F.linear(staged, packed_weight, packed_bias)
    q_size = num_heads * head_dim
    kv_size = num_kv_heads * head_dim
    q, k, v = torch.split(qkv, (q_size, kv_size, kv_size), dim=-1)
    return (
        q.view(-1, num_heads, head_dim),
        k.view(-1, num_kv_heads, head_dim),
        v.view(-1, num_kv_heads, head_dim),
    )


def _partial_prefill_mask(
    q_len: int,
    ctx_len: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Additive mask for suffix-prefill with a cached prefix.

    Query ``j`` corresponds to absolute position ``ctx_len - q_len + j`` and
    may attend through that key position, including the cached prefix.
    """
    prefix_len = ctx_len - q_len
    q_pos = torch.arange(q_len, device=device).unsqueeze(1)
    k_pos = torch.arange(ctx_len, device=device).unsqueeze(0)
    allowed = k_pos <= (prefix_len + q_pos)
    mask = torch.full((q_len, ctx_len), torch.finfo(dtype).min, device=device, dtype=dtype)
    return mask.masked_fill(allowed, 0)


# ──────────────────────────────────────────────────────────────────────────────
# Triton kernel: scatter K/V into physical pool slots
# ──────────────────────────────────────────────────────────────────────────────

@triton.jit
def _store_kvcache_kernel(
    key_ptr,   key_stride,    # [T, HD]
    val_ptr,   val_stride,
    k_cache_ptr,              # [num_blocks * block_size, HD]  (flattened)
    v_cache_ptr,
    cache_stride,             # = HD
    slot_mapping_ptr,
    D: tl.constexpr,          # HD = num_heads * head_dim
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot < 0:
        return                # padding token, skip
    offsets = tl.arange(0, D)
    k = tl.load(key_ptr + idx * key_stride + offsets)
    v = tl.load(val_ptr + idx * val_stride + offsets)
    tl.store(k_cache_ptr + slot * cache_stride + offsets, k)
    tl.store(v_cache_ptr + slot * cache_stride + offsets, v)


def store_kvcache(
    key: torch.Tensor,           # [T, H, D]  contiguous
    value: torch.Tensor,         # [T, H, D]
    k_cache: torch.Tensor,       # [num_blocks, block_size, H, D]
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,  # [T] int32
) -> None:
    """Write key/value tokens into the physical KV cache at given physical slots."""
    T, H, D = key.shape
    HD = H * D
    key_flat    = key.reshape(T, HD).contiguous()
    value_flat  = value.reshape(T, HD).contiguous()
    k_cache_flat = k_cache.view(-1, HD)
    v_cache_flat = v_cache.view(-1, HD)

    # Triton requires constexpr D to be a power of 2
    HD_pow2 = 1
    while HD_pow2 < HD:
        HD_pow2 *= 2

    if HD_pow2 == HD and key.is_cuda:
        _store_kvcache_kernel[(T,)](
            key_flat, key_flat.stride(0),
            value_flat, value_flat.stride(0),
            k_cache_flat, v_cache_flat,
            k_cache_flat.stride(0),
            slot_mapping,
            D=HD,
        )
    else:
        _store_kvcache_pytorch(key, value, k_cache, v_cache, slot_mapping)


def _store_kvcache_pytorch(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    """Pure-PyTorch fallback for store_kvcache (non-power-of-2 HD or CPU)."""
    block_size = k_cache.shape[1]
    for t, slot in enumerate(slot_mapping.tolist()):
        if slot < 0:
            continue
        blk = slot // block_size
        off = slot % block_size
        k_cache[blk, off] = key[t]
        v_cache[blk, off] = value[t]


# ──────────────────────────────────────────────────────────────────────────────
# Triton kernel: decode-only paged attention (q_len = 1)
# ──────────────────────────────────────────────────────────────────────────────

@triton.jit
def _paged_decode_attention_grouped4_kernel(
    q_ptr,                 # [B, H, D]
    k_cache_ptr,           # [num_blocks, block_size, Hkv, D]
    v_cache_ptr,
    block_tables_ptr,      # [B, max_blocks]
    context_lens_ptr,      # [B]
    out_ptr,               # [B, H, D]
    q_stride_b: tl.constexpr,
    q_stride_h: tl.constexpr,
    q_stride_d: tl.constexpr,
    cache_stride_block: tl.constexpr,
    cache_stride_token: tl.constexpr,
    cache_stride_head: tl.constexpr,
    cache_stride_d: tl.constexpr,
    block_table_stride_b: tl.constexpr,
    block_table_stride_blk: tl.constexpr,
    out_stride_b: tl.constexpr,
    out_stride_h: tl.constexpr,
    out_stride_d: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
    MAX_CONTEXT_LEN: tl.constexpr,
    SCALE: tl.constexpr,
):
    seq_id = tl.program_id(0)
    kv_head = tl.program_id(1)
    base_head = kv_head * 4

    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < HEAD_DIM
    h0 = base_head + 0
    h1 = base_head + 1
    h2 = base_head + 2
    h3 = base_head + 3
    q0 = tl.load(
        q_ptr + seq_id * q_stride_b + h0 * q_stride_h + offs_d * q_stride_d,
        mask=d_mask,
        other=0.0,
    ).to(tl.float32)
    q1 = tl.load(
        q_ptr + seq_id * q_stride_b + h1 * q_stride_h + offs_d * q_stride_d,
        mask=d_mask,
        other=0.0,
    ).to(tl.float32)
    q2 = tl.load(
        q_ptr + seq_id * q_stride_b + h2 * q_stride_h + offs_d * q_stride_d,
        mask=d_mask,
        other=0.0,
    ).to(tl.float32)
    q3 = tl.load(
        q_ptr + seq_id * q_stride_b + h3 * q_stride_h + offs_d * q_stride_d,
        mask=d_mask,
        other=0.0,
    ).to(tl.float32)

    ctx_len = tl.load(context_lens_ptr + seq_id)
    m0 = tl.full((), -3.4028234663852886e38, tl.float32)
    m1 = tl.full((), -3.4028234663852886e38, tl.float32)
    m2 = tl.full((), -3.4028234663852886e38, tl.float32)
    m3 = tl.full((), -3.4028234663852886e38, tl.float32)
    l0 = tl.zeros((), tl.float32)
    l1 = tl.zeros((), tl.float32)
    l2 = tl.zeros((), tl.float32)
    l3 = tl.zeros((), tl.float32)
    acc0 = tl.zeros((BLOCK_D,), tl.float32)
    acc1 = tl.zeros((BLOCK_D,), tl.float32)
    acc2 = tl.zeros((BLOCK_D,), tl.float32)
    acc3 = tl.zeros((BLOCK_D,), tl.float32)
    offs_n = tl.arange(0, BLOCK_N)

    for start in tl.range(0, MAX_CONTEXT_LEN, BLOCK_N):
        pos = start + offs_n
        valid_n = pos < ctx_len
        logical_blk = pos // BLOCK_SIZE
        blk_off = pos - logical_blk * BLOCK_SIZE
        phys_blk = tl.load(
            block_tables_ptr
            + seq_id * block_table_stride_b
            + logical_blk * block_table_stride_blk,
            mask=valid_n,
            other=0,
        )

        kv_base = (
            phys_blk[:, None] * cache_stride_block
            + blk_off[:, None] * cache_stride_token
            + kv_head * cache_stride_head
            + offs_d[None, :] * cache_stride_d
        )
        k = tl.load(
            k_cache_ptr + kv_base,
            mask=valid_n[:, None] & d_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        v = tl.load(
            v_cache_ptr + kv_base,
            mask=valid_n[:, None] & d_mask[None, :],
            other=0.0,
        ).to(tl.float32)

        s0 = tl.sum(k * q0[None, :], axis=1) * SCALE
        s1 = tl.sum(k * q1[None, :], axis=1) * SCALE
        s2 = tl.sum(k * q2[None, :], axis=1) * SCALE
        s3 = tl.sum(k * q3[None, :], axis=1) * SCALE
        s0 = tl.where(valid_n, s0, -3.4028234663852886e38)
        s1 = tl.where(valid_n, s1, -3.4028234663852886e38)
        s2 = tl.where(valid_n, s2, -3.4028234663852886e38)
        s3 = tl.where(valid_n, s3, -3.4028234663852886e38)

        m0_new = tl.max(s0, axis=0)
        m1_new = tl.max(s1, axis=0)
        m2_new = tl.max(s2, axis=0)
        m3_new = tl.max(s3, axis=0)
        p0 = tl.exp(s0 - m0_new)
        p1 = tl.exp(s1 - m1_new)
        p2 = tl.exp(s2 - m2_new)
        p3 = tl.exp(s3 - m3_new)
        l0_new = tl.sum(p0, axis=0)
        l1_new = tl.sum(p1, axis=0)
        l2_new = tl.sum(p2, axis=0)
        l3_new = tl.sum(p3, axis=0)
        acc0_new = tl.sum(p0[:, None] * v, axis=0)
        acc1_new = tl.sum(p1[:, None] * v, axis=0)
        acc2_new = tl.sum(p2[:, None] * v, axis=0)
        acc3_new = tl.sum(p3[:, None] * v, axis=0)

        m0_next = tl.maximum(m0, m0_new)
        m1_next = tl.maximum(m1, m1_new)
        m2_next = tl.maximum(m2, m2_new)
        m3_next = tl.maximum(m3, m3_new)
        a0 = tl.exp(m0 - m0_next)
        a1 = tl.exp(m1 - m1_next)
        a2 = tl.exp(m2 - m2_next)
        a3 = tl.exp(m3 - m3_next)
        b0 = tl.exp(m0_new - m0_next)
        b1 = tl.exp(m1_new - m1_next)
        b2 = tl.exp(m2_new - m2_next)
        b3 = tl.exp(m3_new - m3_next)
        acc0 = acc0 * a0 + acc0_new * b0
        acc1 = acc1 * a1 + acc1_new * b1
        acc2 = acc2 * a2 + acc2_new * b2
        acc3 = acc3 * a3 + acc3_new * b3
        l0 = l0 * a0 + l0_new * b0
        l1 = l1 * a1 + l1_new * b1
        l2 = l2 * a2 + l2_new * b2
        l3 = l3 * a3 + l3_new * b3
        m0 = m0_next
        m1 = m1_next
        m2 = m2_next
        m3 = m3_next

    out0 = acc0 / tl.where(l0 > 0, l0, 1.0)
    out1 = acc1 / tl.where(l1 > 0, l1, 1.0)
    out2 = acc2 / tl.where(l2 > 0, l2, 1.0)
    out3 = acc3 / tl.where(l3 > 0, l3, 1.0)
    tl.store(out_ptr + seq_id * out_stride_b + h0 * out_stride_h + offs_d * out_stride_d, out0, mask=d_mask)
    tl.store(out_ptr + seq_id * out_stride_b + h1 * out_stride_h + offs_d * out_stride_d, out1, mask=d_mask)
    tl.store(out_ptr + seq_id * out_stride_b + h2 * out_stride_h + offs_d * out_stride_d, out2, mask=d_mask)
    tl.store(out_ptr + seq_id * out_stride_b + h3 * out_stride_h + offs_d * out_stride_d, out3, mask=d_mask)


@triton.jit
def _paged_decode_attention_grouped_kernel(
    q_ptr,                 # [B, H, D]
    k_cache_ptr,           # [num_blocks, block_size, Hkv, D]
    v_cache_ptr,
    block_tables_ptr,      # [B, max_blocks]
    context_lens_ptr,      # [B]
    out_ptr,               # [B, H, D]
    q_stride_b: tl.constexpr,
    q_stride_h: tl.constexpr,
    q_stride_d: tl.constexpr,
    cache_stride_block: tl.constexpr,
    cache_stride_token: tl.constexpr,
    cache_stride_head: tl.constexpr,
    cache_stride_d: tl.constexpr,
    block_table_stride_b: tl.constexpr,
    block_table_stride_blk: tl.constexpr,
    out_stride_b: tl.constexpr,
    out_stride_h: tl.constexpr,
    out_stride_d: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    NUM_KV_GROUPS: tl.constexpr,
    MAX_CONTEXT_LEN: tl.constexpr,
    SCALE: tl.constexpr,
):
    seq_id = tl.program_id(0)
    kv_head = tl.program_id(1)

    offs_g = tl.arange(0, NUM_KV_GROUPS)
    offs_d = tl.arange(0, BLOCK_D)
    head_ids = kv_head * NUM_KV_GROUPS + offs_g
    g_mask = head_ids < NUM_HEADS
    d_mask = offs_d < HEAD_DIM

    q = tl.load(
        q_ptr
        + seq_id * q_stride_b
        + head_ids[:, None] * q_stride_h
        + offs_d[None, :] * q_stride_d,
        mask=g_mask[:, None] & d_mask[None, :],
        other=0.0,
    ).to(tl.float32)

    ctx_len = tl.load(context_lens_ptr + seq_id)
    m_i = tl.full((NUM_KV_GROUPS,), -3.4028234663852886e38, tl.float32)
    l_i = tl.zeros((NUM_KV_GROUPS,), tl.float32)
    acc = tl.zeros((NUM_KV_GROUPS, BLOCK_D), tl.float32)
    offs_n = tl.arange(0, BLOCK_N)

    for start in tl.range(0, MAX_CONTEXT_LEN, BLOCK_N):
        pos = start + offs_n
        valid_n = pos < ctx_len
        logical_blk = pos // BLOCK_SIZE
        blk_off = pos - logical_blk * BLOCK_SIZE
        phys_blk = tl.load(
            block_tables_ptr
            + seq_id * block_table_stride_b
            + logical_blk * block_table_stride_blk,
            mask=valid_n,
            other=0,
        )

        kv_base = (
            phys_blk[:, None] * cache_stride_block
            + blk_off[:, None] * cache_stride_token
            + kv_head * cache_stride_head
            + offs_d[None, :] * cache_stride_d
        )
        k = tl.load(
            k_cache_ptr + kv_base,
            mask=valid_n[:, None] & d_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        v = tl.load(
            v_cache_ptr + kv_base,
            mask=valid_n[:, None] & d_mask[None, :],
            other=0.0,
        ).to(tl.float32)

        scores = tl.sum(q[:, None, :] * k[None, :, :], axis=2) * SCALE
        scores = tl.where(g_mask[:, None] & valid_n[None, :], scores, -3.4028234663852886e38)

        m_ij = tl.max(scores, axis=1)
        p = tl.exp(scores - m_ij[:, None])
        l_ij = tl.sum(p, axis=1)
        acc_ij = tl.sum(p[:, :, None] * v[None, :, :], axis=1)

        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        beta = tl.exp(m_ij - m_new)
        acc = acc * alpha[:, None] + acc_ij * beta[:, None]
        l_i = l_i * alpha + l_ij * beta
        m_i = tl.where(g_mask, m_new, m_i)

    denom = tl.where(l_i > 0, l_i, 1.0)
    out = acc / denom[:, None]
    tl.store(
        out_ptr
        + seq_id * out_stride_b
        + head_ids[:, None] * out_stride_h
        + offs_d[None, :] * out_stride_d,
        out,
        mask=g_mask[:, None] & d_mask[None, :],
    )


@triton.jit
def _paged_decode_attention_tile_kernel(
    q_ptr,                 # [B, H, D]
    k_cache_ptr,           # [num_blocks, block_size, Hkv, D]
    v_cache_ptr,
    block_tables_ptr,      # [B, max_blocks]
    context_lens_ptr,      # [B]
    partial_m_ptr,         # [B, H, T]
    partial_l_ptr,         # [B, H, T]
    partial_acc_ptr,       # [B, H, T, D]
    q_stride_b: tl.constexpr,
    q_stride_h: tl.constexpr,
    q_stride_d: tl.constexpr,
    cache_stride_block: tl.constexpr,
    cache_stride_token: tl.constexpr,
    cache_stride_head: tl.constexpr,
    cache_stride_d: tl.constexpr,
    block_table_stride_b: tl.constexpr,
    block_table_stride_blk: tl.constexpr,
    partial_scalar_stride_b: tl.constexpr,
    partial_scalar_stride_h: tl.constexpr,
    partial_scalar_stride_t: tl.constexpr,
    partial_acc_stride_b: tl.constexpr,
    partial_acc_stride_h: tl.constexpr,
    partial_acc_stride_t: tl.constexpr,
    partial_acc_stride_d: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_KV_GROUPS: tl.constexpr,
    SCALE: tl.constexpr,
):
    seq_id = tl.program_id(0)
    head_id = tl.program_id(1)
    tile_id = tl.program_id(2)
    kv_head = head_id // NUM_KV_GROUPS

    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < HEAD_DIM
    q = tl.load(
        q_ptr + seq_id * q_stride_b + head_id * q_stride_h + offs_d * q_stride_d,
        mask=d_mask,
        other=0.0,
    ).to(tl.float32)

    ctx_len = tl.load(context_lens_ptr + seq_id)
    m_i = tl.full((), -3.4028234663852886e38, tl.float32)
    l_i = tl.full((), 0.0, tl.float32)
    acc = tl.zeros((BLOCK_D,), tl.float32)

    start = tile_id * BLOCK_N
    offs_n = tl.arange(0, BLOCK_N)
    pos = start + offs_n
    valid_n = pos < ctx_len
    logical_blk = pos // BLOCK_SIZE
    blk_off = pos - logical_blk * BLOCK_SIZE
    phys_blk = tl.load(
        block_tables_ptr
        + seq_id * block_table_stride_b
        + logical_blk * block_table_stride_blk,
        mask=valid_n,
        other=0,
    )

    kv_base = (
        phys_blk[:, None] * cache_stride_block
        + blk_off[:, None] * cache_stride_token
        + kv_head * cache_stride_head
        + offs_d[None, :] * cache_stride_d
    )
    k = tl.load(k_cache_ptr + kv_base, mask=valid_n[:, None] & d_mask[None, :], other=0.0)
    v = tl.load(v_cache_ptr + kv_base, mask=valid_n[:, None] & d_mask[None, :], other=0.0)
    scores = tl.sum(k.to(tl.float32) * q[None, :], axis=1) * SCALE
    scores = tl.where(valid_n, scores, -3.4028234663852886e38)

    m_new = tl.max(scores, axis=0)
    p = tl.exp(scores - m_new)
    acc = tl.sum(p[:, None] * v.to(tl.float32), axis=0)
    l_i = tl.sum(p, axis=0)
    m_i = tl.where(l_i > 0, m_new, -3.4028234663852886e38)

    scalar_base = (
        seq_id * partial_scalar_stride_b
        + head_id * partial_scalar_stride_h
        + tile_id * partial_scalar_stride_t
    )
    acc_base = (
        seq_id * partial_acc_stride_b
        + head_id * partial_acc_stride_h
        + tile_id * partial_acc_stride_t
    )
    tl.store(partial_m_ptr + scalar_base, m_i)
    tl.store(partial_l_ptr + scalar_base, l_i)
    tl.store(
        partial_acc_ptr + acc_base + offs_d * partial_acc_stride_d,
        acc,
        mask=d_mask,
    )


@triton.jit
def _paged_decode_attention_reduce_kernel(
    partial_m_ptr,         # [B, H, T]
    partial_l_ptr,         # [B, H, T]
    partial_acc_ptr,       # [B, H, T, D]
    out_ptr,               # [B, H, D]
    partial_scalar_stride_b: tl.constexpr,
    partial_scalar_stride_h: tl.constexpr,
    partial_scalar_stride_t: tl.constexpr,
    partial_acc_stride_b: tl.constexpr,
    partial_acc_stride_h: tl.constexpr,
    partial_acc_stride_t: tl.constexpr,
    partial_acc_stride_d: tl.constexpr,
    out_stride_b: tl.constexpr,
    out_stride_h: tl.constexpr,
    out_stride_d: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NUM_TILES: tl.constexpr,
):
    seq_id = tl.program_id(0)
    head_id = tl.program_id(1)

    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < HEAD_DIM
    m_i = tl.full((), -3.4028234663852886e38, tl.float32)
    l_i = tl.full((), 0.0, tl.float32)
    acc = tl.zeros((BLOCK_D,), tl.float32)

    for tile_id in tl.range(0, NUM_TILES):
        scalar_base = (
            seq_id * partial_scalar_stride_b
            + head_id * partial_scalar_stride_h
            + tile_id * partial_scalar_stride_t
        )
        acc_base = (
            seq_id * partial_acc_stride_b
            + head_id * partial_acc_stride_h
            + tile_id * partial_acc_stride_t
        )
        m_t = tl.load(partial_m_ptr + scalar_base)
        l_t = tl.load(partial_l_ptr + scalar_base)
        acc_t = tl.load(
            partial_acc_ptr + acc_base + offs_d * partial_acc_stride_d,
            mask=d_mask,
            other=0.0,
        ).to(tl.float32)

        m_new = tl.maximum(m_i, m_t)
        alpha = tl.exp(m_i - m_new)
        beta = tl.exp(m_t - m_new)
        acc = acc * alpha + acc_t * beta
        l_i = l_i * alpha + l_t * beta
        m_i = m_new

    out = acc / l_i
    tl.store(
        out_ptr + seq_id * out_stride_b + head_id * out_stride_h + offs_d * out_stride_d,
        out,
        mask=d_mask,
    )


def _paged_decode_attention_pytorch(
    q: torch.Tensor,              # [B, H, D]
    k_cache: torch.Tensor,        # [num_blocks, block_size, Hkv, D]
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,   # [B, max_blocks]
    context_lens: torch.Tensor,   # [B]
    scale: float,
    num_kv_groups: int,
) -> torch.Tensor:
    B, H, _ = q.shape
    block_size = k_cache.shape[1]
    outs = []
    for b in range(B):
        ctx_len = int(context_lens[b])
        num_blks = (ctx_len + block_size - 1) // block_size
        bt = block_tables[b, :num_blks]
        k_full = k_cache[bt].reshape(-1, k_cache.shape[2], k_cache.shape[3])[:ctx_len]
        v_full = v_cache[bt].reshape(-1, v_cache.shape[2], v_cache.shape[3])[:ctx_len]
        qi = q[b].unsqueeze(0).unsqueeze(2)  # [1, H, 1, D]
        ki = k_full.permute(1, 0, 2).unsqueeze(0).repeat_interleave(num_kv_groups, 1)
        vi = v_full.permute(1, 0, 2).unsqueeze(0).repeat_interleave(num_kv_groups, 1)
        oi = F.scaled_dot_product_attention(qi, ki, vi, scale=scale)
        outs.append(oi.squeeze(0).squeeze(1))
    return torch.stack(outs, dim=0)


def paged_decode_attention(
    q: torch.Tensor,              # [B, H, D]
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    scale: float,
    num_kv_groups: int,
    block_n: Optional[int] = None,
) -> torch.Tensor:
    """Decode-only paged attention for q_len=1.

    On CUDA this avoids materializing per-sequence gathered KV and runs one
    Triton program per (sequence, query head). CPU falls back to the reference
    PyTorch implementation used by tests.
    """
    if not q.is_cuda:
        return _paged_decode_attention_pytorch(
            q, k_cache, v_cache, block_tables, context_lens, scale, num_kv_groups,
        )
    B, H, D = q.shape
    if B == 0:
        return q.new_empty((0, H, D))
    block_tables = block_tables.contiguous()
    context_lens = context_lens.contiguous()
    q = q.contiguous()
    out = torch.empty_like(q)
    block_d = triton.next_power_of_2(D)
    if block_n is None:
        block_n = _decode_block_n(D)
    num_warps = _decode_num_warps()
    num_stages = _decode_num_stages()
    max_context_len = block_tables.shape[1] * k_cache.shape[1]
    if max_context_len <= 0:
        return out.zero_()
    if num_kv_groups == 4 and _decode_grouped_variant() == "g4":
        num_kv_heads = triton.cdiv(H, 4)
        _paged_decode_attention_grouped4_kernel[(B, num_kv_heads)](
            q,
            k_cache,
            v_cache,
            block_tables,
            context_lens,
            out,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k_cache.stride(0),
            k_cache.stride(1),
            k_cache.stride(2),
            k_cache.stride(3),
            block_tables.stride(0),
            block_tables.stride(1),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            BLOCK_SIZE=k_cache.shape[1],
            HEAD_DIM=D,
            BLOCK_D=block_d,
            BLOCK_N=block_n,
            MAX_CONTEXT_LEN=max_context_len,
            SCALE=float(scale),
            num_warps=num_warps,
            num_stages=num_stages,
        )
        return out
    if num_kv_groups > 1:
        num_kv_heads = triton.cdiv(H, num_kv_groups)
        _paged_decode_attention_grouped_kernel[(B, num_kv_heads)](
            q,
            k_cache,
            v_cache,
            block_tables,
            context_lens,
            out,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k_cache.stride(0),
            k_cache.stride(1),
            k_cache.stride(2),
            k_cache.stride(3),
            block_tables.stride(0),
            block_tables.stride(1),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            BLOCK_SIZE=k_cache.shape[1],
            HEAD_DIM=D,
            BLOCK_D=block_d,
            BLOCK_N=block_n,
            NUM_HEADS=H,
            NUM_KV_GROUPS=num_kv_groups,
            MAX_CONTEXT_LEN=max_context_len,
            SCALE=float(scale),
            num_warps=num_warps,
            num_stages=num_stages,
        )
        return out
    num_tiles = triton.cdiv(max_context_len, block_n)
    partial_m = torch.empty((B, H, num_tiles), device=q.device, dtype=torch.float32)
    partial_l = torch.empty_like(partial_m)
    partial_acc = torch.empty((B, H, num_tiles, D), device=q.device, dtype=torch.float32)
    _paged_decode_attention_tile_kernel[(B, H, num_tiles)](
        q,
        k_cache,
        v_cache,
        block_tables,
        context_lens,
        partial_m,
        partial_l,
        partial_acc,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        k_cache.stride(3),
        block_tables.stride(0),
        block_tables.stride(1),
        partial_m.stride(0),
        partial_m.stride(1),
        partial_m.stride(2),
        partial_acc.stride(0),
        partial_acc.stride(1),
        partial_acc.stride(2),
        partial_acc.stride(3),
        BLOCK_SIZE=k_cache.shape[1],
        HEAD_DIM=D,
        BLOCK_D=block_d,
        BLOCK_N=block_n,
        NUM_KV_GROUPS=num_kv_groups,
        SCALE=float(scale),
        num_warps=num_warps,
        num_stages=num_stages,
    )
    _paged_decode_attention_reduce_kernel[(B, H)](
        partial_m,
        partial_l,
        partial_acc,
        out,
        partial_m.stride(0),
        partial_m.stride(1),
        partial_m.stride(2),
        partial_acc.stride(0),
        partial_acc.stride(1),
        partial_acc.stride(2),
        partial_acc.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        HEAD_DIM=D,
        BLOCK_D=block_d,
        NUM_TILES=num_tiles,
        num_warps=1,
        num_stages=1,
    )
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Triton kernel: varlen paged attention for prefill / suffix-prefill
# ──────────────────────────────────────────────────────────────────────────────

@triton.jit
def _paged_varlen_attention_kernel(
    q_ptr,                 # [total_q, H, D]
    k_cache_ptr,           # [num_blocks, block_size, Hkv, D]
    v_cache_ptr,
    block_tables_ptr,      # [B, max_blocks]
    context_lens_ptr,      # [B]
    q_lens_ptr,            # [B]
    q_offsets_ptr,         # [B] offsets into total_q
    out_ptr,               # [total_q, H, D]
    q_stride_t: tl.constexpr,
    q_stride_h: tl.constexpr,
    q_stride_d: tl.constexpr,
    cache_stride_block: tl.constexpr,
    cache_stride_token: tl.constexpr,
    cache_stride_head: tl.constexpr,
    cache_stride_d: tl.constexpr,
    block_table_stride_b: tl.constexpr,
    block_table_stride_blk: tl.constexpr,
    out_stride_t: tl.constexpr,
    out_stride_h: tl.constexpr,
    out_stride_d: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    MAX_CONTEXT_LEN: tl.constexpr,
    NUM_KV_GROUPS: tl.constexpr,
    SCALE: tl.constexpr,
):
    q_block_id = tl.program_id(0)
    seq_id = tl.program_id(1)
    head_id = tl.program_id(2)
    kv_head = head_id // NUM_KV_GROUPS

    q_len = tl.load(q_lens_ptr + seq_id)
    ctx_len = tl.load(context_lens_ptr + seq_id)
    q_start = tl.load(q_offsets_ptr + seq_id)
    prefix_len = ctx_len - q_len

    offs_m = q_block_id * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    q_mask = offs_m < q_len
    d_mask = offs_d < HEAD_DIM
    q_abs_pos = prefix_len + offs_m

    q = tl.load(
        q_ptr
        + (q_start + offs_m[:, None]) * q_stride_t
        + head_id * q_stride_h
        + offs_d[None, :] * q_stride_d,
        mask=q_mask[:, None] & d_mask[None, :],
        other=0.0,
    )

    m_i = tl.full((BLOCK_M,), -3.4028234663852886e38, tl.float32)
    l_i = tl.full((BLOCK_M,), 0.0, tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)

    offs_n = tl.arange(0, BLOCK_N)
    for start in tl.range(0, MAX_CONTEXT_LEN, BLOCK_N):
        pos = start + offs_n
        valid_n = pos < ctx_len
        logical_blk = pos // BLOCK_SIZE
        blk_off = pos - logical_blk * BLOCK_SIZE
        phys_blk = tl.load(
            block_tables_ptr
            + seq_id * block_table_stride_b
            + logical_blk * block_table_stride_blk,
            mask=valid_n,
            other=0,
        )

        kv_base = (
            phys_blk[:, None] * cache_stride_block
            + blk_off[:, None] * cache_stride_token
            + kv_head * cache_stride_head
            + offs_d[None, :] * cache_stride_d
        )
        k = tl.load(k_cache_ptr + kv_base, mask=valid_n[:, None] & d_mask[None, :], other=0.0)
        v = tl.load(v_cache_ptr + kv_base, mask=valid_n[:, None] & d_mask[None, :], other=0.0)

        scores = tl.dot(q, tl.trans(k), input_precision="ieee") * SCALE
        causal = pos[None, :] <= q_abs_pos[:, None]
        scores = tl.where(q_mask[:, None] & valid_n[None, :] & causal, scores, -3.4028234663852886e38)

        m_new = tl.maximum(m_i, tl.max(scores, axis=1))
        p = tl.exp(scores - m_new[:, None])
        alpha = tl.exp(m_i - m_new)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v, input_precision="ieee").to(tl.float32)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_new

    out = acc / l_i[:, None]
    tl.store(
        out_ptr
        + (q_start + offs_m[:, None]) * out_stride_t
        + head_id * out_stride_h
        + offs_d[None, :] * out_stride_d,
        out,
        mask=q_mask[:, None] & d_mask[None, :],
    )


def paged_varlen_attention(
    q: torch.Tensor,              # [total_q, H, D]
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    q_lens: torch.Tensor,
    q_offsets: torch.Tensor,
    scale: float,
    num_kv_groups: int,
    block_m: Optional[int] = None,
    block_n: Optional[int] = None,
) -> torch.Tensor:
    """Varlen paged attention for q_len>1 prefill/suffix-prefill."""
    if not q.is_cuda:
        raise RuntimeError("paged_varlen_attention currently requires CUDA")
    total_q, H, D = q.shape
    if total_q == 0:
        return q.new_empty((0, H, D))
    q = q.contiguous()
    block_tables = block_tables.contiguous()
    context_lens = context_lens.contiguous()
    q_lens = q_lens.contiguous()
    q_offsets = q_offsets.contiguous()
    out = torch.empty_like(q)
    block_d = triton.next_power_of_2(D)
    if block_m is None or block_n is None:
        default_m, default_n = _varlen_blocks(D)
        block_m = default_m if block_m is None else block_m
        block_n = default_n if block_n is None else block_n
    max_q_len = int(q_lens.max().item())
    max_context_len = int(context_lens.max().item())
    if max_q_len <= 0 or max_context_len <= 0:
        return out.zero_()
    grid = (triton.cdiv(max_q_len, block_m), q_lens.numel(), H)
    _paged_varlen_attention_kernel[grid](
        q,
        k_cache,
        v_cache,
        block_tables,
        context_lens,
        q_lens,
        q_offsets,
        out,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        k_cache.stride(3),
        block_tables.stride(0),
        block_tables.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        BLOCK_SIZE=k_cache.shape[1],
        HEAD_DIM=D,
        BLOCK_D=block_d,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        MAX_CONTEXT_LEN=triton.cdiv(max_context_len, block_n) * block_n,
        NUM_KV_GROUPS=num_kv_groups,
        SCALE=float(scale),
    )
    return out


# ──────────────────────────────────────────────────────────────────────────────
# PagedLlamaAttention
# ──────────────────────────────────────────────────────────────────────────────

class PagedLlamaAttention(nn.Module):
    """
    Paged attention: unified varlen forward — QKV projected over all tokens at once,
    KV written to physical pool via store_kvcache, then per-sequence SDPA gathering
    full K/V history from the pool. is_causal=(q_len > 1) handles prefill vs decode.
    """

    @classmethod
    def from_hf(
        cls,
        hf_attn,
        k_cache: torch.Tensor,   # [num_blocks, block_size, H, D]
        v_cache: torch.Tensor,
    ) -> "PagedLlamaAttention":
        obj = cls.__new__(cls)
        nn.Module.__init__(obj)
        obj.config         = hf_attn.config
        obj.layer_idx      = hf_attn.layer_idx
        obj.head_dim       = hf_attn.head_dim
        obj.num_heads      = hf_attn.config.num_attention_heads
        obj.num_kv_heads   = hf_attn.config.num_key_value_heads
        obj.num_kv_groups  = obj.num_heads // obj.num_kv_heads
        obj.scaling        = hf_attn.scaling
        obj.rotary_fn      = hf_attn.rotary_fn
        obj.q_proj  = hf_attn.q_proj
        obj.k_proj  = hf_attn.k_proj
        obj.v_proj  = hf_attn.v_proj
        obj.o_proj  = hf_attn.o_proj
        obj.packed_qkv_weight = torch.cat(
            [obj.q_proj.weight, obj.k_proj.weight, obj.v_proj.weight],
            dim=0,
        ).contiguous()
        q_bias = getattr(obj.q_proj, "bias", None)
        k_bias = getattr(obj.k_proj, "bias", None)
        v_bias = getattr(obj.v_proj, "bias", None)
        if q_bias is None and k_bias is None and v_bias is None:
            obj.packed_qkv_bias = None
        else:
            parts = []
            for bias, out_features in (
                (q_bias, obj.q_proj.out_features),
                (k_bias, obj.k_proj.out_features),
                (v_bias, obj.v_proj.out_features),
            ):
                if bias is None:
                    parts.append(obj.packed_qkv_weight.new_zeros(out_features))
                else:
                    parts.append(bias)
            obj.packed_qkv_bias = torch.cat(parts, dim=0).contiguous()
        obj.use_decode_input_staging = (
            os.environ.get("NANOPOINTLLM_DECODE_QKV_INPUT_STAGING", "0") == "1"
        )
        obj.use_packed_qkv_decode = False
        obj.k_cache = k_cache
        obj.v_cache = v_cache
        return obj

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values=None,
        precomputed_qkv: Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> tuple:
        ctx = get_forward_context()
        _, S, _ = hidden_states.shape   # shape is [1, total_tokens, hidden]
        H, Hkv, D = self.num_heads, self.num_kv_heads, self.head_dim

        if precomputed_qkv is not None:
            with nvtx_stage("pointllm_attention", hidden_states):
                q_decode, k_store, v_store = precomputed_qkv
                q_rot = q_decode.permute(1, 0, 2).unsqueeze(0)
                k_rot = k_store.permute(1, 0, 2).unsqueeze(0)
                cos, sin = position_embeddings
                q_rot, k_rot = self.rotary_fn(q_rot, k_rot, cos, sin)
                q_decode = q_rot[0].permute(1, 0, 2).contiguous()
                k_store = k_rot[0].permute(1, 0, 2).contiguous()
                v_store = v_store.contiguous()
                store_kvcache(k_store, v_store, self.k_cache, self.v_cache, ctx.slot_mapping)
                out_heads = paged_decode_attention(
                    q_decode,
                    self.k_cache,
                    self.v_cache,
                    ctx.block_tables,
                    ctx.context_lens,
                    scale=float(self.scaling),
                    num_kv_groups=self.num_kv_groups,
                )
                out = out_heads.reshape(S, H * D).unsqueeze(0)
            with nvtx_stage("pointllm_o_proj", out):
                projected = _project_output_decode_staged(out, self.o_proj)
            return projected, None

        if (
            self.use_decode_input_staging
            and hidden_states.is_cuda
            and ctx.seq_lens
            and all(q_len == 1 for q_len in ctx.seq_lens)
        ):
            with nvtx_stage("pointllm_qkv", hidden_states):
                q_decode, k_store, v_store = _project_qkv_decode_staged(
                    hidden_states,
                    self.q_proj,
                    self.k_proj,
                    self.v_proj,
                    num_heads=H,
                    num_kv_heads=Hkv,
                    head_dim=D,
                )
            with nvtx_stage("pointllm_attention", hidden_states):
                q_rot = q_decode.permute(1, 0, 2).unsqueeze(0)
                k_rot = k_store.permute(1, 0, 2).unsqueeze(0)
                cos, sin = position_embeddings
                q_rot, k_rot = self.rotary_fn(q_rot, k_rot, cos, sin)
                q_decode = q_rot[0].permute(1, 0, 2).contiguous()
                k_store = k_rot[0].permute(1, 0, 2).contiguous()
                v_store = v_store.contiguous()
                store_kvcache(k_store, v_store, self.k_cache, self.v_cache, ctx.slot_mapping)
                out_heads = paged_decode_attention(
                    q_decode,
                    self.k_cache,
                    self.v_cache,
                    ctx.block_tables,
                    ctx.context_lens,
                    scale=float(self.scaling),
                    num_kv_groups=self.num_kv_groups,
                )
                out = out_heads.reshape(S, H * D).unsqueeze(0)
            with nvtx_stage("pointllm_o_proj", out):
                projected = self.o_proj(out)
            return projected, None

        # QKV projections — single pass over total_tokens
        with nvtx_stage("pointllm_qkv", hidden_states):
            q = self.q_proj(hidden_states).view(1, S, H,   D).transpose(1, 2)  # [1, H, S, D]
            k = self.k_proj(hidden_states).view(1, S, Hkv, D).transpose(1, 2)
            v = self.v_proj(hidden_states).view(1, S, Hkv, D).transpose(1, 2)

        cos, sin = position_embeddings
        q, k = self.rotary_fn(q, k, cos, sin)

        # Write all new K/V to physical pool (slot=-1 positions are skipped)
        k_store = k.transpose(1, 2).reshape(S, Hkv, D).contiguous()
        v_store = v.transpose(1, 2).reshape(S, Hkv, D).contiguous()
        store_kvcache(k_store, v_store, self.k_cache, self.v_cache, ctx.slot_mapping)

        if ctx.seq_lens and all(q_len == 1 for q_len in ctx.seq_lens):
            q_decode = q[0].permute(1, 0, 2).contiguous()
            with nvtx_stage("pointllm_attention", hidden_states):
                out_heads = paged_decode_attention(
                    q_decode,
                    self.k_cache,
                    self.v_cache,
                    ctx.block_tables,
                    ctx.context_lens,
                    scale=float(self.scaling),
                    num_kv_groups=self.num_kv_groups,
                )
                out = out_heads.reshape(S, H * D).unsqueeze(0)
            with nvtx_stage("pointllm_o_proj", out):
                projected = self.o_proj(out)
            return projected, None

        q_offsets = []
        running_offset = 0
        for q_len in ctx.seq_lens:
            q_offsets.append(running_offset)
            running_offset += q_len

        decode_seq_indices = [i for i, q_len in enumerate(ctx.seq_lens) if q_len == 1]
        decode_out_heads = None
        if decode_seq_indices:
            decode_q_offsets = torch.tensor(
                [q_offsets[i] for i in decode_seq_indices],
                dtype=torch.long,
                device=q.device,
            )
            q_decode = q[0].permute(1, 0, 2).index_select(0, decode_q_offsets).contiguous()
            decode_rows = torch.tensor(decode_seq_indices, dtype=torch.long, device=q.device)
            out_heads = paged_decode_attention(
                q_decode,
                self.k_cache,
                self.v_cache,
                ctx.block_tables.index_select(0, decode_rows),
                ctx.context_lens.index_select(0, decode_rows),
                scale=float(self.scaling),
                num_kv_groups=self.num_kv_groups,
            )
            decode_out_heads = {
                seq_idx: out_heads[pos]
                for pos, seq_idx in enumerate(decode_seq_indices)
            }

        prefill_out_heads = None
        prefill_seq_indices = [i for i, q_len in enumerate(ctx.seq_lens) if q_len > 1]
        if prefill_seq_indices and q.is_cuda:
            q_flat = q[0].permute(1, 0, 2).contiguous()
            rows = torch.tensor(prefill_seq_indices, dtype=torch.long, device=q.device)
            q_lens = torch.tensor(
                [ctx.seq_lens[i] for i in prefill_seq_indices],
                dtype=torch.int32,
                device=q.device,
            )
            q_start_offsets = torch.tensor(
                [q_offsets[i] for i in prefill_seq_indices],
                dtype=torch.int32,
                device=q.device,
            )
            prefill_all = paged_varlen_attention(
                q_flat,
                self.k_cache,
                self.v_cache,
                ctx.block_tables.index_select(0, rows),
                ctx.context_lens.index_select(0, rows),
                q_lens,
                q_start_offsets,
                scale=float(self.scaling),
                num_kv_groups=self.num_kv_groups,
            )
            prefill_out_heads = {
                seq_idx: prefill_all[
                    q_offsets[seq_idx]: q_offsets[seq_idx] + ctx.seq_lens[seq_idx]
                ]
                for seq_idx in prefill_seq_indices
            }

        # Per-sequence SDPA reading full K/V history from pool
        block_size = self.k_cache.shape[1]
        seq_outs, q_offset = [], 0
        if not ctx.seq_lens:
            return self.o_proj(hidden_states.new_zeros(1, 0, H * D)), None
        for i, q_len in enumerate(ctx.seq_lens):
            if q_len == 1 and decode_out_heads is not None:
                seq_outs.append(decode_out_heads[i].reshape(1, H * D))
                q_offset += 1
                continue
            if q_len > 1 and prefill_out_heads is not None:
                seq_outs.append(prefill_out_heads[i].reshape(q_len, H * D))
                q_offset += q_len
                continue
            ctx_len  = int(ctx.context_lens[i])
            num_blks = (ctx_len + block_size - 1) // block_size
            bt       = ctx.block_tables[i, :num_blks]                          # valid blocks only
            k_full   = self.k_cache[bt].reshape(-1, Hkv, D)[:ctx_len]         # [ctx_len, Hkv, D]
            v_full   = self.v_cache[bt].reshape(-1, Hkv, D)[:ctx_len]

            qi = q[0, :, q_offset:q_offset+q_len, :].unsqueeze(0)             # [1, H, q_len, D]
            ki = k_full.permute(1, 0, 2).unsqueeze(0).repeat_interleave(self.num_kv_groups, 1)
            vi = v_full.permute(1, 0, 2).unsqueeze(0).repeat_interleave(self.num_kv_groups, 1)

            attn_mask = None
            is_causal = False
            if q_len > 1 and ctx_len == q_len:
                is_causal = True
            elif q_len > 1:
                attn_mask = _partial_prefill_mask(
                    q_len,
                    ctx_len,
                    device=qi.device,
                    dtype=qi.dtype,
                )

            oi = F.scaled_dot_product_attention(
                qi, ki, vi,
                attn_mask=attn_mask,
                is_causal=is_causal,
                scale=self.scaling,
            )  # [1, H, q_len, D]
            seq_outs.append(oi.squeeze(0).permute(1, 0, 2).reshape(q_len, H * D))
            q_offset += q_len

        out = torch.cat(seq_outs).unsqueeze(0)   # [1, total_tokens, H*D]
        return self.o_proj(out), None

    def project_qkv_decode_packed(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return _project_qkv_decode_packed(
            hidden_states,
            self.packed_qkv_weight,
            self.packed_qkv_bias,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
        )

    def project_qkv_decode_fused(
        self,
        hidden_states: torch.Tensor,
        norm_weight: torch.Tensor,
        norm_eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return _rmsnorm_packed_qkv_decode_triton(
            hidden_states,
            norm_weight,
            self.packed_qkv_weight,
            self.packed_qkv_bias,
            eps=norm_eps,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
        )

    def project_qkv_decode_rmsnorm_staged(
        self,
        hidden_states: torch.Tensor,
        norm_weight: torch.Tensor,
        norm_eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return _rmsnorm_staged_packed_qkv_decode(
            hidden_states,
            norm_weight,
            self.packed_qkv_weight,
            self.packed_qkv_bias,
            eps=norm_eps,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
        )
