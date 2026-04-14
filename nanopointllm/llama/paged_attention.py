"""
PagedLlamaAttention + Triton store_kvcache kernel.

store_kvcache writes K/V tokens into the physical KV pool at given slots.
PagedLlamaAttention replaces HF LlamaAttention; reads routing from ForwardContext.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


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
