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

from nanopointllm.engine.forward_context import get_forward_context


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
# PagedLlamaAttention
# ──────────────────────────────────────────────────────────────────────────────

class PagedLlamaAttention(nn.Module):
    """
    Drop-in replacement for HF LlamaAttention.
    Reads routing info from thread-local ForwardContext.
    Prefill: dense causal sdpa (handles left-padded batches via attention_mask).
    Decode: gather K/V from block_tables, per-sequence sdpa.
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
        obj.k_cache = k_cache
        obj.v_cache = v_cache
        return obj

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values=None,
        **kwargs,
    ) -> tuple:
        ctx = get_forward_context()
        _, S, _ = hidden_states.shape   # shape is [1, total_tokens, hidden]
        H, Hkv, D = self.num_heads, self.num_kv_heads, self.head_dim

        # QKV projections — single pass over total_tokens
        q = self.q_proj(hidden_states).view(1, S, H,   D).transpose(1, 2)  # [1, H, S, D]
        k = self.k_proj(hidden_states).view(1, S, Hkv, D).transpose(1, 2)
        v = self.v_proj(hidden_states).view(1, S, Hkv, D).transpose(1, 2)

        cos, sin = position_embeddings
        q, k = self.rotary_fn(q, k, cos, sin)

        # Write all new K/V to physical pool (slot=-1 positions are skipped)
        k_store = k.transpose(1, 2).reshape(S, Hkv, D).contiguous()
        v_store = v.transpose(1, 2).reshape(S, Hkv, D).contiguous()
        store_kvcache(k_store, v_store, self.k_cache, self.v_cache, ctx.slot_mapping)

        # Per-sequence SDPA reading full K/V history from pool
        block_size = self.k_cache.shape[1]
        seq_outs, q_offset = [], 0
        for i, q_len in enumerate(ctx.seq_lens):
            ctx_len  = int(ctx.context_lens[i])
            num_blks = (ctx_len + block_size - 1) // block_size
            bt       = ctx.block_tables[i, :num_blks]                          # valid blocks only
            k_full   = self.k_cache[bt].reshape(-1, Hkv, D)[:ctx_len]         # [ctx_len, Hkv, D]
            v_full   = self.v_cache[bt].reshape(-1, Hkv, D)[:ctx_len]

            qi = q[0, :, q_offset:q_offset+q_len, :].unsqueeze(0)             # [1, H, q_len, D]
            ki = k_full.permute(1, 0, 2).unsqueeze(0).repeat_interleave(self.num_kv_groups, 1)
            vi = v_full.permute(1, 0, 2).unsqueeze(0).repeat_interleave(self.num_kv_groups, 1)

            oi = F.scaled_dot_product_attention(
                qi, ki, vi,
                is_causal=(q_len > 1),   # True for prefill, False for decode
                scale=self.scaling,
            )  # [1, H, q_len, D]
            seq_outs.append(oi.squeeze(0).permute(1, 0, 2).reshape(q_len, H * D))
            q_offset += q_len

        out = torch.cat(seq_outs).unsqueeze(0)   # [1, total_tokens, H*D]
        return self.o_proj(out), None
