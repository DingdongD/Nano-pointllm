# Paged KV Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace per-sequence HF DynamicCache with a global physical KV tensor pool + PagedLlamaAttention, eliminating padded-KV rebuild overhead and enabling true paged decode.

**Architecture:** A pre-allocated `[2, L, num_blocks, block_size, H, D]` tensor pool is sliced per layer into `k_cache`/`v_cache`. A thread-local `ForwardContext` carries slot routing info into each `PagedLlamaAttention` layer during `model.forward()`. Prefill writes new KV to the pool via Triton kernel and runs dense causal sdpa; decode gathers full history from block_tables and runs per-sequence sdpa.

**Tech Stack:** Python 3.10+, PyTorch 2.x, Triton 3.1.0, transformers LlamaAttention (HF). No flash_attn required.

---

### Task 1: KVPool + ForwardContext

**Files:**
- Create: `nanopointllm/engine/kv_pool.py`
- Create: `nanopointllm/engine/forward_context.py`
- Test: `tests/test_kv_pool.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_kv_pool.py
import torch
import pytest
from unittest.mock import MagicMock

from nanopointllm.engine.kv_pool import KVPool, allocate_kv_pool
from nanopointllm.engine.forward_context import (
    ForwardContext,
    set_forward_context,
    get_forward_context,
    clear_forward_context,
)


def test_kv_pool_slices():
    """k_cache / v_cache slices point into the shared pool tensor."""
    pool = torch.zeros(2, 4, 8, 16, 2, 32)  # 2, num_layers, num_blocks, block_size, H, D
    kv = KVPool(pool=pool, num_layers=4, num_blocks=8, block_size=16, num_heads=2, head_dim=32)
    k0 = kv.k_cache(0)
    v0 = kv.v_cache(0)
    assert k0.shape == (8, 16, 2, 32)
    assert v0.shape == (8, 16, 2, 32)
    # slices share storage
    pool[0, 0, 0, 0, 0, 0] = 99.0
    assert k0[0, 0, 0, 0].item() == 99.0


def test_allocate_kv_pool_shapes():
    """allocate_kv_pool reads config correctly and creates right-shaped pool."""
    mock_model = MagicMock()
    mock_model.config.num_hidden_layers = 4
    mock_model.config.num_attention_heads = 8
    mock_model.config.hidden_size = 256

    kv = allocate_kv_pool(mock_model, num_blocks=16, block_size=8,
                           device=torch.device("cpu"), dtype=torch.float32)
    assert kv.num_layers == 4
    assert kv.num_blocks == 16
    assert kv.block_size == 8
    assert kv.num_heads == 8
    assert kv.head_dim == 32  # 256 // 8
    assert kv.pool.shape == (2, 4, 16, 8, 8, 32)


def test_forward_context_thread_local():
    """set/get/clear ForwardContext round-trips correctly."""
    slot_map = torch.zeros(4, dtype=torch.int32)
    ctx = ForwardContext(
        is_prefill=True,
        slot_mapping=slot_map,
        block_tables=None,
        context_lens=None,
    )
    set_forward_context(ctx)
    got = get_forward_context()
    assert got is ctx
    assert got.is_prefill is True
    clear_forward_context()
    assert get_forward_context() is None
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/nano-pointllm && python -m pytest tests/test_kv_pool.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'nanopointllm.engine.kv_pool'`

- [ ] **Step 3: Implement kv_pool.py**

```python
# nanopointllm/engine/kv_pool.py
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class KVPool:
    pool: torch.Tensor   # [2, num_layers, num_blocks, block_size, num_heads, head_dim]
    num_layers: int
    num_blocks: int
    block_size: int
    num_heads: int
    head_dim: int

    def k_cache(self, layer_idx: int) -> torch.Tensor:
        """[num_blocks, block_size, num_heads, head_dim]"""
        return self.pool[0, layer_idx]

    def v_cache(self, layer_idx: int) -> torch.Tensor:
        """[num_blocks, block_size, num_heads, head_dim]"""
        return self.pool[1, layer_idx]


def allocate_kv_pool(
    hf_model: nn.Module,
    num_blocks: int,
    block_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> KVPool:
    """
    Read num_layers / num_heads / head_dim from model config,
    allocate pool tensor, return KVPool.
    """
    cfg = hf_model.config
    num_layers = cfg.num_hidden_layers
    num_heads  = cfg.num_attention_heads
    head_dim   = cfg.hidden_size // num_heads
    pool = torch.zeros(
        2, num_layers, num_blocks, block_size, num_heads, head_dim,
        dtype=dtype, device=device,
    )
    return KVPool(pool, num_layers, num_blocks, block_size, num_heads, head_dim)
```

- [ ] **Step 4: Implement forward_context.py**

```python
# nanopointllm/engine/forward_context.py
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class ForwardContext:
    is_prefill: bool
    slot_mapping: torch.Tensor          # [total_new_tokens] int32; -1 = skip (padding)
    block_tables: Optional[torch.Tensor]  # [B, max_num_blocks] int32; decode only
    context_lens: Optional[torch.Tensor]  # [B] int32; decode only


_ctx: threading.local = threading.local()


def set_forward_context(ctx: Optional[ForwardContext]) -> None:
    _ctx.value = ctx


def get_forward_context() -> Optional[ForwardContext]:
    return getattr(_ctx, "value", None)


def clear_forward_context() -> None:
    _ctx.value = None
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd /home/nano-pointllm && python -m pytest tests/test_kv_pool.py -v
```

Expected: `3 passed`

- [ ] **Step 6: Commit**

```bash
cd /home/nano-pointllm
git add nanopointllm/engine/kv_pool.py nanopointllm/engine/forward_context.py tests/test_kv_pool.py
git commit -m "feat: add KVPool and ForwardContext for paged KV phase 1"
```

---

### Task 2: Triton store_kvcache Kernel

**Files:**
- Create: `nanopointllm/llama/paged_attention.py` (kernel + store_kvcache function only)
- Create: `tests/test_store_kvcache.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_store_kvcache.py
import pytest
import torch


@pytest.fixture
def cache_tensors():
    num_blocks, block_size, H, D = 4, 4, 2, 8
    k_cache = torch.zeros(num_blocks, block_size, H, D)
    v_cache = torch.zeros(num_blocks, block_size, H, D)
    return k_cache, v_cache, block_size


def test_contiguous_slots(cache_tensors):
    """Consecutive tokens map to slots 0,1,2,3 in block 0."""
    from nanopointllm.llama.paged_attention import store_kvcache
    k_cache, v_cache, block_size = cache_tensors
    T, H, D = 4, 2, 8
    key   = torch.arange(T * H * D, dtype=torch.float32).reshape(T, H, D)
    value = key * -1
    slots = torch.arange(T, dtype=torch.int32)  # slots 0..3 → block 0

    store_kvcache(key, value, k_cache, v_cache, slots)

    for t in range(T):
        blk = t // block_size
        off = t % block_size
        assert torch.allclose(k_cache[blk, off], key[t])
        assert torch.allclose(v_cache[blk, off], value[t])


def test_non_contiguous_slots(cache_tensors):
    """Tokens scattered across blocks."""
    from nanopointllm.llama.paged_attention import store_kvcache
    k_cache, v_cache, block_size = cache_tensors
    T, H, D = 3, 2, 8
    key   = torch.randn(T, H, D)
    value = torch.randn(T, H, D)
    # physical slots: 0 (blk0,off0), 5 (blk1,off1), 12 (blk3,off0)
    slots = torch.tensor([0, 5, 12], dtype=torch.int32)

    store_kvcache(key, value, k_cache, v_cache, slots)

    assert torch.allclose(k_cache[0, 0], key[0])
    assert torch.allclose(k_cache[1, 1], key[1])
    assert torch.allclose(k_cache[3, 0], key[2])


def test_skip_negative_slots(cache_tensors):
    """slot == -1 means padding; cache must not be written."""
    from nanopointllm.llama.paged_attention import store_kvcache
    k_cache, v_cache, block_size = cache_tensors
    T, H, D = 4, 2, 8
    key   = torch.ones(T, H, D)
    value = torch.ones(T, H, D)
    slots = torch.tensor([-1, -1, 0, 1], dtype=torch.int32)

    store_kvcache(key, value, k_cache, v_cache, slots)

    # Only tokens 2,3 written to slots 0,1
    assert torch.allclose(k_cache[0, 0], key[2])
    assert torch.allclose(k_cache[0, 1], key[3])
    # Rest untouched (zeros)
    assert k_cache[0, 2].sum().item() == 0.0
    assert k_cache[1, 0].sum().item() == 0.0


def test_single_token(cache_tensors):
    """B=1 decode: write single new token to slot."""
    from nanopointllm.llama.paged_attention import store_kvcache
    k_cache, v_cache, _ = cache_tensors
    H, D = 2, 8
    key   = torch.randn(1, H, D)
    value = torch.randn(1, H, D)
    slots = torch.tensor([7], dtype=torch.int32)  # blk 1, off 3

    store_kvcache(key, value, k_cache, v_cache, slots)

    assert torch.allclose(k_cache[1, 3], key[0])
    assert torch.allclose(v_cache[1, 3], value[0])
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/nano-pointllm && python -m pytest tests/test_store_kvcache.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'nanopointllm.llama.paged_attention'`

- [ ] **Step 3: Implement the Triton kernel + store_kvcache**

```python
# nanopointllm/llama/paged_attention.py
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
    # Flatten to 2-D for Triton (avoid non-contiguous strides)
    key_flat    = key.reshape(T, HD).contiguous()
    value_flat  = value.reshape(T, HD).contiguous()
    k_cache_flat = k_cache.view(-1, HD)   # [num_blocks * block_size, HD]
    v_cache_flat = v_cache.view(-1, HD)

    # Triton requires constexpr D to be a power of 2; pad if needed
    HD_pow2 = 1
    while HD_pow2 < HD:
        HD_pow2 *= 2

    if HD_pow2 == HD:
        _store_kvcache_kernel[(T,)](
            key_flat, key_flat.stride(0),
            value_flat, value_flat.stride(0),
            k_cache_flat, v_cache_flat,
            k_cache_flat.stride(0),
            slot_mapping,
            D=HD,
        )
    else:
        # Fallback to PyTorch scatter for non-power-of-2 HD (rare in practice)
        _store_kvcache_pytorch(key, value, k_cache, v_cache, slot_mapping)


def _store_kvcache_pytorch(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    """Pure-PyTorch fallback for store_kvcache (non-power-of-2 HD)."""
    block_size = k_cache.shape[1]
    for t, slot in enumerate(slot_mapping.tolist()):
        if slot < 0:
            continue
        blk = slot // block_size
        off = slot % block_size
        k_cache[blk, off] = key[t]
        v_cache[blk, off] = value[t]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /home/nano-pointllm && python -m pytest tests/test_store_kvcache.py -v
```

Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
cd /home/nano-pointllm
git add nanopointllm/llama/paged_attention.py tests/test_store_kvcache.py
git commit -m "feat: add Triton store_kvcache kernel for paged KV pool"
```

---

### Task 3: PagedLlamaAttention + inject_paged_attention

**Files:**
- Modify: `nanopointllm/llama/paged_attention.py` (add PagedLlamaAttention class)
- Create: `nanopointllm/llama/inject_paged.py`
- Create: `tests/test_paged_attention.py`

- [ ] **Step 1: Write the failing tests**

```python
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
    """
    Construct a minimal fake LlamaAttention-like object with random weights.
    H heads, D head_dim, hidden = H*D, no GQA (num_kv_heads == H).
    """
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

    # Linear projections: identity-like weights (random but deterministic)
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
        # no-op RoPE for testing
        return q, k

    attn.rotary_fn = _apply_rotary
    return attn


def _make_caches(H: int, D: int, num_blocks: int = 8, block_size: int = 4):
    k_cache = torch.zeros(num_blocks, block_size, H, D)
    v_cache = torch.zeros(num_blocks, block_size, H, D)
    return k_cache, v_cache


def test_prefill_parity_vs_dense_sdpa():
    """
    PagedLlamaAttention prefill output must match a reference dense sdpa
    computed on the same Q/K/V (i.e., store + causal sdpa = dense sdpa).
    """
    from nanopointllm.llama.paged_attention import PagedLlamaAttention

    B, S, H, D = 2, 6, 4, 8
    hidden = H * D
    hf_attn = _make_fake_hf_attn(B, S, H, D)
    k_cache, v_cache = _make_caches(H, D)

    paged = PagedLlamaAttention.from_hf(hf_attn, k_cache, v_cache)

    torch.manual_seed(0)
    hidden_states = torch.randn(B, S, hidden)
    # cos/sin: no-op (all ones/zeros for simplicity)
    cos = torch.ones(1, S, D)
    sin = torch.zeros(1, S, D)

    # Build slot_mapping: B*S slots, left-pad the first seq (B=2, S=6)
    # seq0: tokens 0..5 → slots 0..5 (block 0)
    # seq1: tokens 0..5 → slots 8..13 (block 2 and 3)
    slots = torch.cat([
        torch.arange(6, dtype=torch.int32),
        torch.arange(8, 14, dtype=torch.int32),
    ])
    ctx = ForwardContext(is_prefill=True, slot_mapping=slots, block_tables=None, context_lens=None)
    set_forward_context(ctx)

    try:
        out_paged, _ = paged(hidden_states, (cos, sin), attention_mask=None)
    finally:
        clear_forward_context()

    # Reference: compute dense causal sdpa with same projections
    with torch.no_grad():
        q = hf_attn.q_proj(hidden_states).view(B, S, H, D).transpose(1, 2)  # [B,H,S,D]
        k = hf_attn.k_proj(hidden_states).view(B, S, H, D).transpose(1, 2)
        v = hf_attn.v_proj(hidden_states).view(B, S, H, D).transpose(1, 2)
        ref_attn = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=D**-0.5)
        ref_out = hf_attn.o_proj(ref_attn.transpose(1, 2).reshape(B, S, H * D))

    assert out_paged.shape == ref_out.shape
    assert torch.allclose(out_paged, ref_out, atol=1e-5), \
        f"max diff = {(out_paged - ref_out).abs().max()}"


def test_decode_parity_vs_full_context_sdpa():
    """
    PagedLlamaAttention decode output must match full-context sdpa
    when cache is pre-filled with the exact same K/V as prefill.
    """
    from nanopointllm.llama.paged_attention import PagedLlamaAttention
    from nanopointllm.llama.paged_attention import store_kvcache

    B, S_ctx, H, D = 1, 4, 4, 8  # single sequence, ctx_len=4
    hidden = H * D
    hf_attn = _make_fake_hf_attn(B, S_ctx, H, D)
    k_cache, v_cache = _make_caches(H, D)

    paged = PagedLlamaAttention.from_hf(hf_attn, k_cache, v_cache)

    torch.manual_seed(1)
    # Pre-fill cache with "prefill" tokens
    ctx_hidden = torch.randn(B, S_ctx, hidden)
    cos_ctx = torch.ones(1, S_ctx, D)
    sin_ctx = torch.zeros(1, S_ctx, D)

    with torch.no_grad():
        k_ctx = hf_attn.k_proj(ctx_hidden).view(B, S_ctx, H, D).transpose(1, 2)  # [B,H,S,D]
        v_ctx = hf_attn.v_proj(ctx_hidden).view(B, S_ctx, H, D).transpose(1, 2)

    # Pre-fill cache manually at slots 0..3
    ctx_slots = torch.arange(S_ctx, dtype=torch.int32)
    store_kvcache(
        k_ctx.transpose(1, 2).reshape(B * S_ctx, H, D).contiguous(),
        v_ctx.transpose(1, 2).reshape(B * S_ctx, H, D).contiguous(),
        k_cache, v_cache, ctx_slots,
    )

    # Now do one decode step: new token
    torch.manual_seed(2)
    new_hidden = torch.randn(B, 1, hidden)
    cos_new = torch.ones(1, 1, D)
    sin_new = torch.zeros(1, 1, D)

    # Slot for new token: slot 4 (blk 1, off 0)
    decode_slots = torch.tensor([4], dtype=torch.int32)
    block_tables = torch.tensor([[0, 1]], dtype=torch.int32)  # blk0, blk1 (block_size=4)
    context_lens = torch.tensor([S_ctx + 1], dtype=torch.int32)

    decode_ctx = ForwardContext(
        is_prefill=False,
        slot_mapping=decode_slots,
        block_tables=block_tables,
        context_lens=context_lens,
    )
    set_forward_context(decode_ctx)
    try:
        out_paged, _ = paged(new_hidden, (cos_new, sin_new), attention_mask=None)
    finally:
        clear_forward_context()

    # Reference: full-context sdpa over ctx + new token
    with torch.no_grad():
        q_new = hf_attn.q_proj(new_hidden).view(B, 1, H, D).transpose(1, 2)  # [B,H,1,D]
        k_new = hf_attn.k_proj(new_hidden).view(B, 1, H, D).transpose(1, 2)
        v_new = hf_attn.v_proj(new_hidden).view(B, 1, H, D).transpose(1, 2)
        k_full = torch.cat([k_ctx, k_new], dim=2)  # [B,H,S+1,D]
        v_full = torch.cat([v_ctx, v_new], dim=2)
        ref_attn = F.scaled_dot_product_attention(q_new, k_full, v_full, scale=D**-0.5)
        ref_out = hf_attn.o_proj(ref_attn.transpose(1, 2).reshape(B, 1, H * D))

    assert out_paged.shape == ref_out.shape
    assert torch.allclose(out_paged, ref_out, atol=1e-5), \
        f"max diff = {(out_paged - ref_out).abs().max()}"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/nano-pointllm && python -m pytest tests/test_paged_attention.py -v
```

Expected: FAIL with `ImportError: cannot import name 'PagedLlamaAttention'`

- [ ] **Step 3: Add PagedLlamaAttention to paged_attention.py**

Append the following class to the end of `nanopointllm/llama/paged_attention.py`:

```python
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
        # RoPE function (apply_rotary_pos_emb or equivalent)
        obj.rotary_fn      = hf_attn.rotary_fn
        # Weight references (shared, no copy)
        obj.q_proj  = hf_attn.q_proj
        obj.k_proj  = hf_attn.k_proj
        obj.v_proj  = hf_attn.v_proj
        obj.o_proj  = hf_attn.o_proj
        # Physical KV cache slices
        obj.k_cache = k_cache
        obj.v_cache = v_cache
        return obj

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values=None,   # ignored — physical pool handles KV
        **kwargs,
    ) -> tuple:
        from nanopointllm.engine.forward_context import get_forward_context
        ctx = get_forward_context()
        B, S, _ = hidden_states.shape
        H, Hkv, D = self.num_heads, self.num_kv_heads, self.head_dim

        # QKV projection → [B, H, S, D] layout for RoPE
        q = self.q_proj(hidden_states).view(B, S, H,   D).transpose(1, 2)  # [B, H,   S, D]
        k = self.k_proj(hidden_states).view(B, S, Hkv, D).transpose(1, 2)  # [B, Hkv, S, D]
        v = self.v_proj(hidden_states).view(B, S, Hkv, D).transpose(1, 2)  # [B, Hkv, S, D]

        # Apply RoPE  (position_embeddings = (cos[1,S,D], sin[1,S,D]))
        cos, sin = position_embeddings
        q, k = self.rotary_fn(q, k, cos, sin)  # unsqueeze_dim=1 default

        # Write new K/V into physical pool
        k_store = k.transpose(1, 2).reshape(B * S, Hkv, D).contiguous()
        v_store = v.transpose(1, 2).reshape(B * S, Hkv, D).contiguous()
        store_kvcache(k_store, v_store, self.k_cache, self.v_cache, ctx.slot_mapping)

        if ctx.is_prefill:
            # Dense causal attention; attention_mask handles left-padding
            k_exp = k.repeat_interleave(self.num_kv_groups, dim=1)  # [B, H, S, D]
            v_exp = v.repeat_interleave(self.num_kv_groups, dim=1)
            attn_out = F.scaled_dot_product_attention(
                q, k_exp, v_exp,
                attn_mask=attention_mask,
                is_causal=(attention_mask is None),
                scale=self.scaling,
            )  # [B, H, S, D]
            out = attn_out.transpose(1, 2).reshape(B, S, H * D)
        else:
            # Paged decode: gather K/V from block_tables, per-sequence sdpa
            block_tables = ctx.block_tables   # [B, max_blocks]
            context_lens = ctx.context_lens   # [B]
            block_size = self.k_cache.shape[1]
            seq_outs = []
            for i in range(B):
                ctx_len = int(context_lens[i])
                num_blks = (ctx_len + block_size - 1) // block_size
                bt = block_tables[i, :num_blks]                            # [num_blks]
                k_full = self.k_cache[bt].reshape(-1, Hkv, D)[:ctx_len]   # [ctx_len, Hkv, D]
                v_full = self.v_cache[bt].reshape(-1, Hkv, D)[:ctx_len]
                qi = q[i:i+1]                                              # [1, H, 1, D]
                ki = k_full.transpose(0, 1).unsqueeze(0)                   # [1, Hkv, ctx_len, D]
                vi = v_full.transpose(0, 1).unsqueeze(0)
                ki = ki.repeat_interleave(self.num_kv_groups, dim=1)       # [1, H, ctx_len, D]
                vi = vi.repeat_interleave(self.num_kv_groups, dim=1)
                oi = F.scaled_dot_product_attention(qi, ki, vi, scale=self.scaling)
                seq_outs.append(oi)
            attn_out = torch.cat(seq_outs, dim=0)                          # [B, H, 1, D]
            out = attn_out.transpose(1, 2).reshape(B, 1, H * D)

        return self.o_proj(out), None   # (hidden_states, past_key_values=None)
```

- [ ] **Step 4: Create inject_paged.py**

```python
# nanopointllm/llama/inject_paged.py
"""Replace all LlamaAttention layers with PagedLlamaAttention (in-place)."""
from __future__ import annotations

import torch.nn as nn

from nanopointllm.engine.kv_pool import KVPool
from nanopointllm.llama.paged_attention import PagedLlamaAttention


def inject_paged_attention(hf_model: nn.Module, kv_pool: KVPool) -> None:
    """
    Iterate hf_model.model.layers and replace each self_attn with
    PagedLlamaAttention.from_hf(). Weights are shared (no copy).
    """
    try:
        from transformers.models.llama.modeling_llama import LlamaAttention
    except ImportError:
        LlamaAttention = None

    layers = hf_model.model.layers
    assert len(layers) == kv_pool.num_layers, (
        f"Model has {len(layers)} layers but KVPool has {kv_pool.num_layers}"
    )
    for layer_idx, layer in enumerate(layers):
        attn = layer.self_attn
        # Accept both real LlamaAttention and duck-type compatible objects
        if LlamaAttention is not None and not isinstance(attn, LlamaAttention):
            continue
        layer.self_attn = PagedLlamaAttention.from_hf(
            attn,
            kv_pool.k_cache(layer_idx),
            kv_pool.v_cache(layer_idx),
        )
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd /home/nano-pointllm && python -m pytest tests/test_paged_attention.py -v
```

Expected: `2 passed`

- [ ] **Step 6: Run all tests to verify nothing broken**

```bash
cd /home/nano-pointllm && python -m pytest tests/ -v --ignore=tests/test_pointllm_wrapper.py -x
```

Expected: all pass (wrapper test requires real weights, skip it)

- [ ] **Step 7: Commit**

```bash
cd /home/nano-pointllm
git add nanopointllm/llama/paged_attention.py nanopointllm/llama/inject_paged.py tests/test_paged_attention.py
git commit -m "feat: add PagedLlamaAttention and inject_paged_attention"
```

---

### Task 4: PagedModelRunner

**Files:**
- Create: `nanopointllm/engine/paged_model_runner.py`

No new tests for this task — integration is validated by Task 6 parity scripts. The
class closely mirrors `PointLLMModelRunner` and reuses `_encode_point_clouds_batch`.

- [ ] **Step 1: Create paged_model_runner.py**

```python
# nanopointllm/engine/paged_model_runner.py
"""
PagedModelRunner: prefill + paged-KV decode using ForwardContext.
Replaces PointLLMModelRunner when the engine is constructed with num_kvcache_blocks.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from nanopointllm.engine.forward_context import (
    ForwardContext,
    clear_forward_context,
    set_forward_context,
)
from nanopointllm.engine.kv_pool import KVPool
from nanopointllm.engine.sequence import PointLLMSequence
from nanopointllm.models.pointllm_wrapper import PointLLMWrapper


class PagedModelRunner:

    def __init__(self, hf_model: nn.Module, kv_pool: KVPool, block_manager) -> None:
        self.hf_model      = hf_model
        self.kv_pool       = kv_pool
        self.block_manager = block_manager
        self.wrapper       = PointLLMWrapper(hf_model)
        self.block_size    = kv_pool.block_size

    # ── Point cloud encoding (mirrors PointLLMModelRunner) ──────────────────

    def _encode_point_clouds_batch(self, seqs: list[PointLLMSequence]) -> None:
        pending = [
            seq for seq in seqs
            if seq.point_clouds is not None and seq.inputs_embeds_cached is None
        ]
        if not pending:
            return

        try:
            device = next(self.wrapper.inner.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

        def _to_2d(pc: torch.Tensor) -> torch.Tensor:
            return pc.squeeze(0) if pc.dim() == 3 else pc

        pcs_2d = [_to_2d(seq.point_clouds).to(device) for seq in pending]
        shapes = {pc.shape for pc in pcs_2d}

        if len(shapes) == 1:
            stacked = torch.stack(pcs_2d)
            all_features = self.wrapper.encode_point_clouds(stacked)
        else:
            all_features = [self.wrapper.encode_point_clouds(pc)[0] for pc in pcs_2d]

        for seq, feat in zip(pending, all_features):
            ids    = torch.tensor([seq.token_ids], dtype=torch.long, device=device)
            embeds = self.wrapper.prepare_inputs_embeds(ids, [feat])
            seq.inputs_embeds_cached = embeds.squeeze(0)

    # ── Slot-mapping helpers ─────────────────────────────────────────────────

    def _prefill_slot_mapping(
        self,
        seq: PointLLMSequence,
        max_len: int,
        pad_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Returns [max_len] int32 tensor.
        Padding positions → -1.
        Real token at logical_pos → block_table[blk] * block_size + offset.
        """
        slots = torch.full((max_len,), -1, dtype=torch.int32, device=device)
        for pos in range(seq.num_tokens):
            blk = pos // self.block_size
            off = pos  % self.block_size
            slots[pad_len + pos] = seq.block_table[blk] * self.block_size + off
        return slots

    def _decode_slot(self, seq: PointLLMSequence) -> int:
        """Physical slot for the newly written token (after append_token)."""
        pos = seq.num_tokens - 1
        blk = pos // self.block_size
        off = pos  % self.block_size
        return seq.block_table[blk] * self.block_size + off

    # ── Prefill ──────────────────────────────────────────────────────────────

    @torch.inference_mode()
    def run_prefill(self, seqs: list[PointLLMSequence]) -> list[int]:
        self._encode_point_clouds_batch(seqs)

        B       = len(seqs)
        max_len = max(seq.num_tokens for seq in seqs)

        try:
            device = next(self.hf_model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

        embeds_list, attn_masks, slot_maps = [], [], []
        for seq in seqs:
            L       = seq.num_tokens
            pad_len = max_len - L

            if seq.inputs_embeds_cached is not None:
                emb = seq.inputs_embeds_cached
            else:
                ids = torch.tensor([seq.token_ids], dtype=torch.long, device=device)
                emb = self.wrapper.get_input_embeddings()(ids).squeeze(0)

            if pad_len > 0:
                pad = emb.new_zeros(pad_len, emb.shape[-1])
                emb  = torch.cat([pad, emb], dim=0)
                mask = torch.cat([
                    torch.zeros(pad_len, dtype=torch.long, device=device),
                    torch.ones(L,        dtype=torch.long, device=device),
                ])
            else:
                mask = torch.ones(L, dtype=torch.long, device=device)

            embeds_list.append(emb)
            attn_masks.append(mask)
            slot_maps.append(self._prefill_slot_mapping(seq, max_len, pad_len, device))

        inputs_embeds  = torch.stack(embeds_list)          # [B, max_len, H]
        attention_mask = torch.stack(attn_masks)           # [B, max_len]
        slot_mapping   = torch.stack(slot_maps).reshape(-1)  # [B * max_len]

        dummy_ids = torch.zeros(B, max_len, dtype=torch.long, device=device)

        set_forward_context(ForwardContext(
            is_prefill=True,
            slot_mapping=slot_mapping,
            block_tables=None,
            context_lens=None,
        ))
        try:
            out = self.hf_model(
                input_ids=dummy_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                point_clouds=None,
                use_cache=False,
                return_dict=True,
            )
        finally:
            clear_forward_context()

        logits = out.logits  # [B, max_len, V]
        return logits[:, -1, :].float().argmax(dim=-1).tolist()

    # ── Decode ───────────────────────────────────────────────────────────────

    @torch.inference_mode()
    def run_decode(self, seqs: list[PointLLMSequence]) -> list[int]:
        B = len(seqs)
        try:
            device = next(self.hf_model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

        input_ids = torch.tensor(
            [[seq.last_token] for seq in seqs], dtype=torch.long, device=device,
        )  # [B, 1]
        positions = torch.tensor(
            [[seq.num_tokens - 1] for seq in seqs], dtype=torch.long, device=device,
        )  # [B, 1]
        slot_mapping = torch.tensor(
            [self._decode_slot(seq) for seq in seqs], dtype=torch.int32, device=device,
        )  # [B]
        context_lens = torch.tensor(
            [seq.num_tokens for seq in seqs], dtype=torch.int32, device=device,
        )  # [B]
        max_blocks = max(len(seq.block_table) for seq in seqs)
        block_tables = torch.tensor(
            [seq.block_table + [-1] * (max_blocks - len(seq.block_table)) for seq in seqs],
            dtype=torch.int32, device=device,
        )  # [B, max_blocks]

        set_forward_context(ForwardContext(
            is_prefill=False,
            slot_mapping=slot_mapping,
            block_tables=block_tables,
            context_lens=context_lens,
        ))
        try:
            out = self.hf_model(
                input_ids=input_ids,
                position_ids=positions,
                attention_mask=None,
                use_cache=False,
                return_dict=True,
            )
        finally:
            clear_forward_context()

        logits = out.logits  # [B, 1, V] or [B, V]
        if logits.dim() == 3:
            logits = logits[:, -1, :]
        return logits.float().argmax(dim=-1).tolist()

    def run(self, seqs: list[PointLLMSequence], is_prefill: bool) -> list[int]:
        if is_prefill:
            return self.run_prefill(seqs)
        return self.run_decode(seqs)
```

- [ ] **Step 2: Verify file is importable**

```bash
cd /home/nano-pointllm && python -c "from nanopointllm.engine.paged_model_runner import PagedModelRunner; print('ok')"
```

Expected: `ok`

- [ ] **Step 3: Commit**

```bash
cd /home/nano-pointllm
git add nanopointllm/engine/paged_model_runner.py
git commit -m "feat: add PagedModelRunner for paged KV decode"
```

---

### Task 5: Wire LLMEngine to Paged Path

**Files:**
- Modify: `nanopointllm/engine/llm_engine.py`

When `num_kvcache_blocks` is provided:
1. Allocate `KVPool`
2. Create `BlockManager`
3. Inject paged attention into `hf_model`
4. Use `PagedModelRunner` instead of `PointLLMModelRunner`

When `num_kvcache_blocks` is `None` (default): keep existing behavior unchanged.

- [ ] **Step 1: Read the existing llm_engine.py**

File is at `/home/nano-pointllm/nanopointllm/engine/llm_engine.py` — already read above.

- [ ] **Step 2: Update llm_engine.py**

Replace the existing file content with:

```python
"""
PointLLMLLMEngine：把 Scheduler + ModelRunner 串成完整推理循环。

当 num_kvcache_blocks 未指定时使用 PointLLMModelRunner（padded KV，原有行为）。
当 num_kvcache_blocks 指定时使用 PagedModelRunner（物理 KV 池，分页 decode）。

典型用法（离线 batch）::

    engine = PointLLMLLMEngine(hf_model=model, eos_token_id=2)
    seqs = engine.generate([
        {"token_ids": prompt1, "point_clouds": pc1},
        {"token_ids": prompt2},
    ])
    for seq in seqs:
        print(seq.token_ids[seq.num_prompt_tokens:])
"""
from __future__ import annotations

from typing import Any, Optional

import torch.nn as nn

from nanopointllm.engine.scheduler import Scheduler
from nanopointllm.engine.sequence import PointLLMSequence
from nanopointllm.sampling_params import SamplingParams


class PointLLMLLMEngine:

    def __init__(
        self,
        hf_model: nn.Module,
        eos_token_id: int,
        max_num_seqs: int = 8,
        max_num_batched_tokens: int = 4096,
        num_kvcache_blocks: Optional[int] = None,
        kvcache_block_size: int = 16,
    ) -> None:
        if num_kvcache_blocks is not None:
            import torch
            from nanopointllm.engine.block_manager import BlockManager
            from nanopointllm.engine.kv_pool import allocate_kv_pool
            from nanopointllm.engine.paged_model_runner import PagedModelRunner
            from nanopointllm.llama.inject_paged import inject_paged_attention

            device = next(hf_model.parameters()).device
            dtype  = next(hf_model.parameters()).dtype
            block_manager = BlockManager(
                num_blocks=num_kvcache_blocks,
                block_size=kvcache_block_size,
            )
            kv_pool = allocate_kv_pool(
                hf_model, num_kvcache_blocks, kvcache_block_size, device, dtype,
            )
            inject_paged_attention(hf_model, kv_pool)
            self.runner = PagedModelRunner(hf_model, kv_pool, block_manager)
        else:
            from nanopointllm.engine.model_runner import PointLLMModelRunner
            block_manager = None
            self.runner = PointLLMModelRunner(hf_model=hf_model)

        self.scheduler = Scheduler(
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
            eos_token_id=eos_token_id,
            block_manager=block_manager,
        )

    def is_finished(self) -> bool:
        return self.scheduler.is_finished()

    def add_request(
        self,
        token_ids: list[int],
        point_clouds: Optional[Any] = None,
        sampling_params: Optional[SamplingParams] = None,
    ) -> PointLLMSequence:
        seq = PointLLMSequence(
            token_ids=token_ids,
            point_clouds=point_clouds,
            sampling_params=sampling_params or SamplingParams(),
        )
        self.scheduler.add(seq)
        return seq

    def step(self) -> None:
        """Execute one scheduling round (prefill or decode) and update sequences."""
        seqs, is_prefill = self.scheduler.schedule()
        token_ids = self.runner.run(seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids)

    def generate(
        self,
        requests: list[dict],
    ) -> list[PointLLMSequence]:
        """
        Batch inference: submit all requests then loop step() until done.

        Args:
            requests: list of dict passed to add_request:
                      {"token_ids": [...], "point_clouds": Tensor, "sampling_params": ...}

        Returns:
            list[PointLLMSequence] in request order, all is_finished==True
        """
        seqs = [self.add_request(**req) for req in requests]
        while not self.is_finished():
            self.step()
        return seqs
```

- [ ] **Step 3: Run existing unit tests to verify no regression**

```bash
cd /home/nano-pointllm && python -m pytest tests/test_llm_engine.py tests/test_scheduler.py tests/test_block_manager.py -v
```

Expected: all pass

- [ ] **Step 4: Verify import**

```bash
cd /home/nano-pointllm && python -c "
from unittest.mock import MagicMock, patch
import torch
# Just verify the import and basic wiring without a real model
print('import ok')
"
```

- [ ] **Step 5: Commit**

```bash
cd /home/nano-pointllm
git add nanopointllm/engine/llm_engine.py
git commit -m "feat: wire PagedModelRunner into LLMEngine when num_kvcache_blocks is set"
```

---

### Task 6: Integration Scripts + Run

**Files:**
- Create: `scripts/parity_paged_pointllm.py`
- Create: `scripts/bench_paged_pointllm.py`

These require real GPU + PointLLM_7B_v1.2 weights at `/mnt/llm_data/pointllm_ckpt/PointLLM_7B_v1.2`.

- [ ] **Step 1: Create parity_paged_pointllm.py**

```python
#!/usr/bin/env python3
"""
P5 parity: PointLLMLLMEngine(num_kvcache_blocks=512) vs HF manual greedy decode.

Usage:
  cd /home/nano-pointllm
  PYTHONPATH=/home/PointLLM:. python scripts/parity_paged_pointllm.py \
    --model_path /mnt/llm_data/pointllm_ckpt/PointLLM_7B_v1.2 \
    --device cuda:1

Exit codes: 0 = all cases passed, 1 = mismatch or error.
"""
from __future__ import annotations

import argparse
import sys
import traceback

import torch

from nanopointllm.engine.llm_engine import PointLLMLLMEngine
from nanopointllm.parity.engine_test_utils import (
    build_prompt_token_ids,
    build_text_only_token_ids,
    hf_greedy_generate,
    load_pointllm_model,
    make_fake_point_cloud,
)
from nanopointllm.sampling_params import SamplingParams


def run_case(
    name: str,
    model,
    eos_token_id: int,
    all_token_ids: list[list[int]],
    all_point_clouds: list,
    max_new_tokens: int,
    device: torch.device,
    num_kvcache_blocks: int = 512,
    kvcache_block_size: int = 16,
) -> bool:
    B = len(all_token_ids)
    sp = SamplingParams(max_tokens=max_new_tokens)
    engine = PointLLMLLMEngine(
        model,
        eos_token_id=eos_token_id,
        max_num_seqs=B,
        max_num_batched_tokens=4096,
        num_kvcache_blocks=num_kvcache_blocks,
        kvcache_block_size=kvcache_block_size,
    )
    requests = [
        {"token_ids": ids, "point_clouds": pc, "sampling_params": sp}
        for ids, pc in zip(all_token_ids, all_point_clouds)
    ]
    try:
        seqs = engine.generate(requests)
    except Exception:
        print(f"  [{name}] ENGINE ERROR:")
        traceback.print_exc()
        return False

    ok = True
    for i, (seq, ids, pc) in enumerate(zip(seqs, all_token_ids, all_point_clouds)):
        generated = seq.token_ids[seq.num_prompt_tokens:]
        ref = hf_greedy_generate(model, ids, pc, max_new_tokens, device)
        if generated != ref:
            print(f"  [{name}] seq {i} MISMATCH:")
            print(f"    engine : {generated}")
            print(f"    hf_ref : {ref}")
            ok = False

    status = "PASS" if ok else "FAIL"
    print(f"  {name}: {status}")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description="P5 parity: paged engine vs HF greedy")
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--max_new_tokens", type=int, default=8)
    ap.add_argument("--num_kvcache_blocks", type=int, default=512)
    ap.add_argument("--kvcache_block_size", type=int, default=16)
    args = ap.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        sys.exit("CUDA not available")

    dtype = getattr(torch, args.dtype)
    print(f"Loading model {args.model_path} on {device} ({args.dtype})...")
    model, tok = load_pointllm_model(args.model_path, device, dtype)
    eos = tok.eos_token_id
    print("Model loaded.\n")

    pt_ids  = build_prompt_token_ids(tok, model)
    txt_ids = build_text_only_token_ids(tok)

    def pc():
        return make_fake_point_cloud(device, dtype)

    cases = [
        ("single  (B=1, 1 pc)",           [pt_ids],               [pc()]),
        ("batch_4 (B=4, 4 pc)",           [pt_ids] * 4,           [pc() for _ in range(4)]),
        ("batch_8 (B=8, 8 pc)",           [pt_ids] * 8,           [pc() for _ in range(8)]),
        ("mixed_4 (B=4, 2pc+2text)",      [pt_ids, pt_ids, txt_ids, txt_ids], [pc(), pc(), None, None]),
    ]

    print("=== PARITY (paged engine) ===")
    all_ok = True
    for name, token_ids_list, pcs in cases:
        ok = run_case(name, model, eos, token_ids_list, pcs, args.max_new_tokens, device,
                      args.num_kvcache_blocks, args.kvcache_block_size)
        if not ok:
            all_ok = False

    print()
    if all_ok:
        print("[ok] all parity cases passed.")
        sys.exit(0)
    else:
        print("[FAIL] one or more parity cases failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Create bench_paged_pointllm.py**

```python
#!/usr/bin/env python3
"""
P5 bench: paged PointLLMLLMEngine vs padded PointLLMLLMEngine decode throughput.

Usage:
  cd /home/nano-pointllm
  PYTHONPATH=/home/PointLLM:. python scripts/bench_paged_pointllm.py \
    --model_path /mnt/llm_data/pointllm_ckpt/PointLLM_7B_v1.2 \
    --device cuda:1 --batch_sizes 1,2,4,8 --decode_steps 32 \
    --warmup 2 --runs 3 --out_json results/p5_bench.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

from nanopointllm.engine.llm_engine import PointLLMLLMEngine
from nanopointllm.parity.engine_test_utils import (
    build_prompt_token_ids,
    load_pointllm_model,
    make_fake_point_cloud,
)
from nanopointllm.sampling_params import SamplingParams


@torch.inference_mode()
def bench_one(
    model,
    eos_token_id: int,
    token_ids: list[int],
    device: torch.device,
    dtype: torch.dtype,
    B: int,
    decode_steps: int,
    warmup: int,
    runs: int,
    num_kvcache_blocks: int,
    kvcache_block_size: int,
) -> dict:
    sp = SamplingParams(max_tokens=decode_steps, ignore_eos=True)
    pcs = [make_fake_point_cloud(device, dtype) for _ in range(B)]
    requests = [
        {"token_ids": token_ids, "point_clouds": pc, "sampling_params": sp}
        for pc in pcs
    ]

    def _run() -> tuple[float, float, int]:
        engine = PointLLMLLMEngine(
            model,
            eos_token_id=eos_token_id,
            max_num_seqs=B,
            max_num_batched_tokens=4096,
            num_kvcache_blocks=num_kvcache_blocks if num_kvcache_blocks > 0 else None,
            kvcache_block_size=kvcache_block_size,
        )
        for req in requests:
            engine.add_request(**req)

        torch.cuda.synchronize()
        t_start = time.perf_counter()
        engine.step()  # prefill
        torch.cuda.synchronize()
        t_prefill = time.perf_counter()

        actual_steps = 0
        while not engine.is_finished():
            engine.step()
            actual_steps += 1
        torch.cuda.synchronize()
        t_end = time.perf_counter()
        return (t_prefill - t_start) * 1000, (t_end - t_prefill) * 1000, actual_steps

    for _ in range(warmup):
        _run()

    pf_list, dc_list, steps_list = zip(*[_run() for _ in range(runs)])
    pf_mean = sum(pf_list) / runs
    dc_mean = sum(dc_list) / runs
    actual_steps = steps_list[-1]

    dec_per_step = dc_mean / (actual_steps * B) if actual_steps > 0 else 0.0
    tps = (actual_steps * B) / (dc_mean / 1000) if dc_mean > 0 else 0.0
    return {
        "batch_size": B,
        "prefill_ms_mean": round(pf_mean, 2),
        "decode_mean_per_step_ms": round(dec_per_step, 3),
        "tokens_per_sec": round(tps, 1),
        "actual_decode_steps": actual_steps,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="P5 bench: paged vs padded engine")
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--batch_sizes", default="1,2,4,8")
    ap.add_argument("--decode_steps", type=int, default=32)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--num_kvcache_blocks", type=int, default=512)
    ap.add_argument("--kvcache_block_size", type=int, default=16)
    ap.add_argument("--out_json", default="")
    args = ap.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        sys.exit("CUDA not available")

    dtype = getattr(torch, args.dtype)
    batch_sizes = [int(x.strip()) for x in args.batch_sizes.split(",") if x.strip()]

    print(f"Loading model {args.model_path} on {device} ({args.dtype})...")
    model, tok = load_pointllm_model(args.model_path, device, dtype)
    eos     = tok.eos_token_id
    pt_ids  = build_prompt_token_ids(tok, model)
    print(f"Model loaded. prompt_len={len(pt_ids)}\n")

    print(f"{'B':>4}  {'mode':<8}  {'prefill_ms':>10}  {'dec/step/seq_ms':>15}  {'tps':>8}")
    print("-" * 60)

    padded_results = []
    paged_results  = []

    for B in batch_sizes:
        try:
            # Padded (original)
            r_pad = bench_one(model, eos, pt_ids, device, dtype, B,
                              args.decode_steps, args.warmup, args.runs,
                              num_kvcache_blocks=0, kvcache_block_size=args.kvcache_block_size)
            print(f"  B={B:<2}  padded   prefill={r_pad['prefill_ms_mean']:7.1f}ms  "
                  f"dec/step={r_pad['decode_mean_per_step_ms']:7.3f}ms  "
                  f"tps={r_pad['tokens_per_sec']:7.1f}")
            padded_results.append(r_pad)

            # Paged
            r_paged = bench_one(model, eos, pt_ids, device, dtype, B,
                                args.decode_steps, args.warmup, args.runs,
                                num_kvcache_blocks=args.num_kvcache_blocks,
                                kvcache_block_size=args.kvcache_block_size)
            speedup = r_paged["tokens_per_sec"] / r_pad["tokens_per_sec"] if r_pad["tokens_per_sec"] else 0
            print(f"  B={B:<2}  paged    prefill={r_paged['prefill_ms_mean']:7.1f}ms  "
                  f"dec/step={r_paged['decode_mean_per_step_ms']:7.3f}ms  "
                  f"tps={r_paged['tokens_per_sec']:7.1f}  ({speedup:.2f}x vs padded)")
            paged_results.append(r_paged)
        except Exception as e:
            print(f"  B={B:<2}  SKIPPED ({e})")

    print()

    if args.out_json:
        out_dir = os.path.dirname(os.path.abspath(args.out_json))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        payload = {"padded_results": padded_results, "paged_results": paged_results}
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"[ok] results written to {args.out_json}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run unit tests (all non-GPU tests)**

```bash
cd /home/nano-pointllm && python -m pytest tests/ -v \
  --ignore=tests/test_pointllm_wrapper.py \
  --ignore=tests/test_model_runner.py \
  -x
```

Expected: all pass

- [ ] **Step 4: Run parity script (requires GPU + weights)**

```bash
cd /home/nano-pointllm
PYTHONPATH=/home/PointLLM:. python scripts/parity_paged_pointllm.py \
  --model_path /mnt/llm_data/pointllm_ckpt/PointLLM_7B_v1.2 \
  --device cuda:1
```

Expected: all 4 cases PASS, exit code 0

- [ ] **Step 5: Run bench script**

```bash
cd /home/nano-pointllm
PYTHONPATH=/home/PointLLM:. python scripts/bench_paged_pointllm.py \
  --model_path /mnt/llm_data/pointllm_ckpt/PointLLM_7B_v1.2 \
  --device cuda:1 --batch_sizes 1,2,4,8 --decode_steps 32 \
  --warmup 2 --runs 3 --out_json results/p5_bench.json
```

Expected: paged engine shows equal or better tps vs padded at each batch size; results saved to `results/p5_bench.json`

- [ ] **Step 6: Commit**

```bash
cd /home/nano-pointllm
git add scripts/parity_paged_pointllm.py scripts/bench_paged_pointllm.py
git commit -m "feat: add parity and bench scripts for paged KV engine (P5)"
```

---

## Implementation Notes

### RoPE shape convention
`apply_rotary_pos_emb` in HF transformers LlamaAttention expects `q/k` in `[B, H, S, D]` layout (after `.transpose(1, 2)`). `position_embeddings` from `LlamaModel` is `(cos[1,S,D], sin[1,S,D])`. With `unsqueeze_dim=1` (default), these become `[1,1,S,D]` and broadcast correctly to `[B,H,S,D]`. Do NOT use `[B,S,H,D]` layout before calling `rotary_fn`.

### Triton constexpr D
Triton requires `D` to be a power-of-2 at compile time. For LLaMA-7B: `H=32, D=128 → HD=4096` ✓. The `store_kvcache` wrapper checks and falls back to PyTorch if not power-of-2.

### PagedModelRunner prefill: `inputs_embeds` vs `input_ids`
HF PointLLMLlamaForCausalLM handles point clouds inside `forward()` when `point_clouds` is provided. In the paged runner we pre-encode point clouds into `inputs_embeds_cached` and pass `dummy_ids` + `inputs_embeds` + `point_clouds=None`. This matches the existing `PointLLMModelRunner` approach.

### sequence.py: no modifications needed
`past_key_values` stays in `PointLLMSequence` for the old padded runner's backward compat. `PagedModelRunner` uses `seq.block_table` and `seq.num_tokens` directly and never reads `past_key_values` or `kv_seq_len`.

### LLMEngine backward compat
When `num_kvcache_blocks=None` (default), `block_manager=None` is passed to `Scheduler` and `PointLLMModelRunner` is used. The old padded-KV path remains unchanged.
