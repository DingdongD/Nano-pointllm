"""
Unit tests for varlen continuous batching (Phase 2).
"""
import torch
import pytest
from nanopointllm.engine.forward_context import ForwardContext, set_forward_context, get_forward_context, clear_forward_context
from nanopointllm.engine.scheduler import Scheduler
from nanopointllm.engine.sequence import PointLLMSequence, SequenceStatus
from nanopointllm.sampling_params import SamplingParams


def _make_seq(token_ids):
    sp = SamplingParams(max_tokens=4)
    seq = PointLLMSequence(token_ids=token_ids, point_clouds=None, sampling_params=sp)
    return seq


def test_scheduler_returns_prefill_decode_tuple():
    """schedule() returns (prefill_seqs, decode_seqs) as separate lists."""
    sched = Scheduler(max_num_seqs=4, max_num_batched_tokens=512, block_manager=None)
    seq_a = _make_seq([1, 2, 3])
    seq_b = _make_seq([4, 5])
    sched.add(seq_a)
    sched.add(seq_b)

    result = sched.schedule()
    assert isinstance(result, tuple) and len(result) == 2, "schedule() must return (prefill_seqs, decode_seqs)"
    prefill_seqs, decode_seqs = result
    assert isinstance(prefill_seqs, list)
    assert isinstance(decode_seqs, list)
    # First call: all are new prefill, none are decode
    assert len(prefill_seqs) == 2
    assert len(decode_seqs) == 0


def test_scheduler_decode_seqs_after_prefill():
    """After postprocess, same seqs appear in decode_seqs on next schedule()."""
    sched = Scheduler(max_num_seqs=4, max_num_batched_tokens=512, block_manager=None)
    seq_a = _make_seq([1, 2, 3])
    sched.add(seq_a)

    prefill_seqs, decode_seqs = sched.schedule()
    assert len(prefill_seqs) == 1 and len(decode_seqs) == 0

    # Simulate postprocess: append a token
    sched.postprocess(prefill_seqs + decode_seqs, token_ids=[42])

    prefill_seqs2, decode_seqs2 = sched.schedule()
    assert len(prefill_seqs2) == 0
    assert len(decode_seqs2) == 1
    assert decode_seqs2[0] is seq_a


def test_continuous_batching_scheduler():
    """New requests admitted mid-flight, not waiting for original batch to finish."""
    sched = Scheduler(max_num_seqs=8, max_num_batched_tokens=512, block_manager=None)
    # Add 4 initial requests
    for i in range(4):
        sched.add(_make_seq([i + 1, i + 2]))

    # Step 1: prefill all 4
    prefill1, decode1 = sched.schedule()
    assert len(prefill1) == 4
    sched.postprocess(prefill1 + decode1, token_ids=[10] * 4)

    # Add 4 more while originals are in decode
    for i in range(4, 8):
        sched.add(_make_seq([i + 1, i + 2]))

    # Step 2: new arrivals should be admitted as prefill THIS step
    prefill2, decode2 = sched.schedule()
    assert len(prefill2) == 4, f"Expected 4 new prefill, got {len(prefill2)}"
    assert len(decode2) == 4, f"Expected 4 decode, got {len(decode2)}"


def test_forward_context_has_seq_lens():
    """ForwardContext accepts seq_lens and has no is_prefill field."""
    ctx = ForwardContext(
        slot_mapping=torch.zeros(3, dtype=torch.int32),
        block_tables=torch.zeros(2, 4, dtype=torch.int32),
        context_lens=torch.tensor([2, 1], dtype=torch.int32),
        position_ids=torch.zeros(1, 3, dtype=torch.long),
        seq_lens=[2, 1],
    )
    assert ctx.seq_lens == [2, 1]
    assert not hasattr(ctx, "is_prefill"), "is_prefill must be removed"


def test_forward_context_thread_local():
    """set/get/clear work correctly."""
    ctx = ForwardContext(
        slot_mapping=torch.zeros(1, dtype=torch.int32),
        block_tables=torch.zeros(1, 1, dtype=torch.int32),
        context_lens=torch.tensor([1], dtype=torch.int32),
        position_ids=torch.zeros(1, 1, dtype=torch.long),
        seq_lens=[1],
    )
    set_forward_context(ctx)
    assert get_forward_context() is ctx
    clear_forward_context()
    assert get_forward_context() is None


import torch.nn.functional as F


def _build_kv_cache(num_blocks, block_size, num_kv_heads, head_dim, device):
    """Allocate a fake KV pool (pre-filled with zeros)."""
    k_cache = torch.zeros(num_blocks, block_size, num_kv_heads, head_dim, device=device)
    v_cache = torch.zeros(num_blocks, block_size, num_kv_heads, head_dim, device=device)
    return k_cache, v_cache


def _fill_kv_cache(k_cache, v_cache, block_table, token_kvs_k, token_kvs_v, block_size):
    """Write token KVs into pool according to block_table."""
    for pos, (k_tok, v_tok) in enumerate(zip(token_kvs_k, token_kvs_v)):
        blk = pos // block_size
        off = pos % block_size
        phys = block_table[blk]
        k_cache[phys, off] = k_tok
        v_cache[phys, off] = v_tok


def test_varlen_attention_single_decode():
    """
    KV gather from block pool matches direct construction.
    Verifies the block-table gather logic is correct.
    """
    device = torch.device("cpu")
    H, Hkv, D = 4, 2, 8
    ctx_len = 5
    block_size = 4
    num_blocks = 4

    k_cache, v_cache = _build_kv_cache(num_blocks, block_size, Hkv, D, device)

    torch.manual_seed(0)
    kv_k = torch.randn(ctx_len, Hkv, D)
    kv_v = torch.randn(ctx_len, Hkv, D)
    block_table = [0, 1]
    _fill_kv_cache(k_cache, v_cache, block_table, kv_k, kv_v, block_size)

    q_tok = torch.randn(1, H, 1, D)

    # Expected: SDPA directly from original kv tensors (no cache involved)
    ki_direct = kv_k.permute(1, 0, 2).unsqueeze(0).repeat_interleave(H // Hkv, 1)
    vi_direct = kv_v.permute(1, 0, 2).unsqueeze(0).repeat_interleave(H // Hkv, 1)
    expected = F.scaled_dot_product_attention(q_tok, ki_direct, vi_direct, is_causal=False)

    # Actual: gather from cache using block table
    num_blks = (ctx_len + block_size - 1) // block_size
    bt_valid = block_table[:num_blks]
    k_full = k_cache[bt_valid].reshape(-1, Hkv, D)[:ctx_len]
    v_full = v_cache[bt_valid].reshape(-1, Hkv, D)[:ctx_len]
    ki_gathered = k_full.permute(1, 0, 2).unsqueeze(0).repeat_interleave(H // Hkv, 1)
    vi_gathered = v_full.permute(1, 0, 2).unsqueeze(0).repeat_interleave(H // Hkv, 1)
    actual = F.scaled_dot_product_attention(q_tok, ki_gathered, vi_gathered, is_causal=False)

    torch.testing.assert_close(actual, expected)
