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
