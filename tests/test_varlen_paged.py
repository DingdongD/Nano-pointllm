"""
Unit tests for varlen continuous batching (Phase 2).
"""
import torch
import pytest
from nanopointllm.engine.forward_context import ForwardContext, set_forward_context, get_forward_context, clear_forward_context


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
