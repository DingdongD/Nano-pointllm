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
