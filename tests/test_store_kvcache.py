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
