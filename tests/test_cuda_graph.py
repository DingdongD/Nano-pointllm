import torch

import pytest

from nanopointllm.engine.cuda_graph import DecodeGraphCache, DecodeGraphKey


def test_decode_graph_cache_disabled_raises():
    cache = DecodeGraphCache(enabled=False)
    key = DecodeGraphKey(
        bucket_size=4,
        max_blocks_per_seq=8,
        vocab_size=16,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    with pytest.raises(RuntimeError, match="disabled"):
        cache.run(key, lambda: torch.zeros(4, 1, 16))


def test_decode_graph_cache_cpu_raises_when_enabled():
    cache = DecodeGraphCache(enabled=True)
    key = DecodeGraphKey(
        bucket_size=4,
        max_blocks_per_seq=8,
        vocab_size=16,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    with pytest.raises(RuntimeError, match="CUDA device"):
        cache.run(key, lambda: torch.zeros(4, 1, 16))
