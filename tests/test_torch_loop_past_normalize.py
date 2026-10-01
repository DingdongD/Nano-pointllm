"""legacy tuple past → Cache，供 torch_loop decode 使用。"""
import pytest
import torch
from transformers import LlamaConfig


def test_normalize_legacy_tuple_to_dynamic_cache():
    from nanopointllm.llama.torch_llama_decode_backend import (
        dynamic_cache_cls,
        normalize_past_key_values_for_torch_loop,
        _past_seq_length,
    )

    DC = dynamic_cache_cls()
    if DC is None or not hasattr(DC, "from_legacy_cache"):
        pytest.skip()

    cfg = LlamaConfig(
        hidden_size=8,
        num_attention_heads=2,
        num_key_value_heads=2,
        num_hidden_layers=2,
        max_position_embeddings=32,
    )
    k = torch.randn(1, 2, 5, 4)
    v = torch.randn(1, 2, 5, 4)
    legacy = ((k, v), (k.clone(), v.clone()))
    cache = normalize_past_key_values_for_torch_loop(legacy, cfg, DynamicCache=DC)
    assert hasattr(cache, "get_seq_length")
    assert _past_seq_length(cache) == 5
    assert _past_seq_length(legacy) == 5
