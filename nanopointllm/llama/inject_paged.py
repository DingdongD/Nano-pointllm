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
        if LlamaAttention is not None and not isinstance(attn, LlamaAttention):
            continue
        layer.self_attn = PagedLlamaAttention.from_hf(
            attn,
            kv_pool.k_cache(layer_idx),
            kv_pool.v_cache(layer_idx),
        )
