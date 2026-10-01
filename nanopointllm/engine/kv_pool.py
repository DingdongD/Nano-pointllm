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
    cfg = hf_model.config
    num_layers = cfg.num_hidden_layers
    num_attention_heads = cfg.num_attention_heads
    num_heads = getattr(cfg, "num_key_value_heads", None)
    if not isinstance(num_heads, int):
        num_heads = num_attention_heads
    head_dim = getattr(cfg, "head_dim", None)
    if not isinstance(head_dim, int):
        head_dim = cfg.hidden_size // num_attention_heads
    pool = torch.zeros(
        2, num_layers, num_blocks, block_size, num_heads, head_dim,
        dtype=dtype, device=device,
    )
    return KVPool(pool, num_layers, num_blocks, block_size, num_heads, head_dim)
