"""
从 HF / PointLLM 的 Llama 系 `config` 读取 RoPE、头维等，供 `nanopointllm.llama` 与 HF 对齐使用。
"""
from __future__ import annotations

from typing import Any


def head_dim_from_config(config: Any) -> int:
    """`hidden_size // num_attention_heads`（与 `transformers` Llama 一致）。"""
    hs = int(config.hidden_size)
    nh = int(config.num_attention_heads)
    d = hs // nh
    if d * nh != hs:
        raise ValueError(f"hidden_size {hs} not divisible by num_attention_heads {nh}")
    return d


def rope_theta_from_config(config: Any) -> float:
    """Llama / Vicuna / PointLLM 常用字段 `rope_theta`；缺省 10000。"""
    return float(getattr(config, "rope_theta", 10000.0))


def max_position_embeddings_from_config(config: Any) -> int:
    return int(getattr(config, "max_position_embeddings", 4096))
