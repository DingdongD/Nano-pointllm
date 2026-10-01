"""无 CUDA 依赖：仅测 `nanopointllm.llama.hf_config` 数值。"""
from nanopointllm.llama.hf_config import head_dim_from_config, rope_theta_from_config


class _Cfg:
    hidden_size = 4096
    num_attention_heads = 32
    rope_theta = 10000.0


def test_head_dim_and_rope_theta():
    assert head_dim_from_config(_Cfg()) == 128
    assert rope_theta_from_config(_Cfg()) == 10000.0
    assert rope_theta_from_config(type("X", (), {})()) == 10000.0
