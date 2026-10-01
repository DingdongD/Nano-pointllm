"""
Llama 主干适配（参考 nano-vllm 的层实现思想，语义对齐 HuggingFace `LlamaModel`）。

默认目标：**单卡 CUDA** + **PointLLM-7B v1.2** 上终验 decode（prefill 仍 HF）。
实施顺序见：`docs/nanovllm_port_roadmap.md`（§ 默认目标环境、§0、P2）。

**验证脚本**：`scripts/verify_p2_pointllm_cuda.py`（RoPE 自检 + greedy parity + decode 微 bench）。
"""

from nanopointllm.llama.hf_config import (
    head_dim_from_config,
    max_position_embeddings_from_config,
    rope_theta_from_config,
)
from nanopointllm.llama.eager_attention_kv import eager_attention_forward_ref, repeat_kv
from nanopointllm.llama.mlp_ref import LlamaSwiGLUMLPRef
from nanopointllm.llama.rmsnorm_ref import LlamaRMSNormRef
from nanopointllm.llama.rope_sanity import verify_decode_step_rope
__all__ = [
    "LlamaRMSNormRef",
    "LlamaSwiGLUMLPRef",
    "eager_attention_forward_ref",
    "repeat_kv",
    "head_dim_from_config",
    "max_position_embeddings_from_config",
    "rope_theta_from_config",
    "verify_decode_step_rope",
]
