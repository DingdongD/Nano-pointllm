"""无 GPU：LlamaRMSNormRef / LlamaSwiGLUMLPRef 与 HF 同构层数值对齐。"""
import inspect

import torch
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaMLP, LlamaRMSNorm

from nanopointllm.llama.mlp_ref import LlamaSwiGLUMLPRef
from nanopointllm.llama.rmsnorm_ref import LlamaRMSNormRef


def _hf_llama_mlp(cfg: LlamaConfig) -> LlamaMLP:
    """兼容旧版 ``LlamaMLP(config)`` 与新版 ``(hidden_size, intermediate_size, hidden_act)``。"""
    n = len(inspect.signature(LlamaMLP.__init__).parameters)
    if n == 2:
        return LlamaMLP(cfg)
    return LlamaMLP(cfg.hidden_size, cfg.intermediate_size, cfg.hidden_act)


def _tiny_config() -> LlamaConfig:
    return LlamaConfig(
        hidden_size=128,
        intermediate_size=352,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_hidden_layers=1,
        max_position_embeddings=256,
        rms_norm_eps=1e-5,
        hidden_act="silu",
        mlp_bias=False,
    )


def test_rmsnorm_matches_hf():
    cfg = _tiny_config()
    hf = LlamaRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
    ref = LlamaRMSNormRef(cfg.hidden_size, eps=cfg.rms_norm_eps)
    ref.load_state_dict(hf.state_dict())
    x = torch.randn(2, 17, cfg.hidden_size)
    y_hf = hf(x)
    y_ref = ref(x)
    assert torch.allclose(y_hf, y_ref, rtol=0.0, atol=0.0)


def test_mlp_matches_hf():
    cfg = _tiny_config()
    hf = _hf_llama_mlp(cfg)
    ref = LlamaSwiGLUMLPRef.from_llama_config(cfg)
    ref.load_state_dict(hf.state_dict())
    x = torch.randn(2, 5, cfg.hidden_size)
    y_hf = hf(x)
    y_ref = ref(x)
    assert torch.allclose(y_hf, y_ref, rtol=0.0, atol=0.0)


def test_mlp_bf16_close():
    cfg = _tiny_config()
    hf = _hf_llama_mlp(cfg).to(dtype=torch.bfloat16)
    ref = LlamaSwiGLUMLPRef.from_llama_config(cfg).to(dtype=torch.bfloat16)
    ref.load_state_dict(hf.state_dict())
    x = torch.randn(2, 3, cfg.hidden_size, dtype=torch.bfloat16)
    assert torch.allclose(hf(x).float(), ref(x).float(), rtol=0.02, atol=0.02)
