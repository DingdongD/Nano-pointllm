"""eager_attention_forward_ref 与 HF eager_attention_forward 对齐（无全模型）。"""
import pytest
import torch


def test_repeat_kv_identity():
    from nanopointllm.llama.eager_attention_kv import repeat_kv

    x = torch.randn(2, 2, 5, 8)
    assert torch.equal(repeat_kv(x, 1), x)


def test_eager_ref_matches_hf_eager():
    try:
        from transformers.models.llama.modeling_llama import eager_attention_forward
    except ImportError:
        pytest.skip("transformers Llama eager_attention_forward 不可用")

    from nanopointllm.llama.eager_attention_kv import eager_attention_forward_ref

    class M:
        num_key_value_groups = 2
        training = False

    b, n_h, q, d = 1, 4, 3, 16
    n_kv = 2
    q_t = torch.randn(b, n_h, q, d)
    k_t = torch.randn(b, n_kv, q, d)
    v_t = torch.randn(b, n_kv, q, d)
    mask = torch.zeros(b, 1, q, q, dtype=torch.float32)
    scaling = d**-0.5

    o_hf, _ = eager_attention_forward(M(), q_t, k_t, v_t, mask, scaling=scaling, dropout=0.0)
    o_ref, _ = eager_attention_forward_ref(M(), q_t, k_t, v_t, mask, scaling=scaling, dropout=0.0)
    assert torch.allclose(o_hf, o_ref, rtol=0.0, atol=0.0)
