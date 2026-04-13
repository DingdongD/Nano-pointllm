"""
使用 mock 模型测试 PointLLMWrapper，无需实际加载 PointLLM-7B 权重。
"""
import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock
from nanopointllm.models.pointllm_wrapper import PointLLMWrapper


def _make_mock_hf_model(vocab_size=32, hidden_size=64, num_patch_tokens=4):
    """构造一个最小化的 mock PointLLMLlamaForCausalLM。"""
    PATCH_TOKEN_ID = 30

    inner = MagicMock()
    inner.embed_tokens = nn.Embedding(vocab_size, hidden_size)
    inner.point_backbone_config = {
        "point_patch_token": PATCH_TOKEN_ID,
        "point_token_len": num_patch_tokens,
        "mm_use_point_start_end": False,
        "backbone_output_dim": 32,
        "project_output_dim": hidden_size,
    }

    def _backbone(point_clouds):
        B = point_clouds.shape[0]
        return torch.zeros(B, num_patch_tokens, 32)

    inner.point_backbone = _backbone
    inner.point_proj = nn.Linear(32, hidden_size)

    hf_model = MagicMock()
    hf_model.model = inner
    hf_model.get_model = MagicMock(return_value=inner)
    hf_model.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    return hf_model, PATCH_TOKEN_ID, num_patch_tokens, hidden_size


def test_wrapper_construction():
    hf_model, *_ = _make_mock_hf_model()
    wrapper = PointLLMWrapper(hf_model)
    assert wrapper.inner is hf_model.model


def test_encode_point_clouds_list():
    hf_model, PATCH_TOKEN_ID, num_patch_tokens, hidden_size = _make_mock_hf_model()
    wrapper = PointLLMWrapper(hf_model)

    pc1 = torch.randn(512, 3)
    pc2 = torch.randn(512, 3)
    features = wrapper.encode_point_clouds([pc1, pc2])
    assert len(features) == 2
    for f in features:
        assert f.shape == (num_patch_tokens, hidden_size)


def test_encode_point_clouds_batch_tensor():
    hf_model, PATCH_TOKEN_ID, num_patch_tokens, hidden_size = _make_mock_hf_model()
    wrapper = PointLLMWrapper(hf_model)

    pcs = torch.randn(3, 512, 3)  # batch of 3
    features = wrapper.encode_point_clouds(pcs)
    assert len(features) == 3
    for f in features:
        assert f.shape == (num_patch_tokens, hidden_size)


def test_prepare_inputs_embeds_fuses_patch_tokens():
    hf_model, PATCH_TOKEN_ID, num_patch_tokens, hidden_size = _make_mock_hf_model()
    wrapper = PointLLMWrapper(hf_model)

    L = 1 + num_patch_tokens + 1
    input_ids = torch.tensor([
        [1] + [PATCH_TOKEN_ID] * num_patch_tokens + [2],
        [1] + [PATCH_TOKEN_ID] * num_patch_tokens + [2],
    ])
    point_features = [torch.randn(num_patch_tokens, hidden_size) for _ in range(2)]
    embeds = wrapper.prepare_inputs_embeds(input_ids, point_features)
    assert embeds.shape == (2, L, hidden_size)


def test_prepare_inputs_embeds_no_point_clouds():
    """无点云时退化为纯文本嵌入。"""
    hf_model, PATCH_TOKEN_ID, num_patch_tokens, hidden_size = _make_mock_hf_model()
    wrapper = PointLLMWrapper(hf_model)

    input_ids = torch.tensor([[1, 5, 6, 2]])
    embeds = wrapper.prepare_inputs_embeds(input_ids, point_features=None)
    assert embeds.shape == (1, 4, hidden_size)
