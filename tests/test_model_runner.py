"""
用 mock 模型测试 ModelRunner 的批量 prefill 和批量 decode 逻辑，
无需加载 PointLLM-7B 权重。
"""
import pytest
import torch
import torch.nn as nn
from types import SimpleNamespace
from unittest.mock import MagicMock
from transformers.cache_utils import DynamicCache

from nanopointllm.engine.sequence import PointLLMSequence, SequenceStatus
from nanopointllm.engine.model_runner import PointLLMModelRunner
from nanopointllm.models.pointllm_wrapper import PointLLMWrapper
from nanopointllm.sampling_params import SamplingParams

VOCAB = 64
HIDDEN = 32
PATCH_TOKEN_ID = 55
NUM_PATCH = 4
NUM_LAYERS = 2
NUM_HEADS = 4
HEAD_DIM = HIDDEN // NUM_HEADS   # 8


def _make_mock_hf_model():
    inner = MagicMock()
    inner.embed_tokens = nn.Embedding(VOCAB, HIDDEN)
    inner.point_backbone_config = {
        "point_patch_token": PATCH_TOKEN_ID,
        "point_token_len": NUM_PATCH,
        "mm_use_point_start_end": False,
        "backbone_output_dim": 16,
        "project_output_dim": HIDDEN,
    }

    def _backbone(pcs):
        B = pcs.shape[0]
        return torch.zeros(B, NUM_PATCH, 16)

    inner.point_backbone = _backbone
    inner.point_proj = nn.Linear(16, HIDDEN)

    hf_model = MagicMock()
    hf_model.model = inner
    hf_model.get_model = MagicMock(return_value=inner)
    hf_model.lm_head = nn.Linear(HIDDEN, VOCAB, bias=False)
    return hf_model


def _seq_with_point_cloud(prompt_len: int = 8) -> PointLLMSequence:
    token_ids = [1] + [PATCH_TOKEN_ID] * NUM_PATCH + [2] * (prompt_len - NUM_PATCH - 1)
    return PointLLMSequence(
        token_ids=token_ids,
        point_clouds=torch.randn(512, 3),
        sampling_params=SamplingParams(max_tokens=5),
    )


def _make_fake_cache(batch_size: int, seq_len: int) -> DynamicCache:
    cache = DynamicCache()
    for _ in range(NUM_LAYERS):
        cache.key_cache.append(torch.zeros(batch_size, NUM_HEADS, seq_len, HEAD_DIM))
        cache.value_cache.append(torch.zeros(batch_size, NUM_HEADS, seq_len, HEAD_DIM))
    return cache


def test_runner_construction():
    runner = PointLLMModelRunner(hf_model=_make_mock_hf_model())
    assert runner.wrapper is not None


def test_encode_point_clouds_fills_cache():
    runner = PointLLMModelRunner(hf_model=_make_mock_hf_model())
    seq = _seq_with_point_cloud()
    assert seq.inputs_embeds_cached is None
    runner._encode_point_clouds_batch([seq])
    assert seq.inputs_embeds_cached is not None
    assert seq.inputs_embeds_cached.shape[-1] == HIDDEN


def test_encode_point_clouds_skips_already_cached():
    runner = PointLLMModelRunner(hf_model=_make_mock_hf_model())
    seq = _seq_with_point_cloud()
    dummy = torch.zeros(1, 10, HIDDEN)
    seq.inputs_embeds_cached = dummy
    runner._encode_point_clouds_batch([seq])
    assert seq.inputs_embeds_cached is dummy  # unchanged


def test_run_prefill_sets_past_key_values(monkeypatch):
    """prefill 后每个 seq 应有 past_key_values。"""
    runner = PointLLMModelRunner(hf_model=_make_mock_hf_model())

    def fake_prefill(model, input_ids, attention_mask, point_clouds=None, **kw):
        B, L = input_ids.shape
        cache = _make_fake_cache(B, L)
        from nanopointllm.engine.types import PrefillOutput
        return PrefillOutput(
            logits=torch.randn(B, 1, VOCAB),
            past_key_values=cache,
            prompt_len=L,
        )

    monkeypatch.setattr("nanopointllm.engine.model_runner.hf_prefill", fake_prefill)

    seqs = [_seq_with_point_cloud(8), _seq_with_point_cloud(8)]
    tokens = runner.run_prefill(seqs)
    assert len(tokens) == 2
    for seq in seqs:
        assert seq.past_key_values is not None
        assert seq.kv_seq_len > 0


def test_run_decode_returns_next_tokens(monkeypatch):
    """decode 步应为每个 seq 返回一个 token，且只做一次 forward。"""
    runner = PointLLMModelRunner(hf_model=_make_mock_hf_model())

    seqs = [_seq_with_point_cloud(8), _seq_with_point_cloud(8)]
    for seq in seqs:
        seq.past_key_values = _make_fake_cache(1, 8)

    call_count = [0]
    out_logits = torch.zeros(len(seqs), 1, VOCAB)
    out_logits[0, 0, 10] = 100.0
    out_logits[1, 0, 20] = 100.0

    def fake_batch_decode(model, input_ids, attention_mask, past_key_values, **kw):
        call_count[0] += 1
        new_cache = _make_fake_cache(len(seqs), 9)
        return SimpleNamespace(logits=out_logits, past_key_values=new_cache)

    monkeypatch.setattr("nanopointllm.engine.model_runner._batch_hf_decode", fake_batch_decode)

    tokens = runner.run_decode(seqs)
    assert tokens == [10, 20]
    assert call_count[0] == 1   # single batched forward


def test_run_dispatch():
    """run(is_prefill=True) 走 prefill，run(is_prefill=False) 走 decode。"""
    runner = PointLLMModelRunner(hf_model=_make_mock_hf_model())
    prefill_called = [False]
    decode_called = [False]

    def fake_prefill(seqs): prefill_called[0] = True; return [1]
    def fake_decode(seqs): decode_called[0] = True; return [2]

    runner.run_prefill = fake_prefill
    runner.run_decode = fake_decode

    seq = _seq_with_point_cloud()
    runner.run([seq], is_prefill=True)
    assert prefill_called[0]

    runner.run([seq], is_prefill=False)
    assert decode_called[0]


def test_prefill_kv_has_full_prompt_length(monkeypatch):
    """prefill 后每个 seq 的 kv_seq_len 应等于其 prompt token 数，而不是 1。"""
    runner = PointLLMModelRunner(hf_model=_make_mock_hf_model())

    def fake_prefill(model, input_ids, attention_mask, point_clouds=None, **kw):
        B, L = input_ids.shape
        cache = _make_fake_cache(B, L)   # shape [B, H, max_len, D]
        from nanopointllm.engine.types import PrefillOutput
        return PrefillOutput(
            logits=torch.randn(B, 1, VOCAB),
            past_key_values=cache,
            prompt_len=L,
        )

    monkeypatch.setattr("nanopointllm.engine.model_runner.hf_prefill", fake_prefill)

    prompt_len = 8
    seqs = [_seq_with_point_cloud(prompt_len), _seq_with_point_cloud(prompt_len)]
    runner.run_prefill(seqs)

    for seq in seqs:
        # kv_seq_len 应等于 prompt_len（8），而不是 1
        assert seq.kv_seq_len == prompt_len, (
            f"Expected kv_seq_len={prompt_len}, got {seq.kv_seq_len}"
        )
