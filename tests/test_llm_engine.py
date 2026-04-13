"""
PointLLMLLMEngine 集成测试（mock 模型，不加载权重）。
验证完整的 add_request → step → 生成完成 生命周期。
"""
import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock
from transformers.cache_utils import DynamicCache

from nanopointllm.engine.llm_engine import PointLLMLLMEngine
from nanopointllm.engine.sequence import SequenceStatus
from nanopointllm.sampling_params import SamplingParams

VOCAB = 64
HIDDEN = 32
NUM_PATCH = 4
PATCH_TOKEN_ID = 55
NUM_LAYERS = 2
NUM_HEADS = 4
HEAD_DIM = HIDDEN // NUM_HEADS


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


def _make_engine() -> PointLLMLLMEngine:
    return PointLLMLLMEngine(
        hf_model=_make_mock_hf_model(),
        eos_token_id=2,
        max_num_seqs=4,
        max_num_batched_tokens=256,
    )


def _make_fake_prefill_cache(num_tokens: int) -> DynamicCache:
    cache = DynamicCache()
    for _ in range(NUM_LAYERS):
        cache.key_cache.append(torch.zeros(1, NUM_HEADS, num_tokens, HEAD_DIM))
        cache.value_cache.append(torch.zeros(1, NUM_HEADS, num_tokens, HEAD_DIM))
    return cache


def test_engine_construction():
    engine = _make_engine()
    assert engine.scheduler is not None
    assert engine.runner is not None


def test_engine_is_finished_when_empty():
    engine = _make_engine()
    assert engine.is_finished()


def test_engine_full_lifecycle(monkeypatch):
    """add_request → step loop → all seqs finished。"""
    step_counter = [0]

    def fake_run(seqs, is_prefill):
        if is_prefill:
            for seq in seqs:
                seq.past_key_values = _make_fake_prefill_cache(seq.num_tokens)
            return [5] * len(seqs)
        step_counter[0] += 1
        if step_counter[0] == 1:
            return [9] * len(seqs)
        return [2] * len(seqs)  # EOS

    engine = _make_engine()
    monkeypatch.setattr(engine.runner, "run", fake_run)

    tokens = [1] + [PATCH_TOKEN_ID] * NUM_PATCH + [3]
    seq1 = engine.add_request(token_ids=tokens, point_clouds=torch.randn(512, 3),
                               sampling_params=SamplingParams(max_tokens=10))
    seq2 = engine.add_request(token_ids=tokens,
                               sampling_params=SamplingParams(max_tokens=10))

    assert not engine.is_finished()

    engine.step()   # prefill
    assert seq1.status == SequenceStatus.RUNNING
    assert seq1.last_token == 5

    engine.step()   # decode → token 9
    assert seq1.last_token == 9

    engine.step()   # decode → EOS
    assert seq1.is_finished
    assert seq2.is_finished
    assert engine.is_finished()


def test_engine_generate_returns_completions(monkeypatch):
    def fake_run(seqs, is_prefill):
        if is_prefill:
            for seq in seqs:
                seq.past_key_values = _make_fake_prefill_cache(seq.num_tokens)
            return [5] * len(seqs)
        return [2] * len(seqs)  # immediate EOS

    engine = _make_engine()
    monkeypatch.setattr(engine.runner, "run", fake_run)

    results = engine.generate([
        {"token_ids": [1, 3, 4, 5], "sampling_params": SamplingParams(max_tokens=5)},
        {"token_ids": [1, 3, 4, 5], "sampling_params": SamplingParams(max_tokens=5)},
    ])
    assert len(results) == 2
    for seq in results:
        assert seq.is_finished


def test_engine_add_request_returns_sequence():
    engine = _make_engine()
    seq = engine.add_request(token_ids=[1, 2, 3])
    from nanopointllm.engine.sequence import PointLLMSequence
    assert isinstance(seq, PointLLMSequence)
    assert seq.token_ids == [1, 2, 3]
    assert seq.status == SequenceStatus.WAITING
