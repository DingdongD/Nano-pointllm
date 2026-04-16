"""
Integration tests for continuous batching (Task 6).

Verifies that LLMEngine.step() calls run_mixed() with BOTH prefill_seqs and
decode_seqs non-empty when new requests arrive while existing ones are decoding.
"""
from __future__ import annotations

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


def _make_fake_prefill_cache(num_tokens: int) -> DynamicCache:
    cache = DynamicCache()
    for _ in range(NUM_LAYERS):
        cache.key_cache.append(torch.zeros(1, NUM_HEADS, num_tokens, HEAD_DIM))
        cache.value_cache.append(torch.zeros(1, NUM_HEADS, num_tokens, HEAD_DIM))
    return cache


def _make_engine(max_num_seqs: int = 4) -> PointLLMLLMEngine:
    return PointLLMLLMEngine(
        hf_model=_make_mock_hf_model(),
        eos_token_id=99,  # unlikely EOS so max_tokens controls termination
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=512,
    )


def _fake_run(seqs, is_prefill):
    if is_prefill:
        for seq in seqs:
            seq.past_key_values = _make_fake_prefill_cache(seq.num_tokens)
        return [5] * len(seqs)
    return [7] * len(seqs)  # non-EOS decode token


# ── Test 1: mixed step occurs ─────────────────────────────────────────────────

def test_run_mixed_receives_truly_mixed_batch(monkeypatch):
    """
    True continuous batching: run_mixed must be called with BOTH
    prefill_seqs and decode_seqs non-empty in at least one step.

    Scenario with max_num_seqs=4:
      - Requests 0,1: max_tokens=1  → finish after first decode step
      - Requests 2,3: max_tokens=5  → long-lived
      - Requests 4,5: max_tokens=2  → submitted to waiting queue

    Timeline:
      step 1: prefill [0,1,2,3],         decode []        → pure prefill
      step 2: prefill [],                 decode [0,1,2,3] → pure decode; 0,1 finish
      step 3: prefill [4,5],              decode [2,3]     → MIXED  ← key assertion
      step 4+: running down
    """
    engine = _make_engine(max_num_seqs=4)
    monkeypatch.setattr(engine.runner, "run", _fake_run)

    call_log: list[tuple[int, int]] = []
    orig_run_mixed = engine.runner.run_mixed

    def tracking_run_mixed(prefill_seqs, decode_seqs):
        call_log.append((len(prefill_seqs), len(decode_seqs)))
        return orig_run_mixed(prefill_seqs, decode_seqs)

    monkeypatch.setattr(engine.runner, "run_mixed", tracking_run_mixed)

    # Submit 4 initial requests (fill max_num_seqs)
    for _ in range(2):
        engine.add_request(token_ids=[1, 2, 3],
                           sampling_params=SamplingParams(max_tokens=1))
    for _ in range(2):
        engine.add_request(token_ids=[1, 2, 3],
                           sampling_params=SamplingParams(max_tokens=5, ignore_eos=True))
    # Submit 2 more that start in waiting queue
    for _ in range(2):
        engine.add_request(token_ids=[1, 2, 3],
                           sampling_params=SamplingParams(max_tokens=2, ignore_eos=True))

    while not engine.is_finished():
        engine.step()

    # At least one step must have been truly mixed
    truly_mixed = any(p > 0 and d > 0 for p, d in call_log)
    assert truly_mixed, f"Expected at least one mixed step; step log: {call_log}"


# ── Test 2: correct output count ──────────────────────────────────────────────

def test_run_mixed_output_length_matches_total_seqs(monkeypatch):
    """
    run_mixed must return exactly len(prefill_seqs) + len(decode_seqs) token ids.
    """
    engine = _make_engine(max_num_seqs=4)
    monkeypatch.setattr(engine.runner, "run", _fake_run)

    orig_run_mixed = engine.runner.run_mixed
    output_lengths: list[tuple[int, int]] = []  # (expected, actual)

    def checking_run_mixed(prefill_seqs, decode_seqs):
        result = orig_run_mixed(prefill_seqs, decode_seqs)
        expected = len(prefill_seqs) + len(decode_seqs)
        output_lengths.append((expected, len(result)))
        return result

    monkeypatch.setattr(engine.runner, "run_mixed", checking_run_mixed)

    for _ in range(2):
        engine.add_request(token_ids=[1, 2, 3],
                           sampling_params=SamplingParams(max_tokens=1))
    for _ in range(2):
        engine.add_request(token_ids=[1, 2, 3],
                           sampling_params=SamplingParams(max_tokens=3, ignore_eos=True))
    for _ in range(2):
        engine.add_request(token_ids=[1, 2, 3],
                           sampling_params=SamplingParams(max_tokens=2, ignore_eos=True))

    while not engine.is_finished():
        engine.step()

    for expected, actual in output_lengths:
        assert expected == actual, (
            f"run_mixed returned {actual} tokens but expected {expected}"
        )


# ── Test 3: all sequences complete ────────────────────────────────────────────

def test_continuous_batching_all_sequences_finish(monkeypatch):
    """
    With continuous batching, all submitted sequences must eventually finish.
    """
    engine = _make_engine(max_num_seqs=3)
    monkeypatch.setattr(engine.runner, "run", _fake_run)

    seqs = []
    for _ in range(2):
        seqs.append(engine.add_request(
            token_ids=[1, 2, 3],
            sampling_params=SamplingParams(max_tokens=1),
        ))
    for _ in range(4):
        seqs.append(engine.add_request(
            token_ids=[1, 2, 3],
            sampling_params=SamplingParams(max_tokens=3, ignore_eos=True),
        ))

    max_steps = 100
    steps = 0
    while not engine.is_finished() and steps < max_steps:
        engine.step()
        steps += 1

    assert engine.is_finished(), "Engine should finish all requests"
    for seq in seqs:
        assert seq.is_finished, f"Sequence not finished after {steps} steps"


# ── Test 4: paged engine wires run_mixed ──────────────────────────────────────

def test_paged_engine_uses_run_mixed(monkeypatch):
    """
    PointLLMLLMEngine with paged KV must invoke PagedModelRunner.run_mixed()
    (not runner.run()) when step() is called.
    """
    from types import SimpleNamespace
    from nanopointllm.engine.kv_pool import KVPool

    # Minimal mock that satisfies allocate_kv_pool and inject_paged_attention
    config = SimpleNamespace(
        num_hidden_layers=0,  # no layers → inject_paged is a no-op
        num_attention_heads=NUM_HEADS,
        num_key_value_heads=NUM_HEADS,
        hidden_size=HIDDEN,
    )
    inner = MagicMock()
    inner.embed_tokens = nn.Embedding(VOCAB, HIDDEN)
    inner.point_backbone_config = {
        "point_patch_token": PATCH_TOKEN_ID,
        "point_token_len": NUM_PATCH,
        "mm_use_point_start_end": False,
        "backbone_output_dim": 16,
        "project_output_dim": HIDDEN,
    }
    inner.point_backbone = lambda pcs: torch.zeros(pcs.shape[0], NUM_PATCH, 16)
    inner.point_proj = nn.Linear(16, HIDDEN)
    inner.layers = []  # no attention layers → inject is a no-op

    dummy_param = nn.Parameter(torch.zeros(1))
    hf_model = MagicMock()
    hf_model.config = config
    hf_model.model = inner
    hf_model.get_model = MagicMock(return_value=inner)
    hf_model.lm_head = nn.Linear(HIDDEN, VOCAB, bias=False)
    hf_model.parameters = lambda: iter([dummy_param])

    engine = PointLLMLLMEngine(
        hf_model=hf_model,
        eos_token_id=99,
        max_num_seqs=4,
        max_num_batched_tokens=512,
        num_kvcache_blocks=32,
        kvcache_block_size=4,
    )

    from nanopointllm.engine.paged_model_runner import PagedModelRunner
    assert isinstance(engine.runner, PagedModelRunner), (
        "Paged engine must use PagedModelRunner"
    )

    run_mixed_calls: list[tuple[int, int]] = []

    def fake_run_mixed(prefill_seqs, decode_seqs):
        run_mixed_calls.append((len(prefill_seqs), len(decode_seqs)))
        return [5] * (len(prefill_seqs) + len(decode_seqs))

    monkeypatch.setattr(engine.runner, "run_mixed", fake_run_mixed)

    engine.add_request(token_ids=[1, 2, 3],
                       sampling_params=SamplingParams(max_tokens=2, ignore_eos=True))
    engine.step()   # prefill

    assert len(run_mixed_calls) == 1, "step() must call run_mixed() exactly once"
    assert run_mixed_calls[0] == (1, 0), (
        f"Expected (1, 0) on first step, got {run_mixed_calls[0]}"
    )

    engine.step()   # decode
    assert len(run_mixed_calls) == 2
    assert run_mixed_calls[1] == (0, 1), (
        f"Expected (0, 1) on decode step, got {run_mixed_calls[1]}"
    )
