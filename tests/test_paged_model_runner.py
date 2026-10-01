from types import SimpleNamespace

import torch
import torch.nn as nn
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaModel

from nanopointllm.engine.forward_context import (
    ForwardContext,
    clear_forward_context,
    set_forward_context,
)
from nanopointllm.engine.paged_model_runner import PagedModelRunner
from nanopointllm.engine.paged_model_runner import _install_position_ids_patch


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.ones(1))
        self.embed_tokens = nn.Embedding(8, 1)
        self.config = SimpleNamespace(vocab_size=8)
        self.forward_calls = 0

    def get_input_embeddings(self):
        return self.embed_tokens

    @property
    def model(self):
        return SimpleNamespace(embed_tokens=self.embed_tokens)

    def forward(self, *, input_ids, **kwargs):
        self.forward_calls += 1
        logits = torch.zeros(1, input_ids.shape[1], self.config.vocab_size)
        logits[..., 3] = 1.0
        return SimpleNamespace(logits=logits)


class _DecodeLogits(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, *, input_ids=None, inputs_embeds=None, **kwargs):
        self.calls += 1
        source = input_ids if input_ids is not None else inputs_embeds
        logits = torch.zeros(1, source.shape[1], 8)
        logits[..., 5] = 1.0
        return logits


def _make_runner(monkeypatch, *, graph: bool = False, lightweight: str | None = None):
    calls = []
    sentinel = nn.Identity()

    def fake_build(model, *, decode_graph_mode=False):
        calls.append(decode_graph_mode)
        return sentinel

    monkeypatch.setattr(
        "nanopointllm.engine.paged_model_runner.build_lightweight_llama_runner",
        fake_build,
    )
    monkeypatch.setenv("NANOPOINTLLM_ENABLE_CUDA_GRAPH", "1" if graph else "0")
    monkeypatch.delenv("NANOPOINTLLM_PROFILE_LAYERS", raising=False)
    monkeypatch.delenv("NANOPOINTLLM_COMPILE_LIGHTWEIGHT", raising=False)
    if lightweight is None:
        monkeypatch.delenv("NANOPOINTLLM_LIGHTWEIGHT_DECODE", raising=False)
    else:
        monkeypatch.setenv("NANOPOINTLLM_LIGHTWEIGHT_DECODE", lightweight)

    model = _Model()
    runner = PagedModelRunner(
        model,
        SimpleNamespace(block_size=16),
        block_manager=None,
    )
    return runner, sentinel, calls, model


def test_paged_eager_decode_uses_lightweight_runner_by_default(monkeypatch):
    runner, sentinel, calls, _ = _make_runner(monkeypatch)

    assert runner.lightweight_decode_enabled is True
    assert runner.lightweight_runner is sentinel
    assert calls == [False]


def test_paged_eager_decode_can_restore_hf_forward(monkeypatch):
    runner, _, calls, _ = _make_runner(monkeypatch, lightweight="0")

    assert runner.lightweight_decode_enabled is False
    assert runner.lightweight_runner is None
    assert calls == []


def test_cuda_graph_requires_optimized_lightweight_runner(monkeypatch):
    runner, sentinel, calls, _ = _make_runner(
        monkeypatch,
        graph=True,
        lightweight="0",
    )

    assert runner.lightweight_decode_enabled is True
    assert runner.lightweight_runner is sentinel
    assert calls == [True]


def test_layer_profiling_keeps_eager_lightweight_configuration(monkeypatch):
    monkeypatch.setenv("NANOPOINTLLM_PROFILE_LAYERS", "1")
    calls = []
    sentinel = nn.Identity()

    def fake_build(model, *, decode_graph_mode=False):
        calls.append(decode_graph_mode)
        return sentinel

    monkeypatch.setattr(
        "nanopointllm.engine.paged_model_runner.build_lightweight_llama_runner",
        fake_build,
    )
    monkeypatch.setenv("NANOPOINTLLM_ENABLE_CUDA_GRAPH", "0")
    monkeypatch.setenv("NANOPOINTLLM_LIGHTWEIGHT_DECODE", "0")
    runner = PagedModelRunner(
        _Model(),
        SimpleNamespace(block_size=16),
        block_manager=None,
    )

    assert runner.lightweight_decode_enabled is True
    assert runner.lightweight_runner is sentinel
    assert runner.profile_runtime is True
    assert calls == [False]


def test_run_decode_bypasses_hf_model_forward_by_default(monkeypatch):
    runner, _, _, model = _make_runner(monkeypatch)
    decode = _DecodeLogits()
    runner.lightweight_runner = decode
    seq = SimpleNamespace(
        block_table=[0],
        token_ids=[1, 2],
        num_tokens=2,
        last_token=2,
        sampling_params=SimpleNamespace(
            do_sample=False,
            temperature=1.0,
            top_k=-1,
            top_p=1.0,
            repetition_penalty=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            seed=None,
        ),
    )

    tokens = runner.run_decode([seq])

    assert tokens == [5]
    assert decode.calls == 1
    assert model.forward_calls == 0


def test_paged_prefill_bypasses_hf_model_forward_by_default(monkeypatch):
    runner, _, _, model = _make_runner(monkeypatch)
    decode = _DecodeLogits()
    runner.lightweight_runner = decode
    seq = SimpleNamespace(
        block_table=[0],
        token_ids=[1, 2],
        num_tokens=2,
        num_cached_tokens=0,
        prefill_chunk_end=2,
        inputs_embeds_cached=torch.ones(2, 1),
        point_clouds=None,
        sampling_params=SimpleNamespace(
            do_sample=False,
            temperature=1.0,
            top_k=-1,
            top_p=1.0,
            repetition_penalty=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            seed=None,
        ),
    )

    tokens = runner.run_mixed([seq], [])

    assert tokens == [5]
    assert decode.calls == 1
    assert model.forward_calls == 0


def test_paged_mixed_step_bypasses_hf_model_forward_by_default(monkeypatch):
    runner, _, _, model = _make_runner(monkeypatch)
    decode = _DecodeLogits()
    runner.lightweight_runner = decode
    sampling_params = SimpleNamespace(
        do_sample=False,
        temperature=1.0,
        top_k=-1,
        top_p=1.0,
        repetition_penalty=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        seed=None,
    )
    prefill = SimpleNamespace(
        block_table=[0],
        token_ids=[1, 2],
        num_tokens=2,
        num_cached_tokens=0,
        prefill_chunk_end=2,
        inputs_embeds_cached=torch.ones(2, 1),
        point_clouds=None,
        sampling_params=sampling_params,
    )
    decode_seq = SimpleNamespace(
        block_table=[1],
        token_ids=[3, 4, 5],
        num_tokens=3,
        last_token=5,
        sampling_params=sampling_params,
    )

    tokens = runner.run_mixed([prefill], [decode_seq])

    assert tokens == [5, 5]
    assert decode.calls == 1
    assert model.forward_calls == 0


def test_position_patch_replaces_explicit_none_from_pointllm_base_call(monkeypatch):
    _install_position_ids_patch()
    config = LlamaConfig(
        vocab_size=16,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=32,
    )
    model = LlamaModel(config).eval()
    seen = []
    original_rotary_forward = model.rotary_emb.forward

    def capture_positions(x, position_ids):
        seen.append(position_ids.detach().clone())
        return original_rotary_forward(x, position_ids)

    monkeypatch.setattr(model.rotary_emb, "forward", capture_positions)
    expected = torch.tensor([[7, 7]])
    set_forward_context(ForwardContext(
        slot_mapping=None,
        block_tables=None,
        context_lens=None,
        position_ids=expected,
    ))
    try:
        with torch.inference_mode():
            model(
                input_ids=torch.tensor([[1, 2]]),
                position_ids=None,
                use_cache=False,
            )
    finally:
        clear_forward_context()

    assert len(seen) == 1
    torch.testing.assert_close(seen[0], expected)
