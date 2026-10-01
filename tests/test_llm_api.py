from types import SimpleNamespace

import torch.nn as nn

from nanopointllm import LLM, SamplingParams


class TinyTokenizer:
    eos_token_id = 2

    def encode(self, text):
        return [ord(ch) % 10 + 3 for ch in text]

    def decode(self, token_ids):
        return " ".join(str(t) for t in token_ids)


def _make_model():
    model = SimpleNamespace()
    model.config = SimpleNamespace(eos_token_id=2)
    model.model = SimpleNamespace(embed_tokens=nn.Embedding(32, 8))
    model.get_model = lambda: model.model
    model.lm_head = nn.Linear(8, 32, bias=False)
    return model


def test_llm_generate_accepts_string_prompts(monkeypatch):
    llm = LLM(_make_model(), tokenizer=TinyTokenizer(), max_num_seqs=2)

    def fake_run_mixed(prefill_seqs, decode_seqs):
        return [2] * (len(prefill_seqs) + len(decode_seqs))

    monkeypatch.setattr(llm.runner, "run_mixed", fake_run_mixed)
    outputs = llm.generate(["ab"], SamplingParams(max_tokens=4))

    assert len(outputs) == 1
    assert outputs[0]["token_ids"] == [2]
    assert outputs[0]["text"] == "2"
    assert outputs[0]["sequence"].is_finished


def test_llm_generate_accepts_request_dict_with_point_clouds(monkeypatch):
    llm = LLM(_make_model(), eos_token_id=2, max_num_seqs=2)

    def fake_run_mixed(prefill_seqs, decode_seqs):
        return [2] * (len(prefill_seqs) + len(decode_seqs))

    monkeypatch.setattr(llm.runner, "run_mixed", fake_run_mixed)
    outputs = llm.generate([
        {
            "token_ids": [1, 3, 4],
            "point_clouds": object(),
            "sampling_params": SamplingParams(max_tokens=4),
        }
    ])

    assert outputs[0]["token_ids"] == [2]
    assert outputs[0]["text"] is None
    assert outputs[0]["sequence"].point_clouds is not None


def test_llm_generate_passes_point_cache_fields(monkeypatch):
    llm = LLM(_make_model(), eos_token_id=2, max_num_seqs=2)

    def fake_run_mixed(prefill_seqs, decode_seqs):
        return [2] * (len(prefill_seqs) + len(decode_seqs))

    monkeypatch.setattr(llm.runner, "run_mixed", fake_run_mixed)
    outputs = llm.generate([
        {
            "token_ids": [1, 3, 4],
            "point_clouds": object(),
            "point_cloud_cache_key": "shared-pc-1",
            "point_features_cached": "cached-feat",
            "sampling_params": SamplingParams(max_tokens=4),
        }
    ])

    seq = outputs[0]["sequence"]
    assert seq.point_cloud_cache_key == "shared-pc-1"
    assert seq.point_features_cached == "cached-feat"
