from types import SimpleNamespace

import torch.nn as nn

from nanopointllm.engine.llm_engine import PointLLMLLMEngine


def _make_model():
    model = SimpleNamespace()
    model.config = SimpleNamespace(eos_token_id=2)
    inner = SimpleNamespace(embed_tokens=nn.Embedding(32, 8))
    model.model = inner
    model.get_model = lambda: inner
    model.lm_head = nn.Linear(8, 32, bias=False)
    model.parameters = lambda: iter(model.lm_head.parameters())
    return model


def test_engine_exposes_shared_point_feature_cache():
    engine = PointLLMLLMEngine(_make_model(), eos_token_id=2, max_num_seqs=2)
    assert engine.runner.point_feature_cache is engine.point_feature_cache
    assert engine.get_point_feature_cache_stats()["size"] == 0


def test_engine_add_request_accepts_point_cache_fields():
    engine = PointLLMLLMEngine(_make_model(), eos_token_id=2, max_num_seqs=2)
    seq = engine.add_request(
        token_ids=[1, 2, 3],
        point_clouds="pc",
        point_cloud_cache_key="pc-key",
        point_features_cached="feat",
    )
    assert seq.point_cloud_cache_key == "pc-key"
    assert seq.point_features_cached == "feat"
