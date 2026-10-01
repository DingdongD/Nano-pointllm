from types import SimpleNamespace

import torch

from scripts.profile_pointllm_ncu import build_stage_manifest


def _linear(out_features: int, in_features: int):
    return SimpleNamespace(weight=torch.empty(out_features, in_features, dtype=torch.bfloat16))


def test_stage_manifest_separates_weight_and_kv_traffic():
    attention = SimpleNamespace(
        packed_qkv_weight=torch.empty(8, 4, dtype=torch.bfloat16),
        o_proj=_linear(4, 4),
        k_cache=torch.empty(8, 2, 2, 2, dtype=torch.bfloat16),
        num_heads=2,
        num_kv_heads=2,
        head_dim=2,
    )
    mlp = SimpleNamespace(
        packed_gateup_weight=torch.empty(12, 4, dtype=torch.bfloat16),
        down_proj=_linear(4, 6),
    )
    layer = SimpleNamespace(self_attn=attention, mlp=mlp)
    runner = SimpleNamespace(
        layers=[layer, layer],
        lm_head=_linear(10, 4),
    )
    engine = SimpleNamespace(runner=SimpleNamespace(lightweight_runner=runner))

    manifest = build_stage_manifest(
        engine,
        batch_size=2,
        context_lengths=[[5, 7]],
    )

    assert manifest["runtime"].startswith("LightweightLlamaRunner")
    assert manifest["stages"]["qkv"]["weight_bytes"] > 0
    assert manifest["stages"]["mlp"]["weight_bytes"] > manifest["stages"]["o_proj"]["weight_bytes"]
    assert manifest["stages"]["attention"]["weight_bytes"] == 0
    assert manifest["stages"]["attention"]["kv_read_bytes"] == 384
    assert manifest["stages"]["attention"]["estimated_flops"] == 384

    layer_manifest = build_stage_manifest(
        engine,
        batch_size=2,
        context_lengths=[[5, 7]],
        profiled_layer=0,
    )
    assert layer_manifest["profiled_layers"] == [0]
    assert layer_manifest["stages"]["qkv"]["weight_bytes"] * 2 == manifest["stages"]["qkv"]["weight_bytes"]
    assert layer_manifest["stages"]["attention"]["kv_read_bytes"] == 192
