import pytest
import torch

from nanopointllm.analysis.mlp_block_trace import (
    MLPBlockTraceCollector,
    analyze_mlp_block_trace,
)
from nanopointllm.llama.lightweight_runner import build_lightweight_llama_runner
from tests.test_lightweight_runner import _FakeHF


def _context(sample_id: str, prompt_id: str, step: int) -> dict:
    return {
        "sample_id": sample_id,
        "point_cloud_id": "point-0",
        "prompt_id": prompt_id,
        "decode_step": step,
        "input_token_id": 1,
        "input_generated_token_index": step,
        "output_generated_token_index": step + 1,
    }


def test_lightweight_runner_collects_exact_swiglu_intermediates():
    torch.manual_seed(0)
    model = _FakeHF()
    runner = build_lightweight_llama_runner(model)
    collector = MLPBlockTraceCollector(expected_layers=2, storage_dtype=torch.float32)
    runner.set_mlp_trace_observer(collector)

    input_ids = torch.tensor([[1]])
    collector.begin_step([_context("sample-0", "prompt-0", 0)])
    runner(input_ids)
    collector.end_step([7])

    payload = collector.payload()
    assert payload["activations"].shape == (2, 4)
    assert [record["layer"] for record in payload["records"]] == [0, 1]
    assert all(record["output_token_id"] == 7 for record in payload["records"])

    hidden = model.model.embed_tokens(input_ids)
    first_layer = model.model.layers[0]
    normalized = first_layer.input_layernorm(hidden)
    attention = first_layer.self_attn(normalized)[0]
    post_attention = first_layer.post_attention_layernorm(hidden + attention)
    expected = first_layer.mlp.act_fn(first_layer.mlp.gate_proj(post_attention))
    expected = expected * first_layer.mlp.up_proj(post_attention)
    torch.testing.assert_close(payload["activations"][0], expected.reshape(-1, 4)[0])


def test_block_trace_metrics_and_weight_byte_models():
    records = [
        _context("sample-p0", "prompt-0", 0) | {"layer": 0, "row_index": 0},
        _context("sample-p0", "prompt-0", 1) | {"layer": 0, "row_index": 0},
        _context("sample-p1", "prompt-1", 0) | {"layer": 0, "row_index": 0},
        _context("sample-p1", "prompt-1", 1) | {"layer": 0, "row_index": 0},
    ]
    payload = {
        "schema_version": 1,
        "hidden_size": 2,
        "intermediate_size": 4,
        "weight_element_size": 4,
        "records": records,
        "activations": torch.tensor([
            [4.0, 3.0, 2.0, 1.0],
            [4.0, 3.0, 1.0, 2.0],
            [4.0, 3.0, 2.0, 1.0],
            [1.0, 2.0, 4.0, 3.0],
        ]),
        "down_column_norm_layers": [0],
        "down_column_norms": torch.ones(1, 4),
    }

    result = analyze_mlp_block_trace(payload, block_sizes=[2], block_ratios=[0.5])
    curve = result["curves"][0]
    assert curve["post_gate_weight_byte_reduction_mean"] == pytest.approx(1.0 / 6.0)
    assert curve["oracle_pre_gate_weight_byte_reduction_mean"] == pytest.approx(0.5)
    assert curve["previous_token_weight_byte_reduction_steady_state"] == pytest.approx(0.5)
    assert curve["previous_token_weight_byte_reduction_including_first_step_mean"] == pytest.approx(0.25)
    assert curve["adjacent_token_jaccard_mean"] == pytest.approx(0.5)
    assert curve["previous_token_recall_mean"] == pytest.approx(0.5)
    assert curve["same_point_cross_prompt_jaccard_mean"] == pytest.approx(0.5)
    assert curve["post_gate_contribution_retained_mean"] == pytest.approx(0.7)
    assert curve["previous_token_contribution_retained_mean"] == pytest.approx(0.5)
    assert result["trace"]["baseline_mlp_weight_bytes_per_layer"] == 96


def test_collector_requires_every_expected_layer():
    collector = MLPBlockTraceCollector(expected_layers=2)
    collector.begin_step([_context("sample-0", "prompt-0", 0)])
    collector(
        layer_index=0,
        activations=torch.ones(1, 4),
        down_proj_weight=torch.ones(2, 4),
    )
    with pytest.raises(RuntimeError, match="expected 2"):
        collector.end_step([1])
