import pytest
import torch

from nanopointllm.analysis.neuron_oracle import analyze_neuron_oracle


def _payload(activation: list[float]) -> dict:
    return {
        "records": [{
            "sample_id": "s0",
            "point_cloud_id": "p0",
            "prompt_id": "q0",
            "decode_step": 0,
            "layer": 0,
        }],
        "activations": torch.tensor([activation]),
        "down_column_norm_layers": [0],
        "down_column_norms": torch.ones(1, len(activation)),
        "intermediate_size": len(activation),
        "activation_dtype": "float32",
    }


def test_uniform_neuron_oracle_has_full_effective_support():
    result = analyze_neuron_oracle(
        _payload([1.0, 1.0, 1.0, 1.0]),
        group_sizes=[1],
        budget_ratios=[0.5],
        contribution_targets=[0.9],
    )
    row = result["summary"][0]
    assert row["gini_mean"] == pytest.approx(0.0)
    assert row["normalized_entropy_mean"] == pytest.approx(1.0)
    assert row["effective_support_ratio_mean"] == pytest.approx(1.0)
    assert row["retention_at_50pct_mean"] == pytest.approx(0.5)
    assert row["neuron_ratio_for_90pct_contribution_mean"] == pytest.approx(1.0)


def test_concentrated_neuron_oracle_detects_small_support():
    result = analyze_neuron_oracle(
        _payload([4.0, 0.0, 0.0, 0.0]),
        group_sizes=[1],
        budget_ratios=[0.25],
        contribution_targets=[0.9],
    )
    row = result["summary"][0]
    assert row["gini_mean"] == pytest.approx(0.75)
    assert row["normalized_entropy_mean"] == pytest.approx(0.0)
    assert row["effective_support_ratio_mean"] == pytest.approx(0.25)
    assert row["retention_at_25pct_mean"] == pytest.approx(1.0)
    assert row["neuron_ratio_for_90pct_contribution_mean"] == pytest.approx(0.25)
