import pytest

from nanopointllm.benchmark_metrics import (
    estimate_decode_hbm,
    kv_bytes_per_token,
    kv_cache_metrics,
    resize_point_prompt,
)


def test_resize_point_prompt_preserves_base_and_extends_exactly():
    base = [1, 2, 3]

    assert resize_point_prompt(base, 0) == base
    assert resize_point_prompt(base, 7) == [1, 2, 3, 1, 2, 3, 1]
    with pytest.raises(ValueError, match="cannot be shortened"):
        resize_point_prompt(base, 2)


def test_kv_cache_metrics_include_k_and_v_for_every_layer():
    metrics = kv_cache_metrics(
        num_layers=2,
        num_blocks=8,
        block_size=4,
        num_kv_heads=2,
        head_dim=8,
        dtype_bytes=2,
        batch_size=2,
        prompt_tokens=5,
        output_tokens=2,
    )

    assert kv_bytes_per_token(
        num_layers=2,
        num_kv_heads=2,
        head_dim=8,
        dtype_bytes=2,
    ) == 128
    assert metrics["blocks_per_sequence_peak"] == 2
    assert metrics["active_blocks_peak"] == 4
    assert metrics["pool_utilization_peak"] == 0.5


def test_hbm_estimate_uses_one_weight_read_and_batched_kv_traffic():
    result = estimate_decode_hbm(
        decoder_weight_bytes=1_000_000_000,
        kv_bytes_token=1000,
        batch_size=4,
        average_context_tokens=100,
        batch_step_ms=10,
        peak_hbm_gbps=1000,
    )

    assert result["estimated_decoder_weight_read_gb_per_step"] == 1.0
    assert result["estimated_kv_read_gb_per_step"] == pytest.approx(0.0004)
    assert result["estimated_effective_hbm_gbps"] == pytest.approx(100.0404)
