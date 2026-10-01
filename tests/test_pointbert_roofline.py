import pytest

from nanopointllm.analysis.pointbert_roofline import summarize_stage


def _rows():
    base = {
        "Process ID": "1",
        "Kernel Name": "kernel",
    }
    metrics = {
        "gpu__time_duration.sum": (1000.0, 3000.0),
        "sm__cycles_active.avg.pct_of_peak_sustained_elapsed": (20.0, 40.0),
        "sm__throughput.avg.pct_of_peak_sustained_elapsed": (10.0, 30.0),
        "dram__throughput.avg.pct_of_peak_sustained_elapsed": (50.0, 70.0),
        "dram__bytes_read.sum": (100.0, 300.0),
        "dram__bytes_write.sum": (20.0, 80.0),
    }
    rows = []
    for metric, values in metrics.items():
        for kernel_id, value in enumerate(values):
            rows.append({
                **base,
                "ID": str(kernel_id),
                "Metric Name": metric,
                "Metric Value": str(value),
            })
    return rows


def test_summarize_stage_uses_duration_weighting_and_measured_bytes():
    summary = summarize_stage(
        _rows(),
        {
            "estimated_flops": 1000,
            "compulsory_bytes_lower_bound": 250,
        },
        peak_hbm_gbps=1000,
        peak_compute_tflops=100,
    )

    assert summary["total_profiled_kernel_time_ms"] == pytest.approx(0.004)
    assert summary["duration_weighted_sm_active_pct"] == pytest.approx(35.0)
    assert summary["duration_weighted_dram_throughput_pct"] == pytest.approx(65.0)
    assert summary["measured_dram_bytes"] == pytest.approx(500.0)
    assert summary["measured_ai_flops_per_byte"] == pytest.approx(2.0)
    assert summary["modeled_ai_flops_per_byte"] == pytest.approx(4.0)
    assert summary["achieved_tflops_from_estimated_work"] == pytest.approx(0.00025)


def test_summarize_stage_rejects_incomplete_metrics():
    with pytest.raises(ValueError, match="required NCU metrics"):
        summarize_stage(
            [],
            {"estimated_flops": 1, "compulsory_bytes_lower_bound": 1},
            peak_hbm_gbps=1000,
            peak_compute_tflops=100,
        )
