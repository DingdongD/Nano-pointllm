from scripts.summarize_ncu_pointllm import (
    DRAM_READ_BYTES,
    DRAM_THROUGHPUT,
    DRAM_WRITE_BYTES,
    DURATION,
    SM_ACTIVE,
    SM_THROUGHPUT,
    summarize,
)


def _row(kernel_id, name, metric, value):
    return {
        "Process ID": "1",
        "ID": str(kernel_id),
        "Kernel Name": name,
        "Metric Name": metric,
        "Metric Value": str(value),
    }


def test_ncu_summary_is_weighted_by_kernel_duration():
    rows = []
    for metric, value in ((DURATION, 10), (SM_ACTIVE, 20), (DRAM_THROUGHPUT, 30)):
        rows.append(_row(0, "short", metric, value))
    for metric, value in ((DURATION, 30), (SM_ACTIVE, 60), (DRAM_THROUGHPUT, 70)):
        rows.append(_row(1, "long", metric, value))

    result = summarize(rows, peak_hbm_gbps=1000)

    assert result["kernel_count"] == 2
    assert result["duration_weighted_sm_active_pct"] == 50
    assert result["duration_weighted_dram_throughput_pct"] == 60
    assert result["estimated_duration_weighted_dram_gbps"] == 600


def test_ncu_summary_strict_weight_bound_requires_clean_high_dram_evidence():
    rows = []
    metrics = (
        (DURATION, 100),
        (SM_ACTIVE, 40),
        (SM_THROUGHPUT, 35),
        (DRAM_THROUGHPUT, 75),
        (DRAM_READ_BYTES, 900),
        (DRAM_WRITE_BYTES, 100),
    )
    for metric, value in metrics:
        rows.append(_row(0, "gemv", metric, value))
    manifest = {
        "idle_at_start": True,
        "stages": {
            "qkv": {
                "compulsory_bytes": 1000,
                "estimated_flops": 2000,
                "arithmetic_intensity_flops_per_byte": 2.0,
                "weight_fraction_of_compulsory_bytes": 0.95,
            }
        },
    }

    result = summarize(
        rows,
        peak_hbm_gbps=1000,
        stage="qkv",
        manifest=manifest,
        peak_compute_tflops=100,
    )

    assert result["measured_dram_bytes"] == 1000
    assert result["bottleneck"]["strict_weight_bound_confirmed"] is True
    assert result["bottleneck"]["classification"] == "confirmed_weight_bandwidth_bound"


def test_ncu_summary_rejects_strict_claim_under_external_load():
    rows = []
    for metric, value in (
        (DURATION, 100),
        (SM_ACTIVE, 40),
        (DRAM_THROUGHPUT, 90),
    ):
        rows.append(_row(0, "gemv", metric, value))
    manifest = {
        "idle_at_start": False,
        "stages": {
            "mlp": {
                "compulsory_bytes": 1000,
                "estimated_flops": 2000,
                "arithmetic_intensity_flops_per_byte": 2.0,
                "weight_fraction_of_compulsory_bytes": 0.95,
            }
        },
    }

    result = summarize(
        rows,
        peak_hbm_gbps=1000,
        stage="mlp",
        manifest=manifest,
        peak_compute_tflops=100,
    )

    assert result["bottleneck"]["strict_weight_bound_confirmed"] is False
    assert result["bottleneck"]["classification"] == "invalid_external_gpu_load"


def test_ncu_summary_rejects_high_utilization_without_modeled_dram_traffic():
    rows = []
    for metric, value in (
        (DURATION, 100),
        (SM_ACTIVE, 30),
        (SM_THROUGHPUT, 25),
        (DRAM_THROUGHPUT, 80),
        (DRAM_READ_BYTES, 100),
        (DRAM_WRITE_BYTES, 10),
    ):
        rows.append(_row(0, "gemv", metric, value))
    manifest = {
        "idle_at_start": True,
        "stages": {
            "mlp": {
                "compulsory_bytes": 1000,
                "estimated_flops": 2000,
                "arithmetic_intensity_flops_per_byte": 2.0,
                "weight_fraction_of_compulsory_bytes": 0.95,
            }
        },
    }

    result = summarize(
        rows,
        peak_hbm_gbps=1000,
        stage="mlp",
        manifest=manifest,
        peak_compute_tflops=100,
    )

    assert result["bottleneck"]["measured_traffic_present"] is False
    assert result["bottleneck"]["strict_weight_bound_confirmed"] is False
