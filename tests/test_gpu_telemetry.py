from nanopointllm.gpu_telemetry import parse_samples, summarize_samples


def test_parse_and_summarize_gpu_samples():
    output = "\n".join((
        "2026/09/22 21:00:00.000, P0, 1200, 1215, 200, 400, 80, 60, 1234, 70",
        "2026/09/22 21:00:00.200, P0, 1400, 1215, 240, 400, 100, 80, 1240, 72",
    ))
    samples = parse_samples(output)
    summary = summarize_samples(samples)

    assert len(samples) == 2
    assert samples[0]["sm_clock_mhz"] == 1200.0
    assert summary["sample_count"] == 2
    assert summary["pstate_counts"] == {"P0": 2}
    assert summary["power_w"]["median"] == 220.0
    assert summary["sm_clock_mhz"]["p90"] == 1380.0
    assert summary["sensor_status"]["power_w_available"] is True


def test_parse_samples_skips_malformed_rows():
    assert parse_samples("bad,row") == []


def test_zero_power_is_marked_unavailable():
    samples = parse_samples(
        "2026/09/22 21:00:00.000, P0, 1200, 1215, 0, 400, 80, 60, 1234, 70"
    )
    summary = summarize_samples(samples)

    assert summary["sensor_status"]["power_w_available"] is False
    assert "unavailable" in summary["warnings"][0]
