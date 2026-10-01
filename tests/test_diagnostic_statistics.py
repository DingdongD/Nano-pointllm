import pytest

from scripts.benchmark_pointllm_diagnostics import summarize


def test_steady_state_summary_reports_requested_percentiles():
    stats = summarize([float(value) for value in range(10)])

    assert stats["median"] == 4.5
    assert stats["p10"] == 0.9
    assert stats["p90"] == 8.1
    assert stats["p95"] == pytest.approx(8.55)
