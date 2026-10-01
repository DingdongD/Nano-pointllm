from __future__ import annotations

import math
import statistics
import subprocess
from collections import Counter
from typing import Any


FIELDS = (
    "timestamp",
    "pstate",
    "sm_clock_mhz",
    "memory_clock_mhz",
    "power_w",
    "power_limit_w",
    "gpu_utilization_pct",
    "memory_controller_utilization_pct",
    "memory_used_mb",
    "temperature_c",
)
NUMERIC_FIELDS = FIELDS[2:]


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def parse_samples(output: str) -> list[dict[str, Any]]:
    samples = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != len(FIELDS):
            continue
        sample: dict[str, Any] = {"timestamp": parts[0], "pstate": parts[1]}
        try:
            sample.update({key: float(value) for key, value in zip(NUMERIC_FIELDS, parts[2:])})
        except ValueError:
            continue
        samples.append(sample)
    return samples


def summarize_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "sample_count": len(samples),
        "pstate_counts": dict(Counter(sample["pstate"] for sample in samples)),
    }
    for field in NUMERIC_FIELDS:
        values = [float(sample[field]) for sample in samples]
        summary[field] = {
            "mean": statistics.fmean(values) if values else 0.0,
            "median": statistics.median(values) if values else 0.0,
            "p10": _percentile(values, 0.10),
            "p90": _percentile(values, 0.90),
            "p95": _percentile(values, 0.95),
            "min": min(values) if values else 0.0,
            "max": max(values) if values else 0.0,
        }
    summary["sensor_status"] = {
        "power_w_available": bool(samples) and summary["power_w"]["max"] > 0.0,
        "clock_samples_available": bool(samples) and summary["sm_clock_mhz"]["median"] > 0.0,
    }
    if samples and not summary["sensor_status"]["power_w_available"]:
        summary["warnings"] = [
            "nvidia-smi returned 0W for every sample; treat power as unavailable, not zero"
        ]
    return summary


def _query_args(gpu_uuid: str, *, loop_ms: int | None = None) -> list[str]:
    fields = (
        "timestamp,pstate,clocks.current.graphics,clocks.current.memory,"
        "power.draw,power.limit,utilization.gpu,utilization.memory,"
        "memory.used,temperature.gpu"
    )
    args = [
        "nvidia-smi",
        "-i",
        gpu_uuid,
        f"--query-gpu={fields}",
        "--format=csv,noheader,nounits",
    ]
    if loop_ms is not None:
        args.append(f"--loop-ms={loop_ms}")
    return args


def query_gpu_state(gpu_uuid: str) -> dict[str, Any]:
    output = subprocess.check_output(_query_args(gpu_uuid), text=True, timeout=5)
    samples = parse_samples(output)
    if not samples:
        raise RuntimeError(f"nvidia-smi returned no parseable sample for {gpu_uuid}")
    return samples[0]


class GpuTelemetryMonitor:
    def __init__(self, gpu_uuid: str, *, interval_ms: int = 200) -> None:
        self.gpu_uuid = gpu_uuid
        self.interval_ms = interval_ms
        self._process: subprocess.Popen[str] | None = None

    def start(self) -> None:
        if self._process is not None:
            raise RuntimeError("telemetry monitor is already running")
        self._process = subprocess.Popen(
            _query_args(self.gpu_uuid, loop_ms=self.interval_ms),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    def stop(self) -> dict[str, Any]:
        if self._process is None:
            return {"samples": [], "summary": summarize_samples([])}
        process = self._process
        self._process = None
        process.terminate()
        try:
            stdout, stderr = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate(timeout=5)
        samples = parse_samples(stdout)
        result = {"samples": samples, "summary": summarize_samples(samples)}
        if stderr.strip():
            result["stderr"] = stderr.strip()
        return result

    def __enter__(self) -> "GpuTelemetryMonitor":
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.stop()
