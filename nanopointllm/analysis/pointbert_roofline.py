"""Nsight Compute aggregation and roofline helpers for PointBERT stages."""
from __future__ import annotations

from collections import defaultdict
import csv
from pathlib import Path


SM_ACTIVE = "sm__cycles_active.avg.pct_of_peak_sustained_elapsed"
SM_THROUGHPUT = "sm__throughput.avg.pct_of_peak_sustained_elapsed"
DRAM_THROUGHPUT = "dram__throughput.avg.pct_of_peak_sustained_elapsed"
DRAM_READ_BYTES = "dram__bytes_read.sum"
DRAM_WRITE_BYTES = "dram__bytes_write.sum"
DURATION = "gpu__time_duration.sum"


def read_ncu_csv(path: Path) -> list[dict[str, str]]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    header_index = next(
        (index for index, line in enumerate(lines) if line.startswith('"ID","Process ID"')),
        None,
    )
    if header_index is None:
        raise ValueError("Nsight Compute CSV header was not found")
    return list(csv.DictReader(lines[header_index:]))


def rows_for_range(rows: list[dict[str, str]], nvtx_range: str) -> list[dict[str, str]]:
    if not rows:
        return []
    nvtx_columns = [
        key
        for key in rows[0]
        if key and ("Push/Pop_Range" in key or "Start/Stop_Range" in key)
    ]
    if not nvtx_columns:
        raise ValueError("Nsight Compute CSV contains no NVTX range columns")
    return [
        row
        for row in rows
        if any(nvtx_range in (row.get(column) or "") for column in nvtx_columns)
    ]


def summarize_stage(
    rows: list[dict[str, str]],
    theoretical: dict,
    *,
    peak_hbm_gbps: float,
    peak_compute_tflops: float,
) -> dict:
    """Aggregate one stage with duration weighting and measured-byte AI."""
    kernels: dict[tuple[str, str], dict] = defaultdict(lambda: {"metrics": {}})
    for row in rows:
        key = (row["Process ID"], row["ID"])
        kernel = kernels[key]
        kernel["kernel_name"] = row["Kernel Name"]
        try:
            value = float(row["Metric Value"].replace(",", ""))
        except (KeyError, ValueError):
            continue
        kernel["metrics"][row["Metric Name"]] = value

    complete = []
    required = (DURATION, SM_ACTIVE, DRAM_THROUGHPUT)
    for kernel in kernels.values():
        metrics = kernel["metrics"]
        if all(metric in metrics for metric in required):
            complete.append({
                "kernel_name": kernel["kernel_name"],
                "duration_ns": metrics[DURATION],
                "sm_active_pct": metrics[SM_ACTIVE],
                "sm_throughput_pct": metrics.get(SM_THROUGHPUT),
                "dram_throughput_pct": metrics[DRAM_THROUGHPUT],
                "dram_read_bytes": metrics.get(DRAM_READ_BYTES, 0.0),
                "dram_write_bytes": metrics.get(DRAM_WRITE_BYTES, 0.0),
            })
    if not complete:
        raise ValueError("no kernels contain the required NCU metrics")

    total_duration_ns = sum(kernel["duration_ns"] for kernel in complete)

    def weighted(field: str) -> float | None:
        available = [kernel for kernel in complete if kernel[field] is not None]
        if not available:
            return None
        duration = sum(kernel["duration_ns"] for kernel in available)
        return sum(kernel[field] * kernel["duration_ns"] for kernel in available) / duration

    measured_read = sum(kernel["dram_read_bytes"] for kernel in complete)
    measured_write = sum(kernel["dram_write_bytes"] for kernel in complete)
    measured_bytes = measured_read + measured_write
    estimated_flops = theoretical.get("estimated_flops")
    compulsory_bytes = theoretical.get("compulsory_bytes_lower_bound")
    measured_ai = estimated_flops / measured_bytes if estimated_flops and measured_bytes else None
    modeled_ai = estimated_flops / compulsory_bytes if estimated_flops and compulsory_bytes else None
    achieved_tflops = estimated_flops / total_duration_ns / 1e3 if estimated_flops else None
    ridge = peak_compute_tflops * 1000.0 / peak_hbm_gbps
    roofline_tflops = (
        min(peak_compute_tflops, measured_ai * peak_hbm_gbps / 1000.0)
        if measured_ai is not None
        else None
    )
    return {
        "kernel_count": len(complete),
        "total_profiled_kernel_time_ms": total_duration_ns / 1e6,
        "duration_weighted_sm_active_pct": weighted("sm_active_pct"),
        "duration_weighted_sm_throughput_pct": weighted("sm_throughput_pct"),
        "duration_weighted_dram_throughput_pct": weighted("dram_throughput_pct"),
        "measured_dram_read_bytes": measured_read,
        "measured_dram_write_bytes": measured_write,
        "measured_dram_bytes": measured_bytes,
        "estimated_flops": estimated_flops,
        "compulsory_bytes_lower_bound": compulsory_bytes,
        "modeled_ai_flops_per_byte": modeled_ai,
        "measured_ai_flops_per_byte": measured_ai,
        "ridge_point_flops_per_byte": ridge,
        "achieved_tflops_from_estimated_work": achieved_tflops,
        "roofline_ceiling_tflops_from_measured_ai": roofline_tflops,
        "below_ridge_from_measured_ai": measured_ai < ridge if measured_ai is not None else None,
        "measured_to_compulsory_byte_ratio": (
            measured_bytes / compulsory_bytes if compulsory_bytes else None
        ),
        "work_model_note": theoretical.get("work_model_note"),
        "top_kernels_by_time": sorted(
            complete, key=lambda item: item["duration_ns"], reverse=True
        )[:10],
    }

