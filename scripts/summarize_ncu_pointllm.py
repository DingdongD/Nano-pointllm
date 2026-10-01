#!/usr/bin/env python3
"""Summarize Nsight Compute CSV metrics using kernel-duration weighting."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
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
        (idx for idx, line in enumerate(lines) if line.startswith('"ID","Process ID"')),
        None,
    )
    if header_index is None:
        raise ValueError("Nsight Compute CSV header was not found")
    return list(csv.DictReader(lines[header_index:]))


def _classify_stage(
    *,
    stage: str,
    theoretical: dict,
    weighted_dram: float,
    weighted_sm: float | None,
    measured_dram_bytes: float | None,
    peak_hbm_gbps: float | None,
    peak_compute_tflops: float | None,
    idle_at_start: bool | None,
    min_dram_throughput_pct: float,
) -> dict:
    arithmetic_intensity = theoretical.get("arithmetic_intensity_flops_per_byte")
    weight_fraction = theoretical.get("weight_fraction_of_compulsory_bytes", 0.0)
    compulsory_bytes = theoretical.get("compulsory_bytes")
    ridge_point = None
    if peak_hbm_gbps and peak_compute_tflops:
        ridge_point = peak_compute_tflops * 1000.0 / peak_hbm_gbps
    below_ridge = (
        arithmetic_intensity is not None
        and ridge_point is not None
        and arithmetic_intensity < ridge_point
    )
    measured_to_compulsory = (
        measured_dram_bytes / compulsory_bytes
        if measured_dram_bytes is not None and compulsory_bytes else None
    )
    weight_dominated = weight_fraction >= 0.80
    dram_saturated = weighted_dram >= min_dram_throughput_pct
    measured_traffic_present = (
        measured_to_compulsory is not None and measured_to_compulsory >= 0.50
    )
    memory_pressure_dominates_sm = (
        weighted_sm is not None and weighted_dram > weighted_sm
    )
    clean = idle_at_start is True
    strict_weight_bound = bool(
        clean
        and weight_dominated
        and below_ridge
        and dram_saturated
        and measured_traffic_present
        and memory_pressure_dominates_sm
    )

    if strict_weight_bound:
        classification = "confirmed_weight_bandwidth_bound"
    elif weight_dominated and below_ridge and clean:
        classification = "weight_stream_dominated_but_not_strictly_confirmed"
    elif stage == "attention" and below_ridge and clean:
        classification = "kv_memory_bound_or_bandwidth_underfilled"
    elif idle_at_start is False:
        classification = "invalid_external_gpu_load"
    else:
        classification = "insufficient_evidence"
    return {
        "classification": classification,
        "strict_weight_bound_confirmed": strict_weight_bound,
        "idle_at_start": idle_at_start,
        "weight_dominated_compulsory_traffic": weight_dominated,
        "dram_saturated": dram_saturated,
        "measured_traffic_present": measured_traffic_present,
        "memory_pressure_dominates_sm": memory_pressure_dominates_sm,
        "min_dram_throughput_pct": min_dram_throughput_pct,
        "arithmetic_intensity_flops_per_byte": arithmetic_intensity,
        "ridge_point_flops_per_byte": ridge_point,
        "below_roofline_ridge": below_ridge,
        "weight_fraction_of_compulsory_bytes": weight_fraction,
        "measured_to_compulsory_dram_byte_ratio": measured_to_compulsory,
        "duration_weighted_sm_throughput_pct": weighted_sm,
        "evidence_rule": (
            "Strict confirmation requires an idle launch, >=80% theoretical compulsory bytes "
            "from weights, arithmetic intensity below the hardware ridge point, measured DRAM "
            "traffic >=50% of modeled compulsory traffic, DRAM throughput above the configured "
            "threshold, and DRAM pressure greater than SM throughput."
        ),
    }


def summarize(
    rows: list[dict[str, str]],
    peak_hbm_gbps: float | None,
    *,
    stage: str | None = None,
    manifest: dict | None = None,
    peak_compute_tflops: float | None = None,
    min_dram_throughput_pct: float = 50.0,
) -> dict:
    kernels: dict[tuple[str, str], dict] = defaultdict(lambda: {"metrics": {}})
    for row in rows:
        key = (row["Process ID"], row["ID"])
        kernel = kernels[key]
        kernel["kernel_name"] = row["Kernel Name"]
        try:
            value = float(row["Metric Value"].replace(",", ""))
        except ValueError:
            continue
        kernel["metrics"][row["Metric Name"]] = value

    complete = []
    for kernel in kernels.values():
        metrics = kernel["metrics"]
        if all(metric in metrics for metric in (SM_ACTIVE, DRAM_THROUGHPUT, DURATION)):
            complete.append({
                "kernel_name": kernel["kernel_name"],
                "duration_ns": metrics[DURATION],
                "sm_active_pct": metrics[SM_ACTIVE],
                "dram_throughput_pct": metrics[DRAM_THROUGHPUT],
                "sm_throughput_pct": metrics.get(SM_THROUGHPUT),
                "dram_read_bytes": metrics.get(DRAM_READ_BYTES),
                "dram_write_bytes": metrics.get(DRAM_WRITE_BYTES),
            })
    if not complete:
        raise ValueError("no kernels contain all requested NCU metrics")
    total_duration = sum(kernel["duration_ns"] for kernel in complete)

    def weighted(field: str) -> float:
        return sum(kernel[field] * kernel["duration_ns"] for kernel in complete) / total_duration

    def weighted_optional(field: str) -> float | None:
        available = [kernel for kernel in complete if kernel[field] is not None]
        if not available:
            return None
        duration = sum(kernel["duration_ns"] for kernel in available)
        return sum(kernel[field] * kernel["duration_ns"] for kernel in available) / duration

    def summed_optional(field: str) -> float | None:
        values = [kernel[field] for kernel in complete if kernel[field] is not None]
        return sum(values) if values else None

    weighted_dram = weighted("dram_throughput_pct")
    result = {
        "kernel_count": len(complete),
        "total_profiled_kernel_time_ms": total_duration / 1e6,
        "duration_weighted_sm_active_pct": weighted("sm_active_pct"),
        "duration_weighted_sm_throughput_pct": weighted_optional("sm_throughput_pct"),
        "duration_weighted_dram_throughput_pct": weighted_dram,
        "dram_read_bytes": summed_optional("dram_read_bytes"),
        "dram_write_bytes": summed_optional("dram_write_bytes"),
        "estimated_duration_weighted_dram_gbps": (
            weighted_dram / 100.0 * peak_hbm_gbps if peak_hbm_gbps else None
        ),
        "peak_hbm_gbps": peak_hbm_gbps,
        "top_kernels_by_duration": sorted(
            complete,
            key=lambda kernel: kernel["duration_ns"],
            reverse=True,
        )[:20],
        "method_note": (
            "NCU metrics are weighted by kernel duration. Nsight replay perturbs latency; "
            "use the steady-state benchmark, not this report, for TTFT/TPOT."
        ),
    }
    if stage is not None:
        if manifest is None:
            raise ValueError("a stage manifest is required when --stage is used")
        theoretical = manifest.get("stages", {}).get(stage)
        if theoretical is None:
            raise ValueError(f"stage {stage!r} is absent from the manifest")
        measured_dram_bytes = None
        if result["dram_read_bytes"] is not None and result["dram_write_bytes"] is not None:
            measured_dram_bytes = result["dram_read_bytes"] + result["dram_write_bytes"]
        result["stage"] = stage
        result["theoretical"] = theoretical
        result["measured_dram_bytes"] = measured_dram_bytes
        result["bottleneck"] = _classify_stage(
            stage=stage,
            theoretical=theoretical,
            weighted_dram=weighted_dram,
            weighted_sm=result["duration_weighted_sm_throughput_pct"],
            measured_dram_bytes=measured_dram_bytes,
            peak_hbm_gbps=peak_hbm_gbps,
            peak_compute_tflops=peak_compute_tflops,
            idle_at_start=manifest.get("idle_at_start"),
            min_dram_throughput_pct=min_dram_throughput_pct,
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--peak_hbm_gbps", type=float, default=0.0)
    parser.add_argument("--peak_compute_tflops", type=float, default=0.0)
    parser.add_argument("--stage", choices=("qkv", "attention", "o_proj", "mlp", "lm_head"))
    parser.add_argument("--manifest", default="")
    parser.add_argument("--min_dram_throughput_pct", type=float, default=50.0)
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text()) if args.manifest else None
    result = summarize(
        read_ncu_csv(Path(args.input_csv)),
        args.peak_hbm_gbps or None,
        stage=args.stage,
        manifest=manifest,
        peak_compute_tflops=args.peak_compute_tflops or None,
        min_dram_throughput_pct=args.min_dram_throughput_pct,
    )
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
