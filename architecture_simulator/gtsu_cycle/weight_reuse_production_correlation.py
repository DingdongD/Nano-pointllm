"""Compiled production-shape validation for high-M Dense64 reuse mode."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import tempfile
from typing import Any

from .weight_reuse_dense import (
    WeightReuseDenseConfig, repeated_gemv_weight_bytes,
    run_weight_reuse_dense_model, weight_reuse_bytes,
)


def correlate_pointtransformer_weight_reuse(
    *, rtl_root: str | Path, output_dir: str | Path | None = None,
    config: WeightReuseDenseConfig | None = None,
) -> dict[str, Any]:
    config = config or WeightReuseDenseConfig(
        m=513, n=1152, k=384, m_tile=32, k_block=64,
        memory_lanes=4, weight_stall_mod=0, activation_stall_mod=0,
        output_stall_mod=0, max_cycles=2_000_000,
    )
    model = run_weight_reuse_dense_model(
        config, trace=False, compute_values=False,
    )
    root = Path(rtl_root)
    sources = (
        root / "gtsu_dot4_pe.sv",
        root / "gtsu_dense_dot4_tile.sv",
        root / "gtsu_rv_fifo.sv",
        root / "gtsu_dense64_weight_reuse_fabric.sv",
    )
    with tempfile.TemporaryDirectory(prefix="gtsu_weight_reuse_prod_") as name:
        build_dir = Path(name) / "obj_dir"
        command = [
            "verilator", "--cc", "--exe", "-Wno-fatal",
            "--top-module", "gtsu_dense64_weight_reuse_fabric",
            "--Mdir", str(build_dir),
            f"-GM={config.m}", f"-GN={config.n}", f"-GK={config.k}",
            f"-GM_TILE={config.m_tile}", f"-GK_BLOCK={config.k_block}",
            f"-GN_MEM_LANES={config.memory_lanes}",
            f"-GREAD_LATENCY={config.read_latency}",
            f"-GFIFO_DEPTH={config.fifo_depth}",
            *(str(path.resolve()) for path in sources),
            str((root / "verilator_weight_reuse_driver.cpp").resolve()),
        ]
        compiled = subprocess.run(command, capture_output=True, text=True)
        if compiled.returncode:
            raise RuntimeError(f"weight-reuse Verilator compile failed:\n{compiled.stderr}")
        built = subprocess.run(
            ["make", "-C", str(build_dir), "-f",
             "Vgtsu_dense64_weight_reuse_fabric.mk", "-j2"],
            capture_output=True, text=True,
        )
        if built.returncode:
            raise RuntimeError(f"weight-reuse Verilator build failed:\n{built.stderr}")
        executable = build_dir / "Vgtsu_dense64_weight_reuse_fabric"
        simulated = subprocess.run([
            str(executable), str(config.m), str(config.n), str(config.k),
            str(config.m_tile), str(config.k_block), str(config.memory_lanes),
            str(config.max_cycles), str(config.output_tiles),
        ], capture_output=True, text=True)
        if simulated.returncode:
            raise RuntimeError(
                f"production weight-reuse RTL failed:\n"
                f"{simulated.stdout}\n{simulated.stderr}"
            )
    summary = _parse_summary(simulated.stdout)
    expected_counters = {
        key: model.counters[key] for key in (
            "weight_lines", "wbuf_bank_writes", "wbuf_load_tiles",
            "activation_chunks", "wbuf_read_issues", "wbuf_responses",
            "dot4_chunks", "partial_tiles", "output_tiles", "output_values",
            "response_fifo_peak",
        )
    }
    counter_exact = all(summary[key] == value for key, value in expected_counters.items())
    exact = (
        summary["cycles"] == model.cycles and counter_exact
        and summary["mismatches"] == 0 and summary["done"] == 1
    )
    naive_bytes = repeated_gemv_weight_bytes(config)
    reuse_bytes = weight_reuse_bytes(config)
    report = {
        "schema_version": "0.1",
        "status": "rtl_correlated" if exact else "mismatch",
        "operator": "pointtransformer_qkv_dense64_weight_reuse",
        "config": config.as_dict(),
        "cycle_model_cycles": model.cycles,
        "rtl_cycles": summary["cycles"],
        "cycle_error": summary["cycles"] - model.cycles,
        "counter_trace_exact": counter_exact,
        "functional_outputs_exact": summary["mismatches"] == 0,
        "checked_output_values": summary["output_values"],
        "output_fnv1a64": str(summary["output_hash"]),
        "counters": expected_counters,
        "traffic": {
            "repeated_gemv_weight_bytes": naive_bytes,
            "weight_reuse_bytes": reuse_bytes,
            "weight_byte_reduction_ratio": naive_bytes / reuse_bytes,
            "unique_padded_weight_bytes": config.n_tiles * config.chunks * 256,
            "resident_reload_factor": config.m_tiles,
        },
        "utilization": {
            "dot4_issue_fraction": model.counters["dot4_chunks"] / model.cycles,
            "dot4_pe_count": 64,
        },
        "claim_boundary": {
            "pointtransformer_shape_rtl": exact,
            "all_590976_acc32_outputs_checked": summary["mismatches"] == 0,
            "real_pointtransformer_payload": False,
            "request_side_dramsim3_closed_loop": False,
            "full_pointllm_cycle_accurate": False,
        },
    }
    if output_dir is not None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "verilator_stdout.log").write_text(
            simulated.stdout, encoding="utf-8",
        )
        (destination / "correlation.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8",
        )
    if not exact:
        raise AssertionError(f"production weight-reuse mismatch: {report}")
    return report


def _parse_summary(text: str) -> dict[str, int]:
    for line in text.splitlines():
        fields = line.split()
        if fields[:1] == ["SUMMARY"] and len(fields) == 16:
            keys = (
                "cycles", "weight_lines", "wbuf_bank_writes", "wbuf_load_tiles",
                "activation_chunks", "wbuf_read_issues", "wbuf_responses",
                "dot4_chunks", "partial_tiles", "output_tiles", "output_values",
                "response_fifo_peak", "mismatches", "output_hash", "done",
            )
            return dict(zip(keys, map(int, fields[1:]), strict=True))
    raise RuntimeError(f"production weight-reuse RTL did not emit SUMMARY:\n{text[-4000:]}")
