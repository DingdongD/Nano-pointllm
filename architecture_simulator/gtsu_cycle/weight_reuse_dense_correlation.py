"""Icarus/Yosys lock for K-block-resident high-M Dense64 mode."""
from __future__ import annotations

import csv
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import tempfile
from typing import Any

from .weight_reuse_dense import (
    WeightReuseDenseConfig, WeightReuseEvent, repeated_gemv_weight_bytes,
    run_weight_reuse_dense_model, weight_reuse_bytes,
)


def correlate_weight_reuse_dense(
    config: WeightReuseDenseConfig, *, rtl_root: str | Path,
    output_dir: str | Path | None = None, synthesize: bool = True,
) -> dict[str, Any]:
    model = run_weight_reuse_dense_model(config)
    root = Path(rtl_root)
    sources = (
        root / "gtsu_dot4_pe.sv",
        root / "gtsu_dense_dot4_tile.sv",
        root / "gtsu_rv_fifo.sv",
        root / "gtsu_dense64_weight_reuse_fabric.sv",
        root / "tb_gtsu_dense64_weight_reuse_fabric.sv",
    )
    parameters = {
        "M": config.m, "N": config.n, "K": config.k,
        "M_TILE": config.m_tile, "K_BLOCK": config.k_block,
        "N_MEM_LANES": config.memory_lanes,
        "READ_LATENCY": config.read_latency, "FIFO_DEPTH": config.fifo_depth,
        "WEIGHT_STALL_MOD": config.weight_stall_mod,
        "WEIGHT_STALL_PHASE": config.weight_stall_phase,
        "ACTIVATION_STALL_MOD": config.activation_stall_mod,
        "ACTIVATION_STALL_PHASE": config.activation_stall_phase,
        "OUTPUT_STALL_MOD": config.output_stall_mod,
        "OUTPUT_STALL_PHASE": config.output_stall_phase,
        "MAX_CYCLES": config.max_cycles, "TRACE_INTERNAL": 1,
    }
    with tempfile.TemporaryDirectory(prefix="gtsu_weight_reuse_") as name:
        temporary = Path(name)
        executable = temporary / "weight_reuse.vvp"
        command = [
            "iverilog", "-g2012", "-s", "tb_gtsu_dense64_weight_reuse_fabric",
        ]
        command.extend(
            f"-Ptb_gtsu_dense64_weight_reuse_fabric.{key}={value}"
            for key, value in parameters.items()
        )
        command.extend(["-o", str(executable), *(str(path) for path in sources)])
        compiled = subprocess.run(command, capture_output=True, text=True)
        if compiled.returncode:
            raise RuntimeError(f"weight-reuse RTL compile failed:\n{compiled.stderr}")
        simulated = subprocess.run(
            ["vvp", str(executable)], capture_output=True, text=True,
        )
        if simulated.returncode:
            raise RuntimeError(
                f"weight-reuse RTL failed:\n{simulated.stdout}\n{simulated.stderr}"
            )
        synthesis = _yosys(sources[:-1], config, temporary) if synthesize else {
            "status": "not_run",
        }

    rtl_events, rtl_summary = parse_weight_reuse_output(simulated.stdout)
    expected_summary = {
        "cycles": model.cycles, **model.counters,
        "overflow_error": 0, "done": 1,
    }
    exact = model.events == rtl_events and expected_summary == rtl_summary
    naive_bytes = repeated_gemv_weight_bytes(config)
    reuse_bytes = weight_reuse_bytes(config)
    report = {
        "schema_version": "0.1",
        "status": "rtl_correlated" if exact else "mismatch",
        "operator": "dense64_kblock_resident_weight_reuse",
        "config": config.as_dict(),
        "cycle_model_cycles": model.cycles,
        "rtl_cycles": rtl_summary["cycles"],
        "cycle_error": rtl_summary["cycles"] - model.cycles,
        "event_trace_exact": model.events == rtl_events,
        "counter_trace_exact": expected_summary == rtl_summary,
        "functional_outputs_exact": _outputs(model.events) == _outputs(rtl_events),
        "counters": model.counters,
        "traffic": {
            "repeated_gemv_weight_bytes": naive_bytes,
            "weight_reuse_bytes": reuse_bytes,
            "weight_byte_reduction_ratio": naive_bytes / reuse_bytes,
            "weight_bytes_saved": naive_bytes - reuse_bytes,
            "activation_bytes": config.activation_chunks * 4,
            "wbuf_read_bytes": config.activation_chunks * 256,
        },
        "utilization": {
            "dot4_issue_cycles": model.counters["dot4_chunks"],
            "total_cycles": model.cycles,
            "dot4_issue_fraction": model.counters["dot4_chunks"] / model.cycles,
        },
        "synthesis_check": synthesis,
        "claim_boundary": {
            "kblock_weight_residency": exact,
            "tagged_partial_sum_state": exact,
            "high_m_production_shape": False,
            "real_pointtransformer_payload": False,
            "request_side_dramsim3_closed_loop": False,
            "standalone_report_unlocks_dense_gemm_coverage": False,
            "full_pointllm_cycle_accurate": False,
        },
    }
    if output_dir is not None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        _write_events(destination / "cycle_model_trace.csv", model.events)
        _write_events(destination / "rtl_trace.csv", rtl_events)
        (destination / "rtl_stdout.log").write_text(simulated.stdout, encoding="utf-8")
        if "log" in synthesis:
            (destination / "yosys_check.log").write_text(
                synthesis["log"], encoding="utf-8",
            )
        (destination / "correlation.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8",
        )
    if not exact:
        raise AssertionError(f"weight-reuse Dense mismatch: {report}")
    return report


def parse_weight_reuse_output(
    text: str,
) -> tuple[tuple[WeightReuseEvent, ...], dict[str, int]]:
    events = []
    summary = None
    for line in text.splitlines():
        fields = line.split()
        if fields[:1] == ["TRACE"] and len(fields) == 5:
            events.append(WeightReuseEvent(
                int(fields[1]), fields[2], int(fields[3]), int(fields[4]),
            ))
        elif fields[:1] == ["SUMMARY"] and len(fields) == 18:
            keys = (
                "cycles", "weight_lines", "wbuf_bank_writes", "wbuf_load_tiles",
                "activation_chunks", "wbuf_read_issues", "wbuf_responses",
                "dot4_chunks", "partial_tiles", "output_tiles", "output_values",
                "weight_backpressure_cycles", "activation_backpressure_cycles",
                "output_backpressure_cycles", "response_fifo_peak",
                "overflow_error", "done",
            )
            summary = dict(zip(keys, map(int, fields[1:]), strict=True))
    if summary is None:
        raise RuntimeError(f"weight-reuse RTL did not emit SUMMARY:\n{text[-4000:]}")
    return tuple(events), summary


def _outputs(events):
    return [(event.tag, event.value) for event in events if event.event == "OUTPUT_ACCEPT"]


def _write_events(path: Path, events) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("cycle", "event", "tag", "value"),
            lineterminator="\n",
        )
        writer.writeheader()
        for event in events:
            writer.writerow(asdict(event))


def _yosys(sources, config, temporary) -> dict[str, Any]:
    path = temporary / "weight_reuse.json"
    top = "gtsu_dense64_weight_reuse_fabric"
    command = (
        f"read_verilog -sv {' '.join(str(source) for source in sources)}; "
        "chparam -set M 1 -set N 64 -set K 4 "
        "-set M_TILE 1 -set K_BLOCK 4 -set N_MEM_LANES 4 "
        f"-set READ_LATENCY 1 -set FIFO_DEPTH 2 {top}; "
        f"hierarchy -check -top {top}; proc; opt; check -assert; stat; write_json {path}"
    )
    result = subprocess.run(["yosys", "-p", command], capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"weight-reuse Yosys failed:\n{result.stdout}\n{result.stderr}")
    modules = json.loads(path.read_text())["modules"]
    top_cells = modules[top].get("cells", {}).values()
    dense_instances = sum(
        cell["type"].startswith("$paramod") and "gtsu_dense_dot4_tile" in cell["type"]
        for cell in top_cells
    )
    direct_multipliers = sum(cell["type"] == "$mul" for cell in top_cells)
    dense_modules = [
        module for name, module in modules.items() if "gtsu_dense_dot4_tile" in name
    ]
    dot_instances = {
        sum(cell["type"] == "gtsu_dot4_pe" for cell in module.get("cells", {}).values())
        for module in dense_modules
    }
    if dense_instances != 1 or dot_instances != {64} or direct_multipliers != 0:
        raise AssertionError("weight-reuse top does not preserve exactly 64 shared Dot4 PEs")
    return {
        "status": "passed", "dense_tile_instances": dense_instances,
        "dot4_instances": 64, "top_direct_multipliers": direct_multipliers,
        "wbuf_banks": 16, "wbuf_resident_k_chunks": config.block_chunks,
        "partial_sum_rows": config.m_tile,
        "capacity_reduced_structure_lock": {
            "m": 1, "n": 64, "k": 4, "m_tile": 1, "k_block": 4,
            "reason": "avoid treating behavioral SRAM register expansion as PPA",
        },
        "behavioral_storage_not_foundry_macro": True,
        "log": result.stdout + result.stderr,
    }
