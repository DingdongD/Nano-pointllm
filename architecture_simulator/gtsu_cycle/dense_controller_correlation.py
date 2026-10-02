"""Icarus/Yosys correlation for the Dense GEMM M/N/K ping-pong controller."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import tempfile
from typing import Any

from .dense_controller import (
    DenseControllerConfig, DenseControllerEvent, run_dense_controller_model,
)


def correlate_dense_controller(
    config: DenseControllerConfig, *, rtl_root: str | Path,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    model = run_dense_controller_model(config)
    root = Path(rtl_root)
    sources = (
        root / "gtsu_dense_mnk_controller.sv",
        root / "tb_gtsu_dense_mnk_controller.sv",
    )
    with tempfile.TemporaryDirectory(prefix="gtsu_dense_ctrl_") as temporary_name:
        temporary = Path(temporary_name)
        executable = temporary / "controller.vvp"
        command = ["iverilog", "-g2012", "-s", "tb_gtsu_dense_mnk_controller"]
        for name, value in {
            "M": config.m, "N": config.n, "K": config.k,
            "N_TILE": config.n_tile, "K_BLOCK": config.k_block,
            "SOURCE_STALL_MOD": config.source_stall_mod,
            "SOURCE_STALL_PHASE": config.source_stall_phase,
            "COMPUTE_STALL_MOD": config.compute_stall_mod,
            "COMPUTE_STALL_PHASE": config.compute_stall_phase,
            "MAX_CYCLES": config.max_cycles,
        }.items():
            command.append(f"-Ptb_gtsu_dense_mnk_controller.{name}={value}")
        command.extend(["-o", str(executable), *(str(path) for path in sources)])
        compiled = subprocess.run(command, capture_output=True, text=True)
        if compiled.returncode:
            raise RuntimeError(f"dense controller RTL compile failed:\n{compiled.stderr}")
        simulated = subprocess.run(["vvp", str(executable)], capture_output=True, text=True)
        if simulated.returncode:
            raise RuntimeError(
                f"dense controller RTL failed:\n{simulated.stdout}\n{simulated.stderr}"
            )
        synthesis = _yosys(sources[0], config, temporary)
    rtl_events, rtl_summary = parse_controller_output(simulated.stdout, config)
    expected_summary = {"cycles": model.cycles, **model.counters}
    exact = model.events == rtl_events and expected_summary == rtl_summary
    report = {
        "schema_version": "0.1",
        "status": "rtl_correlated" if exact else "mismatch",
        "operator": "dense_mnk_ping_pong_controller",
        "config": config.as_dict(),
        "cycle_model_cycles": model.cycles,
        "rtl_cycles": rtl_summary["cycles"],
        "cycle_error": rtl_summary["cycles"] - model.cycles,
        "event_trace_exact": model.events == rtl_events,
        "counter_trace_exact": expected_summary == rtl_summary,
        "counters": model.counters,
        "synthesis_check": synthesis,
        "claim_boundary": {
            "mnk_counter_and_ping_pong_lifecycle_exact": exact,
            "dense_dot4_datapath_correlated_separately": True,
            "physical_banked_sram_integrated": False,
            "dma_payload_integrated": False,
            "production_shape_full_trace_correlated": False,
        },
    }
    if output_dir is not None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "rtl_stdout.log").write_text(simulated.stdout, encoding="utf-8")
        (destination / "yosys_check.log").write_text(synthesis["log"], encoding="utf-8")
        (destination / "correlation.json").write_text(json.dumps(report, indent=2) + "\n")
    if not exact:
        raise AssertionError(f"dense controller RTL mismatch: {report}")
    return report


def parse_controller_output(
    text: str, config: DenseControllerConfig,
) -> tuple[tuple[DenseControllerEvent, ...], dict[str, int]]:
    events = []
    summary = None
    for line in text.splitlines():
        fields = line.split()
        if fields[:1] == ["TRACE"] and len(fields) == 11:
            mask = int(fields[8], 16)
            events.append(DenseControllerEvent(
                int(fields[1]), fields[2], int(fields[3]), int(fields[4]),
                int(fields[5]), int(fields[6]), int(fields[7]), mask,
                int(fields[9]), int(fields[10]),
            ))
        elif fields[:1] == ["SUMMARY"] and len(fields) == 9:
            keys = ("cycles", "loads", "read_issues", "blocks_released",
                    "load_backpressure_cycles", "compute_wait_cycles",
                    "downstream_stall_cycles", "ping_pong_overlap_cycles")
            summary = dict(zip(keys, map(int, fields[1:]), strict=True))
    if summary is None:
        raise RuntimeError(f"dense controller did not emit SUMMARY:\n{text[-4000:]}")
    return tuple(events), summary


def _yosys(source: Path, config: DenseControllerConfig, temporary: Path) -> dict[str, Any]:
    path = temporary / "controller.json"
    command = (
        f"read_verilog -sv {source}; hierarchy -check -top gtsu_dense_mnk_controller; "
        f"chparam -set M {config.m} -set N {config.n} -set K {config.k} "
        f"-set N_TILE {config.n_tile} -set K_BLOCK {config.k_block} "
        "gtsu_dense_mnk_controller; hierarchy -check -top gtsu_dense_mnk_controller; "
        f"proc; opt; check -assert; stat; write_json {path}"
    )
    result = subprocess.run(["yosys", "-p", command], capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"dense controller Yosys failed:\n{result.stdout}\n{result.stderr}")
    cells = json.loads(path.read_text())["modules"]["gtsu_dense_mnk_controller"].get("cells", {}).values()
    forbidden = {"$mul", "$div", "$mod"}
    counts = {name: sum(cell["type"] == name for cell in cells) for name in forbidden}
    if any(counts.values()):
        raise AssertionError(f"dense controller contains arithmetic datapath cells: {counts}")
    return {"status": "passed", "forbidden_arithmetic_cells": counts,
            "log": result.stdout + result.stderr}
