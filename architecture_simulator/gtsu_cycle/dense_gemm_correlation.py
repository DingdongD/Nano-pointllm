"""Exact Icarus/Yosys correlation for the shared-Dot4 dense GEMM tile."""
from __future__ import annotations

import csv
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any

from .dense_gemm import (
    DenseBeat, DenseEvent, DenseTileConfig, build_dense_beats,
    pointllm_dense_shape_mapping, run_dense_tile_model,
)


COUNTERS = (
    "input_chunks", "output_rows", "output_values",
    "source_backpressure_cycles", "output_backpressure_cycles",
)


def correlate_dense_tile(
    config: DenseTileConfig,
    *,
    rtl_root: str | Path,
    output_dir: str | Path | None = None,
    beats: tuple[DenseBeat, ...] | None = None,
) -> dict[str, Any]:
    config.validate()
    beats = beats or build_dense_beats(config)
    model = run_dense_tile_model(config, beats)
    root = Path(rtl_root)
    sources = (
        root / "gtsu_dot4_pe.sv",
        root / "gtsu_dense_dot4_tile.sv",
        root / "tb_gtsu_dense_dot4_tile.sv",
    )
    with tempfile.TemporaryDirectory(prefix="gtsu_dense_") as temporary_name:
        temporary = Path(temporary_name)
        trace_path = temporary / "dense.hex"
        write_dense_trace(trace_path, config, beats)
        executable = temporary / "dense.vvp"
        command = ["iverilog", "-g2012", "-s", "tb_gtsu_dense_dot4_tile"]
        for name, value in {
            "N_TILE": config.n_tile, "ROWS": config.rows,
            "BEATS": len(beats), "VALID_COLUMNS": config.columns,
            "SOURCE_STALL_MOD": config.source_stall_mod,
            "SOURCE_STALL_PHASE": config.source_stall_phase,
            "OUTPUT_STALL_MOD": config.output_stall_mod,
            "OUTPUT_STALL_PHASE": config.output_stall_phase,
            "MAX_CYCLES": config.max_cycles,
        }.items():
            command.append(f"-Ptb_gtsu_dense_dot4_tile.{name}={value}")
        command.extend(["-o", str(executable), *(str(path) for path in sources)])
        compiled = subprocess.run(command, capture_output=True, text=True)
        if compiled.returncode:
            raise RuntimeError(f"dense RTL compile failed:\n{compiled.stdout}\n{compiled.stderr}")
        simulated = subprocess.run(
            ["vvp", str(executable), f"+TRACE_FILE={trace_path}"],
            capture_output=True, text=True,
        )
        if simulated.returncode:
            raise RuntimeError(f"dense RTL simulation failed:\n{simulated.stdout}\n{simulated.stderr}")
        synthesis = _yosys(sources[:2], config, temporary)

    events, summary = parse_dense_output(simulated.stdout)
    expected_summary = {"cycles": model.cycles, **model.counters}
    correlated = model.events == events and expected_summary == summary
    report = {
        "schema_version": "0.1",
        "status": "rtl_correlated" if correlated else "mismatch",
        "operator": "shared_dot4_dense_gemm_microtile",
        "config": config.as_dict(),
        "cycle_model_cycles": model.cycles,
        "rtl_cycles": summary["cycles"],
        "cycle_error": summary["cycles"] - model.cycles,
        "event_trace_exact": model.events == events,
        "counter_trace_exact": expected_summary == summary,
        "functional_output_exact": tuple(
            tuple(event.value for event in events if event.event == "OUTPUT_ACCEPT" and event.tag == row)
            for row in range(config.rows)
        ) == model.outputs,
        "counters": model.counters,
        "rtl_summary": summary,
        "synthesis_check": synthesis,
        "pointllm_shape_mapping": pointllm_dense_shape_mapping(64),
        "claim_boundary": {
            "microtile_value_event_cycle_exact": correlated,
            "production_shape_controller_correlated": False,
            "sram_double_buffer_integrated": False,
            "dense_gemm_coverage_enabled": False,
        },
    }
    if output_dir is not None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        write_dense_trace(destination / "dense_trace.hex", config, beats)
        _write_events(destination / "cycle_model_trace.csv", model.events)
        _write_events(destination / "rtl_trace.csv", events)
        (destination / "rtl_stdout.log").write_text(simulated.stdout, encoding="utf-8")
        (destination / "yosys_check.log").write_text(synthesis["log"], encoding="utf-8")
        (destination / "correlation.json").write_text(json.dumps(report, indent=2) + "\n")
    if not correlated:
        raise AssertionError(f"dense tile RTL mismatch: {report}")
    return report


def write_dense_trace(path: Path, config: DenseTileConfig, beats: tuple[DenseBeat, ...]) -> None:
    with path.open("w", encoding="ascii") as handle:
        for beat in beats:
            value = beat.tag
            shift = 16
            value |= int(beat.first) << shift
            value |= int(beat.last) << (shift + 1)
            shift += 2
            value |= beat.column_mask << shift
            shift += config.n_tile
            for item in beat.a:
                value |= (item & 0xFF) << shift
                shift += 8
            for column in beat.b:
                for item in column:
                    value |= (item & 0xFF) << shift
                    shift += 8
            handle.write(f"{value:0{(shift + 3)//4}x}\n")


def parse_dense_output(text: str) -> tuple[tuple[DenseEvent, ...], dict[str, int]]:
    events = []
    summary = None
    for line in text.splitlines():
        fields = line.split()
        if fields[:1] == ["TRACE"] and len(fields) == 6:
            events.append(DenseEvent(int(fields[1]), fields[2], int(fields[3]), int(fields[4]), int(fields[5])))
        elif fields[:1] == ["SUMMARY"] and len(fields) == 7:
            values = [int(item) for item in fields[1:]]
            summary = dict(zip(("cycles", *COUNTERS), values, strict=True))
    if summary is None:
        raise RuntimeError(f"dense RTL did not emit SUMMARY:\n{text[-4000:]}")
    return tuple(events), summary


def _write_events(path: Path, events: tuple[DenseEvent, ...]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("cycle", "event", "tag", "column", "value"), lineterminator="\n")
        writer.writeheader()
        for event in events:
            writer.writerow(asdict(event))


def _yosys(sources: tuple[Path, ...], config: DenseTileConfig, temporary: Path) -> dict[str, Any]:
    path = temporary / "dense.json"
    command = (
        f"read_verilog -sv {' '.join(str(item) for item in sources)}; "
        "hierarchy -check -top gtsu_dense_dot4_tile; "
        f"chparam -set N_TILE {config.n_tile} gtsu_dense_dot4_tile; "
        "hierarchy -check -top gtsu_dense_dot4_tile; proc; opt; check -assert; stat; "
        f"write_json {path}"
    )
    result = subprocess.run(["yosys", "-p", command], capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"dense Yosys failed:\n{result.stdout}\n{result.stderr}")
    modules = json.loads(path.read_text())["modules"]
    top = modules["gtsu_dense_dot4_tile"]
    cells = top.get("cells", {}).values()
    direct_mul = sum(cell["type"] == "$mul" for cell in cells)
    instances = sum(cell["type"] == "gtsu_dot4_pe" for cell in cells)
    dot_mul = sum(cell["type"] == "$mul" for cell in modules["gtsu_dot4_pe"].get("cells", {}).values())
    if direct_mul or instances != config.n_tile or dot_mul != 4:
        raise AssertionError("dense shared-multiplier hierarchy mismatch")
    return {
        "status": "passed", "top_direct_multipliers": direct_mul,
        "dot4_instances": instances, "dot4_multipliers_per_instance": dot_mul,
        "total_shared_int8_multipliers": instances * dot_mul,
        "log": result.stdout + result.stderr,
    }
