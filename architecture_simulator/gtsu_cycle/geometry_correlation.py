"""Icarus/Yosys correlation for the signed 3-D geometry distance tile."""
from __future__ import annotations

import csv
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any

from .geometry import (
    GeometryBeat,
    GeometryConfig,
    GeometryEvent,
    GeometryOutput,
    build_geometry_beats,
    pointllm_geometry_mapping,
    run_geometry_model,
)


COUNTER_NAMES = (
    "input_tiles",
    "output_points",
    "source_backpressure_cycles",
    "output_backpressure_cycles",
)


def correlate_geometry_distance(
    config: GeometryConfig,
    *,
    rtl_root: str | Path,
    output_dir: str | Path | None = None,
    beats: tuple[GeometryBeat, ...] | None = None,
) -> dict[str, Any]:
    config.validate()
    beats = beats or build_geometry_beats(config)
    model = run_geometry_model(config, beats)
    iverilog = shutil.which("iverilog")
    vvp = shutil.which("vvp")
    if not iverilog or not vvp:
        raise RuntimeError("Icarus Verilog is required for geometry RTL correlation")
    root = Path(rtl_root)
    sources = (
        root / "gtsu_geometry_distance_tile.sv",
        root / "tb_gtsu_geometry_distance_tile.sv",
    )
    missing = [str(source) for source in sources if not source.is_file()]
    if missing:
        raise FileNotFoundError(f"missing geometry RTL sources: {missing}")

    with tempfile.TemporaryDirectory(prefix="gtsu_geometry_") as temporary_name:
        temporary = Path(temporary_name)
        trace_path = temporary / "geometry_trace.hex"
        write_geometry_trace(trace_path, config, beats)
        executable = temporary / "geometry.vvp"
        command = [iverilog, "-g2012", "-s", "tb_gtsu_geometry_distance_tile"]
        for name, value in {
            "LANES": config.lanes,
            "COORD_WIDTH": config.coordinate_width,
            "DIST_WIDTH": config.distance_width,
            "TAG_WIDTH": config.tag_width,
            "TILES": config.tiles,
            "SOURCE_STALL_MOD": config.source_stall_mod,
            "SOURCE_STALL_PHASE": config.source_stall_phase,
            "OUTPUT_STALL_MOD": config.output_stall_mod,
            "OUTPUT_STALL_PHASE": config.output_stall_phase,
            "MAX_CYCLES": config.max_cycles,
        }.items():
            command.append(f"-Ptb_gtsu_geometry_distance_tile.{name}={value}")
        command.extend(["-o", str(executable), *(str(source) for source in sources)])
        compiled = subprocess.run(command, check=False, capture_output=True, text=True)
        if compiled.returncode:
            raise RuntimeError(
                f"geometry RTL compile failed:\n{compiled.stdout}\n{compiled.stderr}"
            )
        simulated = subprocess.run(
            [vvp, str(executable), f"+TRACE_FILE={trace_path}"],
            check=False, capture_output=True, text=True,
        )
        if simulated.returncode:
            raise RuntimeError(
                f"geometry RTL simulation failed:\n{simulated.stdout}\n{simulated.stderr}"
            )

    rtl_events, rtl_summary = parse_geometry_rtl_output(simulated.stdout)
    model_events = tuple(sorted(model.events, key=_event_key))
    rtl_events = tuple(sorted(rtl_events, key=_event_key))
    event_exact = model_events == rtl_events
    expected_summary = {"cycles": model.cycles, **model.counters}
    counter_exact = expected_summary == rtl_summary
    rtl_outputs = tuple(
        GeometryOutput(event.tag, event.lane, event.value)
        for event in rtl_events if event.event == "OUTPUT_ACCEPT"
    )
    output_exact = rtl_outputs == model.outputs
    synthesis = _run_yosys_check(sources[0], config)
    correlated = event_exact and counter_exact and output_exact
    report: dict[str, Any] = {
        "schema_version": "0.1",
        "status": "rtl_correlated" if correlated else "mismatch",
        "operator": "geometry_distance_tile",
        "config": config.as_dict(),
        "cycle_model_cycles": model.cycles,
        "rtl_cycles": rtl_summary["cycles"],
        "cycle_error": rtl_summary["cycles"] - model.cycles,
        "event_trace_exact": event_exact,
        "counter_trace_exact": counter_exact,
        "functional_output_exact": output_exact,
        "cycle_model_counters": model.counters,
        "rtl_counters": {
            key: rtl_summary[key] for key in COUNTER_NAMES
        },
        "event_mismatches": _mismatches(model_events, rtl_events),
        "source_contract": {
            "repository": "https://github.com/DingdongD/PointKAN_Accel",
            "commit": "aa356da232a9d30d6c784297603c2a5596c6409a",
            "references": [
                "hw_sim/rtl/dist_engine_rtl/distance_engine.v",
                "hw_sim/rtl/RTL/fps_min_dits2_table_128.v",
            ],
            "adaptation": (
                "clean-room streaming contract with explicit ready/valid, "
                "output backpressure, widened signed subtraction, and exact trace checks"
            ),
        },
        "pointllm_mapping": pointllm_geometry_mapping(config.lanes),
        "claim_boundary": {
            "distance_datapath_rtl_correlated": correlated,
            "pointllm_fp32_to_int16_quantization_validated": False,
            "sram_or_dram_integrated": False,
            "fps_min_argmax_feedback_complete": False,
            "knn_global_topk_complete": False,
            "full_pointllm_cycle_accurate": False,
        },
        "synthesis_check": synthesis,
    }
    if output_dir is not None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        write_geometry_trace(destination / "geometry_trace.hex", config, beats)
        _write_events(destination / "cycle_model_trace.csv", model_events)
        _write_events(destination / "rtl_trace.csv", rtl_events)
        (destination / "rtl_stdout.log").write_text(simulated.stdout, encoding="utf-8")
        (destination / "yosys_check.log").write_text(
            str(synthesis["log"]), encoding="utf-8",
        )
        (destination / "correlation.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8",
        )
    if not correlated:
        raise AssertionError(f"geometry distance RTL mismatch: {report}")
    return report


def write_geometry_trace(
    path: str | Path,
    config: GeometryConfig,
    beats: tuple[GeometryBeat, ...],
) -> None:
    path = Path(path)
    with path.open("w", encoding="ascii") as handle:
        for beat in beats:
            value = beat.tag
            shift = config.tag_width
            for coordinate in (*beat.query, *(v for point in beat.points for v in point)):
                value |= _encode_signed(coordinate, config.coordinate_width) << shift
                shift += config.coordinate_width
            handle.write(f"{value:0{(shift + 3) // 4}x}\n")


def parse_geometry_rtl_output(
    text: str,
) -> tuple[tuple[GeometryEvent, ...], dict[str, int]]:
    events: list[GeometryEvent] = []
    summary: dict[str, int] | None = None
    for raw_line in text.splitlines():
        fields = raw_line.strip().split()
        if fields[:1] == ["TRACE"] and len(fields) == 6:
            events.append(GeometryEvent(
                cycle=int(fields[1]), event=fields[2], tag=int(fields[3]),
                lane=int(fields[4]), value=int(fields[5]),
            ))
        elif fields[:1] == ["SUMMARY"] and len(fields) == 6:
            values = [int(value) for value in fields[1:]]
            summary = dict(zip(("cycles", *COUNTER_NAMES), values, strict=True))
    if summary is None:
        raise RuntimeError(f"geometry RTL did not emit SUMMARY:\n{text[-4000:]}")
    return tuple(events), summary


def _encode_signed(value: int, width: int) -> int:
    lower = -(1 << (width - 1))
    upper = (1 << (width - 1)) - 1
    if not lower <= value <= upper:
        raise ValueError(f"signed value {value} does not fit in {width} bits")
    return value & ((1 << width) - 1)


def _event_key(event: GeometryEvent) -> tuple[int, int, int, int, int]:
    order = 0 if event.event == "INPUT_ACCEPT" else 1
    return event.cycle, order, event.tag, event.lane, event.value


def _mismatches(
    expected: tuple[GeometryEvent, ...],
    actual: tuple[GeometryEvent, ...],
    limit: int = 32,
) -> list[dict[str, Any]]:
    rows = []
    for index in range(max(len(expected), len(actual))):
        left = asdict(expected[index]) if index < len(expected) else None
        right = asdict(actual[index]) if index < len(actual) else None
        if left != right:
            rows.append({"index": index, "cycle_model": left, "rtl": right})
        if len(rows) >= limit:
            break
    return rows


def _write_events(path: Path, events: tuple[GeometryEvent, ...]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("cycle", "event", "tag", "lane", "value"),
            lineterminator="\n",
        )
        writer.writeheader()
        for event in events:
            writer.writerow(asdict(event))


def _run_yosys_check(source: Path, config: GeometryConfig) -> dict[str, Any]:
    yosys = shutil.which("yosys")
    if not yosys:
        return {"status": "unavailable", "version": "unavailable", "log": ""}
    command = (
        f"read_verilog -sv {source}; "
        "hierarchy -check -top gtsu_geometry_distance_tile; "
        f"chparam -set LANES {config.lanes} "
        f"-set COORD_WIDTH {config.coordinate_width} "
        f"-set DIST_WIDTH {config.distance_width} "
        f"-set TAG_WIDTH {config.tag_width} gtsu_geometry_distance_tile; "
        "hierarchy -check -top gtsu_geometry_distance_tile; "
        "proc; memory; opt; check -assert; stat"
    )
    result = subprocess.run(
        [yosys, "-p", command], check=False, capture_output=True, text=True,
    )
    if result.returncode:
        raise RuntimeError(
            f"geometry Yosys structural check failed:\n{result.stdout}\n{result.stderr}"
        )
    version = subprocess.run(
        [yosys, "-V"], check=False, capture_output=True, text=True,
    )
    version_line = next(iter(version.stdout.splitlines()), "unknown")
    return {
        "status": "passed",
        "version": version_line,
        "parameters": {
            "lanes": config.lanes,
            "coordinate_width": config.coordinate_width,
            "distance_width": config.distance_width,
            "tag_width": config.tag_width,
        },
        "log": result.stdout,
    }
