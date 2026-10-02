"""Icarus/Yosys correlation for shared feature-dot and geometry-dot RTL."""
from __future__ import annotations

import csv
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any

from .shared_dot import (
    SharedDotBeat,
    SharedDotConfig,
    SharedDotEvent,
    SharedDotOutput,
    build_shared_dot_beats,
    pointllm_shared_geometry_mapping,
    run_shared_dot_model,
)


COUNTER_NAMES = (
    "input_beats",
    "output_lanes",
    "feature_lanes",
    "geometry_lanes",
    "source_backpressure_cycles",
    "output_backpressure_cycles",
)


def correlate_shared_dot_geometry(
    config: SharedDotConfig,
    *,
    rtl_root: str | Path,
    output_dir: str | Path | None = None,
    beats: tuple[SharedDotBeat, ...] | None = None,
) -> dict[str, Any]:
    config.validate()
    beats = beats or build_shared_dot_beats(config)
    model = run_shared_dot_model(config, beats)
    iverilog = shutil.which("iverilog")
    vvp = shutil.which("vvp")
    if not iverilog or not vvp:
        raise RuntimeError("Icarus Verilog is required for shared-dot correlation")
    root = Path(rtl_root)
    sources = (
        root / "gtsu_dot4_pe.sv",
        root / "gtsu_shared_dot_geometry.sv",
        root / "tb_gtsu_shared_dot_geometry.sv",
    )
    missing = [str(source) for source in sources if not source.is_file()]
    if missing:
        raise FileNotFoundError(f"missing shared-dot RTL sources: {missing}")

    with tempfile.TemporaryDirectory(prefix="gtsu_shared_dot_") as temporary_name:
        temporary = Path(temporary_name)
        trace_path = temporary / "shared_dot_trace.hex"
        write_shared_dot_trace(trace_path, config, beats)
        executable = temporary / "shared_dot.vvp"
        command = ["iverilog", "-g2012", "-s", "tb_gtsu_shared_dot_geometry"]
        for name, value in {
            "PE_COUNT": config.pe_count,
            "NORM_WIDTH": config.norm_width,
            "DIST_WIDTH": config.distance_width,
            "TAG_WIDTH": config.tag_width,
            "BEATS": config.beats,
            "SOURCE_STALL_MOD": config.source_stall_mod,
            "SOURCE_STALL_PHASE": config.source_stall_phase,
            "OUTPUT_STALL_MOD": config.output_stall_mod,
            "OUTPUT_STALL_PHASE": config.output_stall_phase,
            "MAX_CYCLES": config.max_cycles,
        }.items():
            command.append(f"-Ptb_gtsu_shared_dot_geometry.{name}={value}")
        command.extend(["-o", str(executable), *(str(source) for source in sources)])
        compiled = subprocess.run(command, check=False, capture_output=True, text=True)
        if compiled.returncode:
            raise RuntimeError(
                f"shared-dot RTL compile failed:\n{compiled.stdout}\n{compiled.stderr}"
            )
        simulated = subprocess.run(
            [vvp, str(executable), f"+TRACE_FILE={trace_path}"],
            check=False, capture_output=True, text=True,
        )
        if simulated.returncode:
            raise RuntimeError(
                f"shared-dot RTL simulation failed:\n{simulated.stdout}\n{simulated.stderr}"
            )
        synthesis = _run_yosys_check(sources[:2], config, temporary)

    rtl_events, rtl_summary = parse_shared_dot_rtl_output(simulated.stdout)
    event_exact = model.events == rtl_events
    expected_summary = {"cycles": model.cycles, **model.counters}
    counter_exact = expected_summary == rtl_summary
    geometry_by_tag = {beat.tag: beat.geometry for beat in beats}
    rtl_outputs = tuple(
        SharedDotOutput(
            event.tag, event.lane, geometry_by_tag[event.tag],
            event.dot, event.distance,
        )
        for event in rtl_events if event.event == "OUTPUT_ACCEPT"
    )
    output_exact = rtl_outputs == model.outputs
    correlated = event_exact and counter_exact and output_exact
    report: dict[str, Any] = {
        "schema_version": "0.1",
        "status": "rtl_correlated" if correlated else "mismatch",
        "operator": "shared_int8_dot4_feature_geometry",
        "config": config.as_dict(),
        "cycle_model_cycles": model.cycles,
        "rtl_cycles": rtl_summary["cycles"],
        "cycle_error": rtl_summary["cycles"] - model.cycles,
        "event_trace_exact": event_exact,
        "counter_trace_exact": counter_exact,
        "functional_output_exact": output_exact,
        "cycle_model_counters": model.counters,
        "rtl_counters": {key: rtl_summary[key] for key in COUNTER_NAMES},
        "event_mismatches": _mismatches(model.events, rtl_events),
        "pointllm_mapping": pointllm_shared_geometry_mapping(config.pe_count),
        "synthesis_check": synthesis,
        "claim_boundary": {
            "feature_and_geometry_use_same_dot4_instances": correlated,
            "multipliers_outside_dot4_pe": synthesis.get("top_direct_multipliers"),
            "all_fps_knn_hardware_shared": False,
            "int8_geometry_semantic_fidelity_validated": False,
            "full_pointllm_cycle_accurate": False,
        },
    }
    if output_dir is not None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        write_shared_dot_trace(destination / "shared_dot_trace.hex", config, beats)
        _write_events(destination / "cycle_model_trace.csv", model.events)
        _write_events(destination / "rtl_trace.csv", rtl_events)
        (destination / "rtl_stdout.log").write_text(simulated.stdout, encoding="utf-8")
        (destination / "yosys_check.log").write_text(
            str(synthesis.get("log", "")), encoding="utf-8",
        )
        (destination / "correlation.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8",
        )
    if not correlated:
        raise AssertionError(f"shared-dot RTL mismatch: {report}")
    return report


def write_shared_dot_trace(
    path: str | Path,
    config: SharedDotConfig,
    beats: tuple[SharedDotBeat, ...],
) -> None:
    path = Path(path)
    with path.open("w", encoding="ascii") as handle:
        for beat in beats:
            value = beat.tag | (int(beat.geometry) << config.tag_width)
            shift = config.tag_width + 1
            for lane in beat.lanes:
                for item in (*lane.a, *lane.b):
                    value |= _encode_signed(item, 8) << shift
                    shift += 8
                value |= lane.a_norm << shift
                shift += config.norm_width
                value |= lane.b_norm << shift
                shift += config.norm_width
            handle.write(f"{value:0{(shift + 3) // 4}x}\n")


def parse_shared_dot_rtl_output(
    text: str,
) -> tuple[tuple[SharedDotEvent, ...], dict[str, int]]:
    events: list[SharedDotEvent] = []
    summary: dict[str, int] | None = None
    for raw_line in text.splitlines():
        fields = raw_line.strip().split()
        if fields[:1] == ["TRACE"] and len(fields) == 7:
            events.append(SharedDotEvent(
                cycle=int(fields[1]), event=fields[2], tag=int(fields[3]),
                lane=int(fields[4]), dot=int(fields[5]), distance=int(fields[6]),
            ))
        elif fields[:1] == ["SUMMARY"] and len(fields) == 8:
            values = [int(value) for value in fields[1:]]
            summary = dict(zip(("cycles", *COUNTER_NAMES), values, strict=True))
    if summary is None:
        raise RuntimeError(f"shared-dot RTL did not emit SUMMARY:\n{text[-4000:]}")
    return tuple(events), summary


def _encode_signed(value: int, width: int) -> int:
    lower = -(1 << (width - 1))
    upper = (1 << (width - 1)) - 1
    if not lower <= value <= upper:
        raise ValueError(f"signed value {value} does not fit in {width} bits")
    return value & ((1 << width) - 1)


def _mismatches(
    expected: tuple[SharedDotEvent, ...],
    actual: tuple[SharedDotEvent, ...],
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


def _write_events(path: Path, events: tuple[SharedDotEvent, ...]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("cycle", "event", "tag", "lane", "dot", "distance"),
            lineterminator="\n",
        )
        writer.writeheader()
        for event in events:
            writer.writerow(asdict(event))


def _run_yosys_check(
    sources: tuple[Path, ...], config: SharedDotConfig, temporary: Path,
) -> dict[str, Any]:
    yosys = shutil.which("yosys")
    if not yosys:
        return {"status": "unavailable", "version": "unavailable", "log": ""}
    json_path = temporary / "shared_dot_hierarchy.json"
    command = (
        f"read_verilog -sv {' '.join(str(source) for source in sources)}; "
        "hierarchy -check -top gtsu_shared_dot_geometry; "
        f"chparam -set PE_COUNT {config.pe_count} "
        f"-set NORM_WIDTH {config.norm_width} "
        f"-set DIST_WIDTH {config.distance_width} "
        f"-set TAG_WIDTH {config.tag_width} gtsu_shared_dot_geometry; "
        "hierarchy -check -top gtsu_shared_dot_geometry; "
        "proc; opt; check -assert; stat; "
        f"write_json {json_path}"
    )
    result = subprocess.run(
        [yosys, "-p", command], check=False, capture_output=True, text=True,
    )
    if result.returncode:
        raise RuntimeError(f"shared-dot Yosys check failed:\n{result.stdout}\n{result.stderr}")
    netlist = json.loads(json_path.read_text(encoding="utf-8"))
    modules = netlist["modules"]
    top_name = next(
        name for name, module in modules.items()
        if module.get("attributes", {}).get("top") in (1, "00000000000000000000000000000001")
    )
    top_cells = modules[top_name].get("cells", {})
    top_direct_multipliers = sum(
        cell.get("type") == "$mul" for cell in top_cells.values()
    )
    dot_module = modules["gtsu_dot4_pe"]
    dot4_multipliers = sum(
        cell.get("type") == "$mul" for cell in dot_module.get("cells", {}).values()
    )
    dot4_instances = sum(
        cell.get("type") == "gtsu_dot4_pe" for cell in top_cells.values()
    )
    if top_direct_multipliers != 0 or dot4_multipliers != 4 or dot4_instances != config.pe_count:
        raise AssertionError(
            "shared multiplier hierarchy mismatch: "
            f"top={top_direct_multipliers}, dot4={dot4_multipliers}, "
            f"instances={dot4_instances}"
        )
    return {
        "status": "passed",
        "version": _tool_version(yosys),
        "top_direct_multipliers": top_direct_multipliers,
        "dot4_multipliers_per_instance": dot4_multipliers,
        "dot4_instances": dot4_instances,
        "total_shared_int8_multipliers": dot4_multipliers * dot4_instances,
        "hierarchy_proof": "all multipliers are contained in shared gtsu_dot4_pe instances",
        "log": result.stdout + result.stderr,
    }


def _tool_version(executable: str) -> str:
    result = subprocess.run(
        [executable, "-V"], check=False, capture_output=True, text=True,
    )
    lines = [line.strip() for line in (result.stdout + result.stderr).splitlines() if line.strip()]
    return lines[0] if lines else "unknown"
