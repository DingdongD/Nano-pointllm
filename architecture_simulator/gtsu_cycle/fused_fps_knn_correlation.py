"""Icarus/Yosys correlation for the fused FPS/KNN selection controller."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any

from .fused_fps_knn import (
    FusedFpsKnnConfig, FusedRoundOutput, distance_beats,
    run_fused_fps_knn_model,
)


def correlate_fused_fps_knn(
    config: FusedFpsKnnConfig, *, rtl_root: str | Path,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    config.validate()
    model = run_fused_fps_knn_model(config)
    iverilog, vvp = shutil.which("iverilog"), shutil.which("vvp")
    if not iverilog or not vvp:
        raise RuntimeError("Icarus Verilog is required for FPS/KNN correlation")
    root = Path(rtl_root)
    sources = (
        root / "gtsu_fused_fps_knn_controller.sv",
        root / "tb_gtsu_fused_fps_knn_controller.sv",
    )
    with tempfile.TemporaryDirectory(prefix="gtsu_fps_knn_") as name:
        temporary = Path(name)
        trace = temporary / "distance_trace.hex"
        write_distance_trace(trace, config)
        executable = temporary / "fps_knn.vvp"
        command = [iverilog, "-g2012", "-s", "tb_gtsu_fused_fps_knn_controller"]
        for parameter, value in {
            "POINTS": config.points, "CENTERS": config.centers,
            "K": config.neighbors, "LANES": config.lanes,
            "DIST_WIDTH": config.distance_width,
            "SOURCE_STALL_MOD": config.source_stall_mod,
            "SOURCE_STALL_PHASE": config.source_stall_phase,
            "OUTPUT_STALL_MOD": config.output_stall_mod,
            "OUTPUT_STALL_PHASE": config.output_stall_phase,
            "MAX_CYCLES": config.max_cycles,
        }.items():
            command.append(f"-Ptb_gtsu_fused_fps_knn_controller.{parameter}={value}")
        command.extend(["-o", str(executable), *(str(path) for path in sources)])
        compiled = subprocess.run(command, capture_output=True, text=True)
        if compiled.returncode:
            raise RuntimeError(f"FPS/KNN RTL compile failed:\n{compiled.stderr}")
        simulated = subprocess.run(
            [vvp, str(executable), f"+TRACE_FILE={trace}"],
            capture_output=True, text=True,
        )
        if simulated.returncode:
            raise RuntimeError(
                f"FPS/KNN RTL simulation failed:\n{simulated.stdout}\n{simulated.stderr}"
            )
    rtl_outputs, rtl_events, summary = _parse_output(simulated.stdout, config)
    expected_events = tuple(
        (event.cycle, event.event, event.round_index, event.item, event.value)
        for event in model.events
    )
    counters = {
        "input_beats": summary["input_beats"],
        "distance_values": summary["distance_values"],
        "round_outputs": summary["round_outputs"],
        "source_backpressure_cycles": summary["source_backpressure_cycles"],
        "output_backpressure_cycles": summary["output_backpressure_cycles"],
    }
    counter_exact = counters == {
        key: model.counters[key] for key in counters
    }
    exact = (
        summary["cycles"] == model.cycles
        and counter_exact and rtl_events == expected_events
        and rtl_outputs == model.outputs
    )
    synthesis = _run_yosys(sources[0], config)
    composition = _run_composition_yosys(root, config)
    report = {
        "schema_version": "0.1",
        "status": "rtl_correlated" if exact else "mismatch",
        "operator": "fused_fps_knn_selection_controller",
        "config": config.as_dict(),
        "cycle_model_cycles": model.cycles,
        "rtl_cycles": summary["cycles"],
        "cycle_error": summary["cycles"] - model.cycles,
        "event_trace_exact": rtl_events == expected_events,
        "counter_trace_exact": counter_exact,
        "center_sequence_exact": tuple(row.next_center for row in rtl_outputs) ==
                                 tuple(row.next_center for row in model.outputs),
        "topk_exact": rtl_outputs == model.outputs,
        "outputs": [
            {
                "round": row.round_index, "center": row.center,
                "next_center": row.next_center,
                "neighbors": list(row.neighbor_indices),
                "distances": list(row.neighbor_distances),
            }
            for row in rtl_outputs
        ],
        "counters": counters,
        "synthesis_check": synthesis,
        "shared_distance_composition_check": composition,
        "claim_boundary": {
            "persistent_min_state": exact,
            "deterministic_argmax_lower_index_tie": exact,
            "deterministic_topk_distance_index_order": exact,
            "shared_dot_distance_structurally_composed": composition["status"] == "passed",
            "shared_dot_to_selection_value_cycle_correlated": False,
            "production_8192x512_correlated": False,
            "real_modelnet_objaverse_quantization_validated": False,
            "fps_coverage_enabled": False,
            "knn_topk_coverage_enabled": False,
            "full_pointllm_cycle_accurate": False,
        },
        "reference_provenance": {
            "repository": "https://github.com/DingdongD/PointKAN_Accel",
            "commit": "aa356da232a9d30d6c784297603c2a5596c6409a",
            "adaptation": "clean-room banked min-state and bounded deterministic Top-K",
        },
    }
    if output_dir is not None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        write_distance_trace(destination / "distance_trace.hex", config)
        (destination / "rtl_stdout.log").write_text(simulated.stdout, encoding="utf-8")
        (destination / "yosys_check.log").write_text(synthesis["log"], encoding="utf-8")
        (destination / "correlation.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8",
        )
    if not exact:
        raise AssertionError(f"fused FPS/KNN mismatch: {report}")
    return report


def write_distance_trace(path: str | Path, config: FusedFpsKnnConfig) -> None:
    beats = distance_beats(config)
    with Path(path).open("w", encoding="ascii") as handle:
        for beat_index, beat in enumerate(beats):
            tile = beat_index % config.tiles
            mask = 0
            value = 0
            for lane, distance in enumerate(beat):
                if tile * config.lanes + lane < config.points:
                    mask |= 1 << lane
                value |= distance << (config.lanes + lane * config.distance_width)
            value |= mask
            width = config.lanes + config.lanes * config.distance_width
            handle.write(f"{value:0{(width + 3) // 4}x}\n")


def _parse_output(text: str, config: FusedFpsKnnConfig):
    rounds: dict[int, dict[str, Any]] = {}
    events = []
    summary = None
    for line in text.splitlines():
        fields = line.split()
        if fields[:1] == ["TRACE"] and fields[2] == "INPUT":
            cycle, round_index, tile = map(int, (fields[1], fields[3], fields[4]))
            events.append((cycle, "INPUT_BEAT", round_index, tile, 0))
        elif fields[:1] == ["TRACE"] and fields[2] == "ROUND":
            cycle, round_index, center, next_center = map(
                int, (fields[1], fields[3], fields[4], fields[5]),
            )
            rounds[round_index] = {
                "cycle": cycle, "center": center, "next": next_center,
                "neighbors": [None] * config.neighbors,
                "distances": [None] * config.neighbors,
            }
            events.append((cycle, "ROUND_OUTPUT", round_index, center, next_center))
        elif fields[:1] == ["TRACE"] and fields[2] == "NEIGHBOR":
            cycle, round_index, slot, index, distance = map(int, fields[1:2] + fields[3:])
            rounds[round_index]["neighbors"][slot] = index
            rounds[round_index]["distances"][slot] = distance
            events.append((
                cycle, "NEIGHBOR", round_index, slot,
                (index << config.distance_width) | distance,
            ))
        elif fields[:1] == ["SUMMARY"] and len(fields) == 7:
            keys = (
                "cycles", "input_beats", "distance_values", "round_outputs",
                "source_backpressure_cycles", "output_backpressure_cycles",
            )
            summary = dict(zip(keys, map(int, fields[1:]), strict=True))
    if summary is None:
        raise RuntimeError(f"FPS/KNN RTL did not emit summary:\n{text[-4000:]}")
    outputs = tuple(FusedRoundOutput(
        round_index, row["center"], row["next"], tuple(row["neighbors"]),
        tuple(row["distances"]),
    ) for round_index, row in sorted(rounds.items()))
    return outputs, tuple(events), summary


def _run_yosys(source: Path, config: FusedFpsKnnConfig) -> dict[str, Any]:
    yosys = shutil.which("yosys")
    if not yosys:
        return {"status": "unavailable", "log": ""}
    command = (
        f"read_verilog -sv {source}; "
        f"chparam -set POINTS {config.points} -set CENTERS {config.centers} "
        f"-set K {config.neighbors} -set LANES {config.lanes} "
        f"-set DIST_WIDTH {config.distance_width} gtsu_fused_fps_knn_controller; "
        "hierarchy -check -top gtsu_fused_fps_knn_controller; "
        "proc; memory; opt; check -assert; stat"
    )
    result = subprocess.run([yosys, "-p", command], capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"FPS/KNN Yosys failed:\n{result.stdout}\n{result.stderr}")
    return {"status": "passed", "log": result.stdout}


def _run_composition_yosys(root: Path, config: FusedFpsKnnConfig) -> dict[str, Any]:
    yosys = shutil.which("yosys")
    if not yosys:
        return {"status": "unavailable", "log": ""}
    sources = (
        root / "gtsu_dot4_pe.sv", root / "gtsu_shared_dot_geometry.sv",
        root / "gtsu_fused_fps_knn_controller.sv",
        root / "gtsu_fused_geometry_pipeline.sv",
    )
    command = (
        f"read_verilog -sv {' '.join(str(path) for path in sources)}; "
        f"chparam -set POINTS {config.points} -set CENTERS {config.centers} "
        f"-set K {config.neighbors} -set LANES {config.lanes} "
        "gtsu_fused_geometry_pipeline; "
        "hierarchy -check -top gtsu_fused_geometry_pipeline; proc; opt; check -assert; stat"
    )
    result = subprocess.run([yosys, "-p", command], capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(
            f"fused geometry composition Yosys failed:\n{result.stdout}\n{result.stderr}"
        )
    required = (
        "gtsu_shared_dot_geometry", "gtsu_fused_fps_knn_controller",
    )
    missing = [name for name in required if name not in result.stdout]
    if missing:
        raise AssertionError(f"fused geometry hierarchy missing {missing}")
    return {"status": "passed", "required_modules": list(required), "log": result.stdout}
