"""Icarus-based cycle/event correlation for the W8 Split-K GEMV slice."""
from __future__ import annotations

import csv
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any

from .ir import lower_splitk_gemv
from .splitk_gemv import (
    GemvEvent,
    OutputValue,
    SplitKGemvConfig,
    run_cycle_model,
)


COUNTER_NAMES = (
    "input_beats",
    "unpack_beats",
    "compute_beats",
    "partial_sums",
    "outputs",
    "source_backpressure_cycles",
    "unpack_backpressure_cycles",
    "compute_backpressure_cycles",
    "reduction_backpressure_cycles",
)


def correlate_splitk_gemv(
    config: SplitKGemvConfig,
    *,
    rtl_root: str | Path,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    config.validate()
    iverilog = shutil.which("iverilog")
    vvp = shutil.which("vvp")
    if not iverilog or not vvp:
        raise RuntimeError("iverilog and vvp are required for RTL correlation")

    root = Path(rtl_root)
    sources = (
        root / "gtsu_rv_fifo.sv",
        root / "gtsu_w8_splitk_gemv.sv",
        root / "tb_gtsu_w8_splitk_gemv.sv",
    )
    missing = [str(path) for path in sources if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing RTL sources: {missing}")

    cycle_result = run_cycle_model(config)
    micro_ops = lower_splitk_gemv(config)
    synthesis = _run_yosys_check(sources)
    with tempfile.TemporaryDirectory(prefix="gtsu_splitk_rtl_") as temporary:
        executable = Path(temporary) / "splitk.vvp"
        command = [
            iverilog,
            "-g2012",
            "-s", "tb_gtsu_w8_splitk_gemv",
            "-o", str(executable),
            *_parameter_arguments(config),
            *(str(path) for path in sources),
        ]
        compile_result = subprocess.run(
            command, check=False, capture_output=True, text=True,
        )
        if compile_result.returncode:
            raise RuntimeError(
                "RTL compile failed:\n"
                f"stdout:\n{compile_result.stdout}\n"
                f"stderr:\n{compile_result.stderr}"
            )
        run_result = subprocess.run(
            [vvp, str(executable)], check=False, capture_output=True, text=True,
        )
        if run_result.returncode:
            raise RuntimeError(
                "RTL simulation failed:\n"
                f"stdout:\n{run_result.stdout}\n"
                f"stderr:\n{run_result.stderr}"
            )

    rtl_events, rtl_summary = parse_rtl_output(run_result.stdout)
    expected_keys = tuple(event.key() for event in cycle_result.events)
    actual_keys = tuple(event.key() for event in rtl_events)
    event_mismatches = _sequence_mismatches(expected_keys, actual_keys)
    counter_mismatches = {
        key: {"cycle_model": cycle_result.counters[key], "rtl": rtl_summary[key]}
        for key in COUNTER_NAMES
        if cycle_result.counters[key] != rtl_summary[key]
    }
    cycle_match = cycle_result.cycles == rtl_summary["cycles"]
    rtl_outputs = tuple(
        OutputValue(event.n, event.value)
        for event in rtl_events if event.event == "OUTPUT_ACCEPT"
    )
    output_match = rtl_outputs == cycle_result.outputs
    correlated = not event_mismatches and not counter_mismatches and cycle_match and output_match

    report: dict[str, Any] = {
        "schema_version": "0.1",
        "status": "rtl_correlated" if correlated else "mismatch",
        "operator": "w8_splitk_gemv_vertical_slice",
        "config": config.as_dict(),
        "cycle_model_cycles": cycle_result.cycles,
        "rtl_cycles": rtl_summary["cycles"],
        "cycle_error": rtl_summary["cycles"] - cycle_result.cycles,
        "cycle_error_ratio": (
            abs(rtl_summary["cycles"] - cycle_result.cycles)
            / max(1, rtl_summary["cycles"])
        ),
        "event_count": len(rtl_events),
        "micro_op_count": len(micro_ops),
        "event_trace_exact": not event_mismatches,
        "counter_trace_exact": not counter_mismatches,
        "functional_output_exact": output_match,
        "traffic_exact": all(
            cycle_result.counters[key] == rtl_summary[key]
            for key in ("input_beats", "unpack_beats", "compute_beats", "partial_sums", "outputs")
        ),
        "event_mismatches": event_mismatches,
        "counter_mismatches": counter_mismatches,
        "cycle_model_counters": cycle_result.counters,
        "rtl_counters": {key: rtl_summary[key] for key in COUNTER_NAMES},
        "outputs": [asdict(item) for item in cycle_result.outputs],
        "fidelity": {
            "rtl_locked": correlated,
            "full_model_cycle_accurate": False,
            "covered_operators": ["w8_splitk_gemv_vertical_slice"] if correlated else [],
            "unsupported_operators_fail_closed": True,
        },
        "rtl_sources": [str(path) for path in sources],
        "tools": {
            "iverilog": _tool_version(iverilog, "-V"),
            "vvp": _tool_version(vvp, "-V"),
            "yosys": synthesis["version"],
        },
        "synthesis_check": {
            "status": synthesis["status"],
            "flow": "read_verilog; hierarchy; proc; memory; opt; check -assert",
        },
    }
    if output_dir is not None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        _write_events(destination / "cycle_model_trace.csv", cycle_result.events)
        _write_events(destination / "rtl_trace.csv", rtl_events)
        (destination / "rtl_stdout.log").write_text(run_result.stdout, encoding="utf-8")
        (destination / "yosys_check.log").write_text(
            str(synthesis["log"]), encoding="utf-8",
        )
        (destination / "micro_ops.json").write_text(
            json.dumps([operation.as_dict() for operation in micro_ops], indent=2) + "\n",
            encoding="utf-8",
        )
        (destination / "correlation.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8",
        )
    return report


def parse_rtl_output(text: str) -> tuple[tuple[GemvEvent, ...], dict[str, int]]:
    events: list[GemvEvent] = []
    summary: dict[str, int] | None = None
    for raw_line in text.splitlines():
        fields = raw_line.strip().split()
        if not fields:
            continue
        if fields[0] == "TRACE" and len(fields) == 7:
            events.append(GemvEvent(
                cycle=int(fields[1]),
                event=fields[2],
                n=int(fields[3]),
                partition=int(fields[4]),
                chunk=int(fields[5]),
                value=int(fields[6]),
            ))
        elif fields[0] == "SUMMARY" and len(fields) == 11:
            values = [int(item) for item in fields[1:]]
            summary = {"cycles": values[0]}
            summary.update(dict(zip(COUNTER_NAMES, values[1:], strict=True)))
    if summary is None:
        raise ValueError(f"RTL output did not contain a SUMMARY line:\n{text}")
    return tuple(events), summary


def _parameter_arguments(config: SplitKGemvConfig) -> list[str]:
    prefix = "tb_gtsu_w8_splitk_gemv"
    values = {
        "N_OUTPUTS": config.n_outputs,
        "SPLIT_K": config.split_k,
        "K_CHUNKS": config.k_chunks,
        "LANES": config.lanes,
        "SOURCE_LATENCY": config.source_latency,
        "COMP_FIFO_DEPTH": config.compressed_fifo_depth,
        "DECODED_FIFO_DEPTH": config.decoded_fifo_depth,
        "PARTIAL_FIFO_DEPTH": config.partial_fifo_depth,
        "OUTPUT_STALL_MOD": config.output_stall_mod,
        "OUTPUT_STALL_PHASE": config.output_stall_phase,
        "MAX_CYCLES": config.max_cycles,
    }
    result = []
    for name, value in values.items():
        result.extend(("-P", f"{prefix}.{name}={value}"))
    return result


def _sequence_mismatches(
    expected: tuple[tuple[Any, ...], ...],
    actual: tuple[tuple[Any, ...], ...],
    limit: int = 32,
) -> list[dict[str, Any]]:
    mismatches = []
    for index in range(max(len(expected), len(actual))):
        left = expected[index] if index < len(expected) else None
        right = actual[index] if index < len(actual) else None
        if left != right:
            mismatches.append({"index": index, "cycle_model": left, "rtl": right})
        if len(mismatches) >= limit:
            break
    return mismatches


def _write_events(path: Path, events: tuple[GemvEvent, ...]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("cycle", "event", "n", "partition", "chunk", "value"),
            lineterminator="\n",
        )
        writer.writeheader()
        for event in events:
            writer.writerow(asdict(event))


def _run_yosys_check(sources: tuple[Path, ...]) -> dict[str, str]:
    yosys = shutil.which("yosys")
    if not yosys:
        return {"status": "unavailable", "version": "unavailable", "log": ""}
    command = (
        f"read_verilog -sv {sources[0]} {sources[1]}; "
        "hierarchy -check -top gtsu_w8_splitk_gemv; "
        "proc; memory; opt; check -assert; stat"
    )
    result = subprocess.run(
        [yosys, "-p", command], check=False, capture_output=True, text=True,
    )
    if result.returncode:
        raise RuntimeError(
            f"Yosys structural check failed:\n{result.stdout}\n{result.stderr}"
        )
    return {
        "status": "passed",
        "version": _tool_version(yosys, "-V"),
        "log": result.stdout + result.stderr,
    }


def _tool_version(executable: str, argument: str) -> str:
    result = subprocess.run(
        [executable, argument], check=False, capture_output=True, text=True,
    )
    lines = [line.strip() for line in (result.stdout + result.stderr).splitlines() if line.strip()]
    return lines[0] if lines else "unknown"
