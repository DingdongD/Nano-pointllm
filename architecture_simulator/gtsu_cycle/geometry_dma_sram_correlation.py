"""DRAMsim3 completion -> DMA ROB -> geometry SRAM RTL correlation."""
from __future__ import annotations

import csv
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import tempfile
from typing import Any

from .dense_sram_pipeline import DensePipelineEvent, PayloadBeat
from .dram_dma import (
    OrderedDmaPayloadSource, completions_from_timing, pack_payload_lines,
    payload_lines_as_bursts, write_completion_trace,
)
from .dramsim3_backend import DramSim3Paths, time_dram_bursts


def correlate_geometry_dma_sram(
    *, rtl_root: str | Path, dramsim_output_dir: str | Path,
    output_dir: str | Path | None = None, rob_depth: int = 8,
) -> dict[str, Any]:
    points = 128
    beats = tuple(PayloadBeat(index, 0, _point_word(index)) for index in range(points))
    lines = pack_payload_lines(beats, base_address=0x8000_0000)
    timing = time_dram_bursts(
        payload_lines_as_bursts(lines), paths=DramSim3Paths.local_default(),
        output_dir=dramsim_output_dir, max_outstanding=rob_depth,
        reorder_window=rob_depth, max_submissions_per_cycle=1,
    )
    completions = completions_from_timing(timing, lines)
    source = OrderedDmaPayloadSource(
        completions, payload_count=points, block_chunks=1, rob_depth=rob_depth,
    )
    model_events = _run_dma_source(source, points)

    root = Path(rtl_root)
    sources = (
        root / "gtsu_dram_dma_unpack.sv",
        root / "gtsu_lbuf_16bank_macro_model.sv",
        root / "gtsu_geometry_sram_fabric.sv",
        root / "gtsu_geometry_dram_sram_loader.sv",
        root / "tb_gtsu_geometry_dram_sram_loader.sv",
    )
    with tempfile.TemporaryDirectory(prefix="gtsu_geometry_dma_sram_") as name:
        temporary = Path(name)
        trace = temporary / "completion_trace.hex"
        write_completion_trace(trace, completions)
        executable = temporary / "geometry_dma_sram.vvp"
        compiled = subprocess.run([
            "iverilog", "-g2012", "-s", "tb_gtsu_geometry_dram_sram_loader",
            f"-Ptb_gtsu_geometry_dram_sram_loader.ROB_DEPTH={rob_depth}",
            "-o", str(executable), *(str(path) for path in sources),
        ], capture_output=True, text=True)
        if compiled.returncode:
            raise RuntimeError(f"geometry DMA/SRAM compile failed:\n{compiled.stderr}")
        simulated = subprocess.run([
            "vvp", str(executable), f"+TRACE_FILE={trace}",
        ], capture_output=True, text=True)
        if simulated.returncode:
            raise RuntimeError(
                f"geometry DMA/SRAM simulation failed:\n"
                f"{simulated.stdout}\n{simulated.stderr}"
            )
    rtl_events, summary = _parse_output(simulated.stdout)
    rtl_dma_events = tuple(
        event for event in rtl_events
        if event.event in ("DRAM_COMPLETE_ACCEPT", "DMA_WORD_ACCEPT")
    )
    issues = [event for event in rtl_events if event.event == "SRAM_READ_ISSUE"]
    responses = [event for event in rtl_events if event.event == "SRAM_RESPONSE"]
    expected_summary = {
        "dma_completion_accepts": len(lines), "point_words": points,
        "sram_responses": 2, "value_mismatches": 0,
        "dma_protocol_error": 0, "sram_collision": 0,
    }
    counters_exact = all(summary[key] == value for key, value in expected_summary.items())
    exact = (
        rtl_dma_events == model_events and counters_exact
        and len(issues) == len(responses) == 2
        and all(response.cycle - issue.cycle == 3
                for issue, response in zip(issues, responses, strict=True))
    )
    report = {
        "schema_version": "0.1",
        "status": "rtl_correlated" if exact else "mismatch",
        "operator": "dramsim3_dma_geometry_sram_loader",
        "config": {
            "points": points, "dram_lines": len(lines), "rob_depth": rob_depth,
            "sram_lanes": 64, "sram_read_latency": 3,
        },
        "dram_timing": {
            "accelerator_cycles": timing.accelerator_cycles,
            "requests": timing.requests, "bytes_read": timing.bytes_read,
            "latency_min": timing.latency_min,
            "latency_median": timing.latency_median,
            "latency_max": timing.latency_max,
            "provenance": timing.provenance,
        },
        "dma_event_trace_exact": rtl_dma_events == model_events,
        "sram_value_checks": points + 128,
        "sram_value_mismatches": summary["value_mismatches"],
        "sram_read_latency_exact": all(
            response.cycle - issue.cycle == 3
            for issue, response in zip(issues, responses, strict=True)
        ),
        "summary": summary,
        "claim_boundary": {
            "real_dramsim3_completion_timing": True,
            "actual_payload_dma_to_sram_rtl_correlated": exact,
            "finite_completion_rob": True,
            "production_8192_point_payload_rtl_run": False,
            "fp32_arithmetic_and_selector_composed": False,
            "foundry_sram_sta_ppa": False,
            "full_pointllm_cycle_accurate": False,
        },
    }
    if output_dir is not None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        write_completion_trace(destination / "completion_trace.hex", completions)
        _write_events(destination / "cycle_model_trace.csv", model_events)
        _write_events(destination / "rtl_trace.csv", rtl_events)
        (destination / "rtl_stdout.log").write_text(
            simulated.stdout, encoding="utf-8",
        )
        (destination / "correlation.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8",
        )
    if not exact:
        raise AssertionError(f"geometry DMA/SRAM mismatch: {report}")
    return report


def _point_word(index: int) -> int:
    return (
        (0x10000000 + index)
        | ((0x20000000 + index) << 32)
        | ((0x30000000 + index) << 64)
        | ((0x40000000 + index) << 96)
    )


def _run_dma_source(
    source: OrderedDmaPayloadSource, payload_count: int,
) -> tuple[DensePipelineEvent, ...]:
    result = []
    words = 0
    for cycle in range(100_000):
        source.begin_cycle(cycle)
        if source.peek() is not None:
            source.accept(cycle)
            words += 1
        result.extend(source.take_events())
        if words == payload_count:
            source.counters()
            return tuple(result)
    raise TimeoutError("geometry DMA source did not drain")


def _parse_output(text: str):
    events = []
    summary = None
    for line in text.splitlines():
        fields = line.split()
        if fields[:1] == ["TRACE"] and len(fields) == 6:
            events.append(DensePipelineEvent(
                int(fields[1]), fields[2], int(fields[3]), int(fields[4]),
            ))
        elif fields[:1] == ["TRACE"] and len(fields) == 5:
            events.append(DensePipelineEvent(
                int(fields[1]), fields[2], int(fields[3]), int(fields[4]),
            ))
        elif fields[:1] == ["SUMMARY"] and len(fields) == 8:
            keys = (
                "cycles", "dma_completion_accepts", "point_words",
                "sram_responses", "value_mismatches", "dma_protocol_error",
                "sram_collision",
            )
            summary = dict(zip(keys, map(int, fields[1:]), strict=True))
    if summary is None:
        raise RuntimeError(f"geometry DMA/SRAM summary missing:\n{text[-4000:]}")
    return tuple(events), summary


def _write_events(path: Path, events: tuple[DensePipelineEvent, ...]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("cycle", "event", "tag", "value"),
            lineterminator="\n",
        )
        writer.writeheader()
        for event in events:
            writer.writerow(asdict(event))
