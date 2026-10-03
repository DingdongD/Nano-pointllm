"""Compiler-style AXI burst splitting and DRAMsim3 geometry preload timing."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any

from .dramsim3_backend import DramSim3Paths, DramTimingResult, time_dram_bursts
from .pointllm_lowering import DramBurst


@dataclass(frozen=True)
class AxiBurst:
    address: int
    beats: int

    @property
    def arlen(self) -> int:
        return self.beats - 1


def split_axi64_bursts(
    address: int, beats: int, *, enforce_4k: bool = True,
    max_burst_beats: int = 256,
) -> tuple[AxiBurst, ...]:
    if address < 0 or address % 64:
        raise ValueError("AXI64 address must be non-negative and 64-byte aligned")
    if beats <= 0 or not 0 < max_burst_beats <= 256:
        raise ValueError("AXI beat counts must be in the supported range")
    result = []
    remaining = beats
    current = address
    while remaining:
        count = min(remaining, max_burst_beats)
        if enforce_4k:
            count = min(count, 64 - ((current >> 6) & 0x3F))
        result.append(AxiBurst(current, count))
        current += count * 64
        remaining -= count
    return tuple(result)


def geometry_preload_lines(
    *, base_address: int = 0x8000_0000, points: int = 8192,
) -> tuple[DramBurst, ...]:
    payload_bytes = points * 16
    if payload_bytes % 64:
        raise ValueError("geometry point/norm payload must fill 64-byte lines")
    lines = payload_bytes // 64
    return tuple(DramBurst(
        sequence=index, address=base_address + index * 64, size=64,
        stream="geometry_point_norm_preload", tile_id=index // 64,
        output_row=-1, split_index=index // 64, chunk_index=index % 64,
        chunks_in_split=min(64, lines - (index // 64) * 64),
    ) for index in range(lines))


def time_geometry_preload(
    *, output_dir: str | Path, paths: DramSim3Paths | None = None,
    max_outstanding: int = 32,
) -> tuple[DramTimingResult, dict[str, Any]]:
    lines = geometry_preload_lines()
    bursts = split_axi64_bursts(lines[0].address, len(lines))
    timing = time_dram_bursts(
        lines, paths=paths or DramSim3Paths.local_default(),
        output_dir=output_dir, max_outstanding=max_outstanding,
        reorder_window=max_outstanding, max_submissions_per_cycle=1,
    )
    submitted = [event for event in timing.events if event.event == "SUBMIT"]
    completed = [event for event in timing.events if event.event == "COMPLETE"]
    summary = {
        "schema_version": "0.1",
        "status": "dramsim3_closed_loop",
        "payload": {
            "points": 8192, "bytes_per_point_with_norm": 16,
            "bytes": 8192 * 16, "transactions_64b": len(lines),
            "axi_bursts": len(bursts),
            "axi_burst_beats": [burst.beats for burst in bursts],
        },
        "timing": {
            "accelerator_cycles": timing.accelerator_cycles,
            "dram_clock_ticks": timing.dram_clock_ticks,
            "requests": timing.requests,
            "bytes_read": timing.bytes_read,
            "latency_min": timing.latency_min,
            "latency_median": timing.latency_median,
            "latency_max": timing.latency_max,
            "first_submit_cycle": submitted[0].cycle,
            "last_submit_cycle": submitted[-1].cycle,
            "first_completion_cycle": min(event.cycle for event in completed),
            "last_completion_cycle": max(event.cycle for event in completed),
        },
        "provenance": timing.provenance | {
            "compiler_codes_reference_commit": "ad2c31a",
            "compiler_codes_axi_contract": (
                "512-bit beats, ARLEN/AWLEN, optional 4-KiB boundary"
            ),
        },
        "conservation": {
            "all_lines_submitted": len(submitted) == len(lines),
            "all_lines_completed": len(completed) == len(lines),
            "all_bytes_returned": timing.bytes_read == 8192 * 16,
            "one_submit_per_accelerator_cycle": all(
                right.cycle > left.cycle for left, right in zip(submitted, submitted[1:])
            ),
            "no_axi_burst_crosses_4k": all(
                (burst.address >> 12) ==
                ((burst.address + burst.beats * 64 - 1) >> 12)
                for burst in bursts
            ),
        },
        "claim_boundary": {
            "request_side_dramsim3_closed_loop": True,
            "actual_point_payload_transferred": False,
            "dma_to_geometry_sram_rtl_composed": False,
            "full_pointllm_cycle_accurate": False,
        },
    }
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8",
    )
    with (destination / "events.csv").open("w", encoding="ascii", newline="") as handle:
        handle.write("cycle,event,tag,address,latency\n")
        for event in timing.events:
            handle.write(
                f"{event.cycle},{event.event},{event.tag},"
                f"0x{event.address:x},{event.latency}\n"
            )
    return timing, summary


def correlate_axi_burst_splitter(
    *, rtl_root: str | Path, output_dir: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(rtl_root)
    source = root / "gtsu_axi64_burst_splitter.sv"
    testbench = root / "tb_gtsu_axi64_burst_splitter.sv"
    expected = (
        *split_axi64_bursts(0x1000, 130),
        *split_axi64_bursts(0x1FC0, 66),
        *split_axi64_bursts(0x4000, 300, enforce_4k=False),
    )
    with tempfile.TemporaryDirectory(prefix="gtsu_axi_split_") as name:
        executable = Path(name) / "axi_split.vvp"
        compiled = subprocess.run([
            "iverilog", "-g2012", "-s", "tb_gtsu_axi64_burst_splitter",
            "-o", str(executable), str(source), str(testbench),
        ], capture_output=True, text=True)
        if compiled.returncode:
            raise RuntimeError(f"AXI splitter compile failed:\n{compiled.stderr}")
        simulated = subprocess.run(
            ["vvp", str(executable)], capture_output=True, text=True,
        )
        if simulated.returncode:
            raise RuntimeError(
                f"AXI splitter simulation failed:\n{simulated.stdout}\n{simulated.stderr}"
            )
    actual, summary = _parse_axi_output(simulated.stdout)
    values_exact = tuple((row[2], row[3]) for row in actual) == tuple(
        (burst.address, burst.beats) for burst in expected
    )
    synthesis = _synthesize_axi(source)
    report = {
        "schema_version": "0.1",
        "status": "rtl_correlated" if values_exact and summary["mismatches"] == 0
                  else "mismatch",
        "operator": "compiler_style_axi64_burst_splitter",
        "bursts": [asdict(burst) | {"arlen": burst.arlen} for burst in expected],
        "rtl_events": [
            {"cycle": cycle, "ordinal": ordinal, "address": address, "beats": beats}
            for cycle, ordinal, address, beats in actual
        ],
        "value_exact": values_exact,
        "summary": summary,
        "synthesis_check": synthesis,
        "claim_boundary": {
            "axi_splitter_rtl_correlated": values_exact,
            "dramsim3_timing_in_this_report": False,
            "full_dma_rtl_composed": False,
        },
    }
    if output_dir is not None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "rtl_stdout.log").write_text(simulated.stdout, encoding="utf-8")
        (destination / "yosys_check.log").write_text(
            synthesis["log"], encoding="utf-8",
        )
        (destination / "correlation.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8",
        )
    if report["status"] != "rtl_correlated":
        raise AssertionError(f"AXI burst splitter mismatch: {report}")
    return report


def _parse_axi_output(text: str):
    bursts = []
    summary = None
    for line in text.splitlines():
        fields = line.split()
        if fields[:1] == ["TRACE"] and fields[2] == "BURST":
            bursts.append((
                int(fields[1]), int(fields[3]), int(fields[4], 16), int(fields[5]),
            ))
        elif fields[:1] == ["SUMMARY"]:
            summary = {
                "cycles": int(fields[1]), "commands": int(fields[2]),
                "bursts": int(fields[3]), "mismatches": int(fields[4]),
            }
    if summary is None:
        raise RuntimeError(f"AXI splitter summary missing:\n{text[-4000:]}")
    return tuple(bursts), summary


def _synthesize_axi(source: Path) -> dict[str, Any]:
    yosys = shutil.which("yosys")
    if not yosys:
        return {"status": "unavailable", "log": ""}
    result = subprocess.run([
        yosys, "-p", f"read_verilog -sv {source}; hierarchy -check -top "
        "gtsu_axi64_burst_splitter; proc; opt; check -assert; stat",
    ], capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"AXI splitter Yosys failed:\n{result.stdout}\n{result.stderr}")
    return {"status": "passed", "log": result.stdout}
