"""Exact Python-versus-RTL correlation for the locked banked SRAM slice."""
from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
import shutil
import subprocess
import tempfile

from .sram import SramConfig, SramEvent, run_sram_model


def correlate_banked_sram(
    *,
    rtl_root: str | Path,
    output_dir: str | Path | None = None,
) -> dict[str, object]:
    iverilog = shutil.which("iverilog")
    vvp = shutil.which("vvp")
    if not iverilog or not vvp:
        raise RuntimeError("Icarus Verilog is required for SRAM RTL correlation")
    rtl_root = Path(rtl_root)
    sources = (
        rtl_root / "gtsu_banked_sram_2client.sv",
        rtl_root / "tb_gtsu_banked_sram_2client.sv",
    )
    for source in sources:
        if not source.is_file():
            raise FileNotFoundError(source)

    model = run_sram_model(config=SramConfig())
    with tempfile.TemporaryDirectory(prefix="gtsu_sram_") as temporary:
        executable = Path(temporary) / "sram.vvp"
        compile_result = subprocess.run(
            [iverilog, "-g2012", "-s", "tb_gtsu_banked_sram_2client", "-o", str(executable),
             *(str(source) for source in sources)],
            check=False, capture_output=True, text=True,
        )
        if compile_result.returncode:
            raise RuntimeError(
                f"SRAM RTL compile failed:\n{compile_result.stdout}\n{compile_result.stderr}"
            )
        simulation = subprocess.run(
            [vvp, str(executable)], check=False, capture_output=True, text=True,
        )
        if simulation.returncode:
            raise RuntimeError(
                f"SRAM RTL simulation failed:\n{simulation.stdout}\n{simulation.stderr}"
            )

    rtl_events, rtl_cycles = _parse_trace(simulation.stdout)
    model_events = tuple(sorted(model.events, key=_event_key))
    rtl_events = tuple(sorted(rtl_events, key=_event_key))
    event_exact = model_events == rtl_events
    report: dict[str, object] = {
        "schema_version": "0.1",
        "status": "rtl_correlated" if event_exact and model.cycles == rtl_cycles else "mismatch",
        "contract": "gtsu_banked_sram_2client",
        "event_trace_exact": event_exact,
        "cycle_model_cycles": model.cycles,
        "rtl_cycles": rtl_cycles,
        "cycle_error": rtl_cycles - model.cycles,
        "events": len(model_events),
        "counters": model.counters,
        "config": asdict(model.config),
        "source_contract": {
            "repository": "/home/Compiler_Codes",
            "commit": "ad2c31a55227b6d1e3119d33e44a122f705e2099",
            "reference": "rtl/ACTransformer_DS/rtl/hdl/buf_mgr/lbuf_wrap*.sv",
        },
        "limitations": [
            "behavioral synthesizable SRAM arrays, not a foundry SRAM macro",
            "two-client locked arbitration slice, not the complete Compiler_Codes client set",
            "not yet connected to the Split-K compute RTL",
        ],
    }
    if not event_exact:
        report["mismatches"] = _mismatches(model_events, rtl_events)
    if output_dir is not None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        _write_events(destination / "sram_cycle_model.csv", model_events)
        _write_events(destination / "sram_rtl.csv", rtl_events)
        (destination / "sram_correlation.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8",
        )
    if report["status"] != "rtl_correlated":
        raise AssertionError(f"SRAM RTL correlation failed: {report}")
    return report


def _parse_trace(stdout: str) -> tuple[tuple[SramEvent, ...], int]:
    events: list[SramEvent] = []
    cycles: int | None = None
    for line in stdout.splitlines():
        fields = line.split()
        if fields and fields[0] == "TRACE" and len(fields) == 8:
            events.append(SramEvent(
                cycle=int(fields[1]), event=fields[2], client=int(fields[3]),
                tag=int(fields[4]), bank=int(fields[5]), row=int(fields[6]),
                data=int(fields[7]),
            ))
        elif fields and fields[0] == "SUMMARY" and len(fields) == 2:
            cycles = int(fields[1])
    if cycles is None:
        raise RuntimeError(f"SRAM RTL did not emit SUMMARY:\n{stdout}")
    return tuple(events), cycles


def _event_key(event: SramEvent) -> tuple[int, str, int, int, int, int, int]:
    return (
        event.cycle, event.event, event.client, event.tag,
        event.bank, event.row, event.data,
    )


def _mismatches(
    expected: tuple[SramEvent, ...], actual: tuple[SramEvent, ...], limit: int = 20,
) -> list[dict[str, object]]:
    rows = []
    for index in range(max(len(expected), len(actual))):
        left = asdict(expected[index]) if index < len(expected) else None
        right = asdict(actual[index]) if index < len(actual) else None
        if left != right:
            rows.append({"index": index, "cycle_model": left, "rtl": right})
        if len(rows) >= limit:
            break
    return rows


def _write_events(path: Path, events: tuple[SramEvent, ...]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("cycle", "event", "client", "tag", "bank", "row", "data"),
            lineterminator="\n",
        )
        writer.writeheader()
        for event in events:
            writer.writerow(asdict(event))
