"""Exact correlation for the physical SRAM -> Dot4 -> requant Dense slice."""
from __future__ import annotations

import csv
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import tempfile
from typing import Any

from .dense_sram_pipeline import (
    DensePipelineEvent, DenseSramPipelineConfig, PayloadBeat,
    build_payload_beats, run_dense_sram_pipeline_model,
)


def correlate_dense_sram_pipeline(
    config: DenseSramPipelineConfig, *, rtl_root: str | Path,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    beats = build_payload_beats(config)
    model = run_dense_sram_pipeline_model(config, beats)
    root = Path(rtl_root)
    sources = (
        root / "gtsu_dot4_pe.sv",
        root / "gtsu_dense_dot4_tile.sv",
        root / "gtsu_requantize_int32.sv",
        root / "gtsu_rv_fifo.sv",
        root / "gtsu_banked_sram_2client.sv",
        root / "gtsu_dense_mnk_controller.sv",
        root / "gtsu_dense_sram_requant_pipeline.sv",
        root / "tb_gtsu_dense_sram_requant_pipeline.sv",
    )
    with tempfile.TemporaryDirectory(prefix="gtsu_dense_sram_") as temporary_name:
        temporary = Path(temporary_name)
        trace = temporary / "payload.hex"
        write_payload_trace(trace, beats)
        executable = temporary / "pipeline.vvp"
        command = ["iverilog", "-g2012", "-s", "tb_gtsu_dense_sram_requant_pipeline"]
        for name, value in {
            "M": config.m, "N": config.n, "K": config.k,
            "N_TILE": config.n_tile, "K_BLOCK": config.k_block,
            "READ_LATENCY": config.read_latency, "FIFO_DEPTH": config.fifo_depth,
            "REQUANT_MULTIPLIER": config.requant_multiplier,
            "REQUANT_RIGHT_SHIFT": config.requant_right_shift,
            "SOURCE_STALL_MOD": config.source_stall_mod,
            "SOURCE_STALL_PHASE": config.source_stall_phase,
            "OUTPUT_STALL_MOD": config.output_stall_mod,
            "OUTPUT_STALL_PHASE": config.output_stall_phase,
            "MAX_CYCLES": config.max_cycles,
        }.items():
            command.append(f"-Ptb_gtsu_dense_sram_requant_pipeline.{name}={value}")
        command.extend(["-o", str(executable), *(str(path) for path in sources)])
        compiled = subprocess.run(command, capture_output=True, text=True)
        if compiled.returncode:
            raise RuntimeError(f"dense SRAM RTL compile failed:\n{compiled.stderr}")
        simulated = subprocess.run(
            ["vvp", str(executable), f"+TRACE_FILE={trace}"],
            capture_output=True, text=True,
        )
        if simulated.returncode:
            raise RuntimeError(
                f"dense SRAM RTL failed:\n{simulated.stdout}\n{simulated.stderr}"
            )
        synthesis = _yosys(sources[:-1], config, temporary)

    rtl_events, rtl_summary = parse_pipeline_output(simulated.stdout)
    expected_summary = {
        "cycles": model.cycles,
        **model.counters,
        "overflow_error": 0,
        "done": 1,
    }
    exact = model.events == rtl_events and expected_summary == rtl_summary
    report = {
        "schema_version": "0.1",
        "status": "rtl_correlated" if exact else "mismatch",
        "operator": "dense_sram_dot4_requant_physical_slice",
        "config": config.as_dict(),
        "cycle_model_cycles": model.cycles,
        "rtl_cycles": rtl_summary["cycles"],
        "cycle_error": rtl_summary["cycles"] - model.cycles,
        "event_trace_exact": model.events == rtl_events,
        "counter_trace_exact": expected_summary == rtl_summary,
        "functional_outputs_exact": _event_outputs(model.events) == _event_outputs(rtl_events),
        "outputs": [asdict(output) for output in model.outputs],
        "counters": model.counters,
        "rtl_summary": rtl_summary,
        "synthesis_check": synthesis,
        "compiler_codes_reference": {
            "commit": "ad2c31a",
            "lbuf_interface_reference": (
                "rtl/ACTransformer_DS/rtl/hdl/buf_mgr/lbuf_wrap_bank.sv"
            ),
            "postprocess_reference": (
                "rtl/ACTransformer_DS/rtl/hdl/ctc/ct_cluster_ds/pp/pp_unit_ds.sv"
            ),
            "source_copied": False,
        },
        "claim_boundary": {
            "physical_16bank_128bit_sram_payload": exact,
            "sram_1r1w_ping_pong_overlap": exact,
            "shared_dot4_and_requant_composed": exact,
            "n_tile_3_one_word_slice": True,
            "production_n_tile_64_multiword_gather": False,
            "dramsim3_dma_composed": False,
            "floating_bf16_dequant": False,
            "full_pointllm_cycle_accurate": False,
        },
    }
    if output_dir is not None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        write_payload_trace(destination / "payload_trace.hex", beats)
        _write_events(destination / "cycle_model_trace.csv", model.events)
        _write_events(destination / "rtl_trace.csv", rtl_events)
        (destination / "rtl_stdout.log").write_text(simulated.stdout, encoding="utf-8")
        (destination / "yosys_check.log").write_text(synthesis["log"], encoding="utf-8")
        (destination / "correlation.json").write_text(json.dumps(report, indent=2) + "\n")
    if not exact:
        raise AssertionError(f"dense SRAM pipeline mismatch: {report}")
    return report


def write_payload_trace(path: Path, beats: tuple[PayloadBeat, ...]) -> None:
    with path.open("w", encoding="ascii") as handle:
        for beat in beats:
            value = beat.sequence | (beat.chunk << 32) | (beat.data << 40)
            handle.write(f"{value:042x}\n")


def parse_pipeline_output(
    text: str,
) -> tuple[tuple[DensePipelineEvent, ...], dict[str, int]]:
    events = []
    summary = None
    for line in text.splitlines():
        fields = line.split()
        if fields[:1] == ["TRACE"] and len(fields) == 5:
            events.append(DensePipelineEvent(
                int(fields[1]), fields[2], int(fields[3]), int(fields[4], 16),
            ))
        elif fields[:1] == ["SUMMARY"] and len(fields) == 10:
            keys = (
                "cycles", "payload_writes", "sram_read_issues",
                "sram_responses", "dot4_inputs", "output_tiles",
                "read_write_overlap_cycles", "overflow_error", "done",
            )
            summary = dict(zip(keys, map(int, fields[1:]), strict=True))
    if summary is None:
        raise RuntimeError(f"dense SRAM RTL did not emit SUMMARY:\n{text[-4000:]}")
    return tuple(events), summary


def _event_outputs(events: tuple[DensePipelineEvent, ...]) -> list[tuple[int, int]]:
    return [(event.tag, event.value) for event in events if event.event == "OUTPUT_ACCEPT"]


def _write_events(path: Path, events: tuple[DensePipelineEvent, ...]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("cycle", "event", "tag", "value"),
            lineterminator="\n",
        )
        writer.writeheader()
        for event in events:
            writer.writerow(asdict(event))


def _yosys(
    sources: tuple[Path, ...], config: DenseSramPipelineConfig, temporary: Path,
) -> dict[str, Any]:
    path = temporary / "pipeline.json"
    top = "gtsu_dense_sram_requant_pipeline"
    command = (
        f"read_verilog -sv {' '.join(str(source) for source in sources)}; "
        f"chparam -set M {config.m} -set N {config.n} -set K {config.k} "
        f"-set N_TILE {config.n_tile} -set K_BLOCK {config.k_block} "
        f"-set READ_LATENCY {config.read_latency} -set FIFO_DEPTH {config.fifo_depth} "
        f"-set REQUANT_MULTIPLIER {config.requant_multiplier} "
        f"-set REQUANT_RIGHT_SHIFT {config.requant_right_shift} {top}; "
        f"hierarchy -check -top {top}; proc; opt; check -assert; stat; write_json {path}"
    )
    result = subprocess.run(["yosys", "-p", command], capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"dense SRAM Yosys failed:\n{result.stdout}\n{result.stderr}")
    modules = json.loads(path.read_text())["modules"]
    top_cells = modules[top].get("cells", {}).values()
    counts = {
        "sram_instances": sum(cell["type"].startswith("$paramod") and "gtsu_banked_sram_2client" in cell["type"] for cell in top_cells),
        "controller_instances": sum(cell["type"].startswith("$paramod") and "gtsu_dense_mnk_controller" in cell["type"] for cell in top_cells),
        "dense_tile_instances": sum(cell["type"].startswith("$paramod") and "gtsu_dense_dot4_tile" in cell["type"] for cell in top_cells),
        "requant_instances": sum(cell["type"].startswith("$paramod") and "gtsu_requantize_int32" in cell["type"] for cell in top_cells),
        "fifo_instances": sum(cell["type"].startswith("$paramod") and "gtsu_rv_fifo" in cell["type"] for cell in top_cells),
        "top_direct_multipliers": sum(cell["type"] == "$mul" for cell in top_cells),
    }
    expected = {
        "sram_instances": 1, "controller_instances": 1,
        "dense_tile_instances": 1, "requant_instances": config.n_tile,
        "fifo_instances": 1, "top_direct_multipliers": 0,
    }
    if counts != expected:
        raise AssertionError(f"dense SRAM hierarchy mismatch: {counts} != {expected}")
    dot_modules = [module for name, module in modules.items() if name.endswith("gtsu_dot4_pe")]
    requant_modules = [module for name, module in modules.items() if "gtsu_requantize_int32" in name]
    return {
        "status": "passed", **counts,
        "dot4_multipliers_per_instance": _cell_count(dot_modules, "$mul"),
        "requant_multipliers_per_instance": _cell_count(requant_modules, "$mul"),
        "behavioral_sram_not_foundry_macro": True,
        "log": result.stdout + result.stderr,
    }


def _cell_count(modules: list[dict[str, Any]], cell_type: str) -> int:
    if not modules:
        return 0
    counts = {
        sum(cell["type"] == cell_type for cell in module.get("cells", {}).values())
        for module in modules
    }
    if len(counts) != 1:
        raise AssertionError(f"inconsistent {cell_type} counts across parameterized modules")
    return counts.pop()
