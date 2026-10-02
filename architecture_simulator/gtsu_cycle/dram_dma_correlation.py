"""Real DRAMsim3 completion trace through DMA ROB and physical Dense RTL."""
from __future__ import annotations

import csv
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import tempfile
from typing import Any

from .dense_sram_pipeline import (
    DensePipelineEvent, DenseSramPipelineConfig, build_payload_beats,
    run_dense_sram_pipeline_model,
)
from .dram_dma import (
    OrderedDmaPayloadSource, completions_from_timing, pack_payload_lines,
    payload_lines_as_bursts, write_completion_trace,
)
from .dramsim3_backend import DramSim3Paths, time_dram_bursts


def correlate_dense_dramsim_dma(
    config: DenseSramPipelineConfig, *, rtl_root: str | Path,
    dramsim_paths: DramSim3Paths, dramsim_output_dir: str | Path,
    output_dir: str | Path | None = None, rob_depth: int = 4,
) -> dict[str, Any]:
    beats = build_payload_beats(config)
    lines = pack_payload_lines(beats)
    timing = time_dram_bursts(
        payload_lines_as_bursts(lines), paths=dramsim_paths,
        output_dir=dramsim_output_dir, max_outstanding=rob_depth,
        reorder_window=rob_depth,
    )
    completions = completions_from_timing(timing, lines)
    source = OrderedDmaPayloadSource(
        completions, payload_count=len(beats),
        block_chunks=config.block_chunks, rob_depth=rob_depth,
    )
    model = run_dense_sram_pipeline_model(
        config, beats, payload_source=source,
    )

    root = Path(rtl_root)
    sources = (
        root / "gtsu_dot4_pe.sv",
        root / "gtsu_dense_dot4_tile.sv",
        root / "gtsu_requantize_int32.sv",
        root / "gtsu_rv_fifo.sv",
        root / "gtsu_banked_sram_2client.sv",
        root / "gtsu_dense_mnk_controller.sv",
        root / "gtsu_dense_sram_requant_pipeline.sv",
        root / "gtsu_dram_dma_unpack.sv",
        root / "gtsu_dense_dramsim_dma_pipeline.sv",
        root / "tb_gtsu_dense_dramsim_dma_pipeline.sv",
    )
    with tempfile.TemporaryDirectory(prefix="gtsu_dramsim_dma_") as name:
        temporary = Path(name)
        trace = temporary / "completion.hex"
        write_completion_trace(trace, completions)
        executable = temporary / "pipeline.vvp"
        command = ["iverilog", "-g2012", "-s", "tb_gtsu_dense_dramsim_dma_pipeline"]
        for parameter, value in {
            "M": config.m, "N": config.n, "K": config.k,
            "N_TILE": config.n_tile, "K_BLOCK": config.k_block,
            "ROB_DEPTH": rob_depth, "READ_LATENCY": config.read_latency,
            "FIFO_DEPTH": config.fifo_depth,
            "REQUANT_MULTIPLIER": config.requant_multiplier,
            "REQUANT_RIGHT_SHIFT": config.requant_right_shift,
            "OUTPUT_STALL_MOD": config.output_stall_mod,
            "OUTPUT_STALL_PHASE": config.output_stall_phase,
            "MAX_CYCLES": config.max_cycles,
        }.items():
            command.append(f"-Ptb_gtsu_dense_dramsim_dma_pipeline.{parameter}={value}")
        command.extend(["-o", str(executable), *(str(path) for path in sources)])
        compiled = subprocess.run(command, capture_output=True, text=True)
        if compiled.returncode:
            raise RuntimeError(f"DRAMsim3 DMA RTL compile failed:\n{compiled.stderr}")
        simulated = subprocess.run(
            ["vvp", str(executable), f"+TRACE_FILE={trace}"],
            capture_output=True, text=True,
        )
        if simulated.returncode:
            raise RuntimeError(
                f"DRAMsim3 DMA RTL failed:\n{simulated.stdout}\n{simulated.stderr}"
            )
        synthesis = _yosys(sources[:-1], config, rob_depth, temporary)

    rtl_events, rtl_summary = parse_dma_pipeline_output(simulated.stdout)
    expected_summary = {
        "cycles": model.cycles,
        "dma_completion_accepts": model.counters["dma_completion_accepts"],
        "dma_words": model.counters["dma_words"],
        "dma_rob_backpressure_cycles": model.counters["dma_rob_backpressure_cycles"],
        "dma_rob_peak": model.counters["dma_rob_peak"],
        "payload_writes": model.counters["payload_writes"],
        "sram_read_issues": model.counters["sram_read_issues"],
        "sram_responses": model.counters["sram_responses"],
        "dot4_inputs": model.counters["dot4_inputs"],
        "output_tiles": model.counters["output_tiles"],
        "read_write_overlap_cycles": model.counters["read_write_overlap_cycles"],
        "overflow_error": 0, "dma_protocol_error": 0, "done": 1,
    }
    exact = model.events == rtl_events and expected_summary == rtl_summary
    report = {
        "schema_version": "0.1",
        "status": "rtl_correlated" if exact else "mismatch",
        "operator": "dramsim3_dma_sram_dot4_requant_physical_slice",
        "config": config.as_dict() | {"dma_rob_depth": rob_depth},
        "cycle_model_cycles": model.cycles,
        "rtl_cycles": rtl_summary["cycles"],
        "cycle_error": rtl_summary["cycles"] - model.cycles,
        "event_trace_exact": model.events == rtl_events,
        "counter_trace_exact": expected_summary == rtl_summary,
        "functional_outputs_exact": _outputs(model.events) == _outputs(rtl_events),
        "outputs": [asdict(output) for output in model.outputs],
        "counters": model.counters,
        "rtl_summary": rtl_summary,
        "dram_timing": timing.as_dict(),
        "traffic": {
            "dram_lines": len(lines), "dram_bytes": len(lines) * 64,
            "useful_payload_bytes": len(beats) * 16,
            "line_utilization": (len(beats) * 16) / (len(lines) * 64),
            "words_per_line": 4, "sram_word_bytes": 16,
        },
        "synthesis_check": synthesis,
        "compiler_codes_reference": {
            "commit": "ad2c31a",
            "axi_reference": "rtl/ACTransformer_DS/rtl/hdl/ddr_mgr/dmgr_axi_mst.sv",
            "reference_axi_data_bits": 512,
            "reference_fifo_data_bits": 1024,
            "source_copied": False,
        },
        "claim_boundary": {
            "official_dramsim3_controller_timing": True,
            "completion_trace_to_dma_rtl_exact": exact,
            "finite_tagged_completion_rob": exact,
            "physical_sram_payload_and_compute": exact,
            "dram_phy_rtl": False,
            "production_n_tile_64_multiword_gather": False,
            "full_pointllm_cycle_accurate": False,
        },
    }
    if output_dir is not None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        write_completion_trace(destination / "completion_trace.hex", completions)
        _write_events(destination / "cycle_model_trace.csv", model.events)
        _write_events(destination / "rtl_trace.csv", rtl_events)
        (destination / "rtl_stdout.log").write_text(simulated.stdout, encoding="utf-8")
        (destination / "yosys_check.log").write_text(synthesis["log"], encoding="utf-8")
        (destination / "correlation.json").write_text(json.dumps(report, indent=2) + "\n")
    if not exact:
        raise AssertionError(f"DRAMsim3 DMA pipeline mismatch: {report}")
    return report


def parse_dma_pipeline_output(
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
        elif fields[:1] == ["SUMMARY"] and len(fields) == 15:
            keys = (
                "cycles", "dma_completion_accepts", "dma_words",
                "dma_rob_backpressure_cycles", "dma_rob_peak",
                "payload_writes", "sram_read_issues", "sram_responses",
                "dot4_inputs", "output_tiles", "read_write_overlap_cycles",
                "overflow_error", "dma_protocol_error", "done",
            )
            summary = dict(zip(keys, map(int, fields[1:]), strict=True))
    if summary is None:
        raise RuntimeError(f"DRAMsim3 DMA RTL did not emit SUMMARY:\n{text[-4000:]}")
    return tuple(events), summary


def _outputs(events: tuple[DensePipelineEvent, ...]) -> list[tuple[int, int]]:
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
    sources: tuple[Path, ...], config: DenseSramPipelineConfig,
    rob_depth: int, temporary: Path,
) -> dict[str, Any]:
    path = temporary / "dma_pipeline.json"
    top = "gtsu_dense_dramsim_dma_pipeline"
    command = (
        f"read_verilog -sv {' '.join(str(source) for source in sources)}; "
        f"chparam -set M {config.m} -set N {config.n} -set K {config.k} "
        f"-set N_TILE {config.n_tile} -set K_BLOCK {config.k_block} "
        f"-set ROB_DEPTH {rob_depth} -set READ_LATENCY {config.read_latency} "
        f"-set FIFO_DEPTH {config.fifo_depth} {top}; "
        f"hierarchy -check -top {top}; proc; opt; check -assert; stat; write_json {path}"
    )
    result = subprocess.run(["yosys", "-p", command], capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"DRAMsim3 DMA Yosys failed:\n{result.stdout}\n{result.stderr}")
    modules = json.loads(path.read_text())["modules"]
    cells = modules[top].get("cells", {}).values()
    dma_instances = sum(
        cell["type"].startswith("$paramod") and "gtsu_dram_dma_unpack" in cell["type"]
        for cell in cells
    )
    dense_instances = sum(
        cell["type"].startswith("$paramod") and "gtsu_dense_sram_requant_pipeline" in cell["type"]
        for cell in cells
    )
    if (dma_instances, dense_instances) != (1, 1):
        raise AssertionError("DMA wrapper hierarchy is incomplete")
    return {
        "status": "passed", "dma_instances": dma_instances,
        "dense_pipeline_instances": dense_instances,
        "dram_controller_is_external_dramsim3": True,
        "log": result.stdout + result.stderr,
    }
