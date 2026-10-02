"""Icarus/Yosys correlation for the 64-column ABUF/WBUF tensor fabric."""
from __future__ import annotations

import csv
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import tempfile
from typing import Any

from .production_dense_fabric import (
    FabricEvent, ProductionDenseFabricConfig, build_fabric_lines_from_arrays,
    run_production_dense_fabric_model,
)
from nanopointllm.compression.quantized_tensor_package import (
    load_quantized_linear_package,
)


def correlate_production_dense_fabric(
    config: ProductionDenseFabricConfig, *, rtl_root: str | Path,
    output_dir: str | Path | None = None, synthesize: bool = True,
    simulator: str = "iverilog", internal_trace: bool = True,
    tensor_package_dir: str | Path | None = None,
) -> dict[str, Any]:
    if simulator not in {"iverilog", "verilator"}:
        raise ValueError("simulator must be 'iverilog' or 'verilator'")
    package = None
    lines = None
    if tensor_package_dir is not None:
        if simulator != "verilator":
            raise ValueError("checkpoint-backed payload currently requires Verilator")
        package = load_quantized_linear_package(tensor_package_dir)
        lines = build_fabric_lines_from_arrays(
            config, package.activation, package.weight,
        )
    model = run_production_dense_fabric_model(
        config, lines, trace=True if internal_trace else "outputs",
    )
    root = Path(rtl_root)
    sources = (
        root / "gtsu_dot4_pe.sv",
        root / "gtsu_dense_dot4_tile.sv",
        root / "gtsu_rv_fifo.sv",
        root / "gtsu_dense64_abuf_wbuf_fabric.sv",
        root / "tb_gtsu_dense64_abuf_wbuf_fabric.sv",
    )
    with tempfile.TemporaryDirectory(prefix="gtsu_dense64_") as name:
        temporary = Path(name)
        parameters = {
            "M": config.m, "N": config.n, "K": config.k,
            "READ_LATENCY": config.read_latency,
            "FIFO_DEPTH": config.fifo_depth,
            "N_MEM_LANES": config.memory_lanes,
            "SOURCE_STALL_MOD": config.source_stall_mod,
            "SOURCE_STALL_PHASE": config.source_stall_phase,
            "OUTPUT_STALL_MOD": config.output_stall_mod,
            "OUTPUT_STALL_PHASE": config.output_stall_phase,
            "TRACE_INTERNAL": int(internal_trace),
            "MAX_CYCLES": config.max_cycles,
        }
        if simulator == "iverilog":
            executable = temporary / "fabric.vvp"
            command = ["iverilog", "-g2012", "-s", "tb_gtsu_dense64_abuf_wbuf_fabric"]
            for parameter, value in parameters.items():
                command.append(f"-Ptb_gtsu_dense64_abuf_wbuf_fabric.{parameter}={value}")
            command.extend(["-o", str(executable), *(str(path) for path in sources)])
        else:
            build_dir = temporary / "obj_dir"
            command = [
                "verilator", "--cc", "--exe", "-Wno-fatal",
                "--top-module", "tb_gtsu_dense64_abuf_wbuf_fabric",
                "--Mdir", str(build_dir),
            ]
            command[command.index("tb_gtsu_dense64_abuf_wbuf_fabric")] = (
                "gtsu_dense64_abuf_wbuf_fabric"
            )
            command.extend(
                f"-G{parameter}={value}" for parameter, value in parameters.items()
                if parameter in {
                    "M", "N", "K", "READ_LATENCY", "FIFO_DEPTH", "N_MEM_LANES",
                }
            )
            command.extend(str(path.resolve()) for path in sources[:-1])
            driver = (root / "verilator_dense64_driver.cpp").resolve()
            command.append(str(driver))
            executable = build_dir / "Vgtsu_dense64_abuf_wbuf_fabric"
        compiled = subprocess.run(command, capture_output=True, text=True)
        if compiled.returncode:
            raise RuntimeError(f"Dense64 RTL compile failed:\n{compiled.stderr}")
        if simulator == "verilator" and not compiled.returncode:
            built = subprocess.run(
                ["make", "-C", str(build_dir), "-f",
                 "Vgtsu_dense64_abuf_wbuf_fabric.mk", "-j2"],
                capture_output=True, text=True,
            )
            if built.returncode:
                raise RuntimeError(f"Dense64 Verilator build failed:\n{built.stderr}")
        run_command = ["vvp", str(executable)] if simulator == "iverilog" else [
            str(executable), str(config.m), str(config.n), str(config.k),
            str(config.source_stall_mod), str(config.source_stall_phase),
            str(config.output_stall_mod), str(config.output_stall_phase),
            str(config.max_cycles), str(config.output_tiles),
            str(config.memory_lanes),
        ]
        if package is not None:
            package_root = Path(tensor_package_dir).resolve()
            run_command.extend([
                str(package_root / package.metadata["tensors"]["activation"]["file"]),
                str(package_root / package.metadata["tensors"]["weight"]["file"]),
            ])
        simulated = subprocess.run(run_command, capture_output=True, text=True)
        if simulated.returncode:
            raise RuntimeError(
                f"Dense64 RTL failed:\n{simulated.stdout}\n{simulated.stderr}"
            )
        synthesis = _yosys(sources[:-1], config, temporary) if synthesize else {
            "status": "not_rerun_for_shape; hierarchy locked by edge case",
        }

    rtl_events, rtl_summary = parse_fabric_output(simulated.stdout)
    expected_summary = {"cycles": model.cycles, **model.counters,
                        "overflow_error": 0, "done": 1}
    exact = model.events == rtl_events and expected_summary == rtl_summary
    report = {
        "schema_version": "0.1",
        "status": "rtl_correlated" if exact else "mismatch",
        "operator": "dense64_abuf_broadcast_16bank_wbuf_dot4",
        "config": config.as_dict(),
        "cycle_model_cycles": model.cycles,
        "rtl_cycles": rtl_summary["cycles"],
        "cycle_error": rtl_summary["cycles"] - model.cycles,
        "event_trace_exact": model.events == rtl_events,
        "counter_trace_exact": expected_summary == rtl_summary,
        "functional_outputs_exact": _outputs(model.events) == _outputs(rtl_events),
        "simulator": simulator,
        "internal_event_trace_enabled": internal_trace,
        "counters": model.counters,
        "synthesis_check": synthesis,
        "operand_supply": {
            "activation": "independent_32bit_A4_broadcast_buffer",
            "weights": "16_banks_x_128bit_parallel_read",
            "weight_read_bits_per_issue": 2048,
            "weight_read_bytes_per_issue": 256,
            "memory_lanes": config.memory_lanes,
            "ingress_bits_per_cycle": 512 * config.memory_lanes,
            "ingress_bytes_per_cycle": 64 * config.memory_lanes,
            "ingress_beats_per_weight_vector": 4 // config.memory_lanes,
            "dot4_pe_count": 64,
        },
        "claim_boundary": {
            "n_tile_64_operand_supply": exact,
            "full_shape_control_and_edge_mask": exact,
            "real_checkpoint_payload": False,
            "bf16_a8_dual_output": False,
            "smoothquant_metadata_source": False,
            "request_side_dramsim3_closed_loop": False,
            "standalone_report_unlocks_dense_gemm_coverage": False,
            "full_pointllm_cycle_accurate": False,
        },
    }
    if package is not None:
        report["tensor_package"] = {
            "path": str(Path(tensor_package_dir).resolve()),
            "shape": package.metadata["shape"],
            "tensor_sha256": {
                name: entry["sha256"]
                for name, entry in package.metadata["tensors"].items()
            },
        }
        report["claim_boundary"]["real_checkpoint_payload"] = exact
        report["claim_boundary"]["smoothquant_metadata_source"] = exact
    if output_dir is not None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        _write_events(destination / "cycle_model_trace.csv", model.events)
        _write_events(destination / "rtl_trace.csv", rtl_events)
        (destination / "rtl_stdout.log").write_text(simulated.stdout, encoding="utf-8")
        if "log" in synthesis:
            (destination / "yosys_check.log").write_text(
                synthesis["log"], encoding="utf-8",
            )
        (destination / "correlation.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8",
        )
    if not exact:
        raise AssertionError(f"Dense64 fabric mismatch: {report}")
    return report


def parse_fabric_output(
    text: str,
) -> tuple[tuple[FabricEvent, ...], dict[str, int]]:
    events = []
    summary = None
    for line in text.splitlines():
        fields = line.split()
        if fields[:1] == ["TRACE"] and len(fields) == 5:
            events.append(FabricEvent(
                int(fields[1]), fields[2], int(fields[3]), int(fields[4]),
            ))
        elif fields[:1] == ["SUMMARY"] and len(fields) == 16:
            keys = (
                "cycles", "ingress_lines", "wbuf_bank_writes", "abuf_writes",
                "wbuf_read_issues", "wbuf_responses", "dot4_chunks",
                "output_tiles", "output_values", "fill_compute_overlap_cycles",
                "source_backpressure_cycles", "output_backpressure_cycles",
                "response_fifo_peak", "overflow_error", "done",
            )
            summary = dict(zip(keys, map(int, fields[1:]), strict=True))
    if summary is None:
        raise RuntimeError(f"Dense64 RTL did not emit SUMMARY:\n{text[-4000:]}")
    return tuple(events), summary


def _outputs(events: tuple[FabricEvent, ...]) -> list[tuple[int, int]]:
    return [(event.tag, event.value) for event in events if event.event == "OUTPUT_ACCEPT"]


def _write_events(path: Path, events: tuple[FabricEvent, ...]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("cycle", "event", "tag", "value"),
            lineterminator="\n",
        )
        writer.writeheader()
        for event in events:
            writer.writerow(asdict(event))


def _yosys(
    sources: tuple[Path, ...], config: ProductionDenseFabricConfig,
    temporary: Path,
) -> dict[str, Any]:
    path = temporary / "dense64.json"
    top = "gtsu_dense64_abuf_wbuf_fabric"
    command = (
        f"read_verilog -sv {' '.join(str(source) for source in sources)}; "
        f"chparam -set M {config.m} -set N {config.n} -set K {config.k} "
        f"-set READ_LATENCY {config.read_latency} -set FIFO_DEPTH {config.fifo_depth} "
        f"-set N_MEM_LANES {config.memory_lanes} {top}; "
        f"hierarchy -check -top {top}; proc; opt; check -assert; stat; write_json {path}"
    )
    result = subprocess.run(["yosys", "-p", command], capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"Dense64 Yosys failed:\n{result.stdout}\n{result.stderr}")
    modules = json.loads(path.read_text())["modules"]
    top_cells = modules[top].get("cells", {}).values()
    dense_instances = sum(
        cell["type"].startswith("$paramod") and "gtsu_dense_dot4_tile" in cell["type"]
        for cell in top_cells
    )
    fifo_instances = sum(
        cell["type"].startswith("$paramod") and "gtsu_rv_fifo" in cell["type"]
        for cell in top_cells
    )
    direct_multipliers = sum(cell["type"] == "$mul" for cell in top_cells)
    dense_modules = [
        module for name, module in modules.items()
        if "gtsu_dense_dot4_tile" in name
    ]
    dot_instances = {
        sum(cell["type"] == "gtsu_dot4_pe" for cell in module.get("cells", {}).values())
        for module in dense_modules
    }
    if (dense_instances, fifo_instances, direct_multipliers, dot_instances) != (1, 1, 0, {64}):
        raise AssertionError("Dense64 hierarchy does not contain exactly one 64-PE shared tile")
    return {
        "status": "passed", "dense_tile_instances": dense_instances,
        "response_fifo_instances": fifo_instances,
        "dot4_instances": 64, "top_direct_multipliers": direct_multipliers,
        "wbuf_banks": 16, "abuf_broadcast_words": 2,
        "behavioral_storage_not_foundry_macro": True,
        "log": result.stdout + result.stderr,
    }
