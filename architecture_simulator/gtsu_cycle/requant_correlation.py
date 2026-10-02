"""Icarus/Yosys correlation for the post-accumulator requant unit."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import tempfile
from typing import Any

from .requant import (
    RequantConfig, RequantEvent, RequantVector, compile_fp16_requant_scale,
    locked_requant_vectors, run_requant_model,
)


def correlate_requant(
    *, rtl_root: str | Path, output_dir: str | Path | None = None,
    vectors: tuple[RequantVector, ...] | None = None,
    config: RequantConfig | None = None,
) -> dict[str, Any]:
    config = config or RequantConfig()
    vectors = vectors or locked_requant_vectors()
    model = run_requant_model(vectors, config)
    root = Path(rtl_root)
    sources = (root / "gtsu_requantize_int32.sv", root / "tb_gtsu_requantize_int32.sv")
    with tempfile.TemporaryDirectory(prefix="gtsu_requant_") as temporary_name:
        temporary = Path(temporary_name)
        trace = temporary / "requant.hex"
        write_requant_trace(trace, vectors)
        executable = temporary / "requant.vvp"
        command = ["iverilog", "-g2012", "-s", "tb_gtsu_requantize_int32"]
        for name, value in {
            "VECTORS": len(vectors),
            "SOURCE_STALL_MOD": config.source_stall_mod,
            "SOURCE_STALL_PHASE": config.source_stall_phase,
            "OUTPUT_STALL_MOD": config.output_stall_mod,
            "OUTPUT_STALL_PHASE": config.output_stall_phase,
            "MAX_CYCLES": config.max_cycles,
        }.items():
            command.append(f"-Ptb_gtsu_requantize_int32.{name}={value}")
        command.extend(["-o", str(executable), *(str(path) for path in sources)])
        compiled = subprocess.run(command, capture_output=True, text=True)
        if compiled.returncode:
            raise RuntimeError(f"requant RTL compile failed:\n{compiled.stderr}")
        simulated = subprocess.run(
            ["vvp", str(executable), f"+TRACE_FILE={trace}"],
            capture_output=True, text=True,
        )
        if simulated.returncode:
            raise RuntimeError(f"requant RTL failed:\n{simulated.stdout}\n{simulated.stderr}")
        synthesis = _yosys(sources[0], temporary)
    rtl_events, rtl_summary = parse_requant_output(simulated.stdout)
    expected_summary = {
        "cycles": model.cycles,
        "input_accepts": model.counters["input_accepts"],
        "output_accepts": model.counters["output_accepts"],
        "source_backpressure_cycles": model.counters["source_backpressure_cycles"],
        "output_backpressure_cycles": model.counters["output_backpressure_cycles"],
    }
    compiled_scales = [
        compile_fp16_requant_scale(*case) for case in (
            (0.0124359130859375, 0.0036792755126953125, 0.03125),
            (0.0169525146484375, 0.0022430419921875, 0.015625),
            (0.036956787109375, 0.0011730194091796875, 0.0625),
        )
    ]
    exact = model.events == rtl_events and expected_summary == rtl_summary
    report = {
        "schema_version": "0.1",
        "status": "rtl_correlated" if exact else "mismatch",
        "operator": "int32_multiplier_shift_requant",
        "numerical_contract": {
            "bias_domain": "int32_accumulator",
            "multiplier": "unsigned_32bit_offline_compiled",
            "rounding": "round_to_nearest_ties_to_even_on_magnitude",
            "saturation": "symmetric_int8_-127_to_127",
            "zero_point": 0,
        },
        "config": config.as_dict(),
        "vector_count": len(vectors),
        "cycle_model_cycles": model.cycles,
        "rtl_cycles": rtl_summary["cycles"],
        "cycle_error": rtl_summary["cycles"] - model.cycles,
        "event_trace_exact": model.events == rtl_events,
        "counter_trace_exact": expected_summary == rtl_summary,
        "outputs": list(model.outputs),
        "fp16_scale_compiler": {
            "status": "passed",
            "cases": [asdict(item) for item in compiled_scales],
            "max_relative_ratio_error": max(item.relative_error for item in compiled_scales),
        },
        "synthesis_check": synthesis,
        "compiler_codes_reference": {
            "commit": "ad2c31a",
            "structural_references": [
                "rtl/ACTransformer_DS/rtl/hdl/ctc/ct_cluster_ds/pp/act_bit_down.sv",
                "rtl/ACTransformer_DS/rtl/hdl/epu/alu/epu_bf16_to_int8.sv",
                "rtl/ACTransformer_DS/rtl/hdl/ctc/ct_cluster_ds/pp/pp_unit_ds.sv",
            ],
            "source_copied": False,
            "intentional_difference": (
                "GTSU uses an arbitrary offline multiplier plus shift and clips "
                "to [-127,127]; the reference shift path is power-of-two only "
                "and its BF16 converter permits -128"
            ),
        },
        "claim_boundary": {
            "requant_value_event_cycle_exact": exact,
            "fp16_scale_to_multiplier_compiler": True,
            "floating_dequant_bf16_path": False,
            "integrated_dense_sram_path": False,
        },
    }
    if output_dir is not None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        write_requant_trace(destination / "requant_trace.hex", vectors)
        (destination / "rtl_stdout.log").write_text(simulated.stdout, encoding="utf-8")
        (destination / "yosys_check.log").write_text(synthesis["log"], encoding="utf-8")
        (destination / "correlation.json").write_text(json.dumps(report, indent=2) + "\n")
    if not exact:
        raise AssertionError(f"requant RTL mismatch: {report}")
    return report


def write_requant_trace(path: Path, vectors: tuple[RequantVector, ...]) -> None:
    with path.open("w", encoding="ascii") as handle:
        for vector in vectors:
            value = vector.tag & 0xFFFF
            value |= (vector.accumulator & 0xFFFFFFFF) << 16
            value |= (vector.bias & 0xFFFFFFFF) << 48
            value |= vector.multiplier << 80
            value |= vector.right_shift << 112
            handle.write(f"{value:030x}\n")


def parse_requant_output(text: str) -> tuple[tuple[RequantEvent, ...], dict[str, int]]:
    events = []
    summary = None
    for line in text.splitlines():
        fields = line.split()
        if fields[:1] == ["TRACE"] and len(fields) == 5:
            events.append(RequantEvent(int(fields[1]), fields[2], int(fields[3]), int(fields[4])))
        elif fields[:1] == ["SUMMARY"] and len(fields) == 6:
            keys = ("cycles", "input_accepts", "output_accepts",
                    "source_backpressure_cycles", "output_backpressure_cycles")
            summary = dict(zip(keys, map(int, fields[1:]), strict=True))
    if summary is None:
        raise RuntimeError(f"requant RTL did not emit SUMMARY:\n{text[-4000:]}")
    return tuple(events), summary


def _yosys(source: Path, temporary: Path) -> dict[str, Any]:
    json_path = temporary / "requant.json"
    command = (
        f"read_verilog -sv {source}; hierarchy -check -top gtsu_requantize_int32; "
        f"proc; opt; check -assert; stat; write_json {json_path}"
    )
    result = subprocess.run(["yosys", "-p", command], capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"requant Yosys failed:\n{result.stdout}\n{result.stderr}")
    cells = json.loads(json_path.read_text())["modules"]["gtsu_requantize_int32"].get("cells", {}).values()
    multiplier_count = sum(cell["type"] == "$mul" for cell in cells)
    divider_count = sum(cell["type"] in ("$div", "$mod") for cell in cells)
    if multiplier_count != 1 or divider_count:
        raise AssertionError("requant must synthesize as one multiplier and no divider")
    return {"status": "passed", "multipliers": multiplier_count,
            "dividers": divider_count, "log": result.stdout + result.stderr}
