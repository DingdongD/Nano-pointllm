"""Icarus/Yosys correlation for exact BF16 accumulator dequantization."""
from __future__ import annotations

import json
from pathlib import Path
import random
import subprocess
import struct
import tempfile
from typing import Any

from .bf16_dequant import (
    Bf16DequantConfig, Bf16DequantEvent, Bf16DequantVector,
    dequantize_int32_to_bf16,
    locked_bf16_dequant_vectors, run_bf16_dequant_model,
)


def correlate_bf16_dequant(
    *, rtl_root: str | Path, output_dir: str | Path | None = None,
    vectors: tuple[Bf16DequantVector, ...] | None = None,
    config: Bf16DequantConfig | None = None,
) -> dict[str, Any]:
    config = config or Bf16DequantConfig()
    vectors = vectors or locked_bf16_dequant_vectors()
    model = run_bf16_dequant_model(vectors, config)
    root = Path(rtl_root)
    sources = (
        root / "gtsu_dequant_int32_bf16.sv",
        root / "tb_gtsu_dequant_int32_bf16.sv",
    )
    with tempfile.TemporaryDirectory(prefix="gtsu_bf16_dequant_") as name:
        temporary = Path(name)
        trace = temporary / "dequant.hex"
        write_dequant_trace(trace, vectors)
        executable = temporary / "dequant.vvp"
        command = ["iverilog", "-g2012", "-s", "tb_gtsu_dequant_int32_bf16"]
        for parameter, value in {
            "VECTORS": len(vectors),
            "SOURCE_STALL_MOD": config.source_stall_mod,
            "SOURCE_STALL_PHASE": config.source_stall_phase,
            "OUTPUT_STALL_MOD": config.output_stall_mod,
            "OUTPUT_STALL_PHASE": config.output_stall_phase,
            "MAX_CYCLES": config.max_cycles,
        }.items():
            command.append(f"-Ptb_gtsu_dequant_int32_bf16.{parameter}={value}")
        command.extend(["-o", str(executable), *(str(path) for path in sources)])
        compiled = subprocess.run(command, capture_output=True, text=True)
        if compiled.returncode:
            raise RuntimeError(f"BF16 dequant RTL compile failed:\n{compiled.stderr}")
        simulated = subprocess.run(
            ["vvp", str(executable), f"+TRACE_FILE={trace}"],
            capture_output=True, text=True,
        )
        if simulated.returncode:
            raise RuntimeError(
                f"BF16 dequant RTL failed:\n{simulated.stdout}\n{simulated.stderr}"
            )
        synthesis = _yosys(sources[0], temporary)

    rtl_events, rtl_summary = parse_dequant_output(simulated.stdout)
    expected_summary = {"cycles": model.cycles, **model.counters}
    oracle_validation = _validate_torch_bf16_oracle()
    exact = model.events == rtl_events and expected_summary == rtl_summary
    report = {
        "schema_version": "0.1",
        "status": "rtl_correlated" if exact else "mismatch",
        "operator": "int32_fp16x2_to_bf16_dequant",
        "numerical_contract": {
            "operation": "exact_int32_times_fp16_times_fp16",
            "rounding": "direct_bf16_round_to_nearest_ties_to_even",
            "intermediate_host_float": False,
            "bias": "not_in_this_slice",
            "finite_fp16_scales_required": True,
        },
        "config": config.as_dict(),
        "vector_count": len(vectors),
        "cycle_model_cycles": model.cycles,
        "rtl_cycles": rtl_summary["cycles"],
        "cycle_error": rtl_summary["cycles"] - model.cycles,
        "event_trace_exact": model.events == rtl_events,
        "counter_trace_exact": expected_summary == rtl_summary,
        "outputs_bf16_hex": [f"0x{value:04x}" for value in model.outputs],
        "independent_numerical_oracle": oracle_validation,
        "synthesis_check": synthesis,
        "compiler_codes_reference": {
            "commit": "ad2c31a",
            "structural_references": [
                "rtl/ACTransformer_DS/rtl/hdl/ctc/ct_cluster_ds/common/int32_fp32.v",
                "rtl/ACTransformer_DS/rtl/hdl/ctc/ct_cluster_ds/common/fp32_bf16.v",
                "rtl/ACTransformer_DS/rtl/hdl/ctc/ct_cluster_ds/pp/pp_unit_ds.sv",
            ],
            "source_copied": False,
            "intentional_difference": (
                "GTSU rounds the exact fused scale product once to BF16; the "
                "reference stages INT32-to-FP32 and floating multiplies"
            ),
        },
        "claim_boundary": {
            "bf16_value_event_cycle_exact": exact,
            "integrated_dense_output_path": False,
            "bias_and_activation": False,
            "full_pointllm_cycle_accurate": False,
        },
    }
    if output_dir is not None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        write_dequant_trace(destination / "dequant_trace.hex", vectors)
        (destination / "rtl_stdout.log").write_text(simulated.stdout, encoding="utf-8")
        (destination / "yosys_check.log").write_text(synthesis["log"], encoding="utf-8")
        (destination / "correlation.json").write_text(json.dumps(report, indent=2) + "\n")
    if not exact:
        raise AssertionError(f"BF16 dequant RTL mismatch: {report}")
    return report


def write_dequant_trace(path: Path, vectors: tuple[Bf16DequantVector, ...]) -> None:
    with path.open("w", encoding="ascii") as handle:
        for vector in vectors:
            value = vector.tag & 0xFFFF
            value |= (vector.accumulator & 0xFFFFFFFF) << 16
            value |= vector.activation_scale_fp16 << 48
            value |= vector.weight_scale_fp16 << 64
            handle.write(f"{value:020x}\n")


def parse_dequant_output(
    text: str,
) -> tuple[tuple[Bf16DequantEvent, ...], dict[str, int]]:
    events = []
    summary = None
    for line in text.splitlines():
        fields = line.split()
        if fields[:1] == ["TRACE"] and len(fields) == 5:
            value = int(fields[4], 16) if fields[2] == "OUTPUT_ACCEPT" else int(fields[4])
            events.append(Bf16DequantEvent(
                int(fields[1]), fields[2], int(fields[3]), value,
            ))
        elif fields[:1] == ["SUMMARY"] and len(fields) == 6:
            keys = ("cycles", "input_accepts", "output_accepts",
                    "source_backpressure_cycles", "output_backpressure_cycles")
            summary = dict(zip(keys, map(int, fields[1:]), strict=True))
    if summary is None:
        raise RuntimeError(f"BF16 dequant RTL did not emit SUMMARY:\n{text[-4000:]}")
    return tuple(events), summary


def _yosys(source: Path, temporary: Path) -> dict[str, Any]:
    json_path = temporary / "dequant.json"
    command = (
        f"read_verilog -sv {source}; hierarchy -check -top gtsu_dequant_int32_bf16; "
        f"proc; opt; check -assert; stat; write_json {json_path}"
    )
    result = subprocess.run(["yosys", "-p", command], capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"BF16 dequant Yosys failed:\n{result.stdout}\n{result.stderr}")
    cells = json.loads(json_path.read_text())["modules"]["gtsu_dequant_int32_bf16"].get("cells", {}).values()
    multiplier_count = sum(cell["type"] == "$mul" for cell in cells)
    divider_count = sum(cell["type"] in ("$div", "$mod") for cell in cells)
    if multiplier_count != 2 or divider_count:
        raise AssertionError(
            f"BF16 dequant expected two multipliers/no divider, got {multiplier_count}/{divider_count}"
        )
    return {"status": "passed", "multipliers": multiplier_count,
            "dividers": divider_count, "log": result.stdout + result.stderr}


def _validate_torch_bf16_oracle(samples: int = 1000) -> dict[str, Any]:
    import torch

    generator = random.Random(20261002)
    mismatches = 0
    for tag in range(samples):
        accumulator = generator.randint(-(1 << 31), (1 << 31) - 1)
        scale_a = generator.randrange(0x0001, 0x7C00) | (generator.randrange(2) << 15)
        scale_w = generator.randrange(0x0001, 0x7C00) | (generator.randrange(2) << 15)
        value_a = struct.unpack("e", struct.pack("H", scale_a))[0]
        value_w = struct.unpack("e", struct.pack("H", scale_w))[0]
        expected = int(torch.tensor(
            accumulator * value_a * value_w, dtype=torch.float64,
        ).to(torch.bfloat16).view(torch.uint16).item())
        actual = dequantize_int32_to_bf16(Bf16DequantVector(
            tag, accumulator, scale_a, scale_w,
        ))
        mismatches += int(actual != expected)
    if mismatches:
        raise AssertionError(f"BF16 independent oracle found {mismatches} mismatches")
    return {
        "status": "passed", "samples": samples, "mismatches": mismatches,
        "reference": "exactly-representable Float64 product rounded by torch.bfloat16",
        "seed": 20261002,
    }
