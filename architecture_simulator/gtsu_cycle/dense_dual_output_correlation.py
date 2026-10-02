"""RTL correlation and BF16-lane sweep for the Dense64 dual output."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import tempfile
from typing import Any

from .dense_dual_output import (
    DenseDualOutputConfig, DenseDualOutputEvent, locked_dense_output_tiles,
    run_dense_dual_output_model,
)


def correlate_dense_dual_output_sweep(
    *, rtl_root: str | Path, output_dir: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(rtl_root)
    rows = []
    for lanes in (8, 16, 32, 64):
        rows.append(_correlate_one(root, DenseDualOutputConfig(bf16_lanes=lanes)))
    exact = all(row["status"] == "rtl_correlated" for row in rows)
    report = {
        "schema_version": "0.1",
        "status": "rtl_correlated" if exact else "mismatch",
        "operator": "dense64_acc32_a8_bf16_dual_output",
        "a8_lanes": 64,
        "bf16_lane_sweep": rows,
        "numerical_contract": {
            "a8": "compiled_multiplier_shift_rne_symmetric[-127,127]",
            "bf16": "exact_ACC32_times_FP16_times_FP16_direct_BF16_RNE",
        },
        "claim_boundary": {
            "a8_64way_value_event_cycle_exact": exact,
            "bf16_8_16_32_64_lane_value_event_cycle_exact": exact,
            "real_qproj_scale_payload": False,
            "dense64_wrapper_composed": False,
            "bias_and_nonlinearity_after_bf16": False,
            "full_pointllm_cycle_accurate": False,
        },
    }
    if output_dir is not None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "correlation.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8",
        )
    if not exact:
        raise AssertionError(f"Dense64 dual-output sweep mismatch: {report}")
    return report


def _correlate_one(root: Path, config: DenseDualOutputConfig) -> dict[str, Any]:
    tiles = locked_dense_output_tiles()
    model = run_dense_dual_output_model(config, tiles)
    sources = (
        root / "gtsu_requantize_int32.sv",
        root / "gtsu_dequant_int32_bf16.sv",
        root / "gtsu_dense64_dual_output.sv",
        root / "tb_gtsu_dense64_dual_output.sv",
    )
    with tempfile.TemporaryDirectory(prefix="gtsu_dense_dual_") as name:
        temporary = Path(name)
        executable = temporary / "dual.vvp"
        command = ["iverilog", "-g2012", "-s", "tb_gtsu_dense64_dual_output"]
        for key, value in {
            "BF16_LANES": config.bf16_lanes,
            "SOURCE_STALL_MOD": config.source_stall_mod,
            "SOURCE_STALL_PHASE": config.source_stall_phase,
            "OUTPUT_STALL_MOD": config.output_stall_mod,
            "OUTPUT_STALL_PHASE": config.output_stall_phase,
            "MAX_CYCLES": config.max_cycles,
        }.items():
            command.append(f"-Ptb_gtsu_dense64_dual_output.{key}={value}")
        command.extend(["-o", str(executable), *(str(path) for path in sources)])
        compiled = subprocess.run(command, capture_output=True, text=True)
        if compiled.returncode:
            raise RuntimeError(f"dual-output compile failed:\n{compiled.stderr}")
        simulated = subprocess.run(
            ["vvp", str(executable)], capture_output=True, text=True,
        )
        if simulated.returncode:
            raise RuntimeError(
                f"dual-output RTL failed:\n{simulated.stdout}\n{simulated.stderr}"
            )
        hierarchy = _hierarchy(sources[:-1], config.bf16_lanes, temporary)
    rtl_events, rtl_summary = _parse(simulated.stdout)
    expected_summary = {"cycles": model.cycles, **model.counters}
    exact = model.events == rtl_events and expected_summary == rtl_summary
    return {
        "bf16_lanes": config.bf16_lanes,
        "bf16_groups_per_tile": config.groups,
        "status": "rtl_correlated" if exact else "mismatch",
        "cycle_model_cycles": model.cycles,
        "rtl_cycles": rtl_summary["cycles"],
        "cycle_error": rtl_summary["cycles"] - model.cycles,
        "event_trace_exact": model.events == rtl_events,
        "counter_trace_exact": expected_summary == rtl_summary,
        "counters": model.counters,
        "hierarchy_check": hierarchy,
    }


def _parse(text: str):
    events = []
    summary = None
    for line in text.splitlines():
        fields = line.split()
        if fields[:1] == ["TRACE"] and len(fields) == 5:
            value = int(fields[4], 16) if fields[2] in {
                "BF16_VALUE", "BF16_OUTPUT",
            } else int(fields[4])
            events.append(DenseDualOutputEvent(
                int(fields[1]), fields[2], int(fields[3]), value,
            ))
        elif fields[:1] == ["SUMMARY"] and len(fields) == 9:
            keys = (
                "cycles", "input_tiles", "a8_values", "bf16_groups",
                "bf16_values", "output_tiles", "source_backpressure_cycles",
                "output_backpressure_cycles",
            )
            summary = dict(zip(keys, map(int, fields[1:]), strict=True))
    if summary is None:
        raise RuntimeError(f"dual-output RTL did not emit SUMMARY:\n{text[-4000:]}")
    return tuple(events), summary


def _hierarchy(sources, lanes: int, temporary: Path) -> dict[str, Any]:
    path = temporary / "dual.json"
    top = "gtsu_dense64_dual_output"
    command = (
        f"read_verilog -sv {' '.join(str(source) for source in sources)}; "
        f"chparam -set BF16_LANES {lanes} {top}; "
        f"hierarchy -check -top {top}; check -assert; write_json {path}"
    )
    result = subprocess.run(["yosys", "-p", command], capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"dual-output hierarchy check failed:\n{result.stderr}")
    cells = json.loads(path.read_text())["modules"][top].get("cells", {}).values()
    requant = sum("gtsu_requantize_int32" in cell["type"] for cell in cells)
    dequant = sum("gtsu_dequant_int32_bf16" in cell["type"] for cell in cells)
    if requant != 64 or dequant != lanes:
        raise AssertionError(
            f"dual-output hierarchy expected 64/{lanes} A8/BF16 lanes, got {requant}/{dequant}"
        )
    return {
        "status": "passed", "a8_requant_instances": requant,
        "bf16_dequant_instances": dequant,
        "standalone_arithmetic_modules_already_synthesized": True,
        "behavioral_storage_not_ppa_evidence": True,
    }
