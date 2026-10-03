"""Icarus/Yosys correlation for the Compiler-style geometry SRAM fabric."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any


def correlate_geometry_sram(
    *, rtl_root: str | Path, output_dir: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(rtl_root)
    sources = (
        root / "gtsu_lbuf_16bank_macro_model.sv",
        root / "gtsu_geometry_sram_fabric.sv",
        root / "tb_gtsu_geometry_sram_fabric.sv",
    )
    iverilog, vvp = shutil.which("iverilog"), shutil.which("vvp")
    if not iverilog or not vvp:
        raise RuntimeError("Icarus Verilog is required for geometry SRAM correlation")
    with tempfile.TemporaryDirectory(prefix="gtsu_geometry_sram_") as name:
        executable = Path(name) / "geometry_sram.vvp"
        compiled = subprocess.run([
            iverilog, "-g2012", "-s", "tb_gtsu_geometry_sram_fabric",
            "-o", str(executable), *(str(path) for path in sources),
        ], capture_output=True, text=True)
        if compiled.returncode:
            raise RuntimeError(f"geometry SRAM compile failed:\n{compiled.stderr}")
        simulated = subprocess.run(
            [vvp, str(executable)], capture_output=True, text=True,
        )
        if simulated.returncode:
            raise RuntimeError(
                f"geometry SRAM simulation failed:\n"
                f"{simulated.stdout}\n{simulated.stderr}"
            )
    trace, summary = _parse_output(simulated.stdout)
    expected_trace = (
        (130, "ISSUE", 0, 0),
        (131, "ISSUE", 1, 1),
        (133, "RESPONSE", 0, -1),
        (134, "RESPONSE", 1, -1),
        (136, "ISSUE", 2, 0),
        (139, "RESPONSE", 2, -1),
    )
    expected_summary = {
        "cycles": 142, "point_loads": 128, "min_loads": 2,
        "requests": 3, "responses": 3, "mismatches": 0,
        "same_address_rw_collisions": 0,
    }
    exact = trace == expected_trace and summary == expected_summary
    synthesis = _run_yosys(sources[:2])
    report = {
        "schema_version": "0.1",
        "status": "rtl_correlated" if exact else "mismatch",
        "operator": "geometry_compiler_style_sram_fabric",
        "lock_config": {
            "points": 128, "lanes": 64, "rows": 2,
            "point_groups": 4, "banks_per_group": 16,
            "word_bits": 128, "read_latency_cycles": 3,
        },
        "synthesis_structure_lock": {
            "points": 64, "rows_per_bank": 1,
            "reason": "avoid interpreting behavioral-array flop expansion as SRAM PPA",
        },
        "production_mapping": {
            "points": 8192, "lanes": 64, "rows": 128,
            "point_norm_sram_groups": 4,
            "point_norm_banks": 64,
            "point_norm_bytes": 8192 * 16,
            "min_state_sram_groups": 1,
            "min_state_banks": 16,
            "min_state_bytes": 8192 * 4,
            "total_geometry_sram_bytes": 8192 * 20,
        },
        "event_trace_exact": trace == expected_trace,
        "value_checks": 3 * 64 * 2,
        "value_mismatches": summary["mismatches"],
        "read_latency_exact": all(
            response[0] - issue[0] == 3
            for issue, response in zip(
                (trace[0], trace[1], trace[4]),
                (trace[2], trace[3], trace[5]), strict=True,
            )
        ),
        "summary": summary,
        "synthesis_check": synthesis,
        "provenance": {
            "reference_repository": "/home/Compiler_Codes",
            "reference_commit": "ad2c31a",
            "reference_contract": (
                "LBUF 16 banks x 128 bits, byte enables, independent read/write, "
                "configurable 2/3-cycle read latency"
            ),
            "implementation": "clean-room parameterized behavior model",
            "proprietary_reference_source_copied": False,
        },
        "claim_boundary": {
            "sram_behavior_rtl_correlated": exact,
            "production_capacity_mapped": True,
            "foundry_sram_macro_instantiated": False,
            "sram_sta_ppa_characterized": False,
            "fp32_arithmetic_composed": False,
            "selector_composed": False,
            "full_pointllm_cycle_accurate": False,
        },
    }
    if output_dir is not None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "rtl_stdout.log").write_text(
            simulated.stdout, encoding="utf-8",
        )
        (destination / "yosys_check.log").write_text(
            synthesis["log"], encoding="utf-8",
        )
        (destination / "correlation.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8",
        )
    if not exact:
        raise AssertionError(f"geometry SRAM mismatch: {report}")
    return report


def _parse_output(text: str):
    trace = []
    summary = None
    for line in text.splitlines():
        fields = line.split()
        if fields[:1] == ["TRACE"] and fields[2] == "ISSUE":
            trace.append((int(fields[1]), "ISSUE", int(fields[3]), int(fields[4])))
        elif fields[:1] == ["TRACE"] and fields[2] == "RESPONSE":
            trace.append((int(fields[1]), "RESPONSE", int(fields[3]), -1))
        elif fields[:1] == ["SUMMARY"] and len(fields) == 8:
            keys = (
                "cycles", "point_loads", "min_loads", "requests", "responses",
                "mismatches", "same_address_rw_collisions",
            )
            summary = dict(zip(keys, map(int, fields[1:]), strict=True))
    if summary is None:
        raise RuntimeError(f"geometry SRAM summary missing:\n{text[-4000:]}")
    return tuple(trace), summary


def _run_yosys(sources: tuple[Path, ...]) -> dict[str, Any]:
    yosys = shutil.which("yosys")
    if not yosys:
        return {"status": "unavailable", "log": ""}
    command = (
        f"read_verilog -sv {' '.join(str(path) for path in sources)}; "
        "chparam -set POINTS 64 gtsu_geometry_sram_fabric; "
        "hierarchy -check -top gtsu_geometry_sram_fabric; "
        "proc; opt; check -assert; stat"
    )
    result = subprocess.run([yosys, "-p", command], capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(
            f"geometry SRAM Yosys failed:\n{result.stdout}\n{result.stderr}"
        )
    required = ("gtsu_geometry_sram_fabric", "gtsu_lbuf_16bank_macro_model")
    missing = [name for name in required if name not in result.stdout]
    if missing:
        raise AssertionError(f"geometry SRAM hierarchy missing {missing}")
    return {"status": "passed", "required_modules": list(required), "log": result.stdout}
