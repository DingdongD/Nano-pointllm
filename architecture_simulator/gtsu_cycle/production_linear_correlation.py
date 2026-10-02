"""RTL correlation utilities for the production linear controller."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import tempfile

from .pointllm_lowering import LoweredLinear
from .production_linear import (
    DmaArrival,
    ProductionLinearConfig,
    run_production_linear_model,
)


def correlate_production_linear(
    lowered: LoweredLinear,
    arrivals: tuple[DmaArrival, ...],
    *,
    rtl_root: str | Path,
    output_dir: str | Path | None = None,
    config: ProductionLinearConfig | None = None,
) -> dict[str, object]:
    config = config or ProductionLinearConfig()
    model = run_production_linear_model(lowered, arrivals, config=config)
    iverilog = shutil.which("iverilog")
    vvp = shutil.which("vvp")
    if not iverilog or not vvp:
        raise RuntimeError("Icarus Verilog is required")
    rtl_root = Path(rtl_root)
    sources = (
        rtl_root / "gtsu_production_linear_controller.sv",
        rtl_root / "tb_gtsu_production_linear_controller.sv",
    )
    base_chunks, extra = divmod(
        lowered.k // lowered.config.dram_burst_bytes,
        lowered.config.split_k,
    )
    with tempfile.TemporaryDirectory(prefix="gtsu_prod_linear_") as temporary:
        temporary = Path(temporary)
        trace = temporary / "dma_trace.hex"
        write_dma_trace(trace, arrivals)
        executable = temporary / "linear.vvp"
        parameters = {
            "N_OUTPUTS": lowered.n,
            "SPLIT_K": lowered.config.split_k,
            "SCALE_BURSTS": lowered.scale_burst_count,
            "WEIGHT_BURSTS": lowered.weight_burst_count,
            "TOTAL_REQUESTS": len(arrivals),
            "ROB_DEPTH": config.rob_depth,
            "BASE_CHUNKS": base_chunks,
            "EXTRA_PARTITIONS": extra,
            "MAX_CYCLES": config.max_cycles,
        }
        command = [iverilog, "-g2012", "-s", "tb_gtsu_production_linear_controller"]
        for name, value in parameters.items():
            command.extend([f"-Ptb_gtsu_production_linear_controller.{name}={value}"])
        command.extend(["-o", str(executable), *(str(source) for source in sources)])
        compiled = subprocess.run(command, check=False, capture_output=True, text=True)
        if compiled.returncode:
            raise RuntimeError(f"integrated linear RTL compile failed:\n{compiled.stderr}")
        simulated = subprocess.run(
            [vvp, str(executable), f"+TRACE_FILE={trace}"],
            check=False, capture_output=True, text=True,
        )
        if simulated.returncode:
            raise RuntimeError(
                f"integrated linear RTL simulation failed:\n{simulated.stdout}\n{simulated.stderr}"
            )
    rtl = _parse_summary(simulated.stdout)
    expected = {
        "cycles": model.cycles,
        "dma_accept": model.counters["dma_ingress_bursts"],
        "scale_bursts": model.counters["scale_bursts"],
        "weight_bursts": model.counters["weight_bursts"],
        "read_issue": model.counters["compute_bursts"],
        "compute": model.counters["compute_bursts"],
        "partial": model.counters["partial_sums"],
        "output": model.counters["outputs"],
        "bank_conflict": model.counters.get("sram_read_bank_conflicts", 0),
        "max_rob": model.counters["rob_max_occupancy"],
    }
    exact = expected == rtl
    report = {
        "schema_version": "0.1",
        "status": "rtl_correlated" if exact else "mismatch",
        "projection": lowered.projection,
        "shape": {"m": lowered.m, "n": lowered.n, "k": lowered.k},
        "production_shape": lowered.n >= 4096 and lowered.k >= 4096,
        "exact": exact,
        "cycle_error": rtl["cycles"] - expected["cycles"],
        "cycle_model": expected,
        "rtl": rtl,
        "config": config.__dict__,
        "claim_boundary": {
            "dram_completion_trace": True,
            "controller_cycle_rtl_correlated": exact,
            "arithmetic_value_rtl_correlated_separately": True,
            "full_pointllm_cycle_accurate": False,
        },
    }
    if output_dir is not None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        write_dma_trace(destination / "dma_trace.hex", arrivals)
        (destination / "production_linear_correlation.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8",
        )
    if not exact:
        raise AssertionError(f"production linear RTL mismatch: {report}")
    return report


def write_dma_trace(path: str | Path, arrivals: tuple[DmaArrival, ...]) -> None:
    path = Path(path)
    with path.open("w", encoding="ascii") as handle:
        for arrival in arrivals:
            burst = arrival.burst
            value = arrival.cycle & 0xFFFFFFFF
            value |= int(burst.kind == "weight") << 32
            value |= (burst.linear_sequence & 0xFFFFFFFF) << 33
            value |= (burst.output_row & 0xFFFF) << 65
            value |= (burst.split_index & 0xFF) << 81
            value |= (burst.chunk_index & 0xFF) << 89
            value |= (burst.chunks_in_split & 0xFF) << 97
            handle.write(f"{value:032x}\n")


def _parse_summary(stdout: str) -> dict[str, int]:
    for line in stdout.splitlines():
        fields = line.split()
        if fields and fields[0] == "SUMMARY" and len(fields) == 11:
            values = [int(value) for value in fields[1:]]
            keys = (
                "cycles", "dma_accept", "scale_bursts", "weight_bursts",
                "read_issue", "compute", "partial", "output",
                "bank_conflict", "max_rob",
            )
            return dict(zip(keys, values, strict=True))
    raise RuntimeError(f"RTL did not emit a production linear SUMMARY:\n{stdout[-4000:]}")
