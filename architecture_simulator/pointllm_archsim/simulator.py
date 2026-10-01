"""Phase scheduler and aggregate counters for PointLLM architecture studies."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import replace
from typing import Any

from .engines import EngineModels
from .schema import (
    HardwareConfig,
    OperationResult,
    PhaseResult,
    SimulationResult,
    Workload,
)


class ArchitectureSimulator:
    """Serial phase-level model with intra-operation compute/memory overlap.

    The schedule is intentionally conservative between operations. Each engine
    estimate may overlap compute, HBM, unpack, and SRAM work when its contract
    permits it. Cross-operation overlap is deferred until a trace or RTL model
    supplies producer/consumer events.
    """

    def __init__(self, config: HardwareConfig):
        config.validate()
        self.config = config
        self.engines = EngineModels(config)

    def run(self, workload: Workload) -> SimulationResult:
        workload.validate()
        operation_results: list[OperationResult] = []
        phase_cycles: dict[str, int] = defaultdict(int)
        phase_hbm: dict[str, int] = defaultdict(int)
        phase_sram: dict[str, int] = defaultdict(int)
        phase_macs: dict[str, int] = defaultdict(int)
        phase_energy: dict[str, float] = defaultdict(float)
        phase_energy_known: dict[str, bool] = defaultdict(lambda: True)
        phase_ops: Counter[str] = Counter()
        counters: Counter[str] = Counter()
        cycle_cursor = 0
        previous_phase: str | None = None

        for operation in workload.operations:
            if previous_phase is not None and operation.phase != previous_phase:
                cycle_cursor += self.config.phase_reconfiguration_cycles
                phase_cycles[operation.phase] += self.config.phase_reconfiguration_cycles
                counters["phase_reconfigurations"] += 1
                counters["phase_reconfiguration_cycles"] += (
                    self.config.phase_reconfiguration_cycles
                )
            start_cycle = cycle_cursor
            result = self.engines.estimate(operation)
            cycle_cursor += result.cycles
            result = replace(
                result,
                details={
                    **result.details,
                    "start_cycle": start_cycle,
                    "end_cycle": cycle_cursor,
                    "reference": operation.reference,
                },
            )
            operation_results.append(result)
            phase_cycles[result.phase] += result.cycles
            phase_hbm[result.phase] += result.hbm_read_bytes + result.hbm_write_bytes
            phase_sram[result.phase] += result.sram_bytes
            phase_macs[result.phase] += result.macs
            phase_ops[result.phase] += 1
            if result.energy_pj is None:
                phase_energy_known[result.phase] = False
            else:
                phase_energy[result.phase] += result.energy_pj
            counters[f"kind.{result.kind}.cycles"] += result.cycles
            counters[f"dataflow.{result.dataflow}.cycles"] += result.cycles
            counters[f"bottleneck.{result.bottleneck}.cycles"] += result.cycles
            counters[f"phase.{result.phase}.cycles"] += result.cycles
            counters["hbm_read_bytes"] += result.hbm_read_bytes
            counters["hbm_write_bytes"] += result.hbm_write_bytes
            counters["sram_bytes"] += result.sram_bytes
            counters["macs"] += result.macs
            counters["flops"] += result.flops
            previous_phase = operation.phase

        ordered_phases = []
        for operation in workload.operations:
            if operation.phase not in ordered_phases:
                ordered_phases.append(operation.phase)
        phase_results = tuple(
            PhaseResult(
                phase=phase,
                cycles=phase_cycles[phase],
                seconds=phase_cycles[phase] / self.config.clock_hz,
                hbm_bytes=phase_hbm[phase],
                sram_bytes=phase_sram[phase],
                macs=phase_macs[phase],
                energy_pj=(phase_energy[phase] if phase_energy_known[phase] else None),
                operation_count=phase_ops[phase],
            )
            for phase in ordered_phases
        )
        total_energy = (
            sum(result.energy_pj for result in operation_results if result.energy_pj is not None)
            if all(result.energy_pj is not None for result in operation_results)
            else None
        )
        area_values = self.config.characterization.block_area_um2
        area_um2 = sum(area_values.values()) if area_values else None
        seconds = cycle_cursor / self.config.clock_hz
        output_tokens = workload.metadata.get("output_tokens")
        batch_size = workload.metadata.get("batch_size", 1)
        throughput = (
            float(output_tokens) * float(batch_size) / seconds
            if output_tokens and seconds > 0
            else None
        )
        return SimulationResult(
            config=self.config.to_dict(),
            workload={
                "name": workload.name,
                "schema_version": workload.schema_version,
                "operation_count": len(workload.operations),
                "metadata": workload.metadata,
            },
            cycles=cycle_cursor,
            seconds=seconds,
            throughput_tokens_per_second=throughput,
            phase_results=phase_results,
            operation_results=tuple(operation_results),
            counters=dict(counters),
            energy_pj=total_energy,
            area_um2=area_um2,
            methodology={
                "fidelity": "trace-driven phase-level analytical cycle/traffic model",
                "cycle_accuracy_claim": False,
                "cross_operation_overlap": False,
                "intra_operation_overlap": (
                    "compute/HBM/unpack/SRAM overlap is modeled when double-buffer contracts permit"
                ),
                "energy_status": (
                    "characterized" if total_energy is not None else "disabled until RTL/PTPX/CACTI inputs are supplied"
                ),
                "area_status": (
                    "characterized block sum" if area_um2 is not None else "resource counts only; no area claim"
                ),
            },
        )


def summary_row(result: SimulationResult) -> dict[str, Any]:
    phases = {item.phase: item for item in result.phase_results}
    row: dict[str, Any] = {
        "config": result.config["name"],
        "workload": result.workload["name"],
        "cycles": result.cycles,
        "latency_ms": result.seconds * 1e3,
        "tokens_per_second": result.throughput_tokens_per_second,
        "hbm_gb": (
            result.counters.get("hbm_read_bytes", 0)
            + result.counters.get("hbm_write_bytes", 0)
        ) / 1e9,
        "energy_mj": result.energy_pj / 1e9 if result.energy_pj is not None else None,
        "area_mm2": result.area_um2 / 1e6 if result.area_um2 is not None else None,
    }
    for phase, phase_result in phases.items():
        row[f"{phase}_ms"] = phase_result.seconds * 1e3
        row[f"{phase}_share"] = phase_result.cycles / result.cycles
    bottlenecks = {
        key.removeprefix("bottleneck.").removesuffix(".cycles"): value
        for key, value in result.counters.items()
        if key.startswith("bottleneck.") and key.endswith(".cycles")
    }
    if bottlenecks:
        row["dominant_bottleneck"] = max(bottlenecks, key=bottlenecks.get)
        for name, cycles in bottlenecks.items():
            row[f"bottleneck_{name}_share"] = cycles / result.cycles
    phase_bottlenecks: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for operation in result.operation_results:
        phase_bottlenecks[operation.phase][operation.bottleneck] += operation.cycles
    for phase, values in phase_bottlenecks.items():
        denominator = max(1, phases[phase].cycles)
        row[f"{phase}_dominant_bottleneck"] = max(values, key=values.get)
        for name, cycles in values.items():
            row[f"{phase}_bottleneck_{name}_share"] = cycles / denominator
    return row
