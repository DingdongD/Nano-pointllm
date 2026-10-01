"""Small deterministic design-space sweeps for the phase-adaptive architecture."""
from __future__ import annotations

from dataclasses import dataclass, replace
import itertools
from typing import Any, Iterable

from .schema import HardwareConfig, Workload
from .simulator import ArchitectureSimulator, summary_row


@dataclass(frozen=True)
class SweepSpec:
    tensor_rows: tuple[int, ...] = (16, 32, 64)
    tensor_cols: tuple[int, ...] = (16, 32, 64)
    split_k: tuple[int, ...] = (8, 16, 32)
    geometry_lanes: tuple[int, ...] = (64, 128, 256, 512)
    hbm_bandwidth_gbps: tuple[float, ...] = (409.6, 819.2, 1555.0)
    weight_unpack_bytes_per_cycle: tuple[int, ...] = (128, 256, 512, 1024)
    sram_bytes: tuple[int, ...] = (16 * 1024 * 1024, 32 * 1024 * 1024)
    max_designs: int = 10_000

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SweepSpec":
        defaults = cls()
        return cls(
            tensor_rows=tuple(int(item) for item in value.get("tensor_rows", defaults.tensor_rows)),
            tensor_cols=tuple(int(item) for item in value.get("tensor_cols", defaults.tensor_cols)),
            split_k=tuple(int(item) for item in value.get("split_k", defaults.split_k)),
            geometry_lanes=tuple(int(item) for item in value.get("geometry_lanes", defaults.geometry_lanes)),
            hbm_bandwidth_gbps=tuple(
                float(item) for item in value.get("hbm_bandwidth_gbps", defaults.hbm_bandwidth_gbps)
            ),
            weight_unpack_bytes_per_cycle=tuple(
                int(item)
                for item in value.get(
                    "weight_unpack_bytes_per_cycle",
                    defaults.weight_unpack_bytes_per_cycle,
                )
            ),
            sram_bytes=tuple(int(item) for item in value.get("sram_bytes", defaults.sram_bytes)),
            max_designs=int(value.get("max_designs", defaults.max_designs)),
        )


def iter_configs(base: HardwareConfig, spec: SweepSpec) -> Iterable[HardwareConfig]:
    axes = itertools.product(
        spec.tensor_rows,
        spec.tensor_cols,
        spec.split_k,
        spec.geometry_lanes,
        spec.hbm_bandwidth_gbps,
        spec.weight_unpack_bytes_per_cycle,
        spec.sram_bytes,
    )
    seen: set[tuple[int, int, int, int, float, int, int]] = set()
    emitted = 0
    for rows, cols, split_k, lanes, hbm, unpack, sram in axes:
        effective_split_k = min(split_k, rows)
        design = (rows, cols, effective_split_k, lanes, hbm, unpack, sram)
        if design in seen:
            continue
        if emitted >= spec.max_designs:
            break
        seen.add(design)
        emitted += 1
        tensor = replace(base.tensor, rows=rows, cols=cols, split_k=effective_split_k)
        geometry = replace(base.geometry, distance_lanes=lanes)
        memory = replace(
            base.memory,
            hbm_bandwidth_gbps=hbm,
            weight_unpack_bytes_per_cycle=unpack,
            sram_bytes=sram,
        )
        yield replace(
            base,
            name=(
                f"r{rows}_c{cols}_sk{effective_split_k}_g{lanes}_"
                f"hbm{hbm:g}_wu{unpack}_sram{sram // 1048576}m"
            ),
            tensor=tensor,
            geometry=geometry,
            memory=memory,
        )


def run_sweep(base: HardwareConfig, workload: Workload, spec: SweepSpec) -> list[dict[str, Any]]:
    rows = []
    for config in iter_configs(base, spec):
        result = ArchitectureSimulator(config).run(workload)
        row = summary_row(result)
        row.update({
            "tensor_rows": config.tensor.rows,
            "tensor_cols": config.tensor.cols,
            "split_k": config.tensor.split_k,
            "geometry_lanes": config.geometry.distance_lanes,
            "hbm_bandwidth_gbps": config.memory.hbm_bandwidth_gbps,
            "weight_unpack_bytes_per_cycle": config.memory.weight_unpack_bytes_per_cycle,
            "sram_mb": config.memory.sram_bytes / 1048576,
            "tensor_pe_count": config.tensor.rows * config.tensor.cols,
        })
        rows.append(row)
    rows.sort(key=lambda item: item["latency_ms"])
    return rows


def latency_resource_pareto(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return designs not dominated in latency or modeled resource demand."""
    pareto = []
    for candidate in rows:
        dominated = False
        for other in rows:
            if other is candidate:
                continue
            no_worse = (
                other["latency_ms"] <= candidate["latency_ms"]
                and other["tensor_pe_count"] <= candidate["tensor_pe_count"]
                and other["sram_mb"] <= candidate["sram_mb"]
                and other["hbm_bandwidth_gbps"] <= candidate["hbm_bandwidth_gbps"]
                and other["weight_unpack_bytes_per_cycle"]
                <= candidate["weight_unpack_bytes_per_cycle"]
            )
            strictly_better = (
                other["latency_ms"] < candidate["latency_ms"]
                or other["tensor_pe_count"] < candidate["tensor_pe_count"]
                or other["sram_mb"] < candidate["sram_mb"]
                or other["hbm_bandwidth_gbps"] < candidate["hbm_bandwidth_gbps"]
                or other["weight_unpack_bytes_per_cycle"]
                < candidate["weight_unpack_bytes_per_cycle"]
            )
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            pareto.append(candidate)
    return sorted(pareto, key=lambda item: item["latency_ms"])
