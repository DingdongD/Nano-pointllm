"""Real q-projection scale/payload correlation for the Dense64 dual exit."""
from __future__ import annotations

import json
from pathlib import Path
import struct
import subprocess
import tempfile
from typing import Any

from nanopointllm.compression.quantized_tensor_package import (
    load_quantized_linear_package,
)

from .dense_dual_output import (
    DenseDualOutputConfig, DenseOutputTile, run_dense_dual_output_model,
)
from .dense_dual_output_correlation import _parse
from .requant import compile_fp16_requant_scale


def correlate_real_qproj_dual_output(
    *, tensor_package_dir: str | Path, rtl_root: str | Path,
    output_dir: str | Path | None = None, bf16_lanes: int = 16,
) -> dict[str, Any]:
    package = load_quantized_linear_package(tensor_package_dir)
    if package.m != 1 or package.n != 4096 or package.k != 4096:
        raise ValueError("real dual-output lock expects the layer-0 q-projection package")
    if package.output_scale is None:
        raise ValueError("real dual-output lock requires output_scale.fp16")
    accumulators = package.integer_golden()[0]
    activation_scale = float(package.activation_scale[0])
    output_scale = float(package.output_scale[0])
    activation_bits = _fp16_bits(activation_scale)
    tiles = []
    for tile_index in range(64):
        start = tile_index * 64
        scales = package.weight_scale[start:start + 64]
        compiled = [
            compile_fp16_requant_scale(activation_scale, float(scale), output_scale)
            for scale in scales
        ]
        tiles.append(DenseOutputTile(
            tag=tile_index, mask=(1 << 64) - 1,
            accumulators=tuple(int(value) for value in accumulators[start:start + 64]),
            biases=(0,) * 64,
            multipliers=tuple(item.multiplier for item in compiled),
            right_shifts=tuple(item.right_shift for item in compiled),
            activation_scale_fp16=activation_bits,
            weight_scales_fp16=tuple(_fp16_bits(float(scale)) for scale in scales),
        ))
    tiles = tuple(tiles)
    config = DenseDualOutputConfig(
        bf16_lanes=bf16_lanes, source_stall_mod=0, output_stall_mod=0,
    )
    model = run_dense_dual_output_model(config, tiles)
    root = Path(rtl_root)
    sources = (
        root / "gtsu_requantize_int32.sv",
        root / "gtsu_dequant_int32_bf16.sv",
        root / "gtsu_dense64_dual_output.sv",
        root / "tb_gtsu_dense64_dual_output_trace.sv",
    )
    with tempfile.TemporaryDirectory(prefix="gtsu_real_dual_") as name:
        temporary = Path(name)
        trace = temporary / "real_qproj_dual.hex"
        _write_trace(trace, tiles)
        executable = temporary / "dual.vvp"
        command = [
            "iverilog", "-g2012", "-s", "tb_gtsu_dense64_dual_output_trace",
            f"-Ptb_gtsu_dense64_dual_output_trace.BF16_LANES={bf16_lanes}",
            f"-Ptb_gtsu_dense64_dual_output_trace.TILES={len(tiles)}",
            "-o", str(executable), *(str(path) for path in sources),
        ]
        compiled = subprocess.run(command, capture_output=True, text=True)
        if compiled.returncode:
            raise RuntimeError(f"real dual-output compile failed:\n{compiled.stderr}")
        simulated = subprocess.run(
            ["vvp", str(executable), f"+TRACE_FILE={trace}"],
            capture_output=True, text=True,
        )
        if simulated.returncode:
            raise RuntimeError(
                f"real dual-output RTL failed:\n{simulated.stdout}\n{simulated.stderr}"
            )
    rtl_events, rtl_summary = _parse(simulated.stdout)
    expected_summary = {"cycles": model.cycles, **model.counters}
    exact = model.events == rtl_events and expected_summary == rtl_summary
    wrapper_hierarchy = _check_wrapper_hierarchy(root, bf16_lanes)
    report = {
        "schema_version": "0.1",
        "status": "rtl_correlated" if exact else "mismatch",
        "operator": "real_layer0_qproj_acc32_a8_bf16",
        "tensor_package": {
            "path": str(Path(tensor_package_dir).resolve()),
            "activation_scale_fp16": activation_scale,
            "output_scale_fp16": output_scale,
            "tensor_sha256": {
                name: entry["sha256"]
                for name, entry in package.metadata["tensors"].items()
            },
        },
        "bf16_lanes": bf16_lanes,
        "cycle_model_cycles": model.cycles,
        "rtl_cycles": rtl_summary["cycles"],
        "cycle_error": rtl_summary["cycles"] - model.cycles,
        "event_trace_exact": model.events == rtl_events,
        "counter_trace_exact": expected_summary == rtl_summary,
        "a8_outputs_checked": 4096,
        "bf16_outputs_checked": 4096,
        "dense64_wrapper_hierarchy": wrapper_hierarchy,
        "claim_boundary": {
            "real_qproj_a8_value_event_cycle_exact": exact,
            "real_qproj_bf16_value_event_cycle_exact": exact,
            "dense64_wrapper_structurally_composed": True,
            "dense64_wrapper_value_event_cycle_correlated": False,
            "full_production_w8a8_kernel": False,
            "full_pointllm_cycle_accurate": False,
        },
    }
    if output_dir is not None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        _write_trace(destination / "real_qproj_dual.hex", tiles)
        (destination / "rtl_stdout.log").write_text(simulated.stdout, encoding="utf-8")
        (destination / "correlation.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8",
        )
    if not exact:
        raise AssertionError(f"real q-projection dual-output mismatch: {report}")
    return report


def _write_trace(path: Path, tiles: tuple[DenseOutputTile, ...]) -> None:
    with path.open("w", encoding="ascii") as handle:
        for tile in tiles:
            value = tile.tag
            value |= tile.mask << 32
            value |= tile.activation_scale_fp16 << 96
            for column in range(64):
                value |= (tile.accumulators[column] & 0xFFFFFFFF) << (112 + column * 32)
                value |= (tile.biases[column] & 0xFFFFFFFF) << (2160 + column * 32)
                value |= tile.multipliers[column] << (4208 + column * 32)
                value |= tile.right_shifts[column] << (6256 + column * 6)
                value |= tile.weight_scales_fp16[column] << (6640 + column * 16)
            handle.write(f"{value:01916x}\n")


def _fp16_bits(value: float) -> int:
    return struct.unpack("H", struct.pack("e", value))[0]


def _check_wrapper_hierarchy(root: Path, bf16_lanes: int) -> dict[str, Any]:
    sources = (
        root / "gtsu_dot4_pe.sv",
        root / "gtsu_dense_dot4_tile.sv",
        root / "gtsu_rv_fifo.sv",
        root / "gtsu_dense64_abuf_wbuf_fabric.sv",
        root / "gtsu_requantize_int32.sv",
        root / "gtsu_dequant_int32_bf16.sv",
        root / "gtsu_dense64_dual_output.sv",
        root / "gtsu_dense64_stream_dual_output.sv",
    )
    with tempfile.TemporaryDirectory(prefix="gtsu_dual_wrapper_") as name:
        output = Path(name) / "wrapper.json"
        top = "gtsu_dense64_stream_dual_output"
        command = (
            f"read_verilog -sv {' '.join(str(path) for path in sources)}; "
            f"chparam -set M 1 -set N 64 -set K 4 -set N_MEM_LANES 4 "
            f"-set BF16_LANES {bf16_lanes} -set READ_LATENCY 1 "
            f"-set FIFO_DEPTH 2 {top}; hierarchy -check -top {top}; "
            f"check -assert; write_json {output}"
        )
        result = subprocess.run(["yosys", "-p", command], capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(f"Dense64 dual wrapper hierarchy failed:\n{result.stderr}")
        cells = json.loads(output.read_text())["modules"][top].get("cells", {}).values()
    dense = sum("gtsu_dense64_abuf_wbuf_fabric" in cell["type"] for cell in cells)
    post = sum("gtsu_dense64_dual_output" in cell["type"] for cell in cells)
    if (dense, post) != (1, 1):
        raise AssertionError(f"Dense64 dual wrapper hierarchy is {dense}/{post}")
    return {
        "status": "passed", "dense64_instances": dense,
        "dual_output_instances": post,
        "dense_backpressure_directly_driven_by_postprocessor": True,
        "capacity_reduced_structure_lock": True,
    }
