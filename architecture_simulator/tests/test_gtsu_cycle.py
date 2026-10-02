from dataclasses import replace
from pathlib import Path
import shutil

import pytest

from gtsu_cycle.correlation import correlate_splitk_gemv
from gtsu_cycle.coverage import (
    UnsupportedCycleAccurateOperator,
    require_rtl_correlated,
)
from gtsu_cycle.ir import MicroOpcode, lower_splitk_gemv
from gtsu_cycle.dramsim3_backend import DramSim3Paths, time_dram_bursts
from gtsu_cycle.pointllm_lowering import lower_decoder_linear, lower_decoder_manifest
from gtsu_cycle.pointllm_lowering import (
    LinearTile, LoweredLinear, LoweringConfig, MemoryRegion, SourceTensor,
)
from gtsu_cycle.production_linear import (
    DmaArrival, ProductionLinearConfig, build_runtime_bursts,
    run_production_linear_model,
)
from gtsu_cycle.production_linear_correlation import correlate_production_linear
from gtsu_cycle.sram import SramConfig, decode_sram_address, run_sram_model
from gtsu_cycle.sram_correlation import correlate_banked_sram
from gtsu_cycle.geometry import (
    GeometryBeat, GeometryConfig, build_geometry_beats,
    functional_outputs as geometry_outputs, pointllm_geometry_mapping,
    run_geometry_model,
)
from gtsu_cycle.geometry_correlation import correlate_geometry_distance
from gtsu_cycle.shared_dot import (
    SharedDotBeat, SharedDotConfig, SharedDotLane, build_shared_dot_beats,
    functional_outputs as shared_outputs, pointllm_shared_geometry_mapping,
    run_shared_dot_model, signed_int16_product_via_int8,
)
from gtsu_cycle.shared_dot_correlation import correlate_shared_dot_geometry
from gtsu_cycle.dense_gemm import DenseTileConfig, build_dense_beats, run_dense_tile_model
from gtsu_cycle.dense_gemm_correlation import correlate_dense_tile
from gtsu_cycle.dense_controller import DenseControllerConfig, run_dense_controller_model
from gtsu_cycle.dense_controller_correlation import correlate_dense_controller
from gtsu_cycle.requant import (
    RequantConfig, compile_fp16_requant_scale, locked_requant_vectors,
    requantize_int32,
)
from gtsu_cycle.requant_correlation import correlate_requant
from gtsu_cycle.splitk_gemv import (
    SplitKGemvConfig,
    functional_outputs,
    run_cycle_model,
)


ROOT = Path(__file__).resolve().parents[1]
RTL_ROOT = ROOT / "rtl/vertical_slice"
CHECKPOINT = Path("/mnt/llm_data/pointllm_ckpt/PointLLM_7B_v1.2_safetensors")


def test_functional_and_cycle_model_match():
    config = SplitKGemvConfig()
    result = run_cycle_model(config)

    assert result.outputs == functional_outputs(config)
    assert result.counters["input_beats"] == (
        config.n_outputs * config.split_k * config.k_chunks
    )
    assert result.counters["partial_sums"] == config.n_outputs * config.split_k
    assert result.counters["outputs"] == config.n_outputs


def test_splitk_micro_op_lowering_contract():
    config = SplitKGemvConfig(n_outputs=2, split_k=3, k_chunks=2, lanes=8)
    operations = lower_splitk_gemv(config)
    opcodes = [operation.opcode for operation in operations]

    assert opcodes.count(MicroOpcode.LD_W) == 12
    assert opcodes.count(MicroOpcode.UNPACK_W8) == 12
    assert opcodes.count(MicroOpcode.GEMV_SPLITK) == 12
    assert opcodes.count(MicroOpcode.PSUM_REDUCE) == 6
    assert opcodes.count(MicroOpcode.STORE) == 2
    assert all(
        all(dependency < operation.sequence for dependency in operation.dependencies)
        for operation in operations
    )
    reductions = [
        operation for operation in operations
        if operation.opcode == MicroOpcode.PSUM_REDUCE
    ]
    assert all(len(operation.dependencies) == 2 for operation in reductions if operation.partition)


@pytest.mark.skipif(shutil.which("iverilog") is None, reason="iverilog unavailable")
@pytest.mark.parametrize("config", [
    SplitKGemvConfig(),
    SplitKGemvConfig(
        n_outputs=3,
        split_k=3,
        k_chunks=2,
        lanes=8,
        source_latency=1,
        compressed_fifo_depth=1,
        decoded_fifo_depth=1,
        partial_fifo_depth=1,
        output_stall_mod=3,
        output_stall_phase=1,
    ),
    replace(
        SplitKGemvConfig(),
        source_latency=0,
        output_stall_mod=0,
        compressed_fifo_depth=4,
        decoded_fifo_depth=3,
        partial_fifo_depth=3,
    ),
])
def test_splitk_cycle_model_exactly_matches_rtl(config):
    report = correlate_splitk_gemv(config, rtl_root=RTL_ROOT)

    assert report["status"] == "rtl_correlated"
    assert report["event_trace_exact"] is True
    assert report["counter_trace_exact"] is True
    assert report["functional_output_exact"] is True
    assert report["cycle_error"] == 0


def test_full_model_accuracy_remains_fail_closed():
    result = run_cycle_model(SplitKGemvConfig())

    assert result.as_dict()["fidelity"]["full_model_cycle_accurate"] is False
    require_rtl_correlated([
        "w8_splitk_gemv_vertical_slice", "banked_sram_2client",
        "geometry_distance_tile",
    ])
    with pytest.raises(UnsupportedCycleAccurateOperator, match="attention"):
        require_rtl_correlated(["w8_splitk_gemv_vertical_slice", "attention"])


def test_geometry_functional_and_cycle_outputs_match():
    config = GeometryConfig(
        lanes=4, tiles=5, source_stall_mod=0,
        output_stall_mod=2, output_stall_phase=1,
    )
    beats = build_geometry_beats(config)
    result = run_geometry_model(config, beats)

    assert result.outputs == geometry_outputs(beats)
    assert result.counters["input_tiles"] == 5
    assert result.counters["output_points"] == 20
    assert result.counters["source_backpressure_cycles"] > 0
    assert result.counters["output_backpressure_cycles"] > 0

    mapping = pointllm_geometry_mapping(64)
    assert mapping["distance_evaluations_per_operator"] == 4_194_304
    assert mapping["tiles_per_center"] == 128
    assert mapping["distance_tiles_per_operator"] == 65_536
    assert mapping["point_coordinate_sram_bytes_int16"] == 49_152


@pytest.mark.skipif(shutil.which("iverilog") is None, reason="iverilog unavailable")
@pytest.mark.parametrize("config", [
    GeometryConfig(),
    GeometryConfig(
        lanes=4, tiles=9, source_stall_mod=0, output_stall_mod=0,
    ),
    GeometryConfig(
        lanes=16, tiles=6, source_stall_mod=3, source_stall_phase=0,
        output_stall_mod=2, output_stall_phase=1,
    ),
])
def test_geometry_distance_exactly_matches_rtl(config):
    report = correlate_geometry_distance(config, rtl_root=RTL_ROOT)

    assert report["status"] == "rtl_correlated"
    assert report["event_trace_exact"] is True
    assert report["counter_trace_exact"] is True
    assert report["functional_output_exact"] is True
    assert report["cycle_error"] == 0


@pytest.mark.skipif(shutil.which("iverilog") is None, reason="iverilog unavailable")
def test_geometry_distance_signed_int16_extremes_match_rtl():
    config = GeometryConfig(
        lanes=4, tiles=1, source_stall_mod=0, output_stall_mod=0,
    )
    beat = GeometryBeat(
        tag=7,
        query=(-32768, 32767, -32768),
        points=(
            (32767, -32768, 32767),
            (-32768, 32767, -32768),
            (32767, 32767, -32768),
            (-32768, -32768, 32767),
        ),
    )
    report = correlate_geometry_distance(
        config, rtl_root=RTL_ROOT, beats=(beat,),
    )

    assert report["status"] == "rtl_correlated"
    assert report["functional_output_exact"] is True


def test_shared_dot_functional_and_cycle_outputs_match():
    config = SharedDotConfig(
        pe_count=4, beats=6, source_stall_mod=0,
        output_stall_mod=2, output_stall_phase=1,
    )
    beats = build_shared_dot_beats(config)
    result = run_shared_dot_model(config, beats)

    assert result.outputs == shared_outputs(beats)
    assert result.counters["feature_lanes"] == 12
    assert result.counters["geometry_lanes"] == 12
    mapping = pointllm_shared_geometry_mapping(64)
    assert mapping["distance_evaluations"] == 4_194_304
    assert mapping["distance_dot_cycles"] == 65_536
    assert mapping["point_norm_precompute_cycles"] == 128
    assert mapping["int16_fallback_distance_dot_cycles"] == 262_144


@pytest.mark.parametrize("left,right", [
    (-32768, -32768), (-32768, 32767), (32767, 32767),
    (-1, 1), (0, 32767), (12345, -23456),
])
def test_signed_int16_four_int8_product_decomposition(left, right):
    assert signed_int16_product_via_int8(left, right) == left * right


@pytest.mark.skipif(shutil.which("iverilog") is None, reason="iverilog unavailable")
def test_shared_dot_signed_int8_extremes_match_rtl():
    config = SharedDotConfig(
        pe_count=2, beats=1, source_stall_mod=0, output_stall_mod=0,
    )
    lanes = (
        SharedDotLane(
            (-128, 127, -128, 0), (127, -128, 127, 0),
            2 * 128**2 + 127**2, 2 * 127**2 + 128**2,
        ),
        SharedDotLane(
            (127, 127, 127, 0), (-128, -128, -128, 0),
            3 * 127**2, 3 * 128**2,
        ),
    )
    report = correlate_shared_dot_geometry(
        config, rtl_root=RTL_ROOT, beats=(SharedDotBeat(7, True, lanes),),
    )

    assert report["status"] == "rtl_correlated"
    assert report["functional_output_exact"] is True


def test_dense_tile_functional_cycle_model():
    config = DenseTileConfig(rows=3, columns=5, n_tile=8, k=12)
    result = run_dense_tile_model(config, build_dense_beats(config))
    assert len(result.outputs) == 3
    assert all(len(row) == 5 for row in result.outputs)


def test_dense_controller_covers_mnk_edges_and_overlap():
    config = DenseControllerConfig(m=2, n=70, k=132)
    result = run_dense_controller_model(config)
    reads = [event for event in result.events if event.event == "READ_ISSUE"]
    assert result.counters["blocks_released"] == config.total_blocks
    assert result.counters["ping_pong_overlap_cycles"] > 0
    assert reads[0].first == 1
    assert reads[-1].last == 1
    assert reads[-1].column_mask == 0x3F


@pytest.mark.skipif(shutil.which("iverilog") is None, reason="iverilog unavailable")
@pytest.mark.parametrize("config", [
    DenseControllerConfig(m=2, n=70, k=132),
    DenseControllerConfig(
        m=1, n=64, k=64, source_stall_mod=0, compute_stall_mod=0,
    ),
])
def test_dense_controller_exactly_matches_rtl(config):
    report = correlate_dense_controller(config, rtl_root=RTL_ROOT)
    assert report["status"] == "rtl_correlated"
    assert report["cycle_error"] == 0
    assert report["synthesis_check"]["forbidden_arithmetic_cells"] == {
        "$mod": 0, "$mul": 0, "$div": 0,
    }


@pytest.mark.skipif(shutil.which("iverilog") is None, reason="iverilog unavailable")
@pytest.mark.parametrize("config", [
    DenseTileConfig(),
    DenseTileConfig(rows=3, columns=4, n_tile=4, k=12, source_stall_mod=0, output_stall_mod=0),
])
def test_dense_tile_exactly_matches_rtl(config):
    report = correlate_dense_tile(config, rtl_root=RTL_ROOT)
    assert report["status"] == "rtl_correlated"
    assert report["functional_output_exact"] is True
    assert report["cycle_error"] == 0
    assert report["synthesis_check"]["top_direct_multipliers"] == 0


def test_requant_locked_vectors_cover_rne_and_symmetric_saturation():
    outputs = tuple(requantize_int32(vector) for vector in locked_requant_vectors())
    assert outputs[3:7] == (2, 4, -2, -4)
    assert outputs[7:9] == (127, -127)

    compiled = compile_fp16_requant_scale(0.0124359, 0.0036793, 0.03125)
    assert 0 < compiled.multiplier < (1 << 32)
    assert 0 <= compiled.right_shift <= 63
    assert compiled.relative_error < 1e-9


@pytest.mark.skipif(shutil.which("iverilog") is None, reason="iverilog unavailable")
@pytest.mark.parametrize("config", [
    RequantConfig(),
    RequantConfig(source_stall_mod=0, output_stall_mod=0),
])
def test_requant_exactly_matches_rtl(config):
    report = correlate_requant(rtl_root=RTL_ROOT, config=config)
    assert report["status"] == "rtl_correlated"
    assert report["cycle_error"] == 0
    assert report["synthesis_check"]["multipliers"] == 1
    assert report["synthesis_check"]["dividers"] == 0


@pytest.mark.skipif(shutil.which("iverilog") is None, reason="iverilog unavailable")
@pytest.mark.parametrize("config", [
    SharedDotConfig(),
    SharedDotConfig(
        pe_count=4, beats=7, source_stall_mod=0, output_stall_mod=0,
    ),
    SharedDotConfig(
        pe_count=16, beats=6, source_stall_mod=3, source_stall_phase=0,
        output_stall_mod=2, output_stall_phase=1,
    ),
])
def test_shared_dot_geometry_exactly_matches_rtl(config):
    report = correlate_shared_dot_geometry(config, rtl_root=RTL_ROOT)

    assert report["status"] == "rtl_correlated"
    assert report["event_trace_exact"] is True
    assert report["counter_trace_exact"] is True
    assert report["functional_output_exact"] is True
    assert report["cycle_error"] == 0
    assert report["synthesis_check"]["top_direct_multipliers"] == 0
    assert report["synthesis_check"]["dot4_multipliers_per_instance"] == 4
    assert report["synthesis_check"]["dot4_instances"] == config.pe_count


def test_banked_sram_mapping_conflicts_and_data():
    config = SramConfig()
    assert decode_sram_address(config, 0) == (0, 0)
    assert decode_sram_address(config, 15 * 16) == (15, 0)
    assert decode_sram_address(config, 16 * 16) == (0, 1)
    result = run_sram_model(config=config)

    assert result.cycles == 15
    assert result.counters == {
        "read_accepts": 7,
        "write_accepts": 7,
        "read_completions": 7,
        "read_bank_conflicts": 1,
        "write_bank_conflicts": 1,
        "client_stall_cycles": 2,
    }
    returned = {
        event.tag: event.data for event in result.events
        if event.event == "READ_COMPLETE"
    }
    assert returned == {
        10: 0xAA, 11: 0x22, 12: 0x33, 13: 0, 14: 0xCC,
        20: 0xAA, 21: 0xBB,
    }


@pytest.mark.skipif(shutil.which("iverilog") is None, reason="iverilog unavailable")
def test_banked_sram_cycle_model_exactly_matches_rtl():
    report = correlate_banked_sram(rtl_root=RTL_ROOT)
    assert report["status"] == "rtl_correlated"
    assert report["event_trace_exact"] is True
    assert report["cycle_error"] == 0


@pytest.mark.skipif(not CHECKPOINT.is_dir(), reason="PointLLM checkpoint unavailable")
@pytest.mark.parametrize("projection,shape", [
    ("q_proj", (4096, 4096)),
    ("down_proj", (4096, 11008)),
    ("lm_head", (32003, 4096)),
])
def test_checkpoint_backed_pointllm_lowering(projection, shape):
    from itertools import islice

    lowered = lower_decoder_linear(CHECKPOINT, projection)
    validation = lowered.as_dict(sample_tiles=1, sample_bursts=2)["validation"]

    assert lowered.source.dtype == "F32"
    assert lowered.source.shape == shape
    assert lowered.source.payload_bytes == lowered.source.expected_payload_bytes
    assert validation["passed"] is True
    assert lowered.weight_burst_count * 64 == lowered.target_weight_payload_bytes
    assert all(
        burst.address % 64 == 0
        for burst in islice(lowered.iter_weight_bursts(), 64)
    )


@pytest.mark.skipif(
    not DramSim3Paths.local_default().library.is_file()
    or not DramSim3Paths.local_default().config.is_file()
    or not CHECKPOINT.is_dir(),
    reason="local DRAMsim3 or PointLLM checkpoint unavailable",
)
def test_real_dramsim3_times_lowered_pointllm_bursts(tmp_path):
    from itertools import islice

    lowered = lower_decoder_linear(CHECKPOINT, "q_proj")
    bursts = tuple(islice(lowered.iter_weight_bursts(), 8))
    result = time_dram_bursts(
        bursts,
        paths=DramSim3Paths.local_default(),
        output_dir=tmp_path / "dramsim3",
        max_outstanding=4,
    )

    assert result.requests == 8
    assert result.bytes_read == 8 * 64
    assert result.dram_clock_ticks > result.accelerator_cycles
    assert result.latency_min > 0
    assert result.provenance["dramsim3_source_commit"] == (
        "29817593b3389f1337235d63cac515024ab8fd6e"
    )


@pytest.mark.skipif(not CHECKPOINT.is_dir(), reason="PointLLM checkpoint unavailable")
def test_decoder_manifest_uses_global_nonoverlapping_dram_layout():
    manifest = lower_decoder_manifest(CHECKPOINT, layer=0)
    regions = sorted(
        (
            (region["base"], region["end"])
            for operator in manifest["operators"]
            for region in operator["regions"]
            if region["space"] == "dram"
        )
    )

    assert all(left[1] <= right[0] for left, right in zip(regions, regions[1:]))
    assert manifest["totals"]["global_dram_span"]["regions_nonoverlap"] is True


def _small_production_linear():
    config = LoweringConfig(split_k=2, output_tile=16)
    sram = SramConfig()
    n, k = 32, 384
    tiles = []
    tile_id = 0
    for n_start in (0, 16):
        for partition in range(2):
            tiles.append(LinearTile(
                tile_id=tile_id, n_start=n_start, n_count=16,
                split_index=partition, k_start=partition * 192, k_count=192,
                weight_payload_bytes=16 * 192, weight_bursts=48,
                activation_sram_address=partition * 192,
                weight_sram_address=1 << 20,
                partial_sram_address=(3 << 20) + partition * 64,
            ))
            tile_id += 1
    return LoweredLinear(
        projection="q_proj", layer=0, token=0, m=1, n=n, k=k,
        source=SourceTensor("synthetic", "none", "F32", (n, k), (0, n * k * 4)),
        config=config, sram_config=sram,
        regions=(
            MemoryRegion("weight_w8", "dram", 1 << 32, n * k, n * k, "W8"),
            MemoryRegion("weight_scale", "dram", (1 << 32) + n * k, 192, 4096, "FP16"),
            MemoryRegion("activation_a8", "sram", 0, k, 384, "A8"),
            MemoryRegion("weight_tile_w8", "sram", 1 << 20, 3072, 3072, "W8"),
            MemoryRegion("partial_acc32", "sram", 3 << 20, 128, 128, "ACC32"),
            MemoryRegion("output_acc32", "sram", (3 << 20) + 65536, 64, 64, "ACC32"),
        ),
        tiles=tuple(tiles),
    )


@pytest.mark.skipif(shutil.which("iverilog") is None, reason="iverilog unavailable")
def test_integrated_production_linear_controller_matches_rtl():
    lowered = _small_production_linear()
    runtime = build_runtime_bursts(lowered)
    arrivals = tuple(
        DmaArrival(cycle=index * 2, burst=burst)
        for index, burst in enumerate(runtime)
    )
    config = ProductionLinearConfig(rob_depth=8, max_cycles=10_000)
    model = run_production_linear_model(lowered, arrivals, config=config)
    report = correlate_production_linear(
        lowered, arrivals, rtl_root=RTL_ROOT, config=config,
    )

    assert model.counters["outputs"] == 32
    assert report["status"] == "rtl_correlated"
    assert report["cycle_error"] == 0


@pytest.mark.skipif(shutil.which("yosys") is None, reason="yosys unavailable")
def test_splitk_vertical_slice_synthesizes():
    import subprocess

    result = subprocess.run(
        [
            shutil.which("yosys"), "-q", "-p",
            (
                f"read_verilog -sv {RTL_ROOT / 'gtsu_rv_fifo.sv'} "
                f"{RTL_ROOT / 'gtsu_dot4_pe.sv'} "
                f"{RTL_ROOT / 'gtsu_w8_splitk_gemv.sv'}; "
                "hierarchy -check -top gtsu_w8_splitk_gemv; "
                "proc; memory; opt; check -assert"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
