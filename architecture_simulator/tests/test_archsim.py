from dataclasses import replace
from pathlib import Path

import pytest

from pointllm_archsim.dse import SweepSpec, iter_configs, latency_resource_pareto, run_sweep
from pointllm_archsim.engines import EngineModels
from pointllm_archsim.evidence import extract_ncu_evidence
from pointllm_archsim.schema import HardwareConfig, Operation, Workload
from pointllm_archsim.simulator import ArchitectureSimulator
from pointllm_archsim.simulator import summary_row
from pointllm_archsim.workload import build_pointllm_7b_workload


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent


@pytest.fixture
def config():
    return HardwareConfig.load(ROOT / "configs/gtsu_baseline.json")


def test_config_and_workload_json_roundtrip(config, tmp_path):
    workload = build_pointllm_7b_workload(batch_size=1, input_tokens=64, output_tokens=4)
    path = tmp_path / "workload.json"
    workload.save(path)
    loaded = Workload.load(path)

    assert config.tensor.rows == 32
    assert loaded.name == workload.name
    assert len(loaded.operations) == len(workload.operations)
    assert loaded.metadata["output_tokens"] == 4


def test_split_k_improves_decode_compute_cycles(config):
    split = Operation(
        "gemv", "llm_decode", "linear", m=1, n=4096, k=4096,
        weight_bits=8, attributes={"mapping": "split_k"},
    )
    dense = replace(split, name="dense", attributes={"mapping": "dense"})
    engines = EngineModels(config)

    split_result = engines.estimate(split)
    dense_result = engines.estimate(dense)

    assert split_result.compute_cycles < dense_result.compute_cycles
    assert split_result.details["split_k_partitions"] == 32
    assert split_result.dataflow == "split_k_output_channel"


def test_more_fps_lanes_reduce_serial_geometry_cycles(config):
    operation = Operation(
        "fps", "geometry", "fps", attributes={
            "points": 8192, "samples": 512, "coordinate_dims": 3,
            "coordinate_bytes": 2, "distance_bytes": 2,
        },
    )
    slow = replace(config, geometry=replace(config.geometry, distance_lanes=64))
    fast = replace(config, geometry=replace(config.geometry, distance_lanes=512))

    slow_result = EngineModels(slow).estimate(operation)
    fast_result = EngineModels(fast).estimate(operation)

    assert fast_result.cycles < slow_result.cycles
    assert fast_result.details["resident_point_state"] is True
    assert fast_result.bottleneck == "serial_feedback"


def test_w4_reduces_weight_stream_traffic(config):
    base = Operation(
        "decode_mlp", "llm_decode", "linear", repeat=32, m=1,
        n=11008, k=4096, weight_bits=8, attributes={"mapping": "split_k"},
    )
    w4 = replace(base, name="decode_mlp_w4", weight_bits=4)
    engines = EngineModels(config)
    w8_result = engines.estimate(base)
    w4_result = engines.estimate(w4)

    assert w4_result.hbm_read_bytes < w8_result.hbm_read_bytes
    assert w4_result.memory_cycles < w8_result.memory_cycles
    assert w4_result.cycles < w8_result.cycles


def test_fused_attention_avoids_score_hbm(config):
    operation = Operation(
        "attention", "point_encoder", "attention", batch=1, q_len=513,
        kv_len=513, heads=6, hidden=384, weight_bits=8,
    )
    fused = EngineModels(config).estimate(operation)
    materialized_config = replace(
        config,
        attention=replace(config.attention, fused_online_softmax=False),
    )
    materialized = EngineModels(materialized_config).estimate(operation)

    assert fused.hbm_read_bytes + fused.hbm_write_bytes < (
        materialized.hbm_read_bytes + materialized.hbm_write_bytes
    )
    assert fused.details["materialized_score_bytes"] == 0


def test_simulator_records_phase_transitions(config):
    workload = Workload(
        "tiny",
        operations=(
            Operation("fps", "geometry", "fps", attributes={
                "points": 128, "samples": 8, "coordinate_dims": 3,
                "coordinate_bytes": 2, "distance_bytes": 2,
            }),
            Operation("fc", "point_encoder", "linear", m=16, n=32, k=8, weight_bits=8),
            Operation("decode", "llm_decode", "linear", m=1, n=32, k=32, weight_bits=8),
        ),
        metadata={"batch_size": 1, "output_tokens": 1},
    )
    result = ArchitectureSimulator(config).run(workload)

    assert result.counters["phase_reconfigurations"] == 2
    assert result.counters["phase_reconfiguration_cycles"] == 128
    assert [item.phase for item in result.phase_results] == [
        "geometry", "point_encoder", "llm_decode"
    ]
    assert result.methodology["cycle_accuracy_claim"] is False
    assert result.energy_pj is None
    summary = summary_row(result)
    assert summary["geometry_dominant_bottleneck"] == "serial_feedback"
    assert summary["llm_decode_dominant_bottleneck"] in {"compute", "unpack"}


def test_small_dse_and_pareto(config):
    workload = Workload(
        "gemv",
        operations=(Operation(
            "decode", "llm_decode", "linear", m=1, n=1024, k=1024,
            weight_bits=8, attributes={"mapping": "split_k"},
        ),),
        metadata={"batch_size": 1, "output_tokens": 1},
    )
    spec = SweepSpec(
        tensor_rows=(16, 32), tensor_cols=(16,), split_k=(8, 16),
        geometry_lanes=(128,), hbm_bandwidth_gbps=(409.6, 819.2),
        weight_unpack_bytes_per_cycle=(128,),
        sram_bytes=(16 * 1024 * 1024,), max_designs=8,
    )
    rows = run_sweep(config, workload, spec)
    pareto = latency_resource_pareto(rows)

    assert len(rows) == 8
    assert pareto
    assert rows == sorted(rows, key=lambda item: item["latency_ms"])
    assert all("weight_unpack_bytes_per_cycle" in item for item in rows)


def test_dse_deduplicates_clamped_split_k(config):
    spec = SweepSpec(
        tensor_rows=(16,), tensor_cols=(16,), split_k=(8, 16, 32),
        geometry_lanes=(128,), hbm_bandwidth_gbps=(819.2,),
        weight_unpack_bytes_per_cycle=(256,),
        sram_bytes=(16 * 1024 * 1024,), max_designs=3,
    )

    configs = list(iter_configs(config, spec))

    assert [item.tensor.split_k for item in configs] == [8, 16]
    assert len({item.name for item in configs}) == 2


def test_unpack_throughput_changes_weight_bound_decode(config):
    workload = Workload(
        "weight-bound",
        operations=(Operation(
            "decode", "llm_decode", "linear", repeat=32, m=1,
            n=11008, k=4096, weight_bits=8, attributes={"mapping": "split_k"},
        ),),
        metadata={"batch_size": 1, "output_tokens": 1},
    )
    slow = replace(
        config,
        memory=replace(config.memory, weight_unpack_bytes_per_cycle=128),
    )
    fast = replace(
        config,
        memory=replace(config.memory, weight_unpack_bytes_per_cycle=512),
    )

    slow_result = ArchitectureSimulator(slow).run(workload)
    fast_result = ArchitectureSimulator(fast).run(workload)

    assert slow_result.seconds > fast_result.seconds
    assert slow_result.operation_results[0].bottleneck == "unpack"


def test_ncu_evidence_contract():
    evidence = extract_ncu_evidence(REPO)

    assert evidence["pointbert"]["1"]["fps"]["time_ms"] == pytest.approx(2.95088)
    assert evidence["decoder"]["1"]["mlp"]["classification"] == "confirmed_weight_bandwidth_bound"
