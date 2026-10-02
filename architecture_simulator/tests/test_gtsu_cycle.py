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
from gtsu_cycle.splitk_gemv import (
    SplitKGemvConfig,
    functional_outputs,
    run_cycle_model,
)


ROOT = Path(__file__).resolve().parents[1]
RTL_ROOT = ROOT / "rtl/vertical_slice"


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
    require_rtl_correlated(["w8_splitk_gemv_vertical_slice"])
    with pytest.raises(UnsupportedCycleAccurateOperator, match="attention"):
        require_rtl_correlated(["w8_splitk_gemv_vertical_slice", "attention"])


@pytest.mark.skipif(shutil.which("yosys") is None, reason="yosys unavailable")
def test_splitk_vertical_slice_synthesizes():
    import subprocess

    result = subprocess.run(
        [
            shutil.which("yosys"), "-q", "-p",
            (
                f"read_verilog -sv {RTL_ROOT / 'gtsu_rv_fifo.sv'} "
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
