"""Fail-closed RTL/cycle coverage registry."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable


@dataclass(frozen=True)
class OperatorCoverage:
    operator: str
    functional_model: bool
    event_cycle_model: bool
    synthesizable_rtl: bool
    exact_rtl_cycle_correlation: bool
    limitation: str


COVERAGE = {
    "w8_splitk_gemv_vertical_slice": OperatorCoverage(
        operator="w8_splitk_gemv_vertical_slice",
        functional_model=True,
        event_cycle_model=True,
        synthesizable_rtl=True,
        exact_rtl_cycle_correlation=True,
        limitation=(
            "Locked transaction-level slice only; deterministic source timing. "
            "PointLLM lowering and DRAM timing exist separately but are not integrated."
        ),
    ),
    "banked_sram_2client": OperatorCoverage(
        operator="banked_sram_2client",
        functional_model=True,
        event_cycle_model=True,
        synthesizable_rtl=True,
        exact_rtl_cycle_correlation=True,
        limitation=(
            "Locked 16-bank/128-bit/1R1W/3-cycle slice; behavioral SRAM array, "
            "two clients, and no foundry SRAM macro characterization."
        ),
    ),
    "geometry_distance_tile": OperatorCoverage(
        operator="geometry_distance_tile",
        functional_model=True,
        event_cycle_model=True,
        synthesizable_rtl=True,
        exact_rtl_cycle_correlation=True,
        limitation=(
            "Signed INT16 3-D distance tile with elastic ready/valid only; PointLLM "
            "FP32 quantization, SRAM/DRAM, FPS min/argmax feedback, and cross-tile "
            "KNN top-k integration are not validated."
        ),
    ),
    "shared_int8_dot4_feature_geometry": OperatorCoverage(
        operator="shared_int8_dot4_feature_geometry",
        functional_model=True,
        event_cycle_model=True,
        synthesizable_rtl=True,
        exact_rtl_cycle_correlation=True,
        limitation=(
            "Shared arithmetic slice only: INT8 dot4 plus cached-norm distance "
            "reconstruction. Point-cloud INT8 fidelity, norm SRAM, FPS control, "
            "and KNN top-k are not integrated."
        ),
    ),
    "fps": OperatorCoverage(
        "fps", False, False, False, False,
        "Distance tile is correlated, but persistent min/argmax feedback and memory integration are incomplete.",
    ),
    "knn_topk": OperatorCoverage(
        "knn_topk", False, False, False, False,
        "Distance tile is correlated, but exact global top-32 merge and memory integration are incomplete.",
    ),
    "dense_gemm": OperatorCoverage(
        "dense_gemm", True, True, True, True,
        "Production 64-column stream mode is exact for all 4096 real q-projection "
        "outputs; K-block-resident high-M mode is exact for all 590976 outputs of "
        "the PointTransformer QKV shape. Request-side DRAMsim3 feedback, real "
        "PointTransformer payload, foundry SRAM, and full-graph composition remain open.",
    ),
    "dense64_dual_a8_bf16_output": OperatorCoverage(
        "dense64_dual_a8_bf16_output", True, True, True, True,
        "64-way A8 plus 8/16/32/64-lane BF16 adapter is exact; all 4096 real "
        "q-projection outputs use package-backed FP16 scales. Dense-to-adapter "
        "wiring is structurally checked but not yet one composed event trace.",
    ),
    "shared_dot4_dense_gemm_microtile": OperatorCoverage(
        "shared_dot4_dense_gemm_microtile", True, True, True, True,
        "Output-stationary N microtile with K accumulation and edge-column mask; production M/N tiling, SRAM double buffering, and real-shape controller remain incomplete.",
    ),
    "dense_mnk_ping_pong_controller": OperatorCoverage(
        "dense_mnk_ping_pong_controller", True, True, True, True,
        "M/N/K counters, edge masks, partial K blocks, and two-buffer lifecycle "
        "are correlated; physical banked-SRAM payload reads and DMA are not integrated.",
    ),
    "int32_a8_requant": OperatorCoverage(
        "int32_a8_requant", True, True, True, True,
        "Offline FP16-scale to multiplier/shift compilation plus INT32-domain "
        "bias, RNE, and symmetric [-127,127] saturation; floating BF16 dequant "
        "and integration with Dense GEMM/SRAM remain incomplete.",
    ),
    "dense_sram_dot4_requant_physical_slice": OperatorCoverage(
        "dense_sram_dot4_requant_physical_slice", True, True, True, True,
        "Composed 16-bank/128-bit/3-cycle behavioral SRAM, ping-pong 1R1W, "
        "response FIFO, three-column shared-Dot4 tile, and parallel requant; "
        "production 64-column multiword gather and DRAMsim3 DMA remain incomplete.",
    ),
    "pointllm_splitk_linear": OperatorCoverage(
        "pointllm_splitk_linear", True, True, True, True,
        "Production Q and non-uniform-K down-projection controllers are exactly "
        "RTL-correlated against real DRAMsim3 completion traces; arithmetic value "
        "parity is locked by the separate W8 GEMV datapath slice.",
    ),
    "attention": OperatorCoverage(
        "attention", False, False, False, False,
        "No QK/online-softmax/AV RTL slice or paged-KV memory correlation.",
    ),
    "vector_norm_activation": OperatorCoverage(
        "vector_norm_activation", False, False, False, False,
        "No GTSU vector RTL slice or bit/cycle correlation.",
    ),
    "lm_head_sampling": OperatorCoverage(
        "lm_head_sampling", False, False, False, False,
        "No integrated LM-head/argmax RTL slice or cycle correlation.",
    ),
}


class UnsupportedCycleAccurateOperator(RuntimeError):
    pass


def require_rtl_correlated(operators: Iterable[str]) -> None:
    unsupported = []
    for operator in operators:
        coverage = COVERAGE.get(operator)
        if coverage is None or not coverage.exact_rtl_cycle_correlation:
            unsupported.append(operator)
    if unsupported:
        names = ", ".join(sorted(set(unsupported)))
        raise UnsupportedCycleAccurateOperator(
            "cycle-accurate report refused; operators lack exact RTL correlation: "
            f"{names}"
        )


def coverage_report() -> dict[str, object]:
    rows = [asdict(COVERAGE[key]) for key in sorted(COVERAGE)]
    return {
        "schema_version": "0.1",
        "full_pointllm_cycle_accurate": False,
        "policy": "fail_closed",
        "operators": rows,
    }
