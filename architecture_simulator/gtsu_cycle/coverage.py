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
    "fps": OperatorCoverage(
        "fps", False, False, False, False,
        "QuickFPS is an external reference; no GTSU-integrated RTL correlation yet.",
    ),
    "knn_topk": OperatorCoverage(
        "knn_topk", False, False, False, False,
        "No GTSU distance/top-k RTL slice or cycle correlation.",
    ),
    "dense_gemm": OperatorCoverage(
        "dense_gemm", False, False, False, False,
        "No dense-mode tensor-fabric RTL slice or cycle correlation.",
    ),
    "pointllm_splitk_linear": OperatorCoverage(
        "pointllm_splitk_linear", False, False, False, False,
        "Checkpoint-backed tile/address lowering and DRAMsim3 burst timing exist, but "
        "SRAM/DRAM completion is not yet integrated with the compute RTL slice.",
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
