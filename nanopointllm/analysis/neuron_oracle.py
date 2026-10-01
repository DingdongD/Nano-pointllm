from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Iterable

import torch


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else float("nan")


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scalar_keys = [
        key for key, value in rows[0].items()
        if isinstance(value, float) and key not in {"group_size"}
    ]
    return {key + "_mean": _mean(row[key] for row in rows) for key in scalar_keys}


def analyze_neuron_oracle(
    payload: dict[str, Any],
    *,
    group_sizes: Iterable[int] = (1, 4, 8, 16, 32),
    budget_ratios: Iterable[float] = (0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0),
    contribution_targets: Iterable[float] = (0.5, 0.8, 0.9, 0.95, 0.99),
    chunk_size: int = 64,
) -> dict[str, Any]:
    records = payload["records"]
    activations = payload["activations"]
    norm_layers = [int(layer) for layer in payload["down_column_norm_layers"]]
    norms = payload["down_column_norms"].float()
    if len(records) != len(activations):
        raise ValueError("trace metadata and activation counts differ")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    sizes = tuple(sorted(set(int(size) for size in group_sizes)))
    budgets = tuple(sorted(set(float(ratio) for ratio in budget_ratios)))
    targets = tuple(sorted(set(float(target) for target in contribution_targets)))
    if not sizes or any(size <= 0 for size in sizes):
        raise ValueError("group sizes must be positive")
    if any(ratio <= 0.0 or ratio > 1.0 for ratio in budgets):
        raise ValueError("budget ratios must be in (0, 1]")
    if any(target <= 0.0 or target > 1.0 for target in targets):
        raise ValueError("contribution targets must be in (0, 1]")

    layer_to_norm = {layer: index for index, layer in enumerate(norm_layers)}
    record_norm_indices = torch.tensor(
        [layer_to_norm[int(record["layer"])] for record in records], dtype=torch.long
    )
    intermediate_size = int(payload["intermediate_size"])
    details: list[dict[str, Any]] = []

    for group_size in sizes:
        for start in range(0, len(records), chunk_size):
            end = min(start + chunk_size, len(records))
            activation = activations[start:end].float().abs()
            column_norm = norms.index_select(0, record_norm_indices[start:end])
            contribution = activation * column_norm
            padding = (-intermediate_size) % group_size
            if padding:
                contribution = torch.nn.functional.pad(contribution, (0, padding))
            scores = contribution.view(contribution.shape[0], -1, group_size).sum(dim=-1)
            num_groups = scores.shape[1]
            total = scores.sum(dim=1).clamp_min(torch.finfo(torch.float32).tiny)
            probabilities = scores / total[:, None]

            entropy = -(probabilities * probabilities.clamp_min(1e-30).log()).sum(dim=1)
            normalized_entropy = entropy / math.log(num_groups) if num_groups > 1 else entropy * 0.0
            effective_support = entropy.exp()
            sorted_ascending = torch.sort(scores, dim=1).values
            positions = torch.arange(
                1, num_groups + 1, dtype=torch.float32
            ).view(1, -1)
            gini = (
                (2.0 * positions - num_groups - 1.0) * sorted_ascending
            ).sum(dim=1) / (num_groups * total)

            sorted_descending = sorted_ascending.flip(dims=(1,))
            cumulative = sorted_descending.cumsum(dim=1) / total[:, None]
            budget_retention = {}
            budget_neuron_ratio = {}
            for budget in budgets:
                retained_groups = max(1, math.ceil(num_groups * budget))
                key = f"retention_at_{int(round(budget * 100)):02d}pct"
                budget_retention[key] = cumulative[:, retained_groups - 1]
                budget_neuron_ratio[
                    f"actual_neuron_ratio_at_{int(round(budget * 100)):02d}pct"
                ] = min(retained_groups * group_size, intermediate_size) / intermediate_size

            target_support = {}
            for target in targets:
                required_groups = (cumulative < target).sum(dim=1) + 1
                retained_neurons = torch.clamp(
                    required_groups * group_size, max=intermediate_size
                )
                target_support[
                    f"neuron_ratio_for_{int(round(target * 100)):02d}pct_contribution"
                ] = retained_neurons.float() / intermediate_size

            for local_index, record_index in enumerate(range(start, end)):
                row = {
                    "sample_id": str(records[record_index]["sample_id"]),
                    "point_cloud_id": str(records[record_index]["point_cloud_id"]),
                    "prompt_id": str(records[record_index]["prompt_id"]),
                    "decode_step": int(records[record_index]["decode_step"]),
                    "layer": int(records[record_index]["layer"]),
                    "group_size": group_size,
                    "num_groups": num_groups,
                    "gini": float(gini[local_index].item()),
                    "entropy": float(entropy[local_index].item()),
                    "normalized_entropy": float(normalized_entropy[local_index].item()),
                    "effective_support_size": float(effective_support[local_index].item()),
                    "effective_support_ratio": float(
                        effective_support[local_index].item() / num_groups
                    ),
                }
                row.update({
                    key: float(value[local_index].item())
                    for key, value in budget_retention.items()
                })
                row.update(budget_neuron_ratio)
                row.update({
                    key: float(value[local_index].item())
                    for key, value in target_support.items()
                })
                details.append(row)

    global_groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    layer_groups: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    step_groups: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in details:
        global_groups[row["group_size"]].append(row)
        layer_groups[(row["layer"], row["group_size"])].append(row)
        step_groups[(row["decode_step"], row["group_size"])].append(row)
    summary = [
        {"group_size": group_size, "observations": len(rows), **_aggregate(rows)}
        for group_size, rows in sorted(global_groups.items())
    ]
    by_layer = [
        {"layer": key[0], "group_size": key[1], "observations": len(rows), **_aggregate(rows)}
        for key, rows in sorted(layer_groups.items())
    ]
    by_step = [
        {"decode_step": key[0], "group_size": key[1], "observations": len(rows), **_aggregate(rows)}
        for key, rows in sorted(step_groups.items())
    ]

    neuron = next(row for row in summary if row["group_size"] == 1)
    retention_10 = neuron.get("retention_at_10pct_mean")
    support_90 = neuron.get("neuron_ratio_for_90pct_contribution_mean")
    decision_available = retention_10 is not None and support_90 is not None
    close_activation_sparsity = bool(
        decision_available and retention_10 < 0.5 and support_90 > 0.5
    )
    decision = {
        "close_activation_sparsity": close_activation_sparsity,
        "criterion": (
            "close when B=1 oracle retention@10% < 0.50 and the neuron ratio "
            "required for 90% contribution > 0.50"
        ),
        "retention_at_10pct": retention_10,
        "neuron_ratio_for_90pct_contribution": support_90,
        "recommendation": (
            "Stop activation-sparsity and permutation/clustering work."
            if close_activation_sparsity
            else (
                "Proceed to exact permutation/clustering upper-bound only."
                if decision_available
                else "Decision unavailable: include 10% budget and 90% contribution target."
            )
        ),
    }
    return {
        "metric_definitions": {
            "gini": "standard Gini coefficient over non-negative group contribution mass",
            "entropy": "Shannon entropy of normalized group contribution mass",
            "normalized_entropy": "entropy / log(number of groups)",
            "effective_support_size": "exp(entropy)",
            "effective_support_ratio": "exp(entropy) / number of groups",
        },
        "trace": {
            "records": len(records),
            "layers": len(norm_layers),
            "activation_dtype": payload.get("activation_dtype"),
            "intermediate_size": intermediate_size,
        },
        "group_sizes": list(sizes),
        "budget_ratios": list(budgets),
        "contribution_targets": list(targets),
        "decision": decision,
        "summary": summary,
        "by_layer": by_layer,
        "by_step": by_step,
        "details": details,
    }
