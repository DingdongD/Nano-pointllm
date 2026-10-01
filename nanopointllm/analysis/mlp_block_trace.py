from __future__ import annotations

import itertools
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import torch


_SCHEMA_VERSION = 1


def _mean(values: Iterable[float | None]) -> float | None:
    valid = [float(value) for value in values if value is not None]
    return sum(valid) / len(valid) if valid else None


def _jaccard(left: set[int], right: set[int]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def _support_recall(reference: set[int], candidate: set[int]) -> float:
    return len(reference & candidate) / len(reference) if reference else 1.0


class MLPBlockTraceCollector:
    """Collect decode-only SwiGLU intermediates without affecting normal execution."""

    def __init__(
        self,
        *,
        expected_layers: int | None = None,
        storage_dtype: torch.dtype = torch.float16,
    ) -> None:
        self.expected_layers = expected_layers
        self.storage_dtype = storage_dtype
        self.records: list[dict[str, Any]] = []
        self.activations: list[torch.Tensor] = []
        self.down_column_norms: dict[int, torch.Tensor] = {}
        self._contexts: list[dict[str, Any]] | None = None
        self._step_start = 0
        self._layers_seen: set[int] = set()
        self.hidden_size: int | None = None
        self.intermediate_size: int | None = None
        self.weight_element_size: int | None = None

    @property
    def active(self) -> bool:
        return self._contexts is not None

    def begin_step(self, contexts: list[dict[str, Any]]) -> None:
        if self.active:
            raise RuntimeError("previous MLP trace step has not been ended")
        if not contexts:
            raise ValueError("at least one row context is required")
        self._contexts = [dict(context) for context in contexts]
        self._step_start = len(self.records)
        self._layers_seen = set()

    def end_step(self, output_token_ids: list[int] | None = None) -> None:
        if not self.active:
            raise RuntimeError("no active MLP trace step")
        contexts = self._contexts or []
        if output_token_ids is not None and len(output_token_ids) != len(contexts):
            raise ValueError("output_token_ids must match the traced row count")
        expected = self.expected_layers
        if expected is not None and len(self._layers_seen) != expected:
            raise RuntimeError(
                f"traced {len(self._layers_seen)} decoder layers, expected {expected}"
            )
        if output_token_ids is not None:
            for record in self.records[self._step_start:]:
                record["output_token_id"] = int(output_token_ids[record["row_index"]])
        self._contexts = None
        self._layers_seen = set()

    def __call__(
        self,
        *,
        layer_index: int,
        activations: torch.Tensor,
        down_proj_weight: torch.Tensor,
    ) -> None:
        if not self.active:
            return
        rows = activations.detach().reshape(-1, activations.shape[-1])
        contexts = self._contexts or []
        if rows.shape[0] != len(contexts):
            raise RuntimeError(
                f"trace context has {len(contexts)} rows but layer {layer_index} emitted "
                f"{rows.shape[0]} rows"
            )
        if layer_index in self._layers_seen:
            raise RuntimeError(f"decoder layer {layer_index} was observed twice in one step")
        self._layers_seen.add(layer_index)

        hidden_size, intermediate_size = down_proj_weight.shape
        if rows.shape[1] != intermediate_size:
            raise RuntimeError("SwiGLU activation width does not match down projection")
        if self.intermediate_size not in (None, intermediate_size):
            raise RuntimeError("intermediate size changed within one trace")
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.weight_element_size = down_proj_weight.element_size()

        if layer_index not in self.down_column_norms:
            norms = torch.linalg.vector_norm(
                down_proj_weight.detach(), ord=2, dim=0, dtype=torch.float32
            )
            self.down_column_norms[layer_index] = norms.cpu()

        cpu_rows = rows.to(device="cpu", dtype=self.storage_dtype)
        for row_index, (context, row) in enumerate(zip(contexts, cpu_rows)):
            record = dict(context)
            record.update({"layer": int(layer_index), "row_index": row_index})
            self.records.append(record)
            self.activations.append(row.contiguous())

    def payload(self) -> dict[str, Any]:
        if self.active:
            raise RuntimeError("cannot export an active MLP trace step")
        if not self.activations:
            raise RuntimeError("MLP trace is empty")
        layers = sorted(self.down_column_norms)
        return {
            "schema_version": _SCHEMA_VERSION,
            "activation_name": "silu(gate) * up",
            "activation_dtype": str(self.storage_dtype).removeprefix("torch."),
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "weight_element_size": self.weight_element_size,
            "records": self.records,
            "activations": torch.stack(self.activations),
            "down_column_norm_layers": layers,
            "down_column_norms": torch.stack(
                [self.down_column_norms[layer] for layer in layers]
            ),
        }

    def save(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.payload(), output)
        return output


def load_mlp_block_trace(path: str | Path) -> dict[str, Any]:
    return torch.load(Path(path), map_location="cpu", weights_only=False)


def _block_scores(
    activation: torch.Tensor,
    column_norms: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    contribution = activation.float().abs() * column_norms.float()
    padding = (-contribution.numel()) % block_size
    if padding:
        contribution = torch.nn.functional.pad(contribution, (0, padding))
    return contribution.view(-1, block_size).sum(dim=1)


def _selected_neuron_ratio(
    *,
    selected_blocks: int,
    block_size: int,
    intermediate_size: int,
) -> float:
    return min(selected_blocks * block_size, intermediate_size) / intermediate_size


def analyze_mlp_block_trace(
    payload: dict[str, Any],
    *,
    block_sizes: Iterable[int] = (32, 64, 128, 256),
    block_ratios: Iterable[float] = (0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0),
) -> dict[str, Any]:
    records = payload["records"]
    activations = payload["activations"]
    norm_layers = payload["down_column_norm_layers"]
    norms = payload["down_column_norms"]
    if len(records) != len(activations):
        raise ValueError("trace metadata and activation counts differ")
    layer_to_norm = {int(layer): norms[index] for index, layer in enumerate(norm_layers)}
    intermediate_size = int(payload["intermediate_size"])
    hidden_size = int(payload["hidden_size"])
    element_size = int(payload["weight_element_size"])
    baseline_bytes = 3 * hidden_size * intermediate_size * element_size
    sizes = tuple(sorted(set(int(size) for size in block_sizes)))
    ratios = tuple(sorted(set(float(ratio) for ratio in block_ratios)))
    if not sizes or any(size <= 0 for size in sizes):
        raise ValueError("block sizes must be positive")
    if not ratios or any(ratio <= 0.0 or ratio > 1.0 for ratio in ratios):
        raise ValueError("block ratios must be in (0, 1]")

    score_cache: dict[tuple[int, int], torch.Tensor] = {}
    ranking_cache: dict[tuple[int, int], torch.Tensor] = {}
    detailed: list[dict[str, Any]] = []
    previous: dict[tuple[str, int, int, float], tuple[set[int], torch.Tensor]] = {}

    ordered_indices = sorted(
        range(len(records)),
        key=lambda index: (
            str(records[index]["sample_id"]),
            int(records[index]["decode_step"]),
            int(records[index]["layer"]),
        ),
    )
    for index in ordered_indices:
        record = records[index]
        layer = int(record["layer"])
        activation = activations[index]
        for block_size in sizes:
            scores = _block_scores(activation, layer_to_norm[layer], block_size)
            score_cache[(index, block_size)] = scores
            total = float(scores.sum().item())
            num_blocks = scores.numel()
            ranking = torch.argsort(scores, descending=True)
            ranking_cache[(index, block_size)] = ranking
            for requested_ratio in ratios:
                retained_blocks = max(1, math.ceil(num_blocks * requested_ratio))
                selected = set(int(block) for block in ranking[:retained_blocks].tolist())
                selected_ratio = _selected_neuron_ratio(
                    selected_blocks=retained_blocks,
                    block_size=block_size,
                    intermediate_size=intermediate_size,
                )
                retained = (
                    float(scores[list(selected)].sum().item()) / total if total > 0.0 else 1.0
                )
                temporal_key = (
                    str(record["sample_id"]), layer, block_size, requested_ratio
                )
                previous_entry = previous.get(temporal_key)
                if previous_entry is None:
                    adjacent_jaccard = None
                    previous_recall = None
                    previous_retained = None
                else:
                    previous_support, _ = previous_entry
                    adjacent_jaccard = _jaccard(selected, previous_support)
                    previous_recall = _support_recall(selected, previous_support)
                    previous_retained = (
                        float(scores[list(previous_support)].sum().item()) / total
                        if total > 0.0
                        else 1.0
                    )
                previous[temporal_key] = (selected, scores)

                post_gate_bytes = (2.0 + selected_ratio) / 3.0 * baseline_bytes
                pre_gate_bytes = selected_ratio * baseline_bytes
                row = dict(record)
                row.update({
                    "block_size": block_size,
                    "num_blocks": num_blocks,
                    "requested_block_ratio": requested_ratio,
                    "retained_blocks": retained_blocks,
                    "retained_block_ratio": retained_blocks / num_blocks,
                    "retained_neuron_ratio": selected_ratio,
                    "post_gate_contribution_retained": retained,
                    "oracle_pre_gate_contribution_retained": retained,
                    "previous_token_contribution_retained": previous_retained,
                    "adjacent_token_jaccard": adjacent_jaccard,
                    "previous_token_recall": previous_recall,
                    "baseline_weight_bytes": baseline_bytes,
                    "post_gate_weight_bytes": post_gate_bytes,
                    "oracle_pre_gate_weight_bytes": pre_gate_bytes,
                    "previous_token_weight_bytes": pre_gate_bytes if previous_entry else baseline_bytes,
                    "post_gate_weight_byte_reduction": 1.0 - post_gate_bytes / baseline_bytes,
                    "oracle_pre_gate_weight_byte_reduction": 1.0 - pre_gate_bytes / baseline_bytes,
                    "previous_token_weight_byte_reduction": (
                        1.0 - pre_gate_bytes / baseline_bytes if previous_entry else 0.0
                    ),
                })
                detailed.append(row)

    record_lookup = {
        (
            str(record["point_cloud_id"]),
            int(record["decode_step"]),
            int(record["layer"]),
        ): []
        for record in records
    }
    for index, record in enumerate(records):
        key = (
            str(record["point_cloud_id"]),
            int(record["decode_step"]),
            int(record["layer"]),
        )
        record_lookup[key].append(index)

    cross_prompt: list[dict[str, Any]] = []
    for (point_cloud_id, decode_step, layer), indices in record_lookup.items():
        by_prompt = defaultdict(list)
        for index in indices:
            by_prompt[str(records[index]["prompt_id"])].append(index)
        for left_prompt, right_prompt in itertools.combinations(sorted(by_prompt), 2):
            for left_index, right_index in itertools.product(
                by_prompt[left_prompt], by_prompt[right_prompt]
            ):
                for block_size in sizes:
                    for requested_ratio in ratios:
                        left_ranking = ranking_cache[(left_index, block_size)]
                        right_ranking = ranking_cache[(right_index, block_size)]
                        left_count = max(
                            1, math.ceil(left_ranking.numel() * requested_ratio)
                        )
                        right_count = max(
                            1, math.ceil(right_ranking.numel() * requested_ratio)
                        )
                        left = set(int(block) for block in left_ranking[:left_count].tolist())
                        right = set(int(block) for block in right_ranking[:right_count].tolist())
                        cross_prompt.append({
                            "point_cloud_id": point_cloud_id,
                            "decode_step": decode_step,
                            "layer": layer,
                            "left_prompt_id": left_prompt,
                            "right_prompt_id": right_prompt,
                            "block_size": block_size,
                            "requested_block_ratio": requested_ratio,
                            "jaccard": _jaccard(left, right),
                            "left_recall": _support_recall(left, right),
                            "right_recall": _support_recall(right, left),
                        })

    curve_groups: dict[tuple[int, float], list[dict[str, Any]]] = defaultdict(list)
    for row in detailed:
        curve_groups[(row["block_size"], row["requested_block_ratio"])].append(row)
    cross_groups: dict[tuple[int, float], list[dict[str, Any]]] = defaultdict(list)
    for row in cross_prompt:
        cross_groups[(row["block_size"], row["requested_block_ratio"])].append(row)

    def summarize_rows(
        rows: list[dict[str, Any]],
        cross_rows: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return {
            "retained_block_ratio_mean": _mean(row["retained_block_ratio"] for row in rows),
            "retained_neuron_ratio_mean": _mean(row["retained_neuron_ratio"] for row in rows),
            "post_gate_contribution_retained_mean": _mean(
                row["post_gate_contribution_retained"] for row in rows
            ),
            "oracle_pre_gate_contribution_retained_mean": _mean(
                row["oracle_pre_gate_contribution_retained"] for row in rows
            ),
            "previous_token_contribution_retained_mean": _mean(
                row["previous_token_contribution_retained"] for row in rows
            ),
            "adjacent_token_jaccard_mean": _mean(
                row["adjacent_token_jaccard"] for row in rows
            ),
            "previous_token_recall_mean": _mean(
                row["previous_token_recall"] for row in rows
            ),
            "same_point_cross_prompt_jaccard_mean": _mean(
                row["jaccard"] for row in cross_rows
            ),
            "post_gate_weight_byte_reduction_mean": _mean(
                row["post_gate_weight_byte_reduction"] for row in rows
            ),
            "oracle_pre_gate_weight_byte_reduction_mean": _mean(
                row["oracle_pre_gate_weight_byte_reduction"] for row in rows
            ),
            "previous_token_weight_byte_reduction_steady_state": (
                1.0 - float(rows[0]["retained_neuron_ratio"])
            ),
            "previous_token_weight_byte_reduction_including_first_step_mean": _mean(
                row["previous_token_weight_byte_reduction"] for row in rows
            ),
            "observations": len(rows),
            "temporal_pairs": sum(row["adjacent_token_jaccard"] is not None for row in rows),
            "cross_prompt_pairs": len(cross_rows),
        }

    curves = []
    for key in sorted(curve_groups):
        block_size, requested_ratio = key
        rows = curve_groups[key]
        cross_rows = cross_groups.get(key, [])
        curves.append({
            "block_size": block_size,
            "requested_block_ratio": requested_ratio,
            **summarize_rows(rows, cross_rows),
        })

    layer_groups: dict[tuple[int, int, float], list[dict[str, Any]]] = defaultdict(list)
    for row in detailed:
        layer_groups[(row["layer"], row["block_size"], row["requested_block_ratio"])].append(row)
    layer_cross_groups: dict[tuple[int, int, float], list[dict[str, Any]]] = defaultdict(list)
    for row in cross_prompt:
        layer_cross_groups[(row["layer"], row["block_size"], row["requested_block_ratio"])].append(row)
    layer_curves = [
        {
            "layer": key[0],
            "block_size": key[1],
            "requested_block_ratio": key[2],
            **summarize_rows(rows, layer_cross_groups.get(key, [])),
        }
        for key, rows in sorted(layer_groups.items())
    ]

    step_groups: dict[tuple[int, int, float], list[dict[str, Any]]] = defaultdict(list)
    for row in detailed:
        step_groups[(row["decode_step"], row["block_size"], row["requested_block_ratio"])].append(row)
    step_cross_groups: dict[tuple[int, int, float], list[dict[str, Any]]] = defaultdict(list)
    for row in cross_prompt:
        step_cross_groups[(row["decode_step"], row["block_size"], row["requested_block_ratio"])].append(row)
    step_curves = [
        {
            "decode_step": key[0],
            "block_size": key[1],
            "requested_block_ratio": key[2],
            **summarize_rows(rows, step_cross_groups.get(key, [])),
        }
        for key, rows in sorted(step_groups.items())
    ]

    return {
        "schema_version": _SCHEMA_VERSION,
        "metric_definitions": {
            "neuron_contribution": "abs(silu(gate_i) * up_i) * l2_norm(down_proj[:, i])",
            "block_contribution": "sum of neuron contributions inside one contiguous block",
            "post_gate": "current-token top blocks; gate/up dense, down_proj block sparse",
            "oracle_pre_gate": "current-token top blocks known before gate/up; non-causal upper bound",
            "previous_token": "previous decode token's top blocks reused for current token",
            "weight_bytes": "MLP gate/up/down compulsory weight bytes only; index and activation traffic excluded",
        },
        "trace": {
            "records": len(records),
            "layers": len(norm_layers),
            "activation_dtype": payload.get("activation_dtype", str(activations.dtype).removeprefix("torch.")),
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
            "weight_element_size": element_size,
            "baseline_mlp_weight_bytes_per_layer": baseline_bytes,
        },
        "block_sizes": list(sizes),
        "block_ratios": list(ratios),
        "curves": curves,
        "layer_curves": layer_curves,
        "step_curves": step_curves,
        "details": detailed,
        "cross_prompt_overlap": cross_prompt,
    }
