#!/usr/bin/env python3
"""End-to-end PointLLM performance diagnostic benchmark.

Measures wall/CUDA TTFT, request-level TPOT, aggregate throughput, prefill/decode
time, decode CUDA component time, KV-cache capacity/use, and an analytical HBM
traffic lower bound. Hardware-counter bandwidth requires Nsight Compute and is
deliberately not claimed by this script.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from nanopointllm.benchmark_metrics import (
    estimate_decode_hbm,
    identify_peak_hbm_gbps,
    kv_cache_metrics,
    mean,
    resize_point_prompt,
)
from nanopointllm.engine.llm_engine import PointLLMLLMEngine
from nanopointllm.gpu_telemetry import GpuTelemetryMonitor, query_gpu_state
from nanopointllm.parity.engine_test_utils import (
    build_prompt_token_ids,
    load_pointllm_model,
    make_fake_point_cloud,
)
from nanopointllm.sampling_params import SamplingParams


def parse_ints(raw: str) -> list[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values or any(value <= 0 for value in values):
        raise ValueError(f"expected comma-separated positive integers, got {raw!r}")
    return sorted(set(values))


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean": mean(values),
        "median": statistics.median(values) if values else 0.0,
        "p10": percentile(values, 0.10),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "min": min(values) if values else 0.0,
        "max": max(values) if values else 0.0,
    }


def cuda_event() -> torch.cuda.Event:
    return torch.cuda.Event(enable_timing=True)


def unique_parameter_bytes(modules: list[torch.nn.Module]) -> int:
    seen: set[tuple[int, int]] = set()
    total = 0
    for module in modules:
        for parameter in module.parameters():
            key = (parameter.untyped_storage().data_ptr(), parameter.untyped_storage().nbytes())
            if key not in seen:
                seen.add(key)
                total += parameter.untyped_storage().nbytes()
    return total


def decoder_weight_bytes(model: torch.nn.Module) -> int:
    backbone = model.model if hasattr(model, "model") else model
    modules = [backbone.embed_tokens, backbone.layers, backbone.norm, model.lm_head]
    return unique_parameter_bytes(modules)


def gpu_snapshot() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,utilization.memory,power.draw",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.check_output(command, text=True, timeout=5).strip()
        return {"command": " ".join(command), "rows": output.splitlines()}
    except Exception as exc:
        return {"command": " ".join(command), "error": str(exc)}


def set_decode_profiling(runner, enabled: bool) -> None:
    runner.profile_runtime = enabled
    lightweight = runner.lightweight_runner
    if lightweight is None:
        raise RuntimeError("decode profiling requires the lightweight runner")
    lightweight.profile_enabled = enabled
    for layer in lightweight.layers:
        layer.profile_enabled = enabled
        layer.mlp.profile_enabled = enabled


def run_once(
    engine: PointLLMLLMEngine,
    *,
    token_ids: list[int],
    batch_size: int,
    output_tokens: int,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, float | int]:
    engine.clear_point_feature_cache()
    params = SamplingParams(max_tokens=output_tokens, ignore_eos=True)
    seqs = [
        engine.add_request(
            token_ids=token_ids,
            point_clouds=make_fake_point_cloud(device, dtype),
            sampling_params=params,
        )
        for _ in range(batch_size)
    ]

    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    baseline_allocated = torch.cuda.memory_allocated(device)

    prefill_start = cuda_event()
    prefill_end = cuda_event()
    prefill_start.record()
    wall_start = time.perf_counter()
    engine.step()
    prefill_end.record()
    prefill_end.synchronize()
    ttft_wall_ms = (time.perf_counter() - wall_start) * 1000.0
    ttft_cuda_ms = prefill_start.elapsed_time(prefill_end)
    if not all(seq.num_completion_tokens == 1 for seq in seqs):
        raise RuntimeError("TTFT step did not produce exactly one token per request")

    block_manager = engine.runner.block_manager
    peak_active_blocks = len(block_manager.used_block_ids)
    decode_steps = 0
    decode_start = cuda_event()
    decode_end = cuda_event()
    decode_start.record()
    decode_wall_start = time.perf_counter()
    while not all(seq.is_finished for seq in seqs):
        engine.step()
        decode_steps += 1
        peak_active_blocks = max(peak_active_blocks, len(block_manager.used_block_ids))
    decode_end.record()
    decode_end.synchronize()
    decode_wall_ms = (time.perf_counter() - decode_wall_start) * 1000.0
    decode_cuda_ms = decode_start.elapsed_time(decode_end)
    expected_steps = max(output_tokens - 1, 0)
    if decode_steps != expected_steps:
        raise RuntimeError(f"expected {expected_steps} decode steps, got {decode_steps}")

    total_output_tokens = batch_size * output_tokens
    total_wall_ms = ttft_wall_ms + decode_wall_ms
    return {
        "ttft_wall_ms": ttft_wall_ms,
        "ttft_cuda_ms": ttft_cuda_ms,
        "decode_wall_ms": decode_wall_ms,
        "decode_cuda_ms": decode_cuda_ms,
        "decode_steps": decode_steps,
        "request_tpot_wall_ms": decode_wall_ms / decode_steps if decode_steps else 0.0,
        "request_tpot_cuda_ms": decode_cuda_ms / decode_steps if decode_steps else 0.0,
        "decode_tokens_per_sec": (
            batch_size * decode_steps / (decode_wall_ms / 1000.0) if decode_wall_ms else 0.0
        ),
        "e2e_output_tokens_per_sec": total_output_tokens / (total_wall_ms / 1000.0),
        "peak_active_kv_blocks_observed": peak_active_blocks,
        "request_peak_allocated_delta_gib": (
            torch.cuda.max_memory_allocated(device) - baseline_allocated
        ) / (1024 ** 3),
    }


def aggregate_runs(
    raw_runs: list[dict[str, float | int]],
    *,
    batch_size: int,
    input_tokens: int,
    output_tokens: int,
    cold_start_run: dict[str, float | int],
    gpu_telemetry: dict[str, Any],
) -> dict[str, Any]:
    fields = (
        "ttft_wall_ms",
        "ttft_cuda_ms",
        "decode_wall_ms",
        "decode_cuda_ms",
        "request_tpot_wall_ms",
        "request_tpot_cuda_ms",
        "decode_tokens_per_sec",
        "e2e_output_tokens_per_sec",
        "request_peak_allocated_delta_gib",
    )
    result: dict[str, Any] = {
        "batch_size": batch_size,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "decode_steps": max(output_tokens - 1, 0),
        "cold_start_run": cold_start_run,
        "raw_runs": raw_runs,
        "gpu_telemetry": gpu_telemetry,
    }
    for field in fields:
        result[field] = summarize([float(run[field]) for run in raw_runs])
    result["peak_active_kv_blocks_observed"] = max(
        int(run["peak_active_kv_blocks_observed"]) for run in raw_runs
    )
    result["prefill_fraction_e2e"] = (
        result["ttft_wall_ms"]["median"]
        / (result["ttft_wall_ms"]["median"] + result["decode_wall_ms"]["median"])
    )
    return result


def profile_decode(
    engine: PointLLMLLMEngine,
    *,
    token_ids: list[int],
    batch_size: int,
    profile_steps: int,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    set_decode_profiling(engine.runner, False)
    engine.clear_point_feature_cache()
    params = SamplingParams(max_tokens=profile_steps + 1, ignore_eos=True)
    seqs = [
        engine.add_request(
            token_ids=token_ids,
            point_clouds=make_fake_point_cloud(device, dtype),
            sampling_params=params,
        )
        for _ in range(batch_size)
    ]
    engine.step()
    torch.cuda.synchronize(device)
    set_decode_profiling(engine.runner, True)

    samples = []
    for _ in range(profile_steps):
        context_tokens = mean([float(seq.num_tokens) for seq in seqs])
        start = cuda_event()
        end = cuda_event()
        start.record()
        engine.step()
        end.record()
        end.synchronize()
        lightweight = engine.runner.lightweight_runner
        samples.append({
            "context_tokens": context_tokens,
            "decode_cuda_timeline_ms": start.elapsed_time(end),
            "global": dict(lightweight.last_global_profile),
            "runtime": dict(engine.runner.last_cuda_profile),
            "layers": [dict(row) for row in lightweight.last_profile],
        })
    set_decode_profiling(engine.runner, False)

    global_keys = samples[0]["global"].keys()
    runtime_keys = samples[0]["runtime"].keys()
    global_mean = {key: mean(sample["global"][key] for sample in samples) for key in global_keys}
    runtime_mean = {key: mean(sample["runtime"][key] for sample in samples) for key in runtime_keys}
    layer_accum: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for sample in samples:
        for row in sample["layers"]:
            layer = int(row["layer"])
            for key, value in row.items():
                if key != "layer":
                    layer_accum[layer][key].append(float(value))
    layers = [
        {"layer": layer, **{key: mean(values) for key, values in fields.items()}}
        for layer, fields in sorted(layer_accum.items())
    ]
    attention_ms = sum(row["attn_ms"] for row in layers)
    mlp_ms = sum(row["mlp_ms"] for row in layers)
    lm_head_ms = global_mean.get("lm_head_ms", 0.0)
    sampling_ms = runtime_mean.get("sampling_ms", 0.0)
    timeline_ms = mean(sample["decode_cuda_timeline_ms"] for sample in samples)
    categories = {
        "attention_ms": attention_ms,
        "mlp_ms": mlp_ms,
        "lm_head_ms": lm_head_ms,
        "sampling_ms": sampling_ms,
    }
    profiled_cuda_ms = global_mean.get("model_total_ms", 0.0) + sampling_ms
    categories["model_other_ms"] = max(profiled_cuda_ms - sum(categories.values()), 0.0)
    return {
        "batch_size": batch_size,
        "input_tokens": len(token_ids),
        "profile_steps": profile_steps,
        "average_context_tokens": mean(sample["context_tokens"] for sample in samples),
        "decode_cuda_timeline_ms": timeline_ms,
        "profiled_cuda_ms": profiled_cuda_ms,
        "unattributed_timeline_ms": max(timeline_ms - profiled_cuda_ms, 0.0),
        "component_cuda_ms": categories,
        "component_percent_of_profiled_cuda": {
            key.replace("_ms", "_pct"): value / profiled_cuda_ms * 100.0
            for key, value in categories.items()
        },
        "global_cuda_ms": global_mean,
        "layers": layers,
        "raw_samples": samples,
        "scope_note": (
            "attention includes QKV projection, RoPE, KV store/read, attention kernel, and output "
            "projection; MLP includes gate/up, activation, and down projection. Percentages use "
            "model_total + sampling; unattributed_timeline is reported separately and can include "
            "Python enqueue gaps or external GPU scheduling."
        ),
    }


def attach_memory_and_bandwidth(
    result: dict[str, Any],
    *,
    cfg,
    dtype_bytes: int,
    num_blocks: int,
    block_size: int,
    weight_bytes: int,
    peak_hbm_gbps: float | None,
) -> None:
    num_heads = cfg.num_attention_heads
    num_kv_heads = getattr(cfg, "num_key_value_heads", num_heads)
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // num_heads)
    kv = kv_cache_metrics(
        num_layers=cfg.num_hidden_layers,
        num_blocks=num_blocks,
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        dtype_bytes=dtype_bytes,
        batch_size=result["batch_size"],
        prompt_tokens=result["input_tokens"],
        output_tokens=result["output_tokens"],
    )
    result["kv_cache"] = kv
    average_context = result["input_tokens"] + result["output_tokens"] / 2.0
    result["hbm_estimate"] = estimate_decode_hbm(
        decoder_weight_bytes=weight_bytes,
        kv_bytes_token=int(kv["bytes_per_token"]),
        batch_size=result["batch_size"],
        average_context_tokens=average_context,
        batch_step_ms=result["request_tpot_cuda_ms"]["median"],
        peak_hbm_gbps=peak_hbm_gbps,
    )


def analyze(results: list[dict[str, Any]], profiles: list[dict[str, Any]]) -> dict[str, Any]:
    batch_sizes = sorted({row["batch_size"] for row in results})
    input_lengths = sorted({row["input_tokens"] for row in results})
    output_lengths = sorted({row["output_tokens"] for row in results})
    b1 = batch_sizes[0]
    min_input, max_input = input_lengths[0], input_lengths[-1]
    max_output = output_lengths[-1]

    def find(batch: int, inp: int, out: int) -> dict[str, Any]:
        return next(
            row for row in results
            if row["batch_size"] == batch
            and row["input_tokens"] == inp
            and row["output_tokens"] == out
        )

    short = find(b1, min_input, max_output)
    long = find(b1, max_input, max_output)
    small_batch = find(b1, max_input, max_output)
    large_batch = find(batch_sizes[-1], max_input, max_output)
    ttft_growth = long["ttft_wall_ms"]["median"] / max(short["ttft_wall_ms"]["median"], 1e-9)
    tpot_growth = long["request_tpot_wall_ms"]["median"] / max(
        short["request_tpot_wall_ms"]["median"], 1e-9
    )
    throughput_scale = large_batch["decode_tokens_per_sec"]["median"] / max(
        small_batch["decode_tokens_per_sec"]["median"], 1e-9
    )
    batching_efficiency = throughput_scale / batch_sizes[-1] * b1
    dominant = None
    if profiles:
        profile = profiles[0]
        dominant = max(profile["component_cuda_ms"], key=profile["component_cuda_ms"].get)

    priorities = []
    if ttft_growth > 1.25:
        priorities.append(
            "TTFT grows materially with input length: reduce point/prompt tokens and optimize prefill."
        )
    if tpot_growth > 1.15:
        priorities.append(
            "TPOT grows materially with context: prioritize point-token/KV compression and attention."
        )
    weight_share = small_batch["hbm_estimate"]["estimated_decoder_weight_read_gb_per_step"] / max(
        small_batch["hbm_estimate"]["estimated_minimum_hbm_traffic_gb_per_step"], 1e-9
    )
    if weight_share > 0.7:
        priorities.append(
            "Small-batch decode traffic is weight-dominated: weight quantization is the first lever; "
            "speculative decoding is optional only if an accurate cheap draft is available."
        )
    if batching_efficiency >= 0.6 and batch_sizes[-1] > 1:
        priorities.append(
            "Batch throughput scales reasonably: focus on continuous batching, quantization, and "
            "parallel execution before adding speculative decoding complexity."
        )
    if dominant:
        priorities.append(f"Largest measured decode CUDA component: {dominant}.")
    return {
        "ttft_long_vs_short_input_ratio_b1": ttft_growth,
        "tpot_long_vs_short_context_ratio_b1": tpot_growth,
        "decode_throughput_scale_bmax_vs_b1": throughput_scale,
        "batching_scaling_efficiency": batching_efficiency,
        "small_batch_estimated_weight_traffic_share": weight_share,
        "dominant_profile_component": dominant,
        "priorities": priorities,
    }


def make_plots(payload: dict[str, Any], output_dir: Path) -> list[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    results = payload["results"]
    profiles = payload["decode_profiles"]
    batches = sorted({row["batch_size"] for row in results})
    inputs = sorted({row["input_tokens"] for row in results})
    outputs = sorted({row["output_tokens"] for row in results})
    colors = ["#0B6E69", "#D97706", "#B42318", "#3563A8", "#6B4C9A"]
    plt.rcParams.update({"font.size": 10, "axes.titleweight": "bold"})

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for idx, batch in enumerate(batches):
        rows = sorted(
            (r for r in results if r["batch_size"] == batch and r["output_tokens"] == outputs[0]),
            key=lambda r: r["input_tokens"],
        )
        axes[0, 0].plot(
            [r["input_tokens"] for r in rows],
            [r["ttft_wall_ms"]["median"] for r in rows],
            marker="o", color=colors[idx % len(colors)], label=f"B={batch}",
        )
    axes[0, 0].set(title="TTFT vs input length", xlabel="Input tokens", ylabel="Median TTFT wall (ms)")
    axes[0, 0].legend()

    for idx, batch in enumerate(batches):
        rows = sorted(
            (r for r in results if r["batch_size"] == batch and r["input_tokens"] == inputs[-1]),
            key=lambda r: r["output_tokens"],
        )
        axes[0, 1].plot(
            [r["output_tokens"] for r in rows],
            [r["request_tpot_wall_ms"]["median"] for r in rows],
            marker="o", color=colors[idx % len(colors)], label=f"B={batch}",
        )
    axes[0, 1].set(title=f"TPOT at input={inputs[-1]}", xlabel="Output tokens", ylabel="Median request TPOT wall (ms)")
    axes[0, 1].legend()

    representative = [
        next(r for r in results if r["batch_size"] == b and r["input_tokens"] == inputs[-1] and r["output_tokens"] == outputs[-1])
        for b in batches
    ]
    axes[1, 0].bar(
        [str(r["batch_size"]) for r in representative],
        [r["decode_tokens_per_sec"]["median"] for r in representative],
        color=colors[:len(representative)],
    )
    axes[1, 0].set(title="Aggregate decode throughput", xlabel="Batch size", ylabel="tokens/s")

    x = range(len(representative))
    prefill = [r["ttft_wall_ms"]["median"] for r in representative]
    decode = [r["decode_wall_ms"]["median"] for r in representative]
    axes[1, 1].bar(x, prefill, label="Prefill + first token", color="#0B6E69")
    axes[1, 1].bar(x, decode, bottom=prefill, label="Remaining decode", color="#E9A23B")
    axes[1, 1].set_xticks(list(x), [f"B={r['batch_size']}" for r in representative])
    axes[1, 1].set(title="End-to-end latency split", ylabel="Wall time (ms)")
    axes[1, 1].legend()
    for axis in axes.flat:
        axis.grid(alpha=0.2)
    performance_path = output_dir / "latency_throughput.png"
    fig.savefig(performance_path, dpi=180)
    plt.close(fig)

    written = [str(performance_path)]
    if profiles:
        categories = ["attention_ms", "mlp_ms", "lm_head_ms", "sampling_ms", "model_other_ms"]
        labels = ["Attention", "MLP", "LM head", "Sampling", "Model other"]
        fig, ax = plt.subplots(figsize=(11, 5.5), constrained_layout=True)
        bottoms = [0.0] * len(profiles)
        for category, label, color in zip(categories, labels, colors):
            values = [profile["component_cuda_ms"][category] for profile in profiles]
            ax.bar(
                [f"B={profile['batch_size']}" for profile in profiles],
                values, bottom=bottoms, label=label, color=color,
            )
            bottoms = [bottom + value for bottom, value in zip(bottoms, values)]
        ax.set(title="Decode CUDA time breakdown", ylabel="CUDA timeline (ms / batch step)")
        ax.grid(axis="y", alpha=0.2)
        ax.legend(ncol=5, loc="upper center")
        breakdown_path = output_dir / "decode_cuda_breakdown.png"
        fig.savefig(breakdown_path, dpi=180)
        plt.close(fig)
        written.append(str(breakdown_path))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    active = [r["kv_cache"]["active_allocated_peak_gib"] for r in representative]
    reserved = [r["kv_cache"]["reserved_pool_gib"] for r in representative]
    positions = list(range(len(representative)))
    axes[0].bar([x - 0.18 for x in positions], active, width=0.36, label="Active peak", color="#0B6E69")
    axes[0].bar([x + 0.18 for x in positions], reserved, width=0.36, label="Reserved pool", color="#A7C7C5")
    axes[0].set_xticks(positions, [f"B={r['batch_size']}" for r in representative])
    axes[0].set(title="KV-cache memory", ylabel="GiB")
    axes[0].legend()
    bandwidth = [r["hbm_estimate"]["estimated_effective_hbm_gbps"] for r in representative]
    axes[1].bar([str(r["batch_size"]) for r in representative], bandwidth, color=colors[:len(representative)])
    peak = payload["environment"]["peak_hbm_gbps"]
    if peak:
        axes[1].axhline(peak, linestyle="--", color="#333333", label=f"Peak {peak:.0f} GB/s")
        axes[1].legend()
    axes[1].set(title="Estimated minimum effective HBM bandwidth", xlabel="Batch size", ylabel="GB/s")
    for axis in axes:
        axis.grid(axis="y", alpha=0.2)
    memory_path = output_dir / "kv_cache_hbm.png"
    fig.savefig(memory_path, dpi=180)
    plt.close(fig)
    written.append(str(memory_path))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    labels = [f"B={row['batch_size']}" for row in representative]
    positions = list(range(len(representative)))
    width = 0.36
    axes[0, 0].bar(
        [x - width / 2 for x in positions],
        [row["cold_start_run"]["ttft_wall_ms"] for row in representative],
        width=width, label="Cold", color="#B42318",
    )
    axes[0, 0].bar(
        [x + width / 2 for x in positions],
        [row["ttft_wall_ms"]["median"] for row in representative],
        width=width, label="Steady median", color="#0B6E69",
    )
    axes[0, 0].set_xticks(positions, labels)
    axes[0, 0].set(title="Cold vs steady TTFT", ylabel="ms")
    axes[0, 0].legend()

    axes[0, 1].bar(
        [x - width / 2 for x in positions],
        [row["cold_start_run"]["request_tpot_wall_ms"] for row in representative],
        width=width, label="Cold", color="#B42318",
    )
    axes[0, 1].bar(
        [x + width / 2 for x in positions],
        [row["request_tpot_wall_ms"]["median"] for row in representative],
        width=width, label="Steady median", color="#D97706",
    )
    axes[0, 1].set_xticks(positions, labels)
    axes[0, 1].set(title="Cold vs steady TPOT", ylabel="ms")
    axes[0, 1].legend()

    clock_medians = [row["gpu_telemetry"]["summary"]["sm_clock_mhz"]["median"] for row in representative]
    clock_low = [
        median - row["gpu_telemetry"]["summary"]["sm_clock_mhz"]["p10"]
        for median, row in zip(clock_medians, representative)
    ]
    clock_high = [
        row["gpu_telemetry"]["summary"]["sm_clock_mhz"]["p90"] - median
        for median, row in zip(clock_medians, representative)
    ]
    axes[1, 0].errorbar(
        labels, clock_medians, yerr=[clock_low, clock_high],
        fmt="o", capsize=5, color="#3563A8",
    )
    axes[1, 0].set(title="SM clock stability (P10/P50/P90)", ylabel="MHz")

    gpu_util = [row["gpu_telemetry"]["summary"]["gpu_utilization_pct"]["median"] for row in representative]
    mem_util = [
        row["gpu_telemetry"]["summary"]["memory_controller_utilization_pct"]["median"]
        for row in representative
    ]
    axes[1, 1].bar(
        [x - width / 2 for x in positions], gpu_util,
        width=width, label="GPU utilization", color="#6B4C9A",
    )
    axes[1, 1].bar(
        [x + width / 2 for x in positions], mem_util,
        width=width, label="Memory-controller utilization", color="#A7C7C5",
    )
    axes[1, 1].set_xticks(positions, labels)
    axes[1, 1].set(title="Coarse nvidia-smi utilization", ylabel="percent", ylim=(0, 105))
    axes[1, 1].legend()
    for axis in axes.flat:
        axis.grid(axis="y", alpha=0.2)
    quality_path = output_dir / "cold_steady_gpu_telemetry.png"
    fig.savefig(quality_path, dpi=180)
    plt.close(fig)
    written.append(str(quality_path))
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16"))
    parser.add_argument("--batch_sizes", default="1,4,8")
    parser.add_argument("--input_lengths", default="560,768,1024")
    parser.add_argument("--output_lengths", default="16,64,256")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--kvcache_block_size", type=int, default=16)
    parser.add_argument("--num_kvcache_blocks", type=int, default=0, help="0 selects the minimum safe pool")
    parser.add_argument("--profile_batch_sizes", default="1,8")
    parser.add_argument("--profile_steps", type=int, default=3)
    parser.add_argument("--telemetry_interval_ms", type=int, default=200)
    parser.add_argument("--require_idle_gpu", action="store_true")
    parser.add_argument("--idle_utilization_threshold", type=float, default=5.0)
    parser.add_argument("--idle_memory_threshold_mb", type=float, default=1024.0)
    parser.add_argument("--hbm_peak_gbps", type=float, default=0.0)
    parser.add_argument("--output_dir", default="results/pointllm_diagnostics")
    args = parser.parse_args()

    if (
        args.warmup < 0
        or args.runs <= 0
        or args.profile_steps <= 0
        or args.telemetry_interval_ms <= 0
    ):
        parser.error(
            "warmup must be >=0; runs, profile_steps, and telemetry_interval_ms must be >0"
        )
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        parser.error("this diagnostic benchmark requires CUDA")
    dtype = getattr(torch, args.dtype)
    batches = parse_ints(args.batch_sizes)
    requested_inputs = parse_ints(args.input_lengths)
    outputs = parse_ints(args.output_lengths)
    profile_batches = parse_ints(args.profile_batch_sizes)
    if any(batch not in batches for batch in profile_batches):
        parser.error("profile_batch_sizes must be a subset of batch_sizes")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["NANOPOINTLLM_ENABLE_CUDA_GRAPH"] = "0"
    os.environ["NANOPOINTLLM_PROFILE_LAYERS"] = "0"

    properties = torch.cuda.get_device_properties(device)
    gpu_uuid = f"GPU-{properties.uuid}"
    target_gpu_before = query_gpu_state(gpu_uuid)
    target_gpu_idle_at_start = (
        target_gpu_before["gpu_utilization_pct"] <= args.idle_utilization_threshold
        and target_gpu_before["memory_used_mb"] <= args.idle_memory_threshold_mb
    )
    if args.require_idle_gpu and not target_gpu_idle_at_start:
        parser.error(
            "target GPU is not idle: "
            f"util={target_gpu_before['gpu_utilization_pct']:.0f}%, "
            f"memory={target_gpu_before['memory_used_mb']:.0f}MiB"
        )
    before_gpu = gpu_snapshot()
    print(f"Loading {args.model_path} on {device} ({args.dtype})...")
    torch.cuda.synchronize(device)
    model_load_start = time.perf_counter()
    model, tokenizer = load_pointllm_model(args.model_path, device=device, dtype=dtype)
    torch.cuda.synchronize(device)
    model_load_wall_ms = (time.perf_counter() - model_load_start) * 1000.0
    base_prompt = build_prompt_token_ids(tokenizer, model)
    effective_inputs = sorted(set(max(length, len(base_prompt)) for length in requested_inputs))
    prompts = {length: resize_point_prompt(base_prompt, length) for length in effective_inputs}
    max_context = max(effective_inputs) + max(outputs)
    model_limit = int(getattr(model.config, "max_position_embeddings", max_context))
    if max_context > model_limit:
        parser.error(f"input + output ({max_context}) exceeds model context limit ({model_limit})")

    block_size = args.kvcache_block_size
    required_blocks = max(batches) * math.ceil(max_context / block_size)
    num_blocks = args.num_kvcache_blocks or (required_blocks + max(batches))
    if num_blocks < required_blocks:
        parser.error(f"num_kvcache_blocks={num_blocks} is below required minimum {required_blocks}")
    max_batched_tokens = max(batches) * max(effective_inputs)
    torch.cuda.synchronize(device)
    engine_init_start = time.perf_counter()
    engine = PointLLMLLMEngine(
        model,
        eos_token_id=tokenizer.eos_token_id or 2,
        max_num_seqs=max(batches),
        max_num_batched_tokens=max_batched_tokens,
        num_kvcache_blocks=num_blocks,
        kvcache_block_size=block_size,
    )
    torch.cuda.synchronize(device)
    engine_init_wall_ms = (time.perf_counter() - engine_init_start) * 1000.0
    set_decode_profiling(engine.runner, False)

    cfg = model.config
    dtype_bytes = torch.empty((), dtype=dtype).element_size()
    weight_bytes = decoder_weight_bytes(model)
    device_name = torch.cuda.get_device_name(device)
    inferred_peak, peak_source = identify_peak_hbm_gbps(device_name)
    peak_hbm = args.hbm_peak_gbps or inferred_peak
    if args.hbm_peak_gbps:
        peak_source = "command line"

    results = []
    total_configs = len(batches) * len(effective_inputs) * len(outputs)
    config_index = 0
    with torch.inference_mode():
        for input_tokens in effective_inputs:
            for output_tokens in outputs:
                for batch_size in batches:
                    config_index += 1
                    print(
                        f"[{config_index}/{total_configs}] B={batch_size} "
                        f"input={input_tokens} output={output_tokens}",
                        flush=True,
                    )
                    cold_start_run = run_once(
                        engine,
                        token_ids=prompts[input_tokens],
                        batch_size=batch_size,
                        output_tokens=output_tokens,
                        device=device,
                        dtype=dtype,
                    )
                    for _ in range(args.warmup):
                        run_once(
                            engine,
                            token_ids=prompts[input_tokens],
                            batch_size=batch_size,
                            output_tokens=output_tokens,
                            device=device,
                            dtype=dtype,
                        )
                    telemetry_monitor = GpuTelemetryMonitor(
                        gpu_uuid,
                        interval_ms=args.telemetry_interval_ms,
                    )
                    telemetry_monitor.start()
                    try:
                        raw = [
                            run_once(
                                engine,
                                token_ids=prompts[input_tokens],
                                batch_size=batch_size,
                                output_tokens=output_tokens,
                                device=device,
                                dtype=dtype,
                            )
                            for _ in range(args.runs)
                        ]
                    finally:
                        gpu_telemetry = telemetry_monitor.stop()
                    aggregated = aggregate_runs(
                        raw,
                        batch_size=batch_size,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        cold_start_run=cold_start_run,
                        gpu_telemetry=gpu_telemetry,
                    )
                    attach_memory_and_bandwidth(
                        aggregated,
                        cfg=cfg,
                        dtype_bytes=dtype_bytes,
                        num_blocks=num_blocks,
                        block_size=block_size,
                        weight_bytes=weight_bytes,
                        peak_hbm_gbps=peak_hbm,
                    )
                    results.append(aggregated)
                    print(
                        f"  steady median TTFT={aggregated['ttft_wall_ms']['median']:.2f} ms, "
                        f"TPOT={aggregated['request_tpot_wall_ms']['median']:.2f} ms, "
                        f"decode={aggregated['decode_tokens_per_sec']['median']:.1f} tok/s",
                        flush=True,
                    )

        profiles = []
        for batch_size in profile_batches:
            print(f"Profiling decode CUDA components at B={batch_size}...", flush=True)
            profiles.append(profile_decode(
                engine,
                token_ids=prompts[max(effective_inputs)],
                batch_size=batch_size,
                profile_steps=args.profile_steps,
                device=device,
                dtype=dtype,
            ))

    environment = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": str(device),
        "device_name": device_name,
        "gpu_uuid": gpu_uuid,
        "device_total_memory_gib": properties.total_memory / (1024 ** 3),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "peak_hbm_gbps": peak_hbm,
        "peak_hbm_source": peak_source,
        "gpu_snapshot_before": before_gpu,
        "gpu_snapshot_after": gpu_snapshot(),
        "target_gpu_before_model_load": target_gpu_before,
        "target_gpu_after_benchmark": query_gpu_state(gpu_uuid),
        "target_gpu_idle_at_start": target_gpu_idle_at_start,
        "model_load_wall_ms": model_load_wall_ms,
        "engine_init_wall_ms": engine_init_wall_ms,
    }
    payload = {
        "schema_version": 1,
        "metric_definitions": {
            "ttft_wall_ms": "submission to first generated token; includes point encoder, prefill, LM head, and sampling",
            "request_tpot_wall_ms": "decode wall time divided by output_tokens-1; not divided by batch size",
            "decode_tokens_per_sec": "batch_size * decode_steps / decode wall time",
            "prefill": "TTFT phase, including first-token sampling",
            "decode": "all output tokens after the first token",
            "hbm_estimate": "analytical lower bound from one decoder-weight read plus KV reads/writes; not a hardware counter",
            "cold_start_run": "first request for each exact input/output/batch shape before warmup",
            "steady_state": "distribution after shape cold run and configured warmup iterations",
            "gpu_telemetry": (
                f"{args.telemetry_interval_ms}ms nvidia-smi samples; memory-controller "
                "utilization is not DRAM bandwidth"
            ),
        },
        "config": {
            "model_path": args.model_path,
            "dtype": args.dtype,
            "base_point_prompt_tokens": len(base_prompt),
            "requested_input_lengths": requested_inputs,
            "effective_input_lengths": effective_inputs,
            "output_lengths": outputs,
            "batch_sizes": batches,
            "warmup": args.warmup,
            "runs": args.runs,
            "require_idle_gpu": args.require_idle_gpu,
            "telemetry_interval_ms": args.telemetry_interval_ms,
            "num_kvcache_blocks": num_blocks,
            "kvcache_block_size": block_size,
            "decoder_weight_gib": weight_bytes / (1024 ** 3),
        },
        "environment": environment,
        "results": results,
        "decode_profiles": profiles,
    }
    methodology_warnings = []
    if args.warmup < 5:
        methodology_warnings.append(
            f"warmup={args.warmup} is below the recommended minimum of 5"
        )
    if args.runs < 20:
        methodology_warnings.append(
            f"runs={args.runs} is below the recommended minimum of 20"
        )
    if not target_gpu_idle_at_start:
        methodology_warnings.append(
            "target GPU was not idle before model load; absolute latency is not publishable"
        )
    if any(
        not result["gpu_telemetry"]["summary"]["sensor_status"]["power_w_available"]
        for result in results
    ):
        methodology_warnings.append(
            "power telemetry was unavailable for at least one configuration"
        )
    payload["methodology_warnings"] = methodology_warnings
    payload["analysis"] = analyze(results, profiles)
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    plots = make_plots(payload, output_dir)
    summary = {
        "metrics_json": str(metrics_path),
        "plots": plots,
        "analysis": payload["analysis"],
        "methodology_warnings": methodology_warnings,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
