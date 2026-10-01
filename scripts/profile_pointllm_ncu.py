#!/usr/bin/env python3
"""Run one warmed PointLLM decode region for external Nsight Compute capture."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import torch

from nanopointllm.benchmark_metrics import resize_point_prompt
from nanopointllm.engine.llm_engine import PointLLMLLMEngine
from nanopointllm.gpu_telemetry import query_gpu_state
from nanopointllm.profiling import set_nvtx_stages
from nanopointllm.parity.engine_test_utils import (
    build_prompt_token_ids,
    load_pointllm_model,
    make_fake_point_cloud,
)
from nanopointllm.sampling_params import SamplingParams


def _tensor_bytes(tensor: torch.Tensor | None) -> int:
    return 0 if tensor is None else tensor.numel() * tensor.element_size()


def _linear_cost(weights: list[torch.Tensor], batch_size: int, steps: int) -> dict:
    weight_bytes_per_step = sum(_tensor_bytes(weight) for weight in weights)
    weight_elements = sum(weight.numel() for weight in weights)
    activation_elements = sum(batch_size * (weight.shape[0] + weight.shape[1]) for weight in weights)
    activation_bytes_per_step = activation_elements * weights[0].element_size()
    compulsory_bytes = (weight_bytes_per_step + activation_bytes_per_step) * steps
    return {
        "weight_bytes_per_step": weight_bytes_per_step,
        "weight_bytes": weight_bytes_per_step * steps,
        "activation_bytes": activation_bytes_per_step * steps,
        "compulsory_bytes": compulsory_bytes,
        "estimated_flops": 2 * batch_size * weight_elements * steps,
        "weight_fraction_of_compulsory_bytes": (
            weight_bytes_per_step / (weight_bytes_per_step + activation_bytes_per_step)
        ),
    }


def build_stage_manifest(
    engine: PointLLMLLMEngine,
    *,
    batch_size: int,
    context_lengths: list[list[int]],
) -> dict:
    """Build a per-stage compulsory-traffic/FLOP model for the profiled decode steps."""
    runner = engine.runner.lightweight_runner
    if runner is None:
        raise RuntimeError("NCU stage profiling requires LightweightLlamaRunner")
    steps = len(context_lengths)
    qkv_weights: list[torch.Tensor] = []
    o_weights: list[torch.Tensor] = []
    mlp_weights: list[torch.Tensor] = []
    for layer in runner.layers:
        attn = layer.self_attn
        packed_qkv = getattr(attn, "packed_qkv_weight", None)
        if packed_qkv is not None:
            qkv_weights.append(packed_qkv)
        else:
            qkv_weights.extend((attn.q_proj.weight, attn.k_proj.weight, attn.v_proj.weight))
        o_weights.append(attn.o_proj.weight)
        packed_gateup = getattr(layer.mlp, "packed_gateup_weight", None)
        if packed_gateup is not None:
            mlp_weights.append(packed_gateup)
        else:
            mlp_weights.extend((layer.mlp.gate_proj.weight, layer.mlp.up_proj.weight))
        mlp_weights.append(layer.mlp.down_proj.weight)

    first_attn = runner.layers[0].self_attn
    element_size = first_attn.k_cache.element_size()
    kv_elements_per_token = 2 * first_attn.num_kv_heads * first_attn.head_dim
    kv_read_bytes = sum(
        sum(lengths) * kv_elements_per_token * element_size
        for lengths in context_lengths
    ) * len(runner.layers)
    kv_write_bytes = (
        steps * batch_size * kv_elements_per_token * element_size * len(runner.layers)
    )
    attention_flops = sum(
        4 * sum(lengths) * first_attn.num_heads * first_attn.head_dim
        for lengths in context_lengths
    ) * len(runner.layers)
    attention_compulsory = kv_read_bytes + kv_write_bytes
    stages = {
        "qkv": _linear_cost(qkv_weights, batch_size, steps),
        "attention": {
            "weight_bytes_per_step": 0,
            "weight_bytes": 0,
            "activation_bytes": kv_write_bytes,
            "kv_read_bytes": kv_read_bytes,
            "kv_write_bytes": kv_write_bytes,
            "compulsory_bytes": attention_compulsory,
            "estimated_flops": attention_flops,
            "weight_fraction_of_compulsory_bytes": 0.0,
        },
        "o_proj": _linear_cost(o_weights, batch_size, steps),
        "mlp": _linear_cost(mlp_weights, batch_size, steps),
        "lm_head": _linear_cost([runner.lm_head.weight], batch_size, steps),
    }
    for stage in stages.values():
        stage["arithmetic_intensity_flops_per_byte"] = (
            stage["estimated_flops"] / stage["compulsory_bytes"]
            if stage["compulsory_bytes"] else None
        )
    return {
        "runtime": "LightweightLlamaRunner + paged KV (HF model.forward bypassed for decode)",
        "batch_size": batch_size,
        "profile_steps": steps,
        "context_lengths_by_step": context_lengths,
        "num_layers": len(runner.layers),
        "dtype": str(runner.lm_head.weight.dtype),
        "stage_ranges": {
            "qkv": "pointllm_qkv",
            "attention": "pointllm_attention",
            "o_proj": "pointllm_o_proj",
            "mlp": "pointllm_mlp",
            "lm_head": "pointllm_lm_head",
            "sampling": "pointllm_sampling",
        },
        "stages": stages,
        "scope_note": (
            "Linear-stage compulsory bytes include the weights actually read by the optimized "
            "runner plus input/output activation traffic. Attention bytes model paged K/V reads "
            "and current-token K/V writes. Kernel temporaries and cache effects are measured by NCU."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16"))
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--input_length", type=int, default=768)
    parser.add_argument("--warmup_decode_steps", type=int, default=5)
    parser.add_argument("--profile_steps", type=int, default=1)
    parser.add_argument("--kvcache_block_size", type=int, default=16)
    parser.add_argument("--num_kvcache_blocks", type=int, default=0)
    parser.add_argument("--manifest_output", default="")
    parser.add_argument("--require_idle_gpu", action="store_true")
    parser.add_argument("--idle_utilization_threshold", type=float, default=5.0)
    parser.add_argument("--idle_memory_threshold_mb", type=float, default=1024.0)
    args = parser.parse_args()

    if args.batch_size <= 0 or args.warmup_decode_steps < 0 or args.profile_steps <= 0:
        parser.error("batch_size/profile_steps must be positive and warmup_decode_steps non-negative")
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    os.environ["NANOPOINTLLM_ENABLE_CUDA_GRAPH"] = "0"
    os.environ["NANOPOINTLLM_PROFILE_LAYERS"] = "0"
    os.environ["NANOPOINTLLM_LIGHTWEIGHT_DECODE"] = "1"
    os.environ["NANOPOINTLLM_RMSNORM_STAGED_PACKED_QKV"] = "1"
    os.environ["NANOPOINTLLM_PACKED_GATEUP_DECODE"] = "1"
    os.environ["NANOPOINTLLM_NVTX_STAGES"] = "0"

    properties = torch.cuda.get_device_properties(device)
    gpu_uuid = f"GPU-{properties.uuid}"
    gpu_before = query_gpu_state(gpu_uuid)
    idle_at_start = (
        gpu_before["gpu_utilization_pct"] <= args.idle_utilization_threshold
        and gpu_before["memory_used_mb"] <= args.idle_memory_threshold_mb
    )
    if args.require_idle_gpu and not idle_at_start:
        parser.error(
            "target GPU is not idle: "
            f"util={gpu_before['gpu_utilization_pct']:.0f}%, "
            f"memory={gpu_before['memory_used_mb']:.0f}MiB"
        )

    model, tokenizer = load_pointllm_model(args.model_path, device=device, dtype=dtype)
    base_prompt = build_prompt_token_ids(tokenizer, model)
    token_ids = resize_point_prompt(base_prompt, max(args.input_length, len(base_prompt)))
    output_tokens = 1 + args.warmup_decode_steps + args.profile_steps
    required_blocks = args.batch_size * math.ceil(
        (len(token_ids) + output_tokens) / args.kvcache_block_size
    )
    num_blocks = args.num_kvcache_blocks or (required_blocks + args.batch_size)
    engine = PointLLMLLMEngine(
        model,
        eos_token_id=tokenizer.eos_token_id or 2,
        max_num_seqs=args.batch_size,
        max_num_batched_tokens=args.batch_size * len(token_ids),
        num_kvcache_blocks=num_blocks,
        kvcache_block_size=args.kvcache_block_size,
    )
    params = SamplingParams(max_tokens=output_tokens, ignore_eos=True)
    seqs = [
        engine.add_request(
            token_ids=token_ids,
            point_clouds=make_fake_point_cloud(device, dtype),
            sampling_params=params,
        )
        for _ in range(args.batch_size)
    ]

    with torch.inference_mode():
        engine.step()  # point encoder + prefill + first token; excluded from NCU range
        for _ in range(args.warmup_decode_steps):
            engine.step()
        torch.cuda.synchronize(device)
        contexts = [
            [seq.num_tokens + step for seq in seqs]
            for step in range(args.profile_steps)
        ]
        manifest = build_stage_manifest(
            engine,
            batch_size=args.batch_size,
            context_lengths=contexts,
        )
        manifest["gpu_before"] = gpu_before
        manifest["idle_at_start"] = idle_at_start
        if args.manifest_output:
            output = Path(args.manifest_output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        set_nvtx_stages(True)
        torch.cuda.nvtx.range_push("pointllm_decode_profile")
        for _ in range(args.profile_steps):
            engine.step()
        torch.cuda.nvtx.range_pop()
        set_nvtx_stages(False)
        torch.cuda.synchronize(device)

    print(json.dumps({
        "nvtx_range": "pointllm_decode_profile",
        "batch_size": args.batch_size,
        "input_tokens": len(token_ids),
        "warmup_decode_steps": args.warmup_decode_steps,
        "profile_steps": args.profile_steps,
        "context_tokens_at_end": [seq.num_tokens for seq in seqs],
        "manifest_output": args.manifest_output or None,
        "idle_at_start": idle_at_start,
        "gpu_before": gpu_before,
    }, indent=2))


if __name__ == "__main__":
    main()
