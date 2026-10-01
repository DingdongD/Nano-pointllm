#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time

import torch

from nanopointllm.engine.model_runner import PointLLMModelRunner
from nanopointllm.engine.sequence import PointLLMSequence
from nanopointllm.parity.engine_test_utils import (
    build_prompt_token_ids,
    load_pointllm_model,
    make_fake_point_cloud,
)
from nanopointllm.sampling_params import SamplingParams


def cuda_ms(fn, warmup: int, runs: int) -> tuple[float, list[float]]:
    for _ in range(warmup):
        fn()
    vals = []
    for _ in range(runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        vals.append((time.perf_counter() - t0) * 1000)
    return sum(vals) / len(vals), vals


def make_seq(token_ids: list[int], pc: torch.Tensor) -> PointLLMSequence:
    return PointLLMSequence(
        token_ids=token_ids,
        point_clouds=pc,
        sampling_params=SamplingParams(max_tokens=1, ignore_eos=True),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Profile point encoder / projector / splice / cache reuse")
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--out_json", default="")
    args = ap.parse_args()

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    print(f"Loading model {args.model_path} on {device} ({args.dtype})...")
    model, tok = load_pointllm_model(args.model_path, device=device, dtype=dtype)
    token_ids = build_prompt_token_ids(tok, model)
    runner = PointLLMModelRunner(model)
    wrapper = runner.wrapper
    print(f"Model loaded. prompt_len={len(token_ids)}")

    pc = make_fake_point_cloud(device, dtype)
    pc_batch = torch.stack([pc.clone() for _ in range(4)])
    raw_backbone = wrapper.inner.point_backbone(pc.unsqueeze(0))

    encode_mean, encode_runs = cuda_ms(lambda: wrapper.encode_point_clouds(pc.unsqueeze(0)), args.warmup, args.runs)
    batch_encode_mean, batch_encode_runs = cuda_ms(lambda: wrapper.encode_point_clouds(pc_batch), args.warmup, args.runs)
    proj_mean, proj_runs = cuda_ms(lambda: wrapper.inner.point_proj(raw_backbone[0]), args.warmup, args.runs)

    layout = wrapper.analyze_input_layout(torch.tensor(token_ids, device=device))
    text_embeds = wrapper.get_input_embeddings()(torch.tensor([token_ids], device=device))[0]
    point_features = wrapper.encode_point_clouds(pc.unsqueeze(0))[0]
    splice_mean, splice_runs = cuda_ms(
        lambda: wrapper.splice_point_features(text_embeds, point_features, layout),
        args.warmup,
        args.runs,
    )

    repeated_pc = pc.clone()
    cache_runner = PointLLMModelRunner(model)

    def _fresh_prefill_encode():
        seq = make_seq(token_ids, repeated_pc.clone())
        cache_runner._encode_point_clouds_batch([seq])

    first_mean, first_runs = cuda_ms(_fresh_prefill_encode, args.warmup, args.runs)

    persistent_seq = make_seq(token_ids, repeated_pc.clone())
    cache_runner._encode_point_clouds_batch([persistent_seq])

    def _reuse_prefill_encode():
        seq = make_seq(token_ids, repeated_pc.clone())
        cache_runner._encode_point_clouds_batch([seq])

    reuse_mean, reuse_runs = cuda_ms(_reuse_prefill_encode, args.warmup, args.runs)

    payload = {
        "encode_single_ms_mean": round(encode_mean, 2),
        "encode_single_ms_runs": [round(v, 2) for v in encode_runs],
        "encode_batch4_ms_mean": round(batch_encode_mean, 2),
        "encode_batch4_ms_runs": [round(v, 2) for v in batch_encode_runs],
        "point_proj_ms_mean": round(proj_mean, 2),
        "point_proj_ms_runs": [round(v, 2) for v in proj_runs],
        "splice_ms_mean": round(splice_mean, 2),
        "splice_ms_runs": [round(v, 2) for v in splice_runs],
        "runner_encode_first_ms_mean": round(first_mean, 2),
        "runner_encode_first_ms_runs": [round(v, 2) for v in first_runs],
        "runner_encode_reuse_ms_mean": round(reuse_mean, 2),
        "runner_encode_reuse_ms_runs": [round(v, 2) for v in reuse_runs],
        "point_feature_cache_stats": {
            "hits": cache_runner.point_feature_cache.stats.hits,
            "misses": cache_runner.point_feature_cache.stats.misses,
            "inserts": cache_runner.point_feature_cache.stats.inserts,
            "evictions": cache_runner.point_feature_cache.stats.evictions,
            "size": len(cache_runner.point_feature_cache),
        },
    }
    print(json.dumps(payload, indent=2))
    if args.out_json:
        out_dir = os.path.dirname(os.path.abspath(args.out_json))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)


if __name__ == "__main__":
    main()
