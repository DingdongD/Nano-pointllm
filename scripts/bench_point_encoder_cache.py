#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time

import torch

from nanopointllm.engine.llm_engine import PointLLMLLMEngine
from nanopointllm.parity.engine_test_utils import (
    build_prompt_token_ids,
    load_pointllm_model,
    make_fake_point_cloud,
)
from nanopointllm.sampling_params import SamplingParams


def _make_requests(token_ids, pcs, decode_steps):
    sp = SamplingParams(max_tokens=decode_steps, ignore_eos=True)
    return [{"token_ids": token_ids, "point_clouds": pc, "sampling_params": sp} for pc in pcs]


@torch.inference_mode()
def bench_prefill_step(
    model,
    eos_token_id: int,
    requests: list[dict],
    *,
    max_num_seqs: int,
    max_num_batched_tokens: int,
    num_kvcache_blocks: int | None,
    kvcache_block_size: int,
    warmup: int,
    runs: int,
    reuse_engine: bool,
) -> dict:
    def _new_engine():
        return PointLLMLLMEngine(
            model,
            eos_token_id=eos_token_id,
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
            num_kvcache_blocks=num_kvcache_blocks,
            kvcache_block_size=kvcache_block_size,
        )

    persistent_engine = _new_engine() if reuse_engine else None

    def _run_once() -> float:
        engine = persistent_engine if persistent_engine is not None else _new_engine()
        for req in requests:
            engine.add_request(**req)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        engine.step()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1000

    for _ in range(warmup):
        _run_once()
    vals = [_run_once() for _ in range(runs)]
    mean_ms = sum(vals) / len(vals)
    return {
        "prefill_ms_mean": round(mean_ms, 2),
        "prefill_ms_runs": [round(v, 2) for v in vals],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Bench point-encoder cache reuse across repeated requests")
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--decode_steps", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--paged", action="store_true")
    ap.add_argument("--num_kvcache_blocks", type=int, default=1024)
    ap.add_argument("--kvcache_block_size", type=int, default=16)
    ap.add_argument("--out_json", default="")
    args = ap.parse_args()

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    print(f"Loading model {args.model_path} on {device} ({args.dtype})...")
    model, tok = load_pointllm_model(args.model_path, device=device, dtype=dtype)
    eos = tok.eos_token_id or 2
    token_ids = build_prompt_token_ids(tok, model)
    print(f"Model loaded. prompt_len={len(token_ids)}")

    base_pc = make_fake_point_cloud(device, dtype)
    unique_pcs = [make_fake_point_cloud(device, dtype) for _ in range(args.batch_size)]
    repeated_pcs = [base_pc.clone() for _ in range(args.batch_size)]
    cfg = {
        "max_num_seqs": args.batch_size,
        "max_num_batched_tokens": 4096,
        "num_kvcache_blocks": args.num_kvcache_blocks if args.paged else None,
        "kvcache_block_size": args.kvcache_block_size,
        "warmup": args.warmup,
        "runs": args.runs,
    }

    print("\nRunning unique point-cloud prefill...")
    unique = bench_prefill_step(
        model,
        eos,
        _make_requests(token_ids, unique_pcs, args.decode_steps),
        reuse_engine=False,
        **cfg,
    )
    print("Running repeated point-cloud prefill with fresh engine...")
    repeated_fresh = bench_prefill_step(
        model,
        eos,
        _make_requests(token_ids, repeated_pcs, args.decode_steps),
        reuse_engine=False,
        **cfg,
    )
    print("Running repeated point-cloud prefill with persistent engine cache...")
    repeated_reuse = bench_prefill_step(
        model,
        eos,
        _make_requests(token_ids, repeated_pcs, args.decode_steps),
        reuse_engine=True,
        **cfg,
    )

    payload = {
        "batch_size": args.batch_size,
        "paged": args.paged,
        "unique": unique,
        "repeated_fresh_engine": repeated_fresh,
        "repeated_reuse_engine": repeated_reuse,
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
