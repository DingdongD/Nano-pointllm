#!/usr/bin/env python3
"""
P5 bench: paged PointLLMLLMEngine vs padded PointLLMLLMEngine decode throughput.

Usage:
  cd /home/nano-pointllm
  PYTHONPATH=/home/PointLLM:. python scripts/bench_paged_pointllm.py \
    --model_path /mnt/llm_data/pointllm_ckpt/PointLLM_7B_v1.2 \
    --device cuda:1 --batch_sizes 1,2,4,8 --decode_steps 32 \
    --warmup 2 --runs 3 --out_json results/p5_bench.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

from nanopointllm.engine.llm_engine import PointLLMLLMEngine
from nanopointllm.parity.engine_test_utils import (
    build_prompt_token_ids,
    load_pointllm_model,
    make_fake_point_cloud,
)
from nanopointllm.sampling_params import SamplingParams


def bench_batch(
    model,
    eos_token_id: int,
    token_ids: list[int],
    device: torch.device,
    dtype: torch.dtype,
    B: int,
    decode_steps: int,
    warmup: int,
    runs: int,
    num_kvcache_blocks: int,   # 0 → padded (no paged KV)
    kvcache_block_size: int,
) -> dict:
    """
    Benchmark one (B, mode) configuration.

    IMPORTANT: inject_paged_attention permanently replaces LlamaAttention layers.
    This function must NOT be called with num_kvcache_blocks=0 after a prior call
    with num_kvcache_blocks>0 — the padded runner would invoke PagedLlamaAttention
    without a ForwardContext and crash.  Callers must run ALL padded configs first,
    then all paged configs.
    """
    use_paged = num_kvcache_blocks > 0
    sp  = SamplingParams(max_tokens=decode_steps, ignore_eos=True)
    pcs = [make_fake_point_cloud(device, dtype) for _ in range(B)]
    reqs = [
        {"token_ids": token_ids, "point_clouds": pc, "sampling_params": sp}
        for pc in pcs
    ]

    def _run() -> tuple[float, float, int]:
        engine = PointLLMLLMEngine(
            model,
            eos_token_id=eos_token_id,
            max_num_seqs=B,
            max_num_batched_tokens=4096,
            num_kvcache_blocks=num_kvcache_blocks if use_paged else None,
            kvcache_block_size=kvcache_block_size,
        )
        for req in reqs:
            engine.add_request(**req)

        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        engine.step()   # prefill
        torch.cuda.synchronize(device)
        t1 = time.perf_counter()

        steps = 0
        while not engine.is_finished():
            engine.step()
            steps += 1
        torch.cuda.synchronize(device)
        t2 = time.perf_counter()
        return (t1 - t0) * 1000, (t2 - t1) * 1000, steps

    with torch.inference_mode():
        for _ in range(warmup):
            _run()
        pf_list, dc_list, s_list = zip(*[_run() for _ in range(runs)])

    pf_mean = sum(pf_list) / runs
    dc_mean = sum(dc_list) / runs
    actual  = s_list[-1]
    dec_per_step = dc_mean / (actual * B) if actual > 0 else 0.0
    tps          = (actual * B) / (dc_mean / 1000) if dc_mean > 0 else 0.0
    return {
        "batch_size": B,
        "mode": "paged" if use_paged else "padded",
        "prefill_ms_mean": round(pf_mean, 2),
        "decode_mean_per_step_ms": round(dec_per_step, 3),
        "tokens_per_sec": round(tps, 1),
        "actual_decode_steps": actual,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="P5 bench: paged vs padded engine")
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--batch_sizes", default="1,2,4,8")
    ap.add_argument("--decode_steps", type=int, default=32)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--num_kvcache_blocks", type=int, default=512)
    ap.add_argument("--kvcache_block_size", type=int, default=16)
    ap.add_argument("--out_json", default="")
    args = ap.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        sys.exit("CUDA not available")

    dtype        = getattr(torch, args.dtype)
    batch_sizes  = [int(x.strip()) for x in args.batch_sizes.split(",") if x.strip()]
    kw = dict(
        decode_steps=args.decode_steps,
        warmup=args.warmup,
        runs=args.runs,
        kvcache_block_size=args.kvcache_block_size,
    )

    print(f"Loading model {args.model_path} on {device} ({args.dtype})...")
    model, tok = load_pointllm_model(args.model_path, device, dtype)
    eos     = tok.eos_token_id
    pt_ids  = build_prompt_token_ids(tok, model)
    print(f"Model loaded. prompt_len={len(pt_ids)}\n")

    fmt = f"{'B':>4}  {'mode':<8}  {'prefill_ms':>10}  {'dec/step/seq_ms':>15}  {'tps':>8}"
    sep = "-" * 60
    print(fmt)
    print(sep)

    padded_results, paged_results = [], []

    # ── Pass 1: padded (MUST run before any paged injection) ─────────────
    for B in batch_sizes:
        try:
            r = bench_batch(model, eos, pt_ids, device, dtype, B,
                            num_kvcache_blocks=0, **kw)
            print(f"  B={B:<2}  padded   prefill={r['prefill_ms_mean']:7.1f}ms  "
                  f"dec/step={r['decode_mean_per_step_ms']:7.3f}ms  "
                  f"tps={r['tokens_per_sec']:7.1f}")
            padded_results.append(r)
        except Exception as e:
            print(f"  B={B:<2}  padded   SKIPPED ({e})")

    print(sep)

    # ── Pass 2: paged (inject_paged_attention runs on first call here) ───
    for B in batch_sizes:
        try:
            r = bench_batch(model, eos, pt_ids, device, dtype, B,
                            num_kvcache_blocks=args.num_kvcache_blocks, **kw)
            pad_r = next((p for p in padded_results if p["batch_size"] == B), None)
            speedup = r["tokens_per_sec"] / pad_r["tokens_per_sec"] if pad_r else 0
            print(f"  B={B:<2}  paged    prefill={r['prefill_ms_mean']:7.1f}ms  "
                  f"dec/step={r['decode_mean_per_step_ms']:7.3f}ms  "
                  f"tps={r['tokens_per_sec']:7.1f}  ({speedup:.2f}x vs padded)")
            paged_results.append(r)
        except Exception as e:
            print(f"  B={B:<2}  paged    SKIPPED ({e})")

    print()

    if args.out_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_json)) or ".", exist_ok=True)
        payload = {"padded_results": padded_results, "paged_results": paged_results}
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"[ok] results written to {args.out_json}")


if __name__ == "__main__":
    main()
