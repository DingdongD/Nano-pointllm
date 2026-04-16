#!/usr/bin/env python3
"""
P5/T6 bench: paged PointLLMLLMEngine vs padded, plus continuous batching.

Usage:
  cd /home/nano-pointllm
  PYTHONPATH=/home/PointLLM:. python scripts/bench_paged_pointllm.py \
    --model_path /mnt/llm_data/pointllm_ckpt/PointLLM_7B_v1.2 \
    --device cuda:1 --batch_sizes 1,2,4,8 --decode_steps 32 \
    --warmup 2 --runs 3 --out_json results/p5_bench.json

Task 6 continuous batching bench:
  Add --continuous_total 8 --continuous_window 4 to also measure throughput
  when more requests than max_num_seqs are queued (mixed prefill+decode steps).
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


def bench_continuous(
    model,
    eos_token_id: int,
    token_ids: list[int],
    device: torch.device,
    dtype: torch.dtype,
    total_requests: int,
    window: int,
    decode_steps: int,
    warmup: int,
    runs: int,
    num_kvcache_blocks: int,
    kvcache_block_size: int,
) -> dict:
    """
    Benchmark continuous batching: submit total_requests with max_num_seqs=window.
    When window < total_requests, some steps will have mixed prefill+decode.

    To guarantee mixed steps, sequences get staggered max_tokens (half at
    decode_steps // 2, half at decode_steps) so the first sub-batch finishes
    while the second sub-batch is still waiting — forcing a mixed step when the
    new prefill is admitted.

    Returns throughput metrics plus the fraction of steps that were truly mixed.
    """
    pcs = [make_fake_point_cloud(device, dtype) for _ in range(total_requests)]
    reqs = []
    for i, pc in enumerate(pcs):
        # Stagger lifetimes: even-indexed seqs run for decode_steps // 2,
        # odd-indexed seqs run for decode_steps. This guarantees that when an
        # even seq finishes mid-flight, a waiting seq gets admitted while
        # odd seqs are still decoding → truly mixed step.
        mt = decode_steps // 2 if i % 2 == 0 else decode_steps
        sp = SamplingParams(max_tokens=mt, ignore_eos=True)
        reqs.append({"token_ids": token_ids, "point_clouds": pc, "sampling_params": sp})

    def _run() -> tuple[float, float, int, float]:
        engine = PointLLMLLMEngine(
            model,
            eos_token_id=eos_token_id,
            max_num_seqs=window,
            max_num_batched_tokens=4096,
            num_kvcache_blocks=num_kvcache_blocks,
            kvcache_block_size=kvcache_block_size,
        )
        step_log: list[tuple[int, int]] = []
        orig = engine.runner.run_mixed

        def tracked(prefill_seqs, decode_seqs):
            step_log.append((len(prefill_seqs), len(decode_seqs)))
            return orig(prefill_seqs, decode_seqs)

        engine.runner.run_mixed = tracked

        for req in reqs:
            engine.add_request(**req)

        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        engine.step()   # first prefill
        torch.cuda.synchronize(device)
        t1 = time.perf_counter()

        while not engine.is_finished():
            engine.step()
        torch.cuda.synchronize(device)
        t2 = time.perf_counter()

        total_decode_steps = sum(1 for p, d in step_log if d > 0)
        mixed_steps = sum(1 for p, d in step_log if p > 0 and d > 0)
        mixed_frac = mixed_steps / max(total_decode_steps, 1)
        # Half of reqs run decode_steps//2, half run decode_steps
        half = total_requests // 2
        total_tokens = half * (decode_steps // 2) + (total_requests - half) * decode_steps
        return (t1 - t0) * 1000, (t2 - t1) * 1000, total_tokens, mixed_frac

    with torch.inference_mode():
        for _ in range(warmup):
            _run()
        results = [_run() for _ in range(runs)]

    pf_ms   = sum(r[0] for r in results) / runs
    tot_ms  = sum(r[1] for r in results) / runs
    tokens  = results[-1][2]
    mf      = results[-1][3]
    tps     = tokens / (tot_ms / 1000) if tot_ms > 0 else 0.0
    return {
        "total_requests": total_requests,
        "window": window,
        "mode": "continuous",
        "prefill_ms_mean": round(pf_ms, 2),
        "total_decode_ms_mean": round(tot_ms, 2),
        "tokens_per_sec": round(tps, 1),
        "mixed_step_fraction": round(mf, 3),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="P5/T6 bench: paged vs padded + continuous batching")
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
    # Task 6: continuous batching bench
    ap.add_argument("--continuous_total", type=int, default=0,
                    help="If >0, also bench continuous batching with this many total requests")
    ap.add_argument("--continuous_window", type=int, default=4,
                    help="max_num_seqs (sliding window) for continuous batching bench")
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

    # ── Pass 3 (Task 6): continuous batching bench ────────────────────────
    continuous_results = []
    if args.continuous_total > 0:
        print("=== Continuous Batching (Task 6) ===")
        print(f"  total_requests={args.continuous_total}, window={args.continuous_window}")
        fmt_cb = (f"{'total':>6}  {'window':>6}  {'prefill_ms':>10}  "
                  f"{'total_ms':>10}  {'tps':>8}  {'mixed%':>7}")
        print(fmt_cb)
        print("-" * 60)
        try:
            r = bench_continuous(
                model=model,
                eos_token_id=eos,
                token_ids=pt_ids,
                device=device,
                dtype=dtype,
                total_requests=args.continuous_total,
                window=args.continuous_window,
                decode_steps=args.decode_steps,
                warmup=args.warmup,
                runs=args.runs,
                num_kvcache_blocks=args.num_kvcache_blocks,
                kvcache_block_size=args.kvcache_block_size,
            )
            mixed_pct = r["mixed_step_fraction"] * 100
            print(f"  {r['total_requests']:>6}  {r['window']:>6}  "
                  f"{r['prefill_ms_mean']:>10.1f}  "
                  f"{r['total_decode_ms_mean']:>10.1f}  "
                  f"{r['tokens_per_sec']:>8.1f}  "
                  f"{mixed_pct:>6.1f}%")
            continuous_results.append(r)
        except Exception as e:
            print(f"  continuous batching SKIPPED ({e})")
        print()

    if args.out_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_json)) or ".", exist_ok=True)
        payload = {
            "padded_results": padded_results,
            "paged_results": paged_results,
            "continuous_results": continuous_results,
        }
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"[ok] results written to {args.out_json}")


if __name__ == "__main__":
    main()
