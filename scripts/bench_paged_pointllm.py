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


@torch.inference_mode()
def bench_one(
    model,
    eos_token_id: int,
    token_ids: list[int],
    device: torch.device,
    dtype: torch.dtype,
    B: int,
    decode_steps: int,
    warmup: int,
    runs: int,
    num_kvcache_blocks: int,
    kvcache_block_size: int,
) -> dict:
    sp = SamplingParams(max_tokens=decode_steps, ignore_eos=True)
    pcs = [make_fake_point_cloud(device, dtype) for _ in range(B)]
    requests = [
        {"token_ids": token_ids, "point_clouds": pc, "sampling_params": sp}
        for pc in pcs
    ]

    def _run() -> tuple[float, float, int]:
        engine = PointLLMLLMEngine(
            model,
            eos_token_id=eos_token_id,
            max_num_seqs=B,
            max_num_batched_tokens=4096,
            num_kvcache_blocks=num_kvcache_blocks if num_kvcache_blocks > 0 else None,
            kvcache_block_size=kvcache_block_size,
        )
        for req in requests:
            engine.add_request(**req)

        torch.cuda.synchronize()
        t_start = time.perf_counter()
        engine.step()  # prefill
        torch.cuda.synchronize()
        t_prefill = time.perf_counter()

        actual_steps = 0
        while not engine.is_finished():
            engine.step()
            actual_steps += 1
        torch.cuda.synchronize()
        t_end = time.perf_counter()
        return (t_prefill - t_start) * 1000, (t_end - t_prefill) * 1000, actual_steps

    for _ in range(warmup):
        _run()

    pf_list, dc_list, steps_list = zip(*[_run() for _ in range(runs)])
    pf_mean = sum(pf_list) / runs
    dc_mean = sum(dc_list) / runs
    actual_steps = steps_list[-1]

    dec_per_step = dc_mean / (actual_steps * B) if actual_steps > 0 else 0.0
    tps = (actual_steps * B) / (dc_mean / 1000) if dc_mean > 0 else 0.0
    return {
        "batch_size": B,
        "prefill_ms_mean": round(pf_mean, 2),
        "decode_mean_per_step_ms": round(dec_per_step, 3),
        "tokens_per_sec": round(tps, 1),
        "actual_decode_steps": actual_steps,
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

    dtype = getattr(torch, args.dtype)
    batch_sizes = [int(x.strip()) for x in args.batch_sizes.split(",") if x.strip()]

    print(f"Loading model {args.model_path} on {device} ({args.dtype})...")
    model, tok = load_pointllm_model(args.model_path, device, dtype)
    eos     = tok.eos_token_id
    pt_ids  = build_prompt_token_ids(tok, model)
    print(f"Model loaded. prompt_len={len(pt_ids)}\n")

    print(f"{'B':>4}  {'mode':<8}  {'prefill_ms':>10}  {'dec/step/seq_ms':>15}  {'tps':>8}")
    print("-" * 60)

    padded_results = []
    paged_results  = []

    for B in batch_sizes:
        try:
            # Padded (original)
            r_pad = bench_one(model, eos, pt_ids, device, dtype, B,
                              args.decode_steps, args.warmup, args.runs,
                              num_kvcache_blocks=0, kvcache_block_size=args.kvcache_block_size)
            print(f"  B={B:<2}  padded   prefill={r_pad['prefill_ms_mean']:7.1f}ms  "
                  f"dec/step={r_pad['decode_mean_per_step_ms']:7.3f}ms  "
                  f"tps={r_pad['tokens_per_sec']:7.1f}")
            padded_results.append(r_pad)

            # Paged
            r_paged = bench_one(model, eos, pt_ids, device, dtype, B,
                                args.decode_steps, args.warmup, args.runs,
                                num_kvcache_blocks=args.num_kvcache_blocks,
                                kvcache_block_size=args.kvcache_block_size)
            speedup = r_paged["tokens_per_sec"] / r_pad["tokens_per_sec"] if r_pad["tokens_per_sec"] else 0
            print(f"  B={B:<2}  paged    prefill={r_paged['prefill_ms_mean']:7.1f}ms  "
                  f"dec/step={r_paged['decode_mean_per_step_ms']:7.3f}ms  "
                  f"tps={r_paged['tokens_per_sec']:7.1f}  ({speedup:.2f}x vs padded)")
            paged_results.append(r_paged)
        except Exception as e:
            print(f"  B={B:<2}  SKIPPED ({e})")

    print()

    if args.out_json:
        out_dir = os.path.dirname(os.path.abspath(args.out_json))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        payload = {"padded_results": padded_results, "paged_results": paged_results}
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"[ok] results written to {args.out_json}")


if __name__ == "__main__":
    main()
