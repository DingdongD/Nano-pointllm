#!/usr/bin/env python3
"""
P4 bench: PointLLMLLMEngine throughput across batch sizes.

Usage:
  cd /home/nano-pointllm
  PYTHONPATH=/home/PointLLM:. python scripts/bench_engine_pointllm.py \
    --model_path /mnt/llm_data/pointllm_ckpt/PointLLM_7B_v1.2 \
    --device cuda:1 --batch_sizes 1,2,4,8 --decode_steps 32 \
    --warmup 2 --runs 3 --out_json results/p4_bench.json

Metrics:
  decode_mean_per_step_ms  -- wall time per (step x sequence), matches bench_m3 naming
  tokens_per_sec           -- aggregate tokens generated per second across all sequences
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

from nanopointllm.engine.decode_backend import HFDecodeBackend
from nanopointllm.engine.hf_hybrid import hf_prefill
from nanopointllm.engine.llm_engine import PointLLMLLMEngine
from nanopointllm.engine.types import DecodeStepInput
from nanopointllm.parity.engine_test_utils import (
    build_prompt_token_ids,
    load_pointllm_model,
    make_fake_point_cloud,
)
from nanopointllm.sampling_params import SamplingParams


# -- HF B=1 baseline ---------------------------------------------------------

@torch.inference_mode()
def bench_hf_baseline(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    point_clouds: torch.Tensor,
    decode_steps: int,
    warmup: int,
    runs: int,
) -> dict:
    """Single-sequence HF prefill + HFDecodeBackend decode loop."""
    backend = HFDecodeBackend()
    pc = point_clouds.unsqueeze(0)  # [1, N, C]

    def _run() -> tuple[float, float]:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        po = hf_prefill(model, input_ids, attention_mask, point_clouds=pc)
        torch.cuda.synchronize()
        prefill_ms = (time.perf_counter() - t0) * 1000

        nid = po.logits[:, -1, :].float().argmax(-1, keepdim=True)
        mask = attention_mask
        past = po.past_key_values
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        for _ in range(decode_steps):
            mask = torch.cat([mask, mask.new_ones((1, 1))], dim=-1)
            logits, past = backend.decode_step(
                model,
                DecodeStepInput(input_ids=nid, attention_mask=mask, past_key_values=past),
            )
            nid = logits[:, -1, :].float().argmax(-1, keepdim=True)
        torch.cuda.synchronize()
        decode_ms = (time.perf_counter() - t1) * 1000
        return prefill_ms, decode_ms

    for _ in range(warmup):
        _run()

    pf_list, dc_list = zip(*[_run() for _ in range(runs)])
    pf_mean = sum(pf_list) / runs
    dc_mean = sum(dc_list) / runs
    return {
        "prefill_ms_mean": round(pf_mean, 2),
        "decode_mean_per_step_ms": round(dc_mean / decode_steps, 3),
        "tokens_per_sec": round(decode_steps / (dc_mean / 1000), 1),
    }


# -- Engine bench ------------------------------------------------------------

@torch.inference_mode()
def bench_engine_one_b(
    model,
    eos_token_id: int,
    token_ids: list[int],
    device: torch.device,
    dtype: torch.dtype,
    B: int,
    decode_steps: int,
    warmup: int,
    runs: int,
) -> dict:
    """Bench PointLLMLLMEngine at batch size B. decode_steps controls max_tokens."""
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
        )
        for req in requests:
            engine.add_request(**req)

        # First step = prefill
        torch.cuda.synchronize()
        t_start = time.perf_counter()
        engine.step()
        torch.cuda.synchronize()
        t_prefill = time.perf_counter()

        # Remaining steps = decode
        actual_decode_steps = 0
        while not engine.is_finished():
            engine.step()
            actual_decode_steps += 1
        torch.cuda.synchronize()
        t_end = time.perf_counter()

        return (
            (t_prefill - t_start) * 1000,
            (t_end - t_prefill) * 1000,
            actual_decode_steps,
        )

    for _ in range(warmup):
        _run()

    pf_list, dc_list, steps_list = zip(*[_run() for _ in range(runs)])
    pf_mean = sum(pf_list) / runs
    dc_mean = sum(dc_list) / runs
    # Use actual decode steps from last run (should equal decode_steps with ignore_eos=True)
    actual_steps = steps_list[-1]

    decode_mean_per_step_ms = dc_mean / (actual_steps * B) if actual_steps > 0 else 0.0
    tokens_per_sec = (actual_steps * B) / (dc_mean / 1000) if dc_mean > 0 else 0.0

    return {
        "batch_size": B,
        "prefill_ms_mean": round(pf_mean, 2),
        "decode_mean_per_step_ms": round(decode_mean_per_step_ms, 3),
        "tokens_per_sec": round(tokens_per_sec, 1),
        "actual_decode_steps": actual_steps,
    }


# -- main --------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="P4 bench: PointLLMLLMEngine throughput")
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--batch_sizes", default="1,2,4,8")
    ap.add_argument("--decode_steps", type=int, default=32)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--out_json", default="")
    args = ap.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        sys.exit("CUDA not available")

    dtype = getattr(torch, args.dtype)
    batch_sizes = [int(x.strip()) for x in args.batch_sizes.split(",") if x.strip()]

    print(f"Loading model {args.model_path} on {device} ({args.dtype})...")
    model, tok = load_pointllm_model(args.model_path, device, dtype)
    eos = tok.eos_token_id
    pt_ids = build_prompt_token_ids(tok, model)
    print(f"Model loaded. prompt_len={len(pt_ids)}\n")

    # HF B=1 baseline
    ids_1 = torch.tensor([pt_ids], dtype=torch.long, device=device)
    mask_1 = torch.ones_like(ids_1)
    pc_1 = make_fake_point_cloud(device, dtype)
    print("Running HF B=1 baseline...")
    hf_result = bench_hf_baseline(model, ids_1, mask_1, pc_1,
                                   args.decode_steps, args.warmup, args.runs)
    print(f"  HF B=1  prefill={hf_result['prefill_ms_mean']:.1f}ms  "
          f"decode/step={hf_result['decode_mean_per_step_ms']:.3f}ms  "
          f"tps={hf_result['tokens_per_sec']:.1f}\n")

    # Engine bench per batch size
    print(f"{'B':>4}  {'prefill_ms':>10}  {'dec/step/seq_ms':>15}  {'tps':>8}")
    print("-" * 50)
    all_results = []
    for B in batch_sizes:
        try:
            r = bench_engine_one_b(
                model, eos, pt_ids, device, dtype, B,
                args.decode_steps, args.warmup, args.runs,
            )
            if B == 1:
                r["hf_b1_decode_mean_per_step_ms"] = hf_result["decode_mean_per_step_ms"]
                r["hf_b1_tokens_per_sec"] = hf_result["tokens_per_sec"]
            speedup = r["tokens_per_sec"] / hf_result["tokens_per_sec"] if hf_result["tokens_per_sec"] else 0
            print(f"  B={B:<2}  prefill={r['prefill_ms_mean']:7.1f}ms  "
                  f"dec/step={r['decode_mean_per_step_ms']:7.3f}ms  "
                  f"tps={r['tokens_per_sec']:7.1f}  ({speedup:.2f}x vs HF B=1)")
            all_results.append(r)
        except Exception as e:
            print(f"  B={B:<2}  SKIPPED ({e})")

    print()

    if args.out_json:
        out_dir = os.path.dirname(os.path.abspath(args.out_json))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        payload = {"hf_b1_baseline": hf_result, "engine_results": all_results}
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"[ok] results written to {args.out_json}")


if __name__ == "__main__":
    main()
