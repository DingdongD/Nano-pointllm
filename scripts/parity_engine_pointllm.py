#!/usr/bin/env python3
"""
P4 parity: PointLLMLLMEngine.generate() vs HF manual greedy decode.

Usage:
  cd /home/nano-pointllm
  PYTHONPATH=/home/PointLLM:. python scripts/parity_engine_pointllm.py \
    --model_path /mnt/llm_data/pointllm_ckpt/PointLLM_7B_v1.2 \
    --device cuda:1

Exit codes: 0 = all cases passed, 1 = mismatch or error.
"""
from __future__ import annotations

import argparse
import sys
import traceback

import torch

from nanopointllm.engine.llm_engine import PointLLMLLMEngine
from nanopointllm.parity.engine_test_utils import (
    build_prompt_token_ids,
    build_text_only_token_ids,
    hf_greedy_generate,
    load_pointllm_model,
    make_fake_point_cloud,
)
from nanopointllm.sampling_params import SamplingParams


def run_case(
    name: str,
    model,
    eos_token_id: int,
    all_token_ids: list[list[int]],
    all_point_clouds: list,
    max_new_tokens: int,
    device: torch.device,
) -> bool:
    """
    Run one parity case. Returns True if all sequences match HF greedy.
    Prints PASS or per-sequence FAIL details.
    """
    B = len(all_token_ids)
    sp = SamplingParams(max_tokens=max_new_tokens)
    engine = PointLLMLLMEngine(
        model,
        eos_token_id=eos_token_id,
        max_num_seqs=B,
        max_num_batched_tokens=4096,
    )
    requests = [
        {"token_ids": ids, "point_clouds": pc, "sampling_params": sp}
        for ids, pc in zip(all_token_ids, all_point_clouds)
    ]

    try:
        seqs = engine.generate(requests)
    except Exception:
        print(f"  [{name}] ENGINE ERROR:")
        traceback.print_exc()
        return False

    ok = True
    for i, (seq, ids, pc) in enumerate(zip(seqs, all_token_ids, all_point_clouds)):
        generated = seq.token_ids[seq.num_prompt_tokens:]
        ref = hf_greedy_generate(model, ids, pc, max_new_tokens, device)
        if generated != ref:
            print(f"  [{name}] seq {i} MISMATCH:")
            print(f"    engine : {generated}")
            print(f"    hf_ref : {ref}")
            ok = False

    status = "PASS" if ok else "FAIL"
    print(f"  {name}: {status}")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description="P4 parity: PointLLMLLMEngine vs HF greedy")
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--max_new_tokens", type=int, default=8)
    args = ap.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        sys.exit("CUDA not available")

    dtype = getattr(torch, args.dtype)
    print(f"Loading model {args.model_path} on {device} ({args.dtype})...")
    model, tok = load_pointllm_model(args.model_path, device, dtype)
    eos = tok.eos_token_id
    print("Model loaded.\n")

    pt_ids = build_prompt_token_ids(tok, model)
    txt_ids = build_text_only_token_ids(tok)

    def pc() -> torch.Tensor:
        return make_fake_point_cloud(device, dtype)

    cases = [
        (
            "single  (B=1, 1 point cloud)",
            [pt_ids],
            [pc()],
        ),
        (
            "batch_4 (B=4, 4 point clouds)",
            [pt_ids] * 4,
            [pc() for _ in range(4)],
        ),
        (
            "batch_8 (B=8, 8 point clouds)",
            [pt_ids] * 8,
            [pc() for _ in range(8)],
        ),
        (
            "mixed_4 (B=4, 2 pc + 2 text-only)",
            [pt_ids, pt_ids, txt_ids, txt_ids],
            [pc(), pc(), None, None],
        ),
    ]

    print("=== PARITY ===")
    all_ok = True
    for name, token_ids_list, pcs in cases:
        ok = run_case(name, model, eos, token_ids_list, pcs, args.max_new_tokens, device)
        if not ok:
            all_ok = False

    print()
    if all_ok:
        print("[ok] all parity cases passed.")
        sys.exit(0)
    else:
        print("[FAIL] one or more parity cases failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
