#!/usr/bin/env python3
"""
P5 parity: PointLLMLLMEngine(num_kvcache_blocks=512) vs HF manual greedy decode.

Usage:
  cd /home/nano-pointllm
  PYTHONPATH=/home/PointLLM:. python scripts/parity_paged_pointllm.py \
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


def main() -> None:
    ap = argparse.ArgumentParser(description="P5 parity: paged engine vs HF greedy")
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--max_new_tokens", type=int, default=8)
    ap.add_argument("--num_kvcache_blocks", type=int, default=512)
    ap.add_argument("--kvcache_block_size", type=int, default=16)
    args = ap.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        sys.exit("CUDA not available")

    dtype = getattr(torch, args.dtype)
    print(f"Loading model {args.model_path} on {device} ({args.dtype})...")
    model, tok = load_pointllm_model(args.model_path, device, dtype)
    eos = tok.eos_token_id
    print("Model loaded.\n")

    pt_ids  = build_prompt_token_ids(tok, model)
    txt_ids = build_text_only_token_ids(tok)

    def pc():
        return make_fake_point_cloud(device, dtype)

    cases = [
        ("single  (B=1, 1 pc)",      [pt_ids],                                  [pc()]),
        ("batch_4 (B=4, 4 pc)",      [pt_ids] * 4,                              [pc() for _ in range(4)]),
        ("batch_8 (B=8, 8 pc)",      [pt_ids] * 8,                              [pc() for _ in range(8)]),
        ("mixed_4 (B=4, 2pc+2text)", [pt_ids, pt_ids, txt_ids, txt_ids],        [pc(), pc(), None, None]),
    ]

    # ── Step 1: compute all HF references BEFORE patching the model ────────
    # inject_paged_attention (called inside PointLLMLLMEngine.__init__) permanently
    # replaces LlamaAttention layers with PagedLlamaAttention.  We must generate
    # all reference outputs while the model is still in its original state.
    print("=== HF GREEDY REFERENCE (unpatched model) ===")
    all_refs: list[list[list[int]]] = []
    for name, token_ids_list, pcs in cases:
        case_refs = [
            hf_greedy_generate(model, ids, p, args.max_new_tokens, device)
            for ids, p in zip(token_ids_list, pcs)
        ]
        all_refs.append(case_refs)
        print(f"  {name}: {len(case_refs)} ref(s) computed")
    print()

    # ── Step 2: create ONE engine (patches model once) and run all cases ───
    max_batch = max(len(ids_list) for _, ids_list, _ in cases)
    engine = PointLLMLLMEngine(
        model,
        eos_token_id=eos,
        max_num_seqs=max_batch,
        max_num_batched_tokens=4096,
        num_kvcache_blocks=args.num_kvcache_blocks,
        kvcache_block_size=args.kvcache_block_size,
    )

    print("=== PARITY (paged engine) ===")
    all_ok = True
    sp = SamplingParams(max_tokens=args.max_new_tokens)

    for (name, token_ids_list, pcs), case_refs in zip(cases, all_refs):
        requests = [
            {"token_ids": ids, "point_clouds": p, "sampling_params": sp}
            for ids, p in zip(token_ids_list, pcs)
        ]
        try:
            seqs = engine.generate(requests)
        except Exception:
            print(f"  [{name}] ENGINE ERROR:")
            traceback.print_exc()
            all_ok = False
            continue

        ok = True
        for i, (seq, case_ref) in enumerate(zip(seqs, case_refs)):
            generated = seq.token_ids[seq.num_prompt_tokens:]
            if generated != case_ref:
                print(f"  [{name}] seq {i} MISMATCH:")
                print(f"    engine : {generated}")
                print(f"    hf_ref : {case_ref}")
                ok = False

        status = "PASS" if ok else "FAIL"
        print(f"  {name}: {status}")
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
