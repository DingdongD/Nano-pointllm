#!/usr/bin/env python3
"""
Task 6 parity: paged LLMEngine in continuous batching mode vs HF greedy.

Tests the case where new requests arrive while existing ones are already
decoding, so engine.step() issues run_mixed(prefill_seqs, decode_seqs)
with BOTH lists non-empty — the core of continuous batching.

Usage:
  cd /home/nano-pointllm
  PYTHONPATH=/home/PointLLM:. python scripts/parity_mixed_continuous_pointllm.py \\
    --model_path /mnt/llm_data/pointllm_ckpt/PointLLM_7B_v1.2 \\
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


def run_paged_continuous(
    model,
    eos_token_id: int,
    all_token_ids: list[list[int]],
    all_point_clouds: list,
    max_new_tokens: int,
    max_num_seqs: int,
    num_kvcache_blocks: int,
    kvcache_block_size: int,
) -> tuple[list[list[int]], list[tuple[int, int]]]:
    """
    Start part of the batch, then inject the remaining requests after its
    prefill so that a later scheduling round contains both prefill and decode.

    Returns (generated_tokens_per_seq, step_log) where each entry in
    step_log is (num_prefill, num_decode) for that step.
    """
    B = len(all_token_ids)
    sp = SamplingParams(max_tokens=max_new_tokens)
    engine = PointLLMLLMEngine(
        model,
        eos_token_id=eos_token_id,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=4096,
        num_kvcache_blocks=num_kvcache_blocks,
        kvcache_block_size=kvcache_block_size,
    )

    step_log: list[tuple[int, int]] = []

    # Instrument run_mixed to record batch composition per step
    orig_run_mixed = engine.runner.run_mixed

    def tracked_run_mixed(prefill_seqs, decode_seqs):
        step_log.append((len(prefill_seqs), len(decode_seqs)))
        return orig_run_mixed(prefill_seqs, decode_seqs)

    engine.runner.run_mixed = tracked_run_mixed

    if max_num_seqs < 2:
        raise ValueError("mixed-step parity requires max_num_seqs >= 2")

    # Leave one admission slot free.  After this first prefill step, adding the
    # rest guarantees that the next round combines a new prefill with decode.
    initial_count = min(B, max_num_seqs - 1)
    seqs = [
        engine.add_request(token_ids=ids, point_clouds=pc, sampling_params=sp)
        for ids, pc in zip(
            all_token_ids[:initial_count],
            all_point_clouds[:initial_count],
        )
    ]
    engine.step()
    seqs.extend(
        engine.add_request(token_ids=ids, point_clouds=pc, sampling_params=sp)
        for ids, pc in zip(
            all_token_ids[initial_count:],
            all_point_clouds[initial_count:],
        )
    )

    while not engine.is_finished():
        engine.step()

    generated = [seq.token_ids[seq.num_prompt_tokens:] for seq in seqs]
    return generated, step_log


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Task 6 parity: continuous batching paged engine vs HF greedy"
    )
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--max_new_tokens", type=int, default=6)
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

    # Cases: (name, token_ids_list, point_clouds_list, max_num_seqs)
    # Requests are staggered inside run_paged_continuous to force mixed steps.
    cases = [
        (
            "2pc+2text, max_num_seqs=2 (forces mixed steps)",
            [pt_ids, pt_ids, txt_ids, txt_ids],
            [pc(), pc(), None, None],
            2,
        ),
        (
            "4pc, max_num_seqs=2 (forces mixed steps)",
            [pt_ids] * 4,
            [pc() for _ in range(4)],
            2,
        ),
        (
            "6pc, max_num_seqs=4 (forces mixed steps)",
            [pt_ids] * 6,
            [pc() for _ in range(6)],
            4,
        ),
    ]

    # Compute all HF references BEFORE patching the model
    print("=== HF GREEDY REFERENCE (unpatched model) ===")
    all_refs: list[list[list[int]]] = []
    for name, token_ids_list, pcs, _ in cases:
        case_refs = [
            hf_greedy_generate(model, ids, p, args.max_new_tokens, device)
            for ids, p in zip(token_ids_list, pcs)
        ]
        all_refs.append(case_refs)
        print(f"  {name[:40]}: {len(case_refs)} ref(s) computed")
    print()

    print("=== PARITY (continuous batching, paged engine) ===")
    all_ok = True

    for (name, token_ids_list, pcs, max_num_seqs), case_refs in zip(cases, all_refs):
        try:
            generated, step_log = run_paged_continuous(
                model=model,
                eos_token_id=eos,
                all_token_ids=token_ids_list,
                all_point_clouds=pcs,
                max_new_tokens=args.max_new_tokens,
                max_num_seqs=max_num_seqs,
                num_kvcache_blocks=args.num_kvcache_blocks,
                kvcache_block_size=args.kvcache_block_size,
            )
        except Exception:
            print(f"  [{name}] ENGINE ERROR:")
            traceback.print_exc()
            all_ok = False
            continue

        # Check that at least one step was truly mixed
        mixed_steps = [(p, d) for p, d in step_log if p > 0 and d > 0]
        if not mixed_steps:
            print(f"  [{name}] FAIL: no mixed step occurred; step_log={step_log}")
            all_ok = False
            ok = False
        else:
            print(f"  [{name}] mixed steps: {mixed_steps}")
            ok = True
        for i, (gen, ref) in enumerate(zip(generated, case_refs)):
            if gen != ref:
                print(f"    seq {i} MISMATCH:")
                print(f"      engine : {gen}")
                print(f"      hf_ref : {ref}")
                ok = False

        status = "PASS" if ok else "FAIL"
        print(f"  {name[:50]}: {status}")
        if not ok:
            all_ok = False

    print()
    if all_ok:
        print("[ok] all continuous batching parity cases passed.")
        sys.exit(0)
    else:
        print("[FAIL] one or more cases failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
