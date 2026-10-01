#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os

import torch

from nanopointllm.engine.llm_engine import PointLLMLLMEngine
from nanopointllm.parity.engine_test_utils import (
    build_prompt_token_ids,
    load_pointllm_model,
    make_fake_point_cloud,
)
from nanopointllm.sampling_params import SamplingParams


def main() -> None:
    ap = argparse.ArgumentParser(description="Profile lightweight decode layer timings")
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--decode_steps", type=int, default=4)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_kvcache_blocks", type=int, default=1024)
    ap.add_argument("--kvcache_block_size", type=int, default=16)
    ap.add_argument("--out_json", default="")
    args = ap.parse_args()

    os.environ["NANOPOINTLLM_ENABLE_CUDA_GRAPH"] = "0"
    os.environ["NANOPOINTLLM_PROFILE_LAYERS"] = "1"

    device = torch.device(args.device)
    model, tok = load_pointllm_model(args.model_path, device=device, dtype=torch.bfloat16)
    eos = tok.eos_token_id or 2
    token_ids = build_prompt_token_ids(tok, model)
    pc = make_fake_point_cloud(device, torch.bfloat16)
    sp = SamplingParams(max_tokens=args.decode_steps, ignore_eos=True)

    engine = PointLLMLLMEngine(
        model,
        eos_token_id=eos,
        max_num_seqs=args.batch_size,
        max_num_batched_tokens=4096,
        num_kvcache_blocks=args.num_kvcache_blocks,
        kvcache_block_size=args.kvcache_block_size,
    )
    seqs = [
        engine.add_request(token_ids=token_ids, point_clouds=pc.clone(), sampling_params=sp)
        for _ in range(args.batch_size)
    ]

    while True:
        engine.step()
        if all(seq.num_completion_tokens > 0 or seq.is_finished for seq in seqs):
            break
        if engine.is_finished():
            break

    if not engine.is_finished():
        engine.step()

    runner = getattr(engine.runner, "lightweight_runner", None)
    profile = getattr(runner, "last_profile", []) if runner is not None else []
    totals: dict[str, float] = {}
    for row in profile:
        for key, value in row.items():
            if key == "layer":
                continue
            totals[key] = totals.get(key, 0.0) + float(value)

    result = {
        "batch_size": args.batch_size,
        "decode_steps": args.decode_steps,
        "layers": profile,
        "totals_ms": totals,
    }
    print(json.dumps(result, indent=2))
    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
