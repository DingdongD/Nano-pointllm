#!/usr/bin/env python3
"""
M1：标准 HuggingFace Llama（无点云）上，对比 `manual_greedy_decode` 与 `model.generate`（greedy）。
"""
from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from nanopointllm.parity.hf_manual_greedy import manual_greedy_decode
from nanopointllm.parity.model_path import validate_pretrained_local_or_hub


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--max_new_tokens", type=int, default=16)
    ap.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    args = ap.parse_args()

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    model_path = validate_pretrained_local_or_hub(args.model_path)

    tok = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    prompt = "The capital of France is"
    enc = tok(prompt, return_tensors="pt").to(device)
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    manual = manual_greedy_decode(
        model,
        input_ids,
        attention_mask,
        args.max_new_tokens,
    )

    gen = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        pad_token_id=tok.pad_token_id,
    )

    ok = torch.equal(manual, gen)
    print("manual shape:", manual.shape, "generate shape:", gen.shape)
    print("parity (exact match):", ok)
    if not ok:
        print("manual:", tok.decode(manual[0]))
        print("gen   :", tok.decode(gen[0]))
        raise SystemExit(1)
    print("[ok] M1 Llama greedy parity passed.")


if __name__ == "__main__":
    main()
