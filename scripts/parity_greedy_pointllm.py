#!/usr/bin/env python3
"""
M1：PointLLM 上对比 `manual_greedy_decode`（首步带点云）与 `model.generate`（greedy）。
需：export PYTHONPATH=/home/PointLLM:$PYTHONPATH 且已安装 PointLLM 依赖的 conda 环境。
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

# PointLLM 仓库（默认 /home/PointLLM）
_POINTLLM = os.environ.get("POINTLLM_ROOT", "/home/PointLLM")
if _POINTLLM not in sys.path:
    sys.path.insert(0, _POINTLLM)

from pointllm.conversation import conv_templates
from pointllm.model import PointLLMLlamaForCausalLM
from transformers import AutoTokenizer

from nanopointllm.parity.hf_manual_greedy import manual_greedy_decode
from nanopointllm.parity.model_path import validate_pretrained_local_or_hub


def build_prompt_ids(tok, model, device, question: str = "Hi."):
    conv = conv_templates["vicuna_v1_1"].copy()
    cfg = model.get_model().point_backbone_config
    pt_len = cfg["point_token_len"]
    patch = cfg["default_point_patch_token"]
    if cfg["mm_use_point_start_end"]:
        s, e = cfg["default_point_start_token"], cfg["default_point_end_token"]
        qs = s + patch * pt_len + e + "\n" + question
    else:
        qs = patch * pt_len + "\n" + question
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    enc = tok([conv.get_prompt()], return_tensors="pt")
    return enc["input_ids"].to(device), enc["attention_mask"].to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda:1")
    ap.add_argument("--max_new_tokens", type=int, default=8)
    ap.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    args = ap.parse_args()

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    model_path = validate_pretrained_local_or_hub(args.model_path)

    tok = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = PointLLMLlamaForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        use_cache=True,
    ).to(device)
    model.initialize_tokenizer_point_backbone_config_wo_embedding(tok)
    model.eval()

    input_ids, attention_mask = build_prompt_ids(tok, model, device)
    point_clouds = torch.randn(1, 8192, 6, device=device, dtype=dtype)

    manual = manual_greedy_decode(
        model,
        input_ids,
        attention_mask,
        args.max_new_tokens,
        point_clouds=point_clouds,
    )

    gen = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        point_clouds=point_clouds,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
    )

    ok = torch.equal(manual, gen)
    print("manual shape:", manual.shape, "generate shape:", gen.shape)
    print("parity (exact match):", ok)
    if not ok:
        # 允许因实现细节产生 1 token 边界差异时，可改为比较 decode 段
        print("manual:", repr(tok.decode(manual[0], skip_special_tokens=True)[:200]))
        print("gen   :", repr(tok.decode(gen[0], skip_special_tokens=True)[:200]))
        raise SystemExit(1)
    print("[ok] M1 PointLLM greedy parity passed.")


if __name__ == "__main__":
    main()
