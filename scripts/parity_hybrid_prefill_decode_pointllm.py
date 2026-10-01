#!/usr/bin/env python3
"""
M2 验收：HF **prefill + 分步 decode**（`hybrid_greedy_decode`）与整段 `manual_greedy_decode` greedy 结果一致。
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

# 减轻 bf16 下不同子图顺序导致的 logits 细微差异（进而 argmax 分叉）
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

_POINTLLM = os.environ.get("POINTLLM_ROOT", "/home/PointLLM")
if _POINTLLM not in sys.path:
    sys.path.insert(0, _POINTLLM)

from pointllm.conversation import conv_templates
from pointllm.model import PointLLMLlamaForCausalLM
from transformers import AutoTokenizer

from nanopointllm.engine.hf_hybrid import hybrid_greedy_decode
from nanopointllm.parity.attn_stability import set_eager_attention_if_supported
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
    ap.add_argument(
        "--parity_keep_sdpa",
        action="store_true",
        help="parity 不切 eager 注意力（默认 eager，减轻 bf16+SDPA 下 greedy 分叉）",
    )
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
    if not args.parity_keep_sdpa:
        set_eager_attention_if_supported(model)

    input_ids, attention_mask = build_prompt_ids(tok, model, device)
    point_clouds = torch.randn(1, 8192, 6, device=device, dtype=dtype)

    manual = manual_greedy_decode(
        model,
        input_ids,
        attention_mask,
        args.max_new_tokens,
        point_clouds=point_clouds,
    )
    hybrid = hybrid_greedy_decode(
        model,
        input_ids,
        attention_mask,
        args.max_new_tokens,
        point_clouds=point_clouds,
    )

    ok = torch.equal(manual, hybrid)
    print("manual shape:", manual.shape, "hybrid shape:", hybrid.shape)
    print("parity (exact match):", ok)
    if not ok:
        diff = (manual != hybrid).nonzero(as_tuple=True)
        if diff[0].numel():
            i = int(diff[1][0].item())
            print("first diff index:", i, "manual:", manual[0, i].item(), "hybrid:", hybrid[0, i].item())
        print("提示: 若仍不一致，可试 `--dtype float32` 排除半精度累加差异。")
        raise SystemExit(1)
    print("[ok] M2 PointLLM hybrid (prefill + decode) greedy parity passed.")


if __name__ == "__main__":
    main()
