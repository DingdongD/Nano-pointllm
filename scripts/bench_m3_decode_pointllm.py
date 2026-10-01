#!/usr/bin/env python3
"""
M3：经 `nanopointllm.engine` 的 **prefill + 可插拔 decode 后端** 计时，指标与
`PointLLM/docs/scripts/phase1_prefill_decode_bench.py` 对齐，便于对比 `decode_mean_per_step_ms`。

  conda activate pointllm
  export PYTHONPATH=/home/PointLLM:$PYTHONPATH
  cd /home/nano-pointllm
  python scripts/bench_m3_decode_pointllm.py \\
    --model_path /mnt/llm_data/pointllm_ckpt/PointLLM_7B_v1.2 \\
    --device cuda:1 --decode_steps 32 --warmup 2 --runs 1

  # decode 栈：--decode_stack hf|compile （内 hf，外 torch.compile）
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

_POINTLLM = os.environ.get("POINTLLM_ROOT", "/home/PointLLM")
if _POINTLLM not in sys.path:
    sys.path.insert(0, _POINTLLM)

import torch
from transformers import AutoTokenizer

from nanopointllm.engine.decode_backend import DecodeBackend
from nanopointllm.engine.decode_stack import build_decode_stack
from nanopointllm.engine.hf_hybrid import hf_prefill
from nanopointllm.engine.types import DecodeStepInput
from nanopointllm.parity.model_path import (
    validate_pretrained_local_or_hub,
    validate_safetensors_weights_present,
    validate_torch_version_for_bin_weights,
)

from pointllm.conversation import conv_templates
from pointllm.model import PointLLMLlamaForCausalLM
from pointllm.utils import disable_torch_init


def build_prompt_ids(tok, model, device, question: str = "What is this?"):
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


@torch.inference_mode()
def bench_once(
    model: torch.nn.Module,
    backend: DecodeBackend,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    point_clouds: torch.Tensor,
    decode_steps: int,
    device: torch.device,
) -> dict:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    po = hf_prefill(model, input_ids, attention_mask, point_clouds=point_clouds)
    torch.cuda.synchronize()
    prefill_ms = (time.perf_counter() - t0) * 1000.0

    next_id = po.logits[:, -1, :].float().argmax(dim=-1, keepdim=True)
    mask = attention_mask
    past = po.past_key_values

    decode_times: list[float] = []
    for _ in range(decode_steps):
        # 与 phase1_prefill_decode_bench：每步先延长 mask 再 forward
        mask = torch.cat([mask, mask.new_ones((mask.shape[0], 1))], dim=-1)
        step_in = DecodeStepInput(
            input_ids=next_id,
            attention_mask=mask,
            past_key_values=past,
        )
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        logits, past = backend.decode_step(model, step_in)
        torch.cuda.synchronize()
        decode_times.append((time.perf_counter() - t1) * 1000.0)
        next_id = logits[:, -1, :].float().argmax(dim=-1, keepdim=True)

    return {
        "prefill_ms": prefill_ms,
        "decode_step_ms": decode_times,
        "decode_total_ms": sum(decode_times),
        "decode_mean_ms": sum(decode_times) / len(decode_times) if decode_times else 0.0,
        "prompt_len": int(input_ids.shape[1]),
        "decode_steps": decode_steps,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda:1")
    ap.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--decode_steps", type=int, default=32)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--out_json", type=str, default="")
    ap.add_argument(
        "--decode_stack",
        type=str,
        default="hf",
        help="单步 decode 栈：内→外用 | 连接。例：`hf`、`hf_inner`、`torch_loop`（显式逐层+lm_head）、`hf|compile`、`torch_loop|compile`。",
    )
    ap.add_argument(
        "--compile_mode",
        type=str,
        default="reduce-overhead",
        help="仅当栈含 compile 时传给 torch.compile(mode=...)",
    )
    ap.add_argument(
        "--compile_fullgraph",
        action="store_true",
        help="仅当栈含 compile 时传给 torch.compile(fullgraph=True)",
    )
    ap.add_argument(
        "--compile_allow_triton_cudagraphs",
        action="store_true",
        help="默认关闭 Triton CUDAGraph（与自回归 KV 兼容）。仅当你了解风险且需极致延迟时可打开；将尝试每步 cudagraph_mark_step_begin()",
    )
    ap.add_argument(
        "--use_safetensors",
        action="store_true",
        help="from_pretrained(..., use_safetensors=True)：在 torch<2.6 且 transformers≥5.5 时避免对 .bin 调用 torch.load（CVE-2025-32434）；**要求目录内已有 *.safetensors**（仅有 .bin 时不要加此开关）",
    )
    args = ap.parse_args()

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise SystemExit("需要 CUDA")

    dtype = getattr(torch, args.dtype)
    disable_torch_init()
    model_path = validate_pretrained_local_or_hub(args.model_path)
    if args.use_safetensors:
        validate_safetensors_weights_present(model_path)
    validate_torch_version_for_bin_weights(model_path, args.use_safetensors)

    tok = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    load_kw = dict(
        low_cpu_mem_usage=True,
        torch_dtype=dtype,
        use_cache=True,
    )
    if args.use_safetensors:
        load_kw["use_safetensors"] = True
    model = PointLLMLlamaForCausalLM.from_pretrained(model_path, **load_kw).to(device)
    model.initialize_tokenizer_point_backbone_config_wo_embedding(tok)
    model.eval()

    input_ids, attention_mask = build_prompt_ids(tok, model, device)
    point_clouds = torch.randn(1, 8192, 6, device=device, dtype=dtype)

    try:
        backend = build_decode_stack(
            args.decode_stack,
            compile_mode=args.compile_mode,
            compile_fullgraph=args.compile_fullgraph,
            compile_allow_triton_cudagraphs=args.compile_allow_triton_cudagraphs,
        )
    except ValueError as e:
        raise SystemExit(str(e)) from e

    for _ in range(args.warmup):
        bench_once(model, backend, input_ids, attention_mask, point_clouds, args.decode_steps, device)

    agg_prefill: list[float] = []
    agg_decode_mean: list[float] = []
    agg_decode_total: list[float] = []
    last_detail: dict = {}

    for _ in range(args.runs):
        last_detail = bench_once(
            model, backend, input_ids, attention_mask, point_clouds, args.decode_steps, device
        )
        agg_prefill.append(last_detail["prefill_ms"])
        agg_decode_mean.append(last_detail["decode_mean_ms"])
        agg_decode_total.append(last_detail["decode_total_ms"])

    summary = {
        "engine": "nanopointllm",
        "decode_stack": args.decode_stack,
        "compile_mode": args.compile_mode if "compile" in args.decode_stack else None,
        "compile_fullgraph": bool(args.compile_fullgraph) if "compile" in args.decode_stack else None,
        "compile_allow_triton_cudagraphs": bool(args.compile_allow_triton_cudagraphs)
        if "compile" in args.decode_stack
        else None,
        "model_path": model_path,
        "device": str(device),
        "dtype": args.dtype,
        "prompt_len": last_detail["prompt_len"],
        "decode_steps": args.decode_steps,
        "warmup": args.warmup,
        "runs": args.runs,
        "prefill_ms_mean": sum(agg_prefill) / len(agg_prefill),
        "decode_mean_per_step_ms": sum(agg_decode_mean) / len(agg_decode_mean),
        "decode_total_ms_mean": sum(agg_decode_total) / len(agg_decode_total),
        "last_run_decode_step_ms": last_detail["decode_step_ms"],
        "note_m3": "经 hf_prefill + decode_stack（DecodeBackend.decode_step）；与 phase1 指标名对齐。首含 compile 的跑次含编译预热，可适当增大 --warmup。",
    }

    print("=== nano-pointllm M3 decode bench (PointLLM) ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if args.out_json:
        out_dir = os.path.dirname(os.path.abspath(args.out_json))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"[ok] wrote {args.out_json}")


if __name__ == "__main__":
    main()
