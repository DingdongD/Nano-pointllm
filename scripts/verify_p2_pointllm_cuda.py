#!/usr/bin/env python3
"""
P2 环境验证（单卡 CUDA + PointLLM-7B）：

1. greedy：`manual_greedy_decode` vs `hybrid_greedy_decode`（HF 后端）
2. RoPE：`transformers` LlamaRoPE + decode offset 形状与有限性自检
3. 微 bench：`hf_prefill` 后若干步 `HFDecodeBackend.decode_step` 平均耗时（毫秒）

  conda activate pointllm
  export PYTHONPATH=/home/PointLLM:$PYTHONPATH
  cd /home/nano-pointllm
  python scripts/verify_p2_pointllm_cuda.py \\
    --model_path /mnt/llm_data/pointllm_ckpt/PointLLM_7B_v1.2 \\
    --device cuda:1

  # 若仍报 greedy mismatch：勿加 --parity_keep_sdpa，或试 --dtype float32
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

_POINTLLM = os.environ.get("POINTLLM_ROOT", "/home/PointLLM")
if _POINTLLM not in sys.path:
    sys.path.insert(0, _POINTLLM)

from pointllm.conversation import conv_templates
from pointllm.model import PointLLMLlamaForCausalLM
from transformers import AutoTokenizer

from nanopointllm.engine.decode_backend import HFDecodeBackend, HFInnerDecodeBackend
from nanopointllm.llama.torch_llama_decode_backend import TorchLlamaExplicitLoopDecodeBackend
from nanopointllm.engine.decode_stack import build_decode_stack
from nanopointllm.engine.hf_hybrid import hf_prefill, hybrid_greedy_decode
from nanopointllm.engine.types import DecodeStepInput
from nanopointllm.llama.hf_config import (
    head_dim_from_config,
    max_position_embeddings_from_config,
    rope_theta_from_config,
)
from nanopointllm.llama.rope_sanity import verify_decode_step_rope
from nanopointllm.parity.attn_stability import set_eager_attention_if_supported
from nanopointllm.parity.hf_manual_greedy import manual_greedy_decode
from nanopointllm.parity.model_path import (
    validate_pretrained_local_or_hub,
    validate_safetensors_weights_present,
    validate_torch_version_for_bin_weights,
)


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


@torch.inference_mode()
def micro_bench_decode_steps(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    point_clouds: torch.Tensor,
    decode_steps: int,
) -> float:
    po = hf_prefill(model, input_ids, attention_mask, point_clouds=point_clouds)
    next_id = po.logits[:, -1, :].float().argmax(dim=-1, keepdim=True)
    mask = attention_mask
    past = po.past_key_values
    backend = HFDecodeBackend()
    times: list[float] = []
    for _ in range(decode_steps):
        mask = torch.cat([mask, mask.new_ones((mask.shape[0], 1))], dim=-1)
        inp = DecodeStepInput(
            input_ids=next_id,
            attention_mask=mask,
            past_key_values=past,
        )
        if next_id.device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        logits, past = backend.decode_step(model, inp)
        if next_id.device.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
        next_id = logits[:, -1, :].float().argmax(dim=-1, keepdim=True)
    return sum(times) / len(times)


def main():
    ap = argparse.ArgumentParser(description="P2 PointLLM-7B + CUDA 验证")
    ap.add_argument(
        "--model_path",
        type=str,
        default="/mnt/llm_data/pointllm_ckpt/PointLLM_7B_v1.2",
        help="PointLLM-7B 权重目录",
    )
    ap.add_argument("--device", type=str, default="cuda:1")
    ap.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--max_new_tokens", type=int, default=8)
    ap.add_argument("--decode_micro_steps", type=int, default=8, help="微 bench 的 decode 步数")
    ap.add_argument(
        "--parity_keep_sdpa",
        action="store_true",
        help="greedy parity 不切 eager 注意力（默认切 eager，避免 SDPA 在 prefill vs decode 形状下 bf16 分叉）",
    )
    ap.add_argument(
        "--use_safetensors",
        action="store_true",
        help="from_pretrained(..., use_safetensors=True)；见 transformers≥5.5 + torch<2.6 与 CVE-2025-32434 说明（setup_inference 常见问题表）",
    )
    args = ap.parse_args()

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise SystemExit("本脚本要求单卡 CUDA：请设置 --device cuda:0 或 cuda:1 并确保 torch.cuda.is_available()")

    dtype = getattr(torch, args.dtype)
    model_path = validate_pretrained_local_or_hub(args.model_path)
    if args.use_safetensors:
        validate_safetensors_weights_present(model_path)
    validate_torch_version_for_bin_weights(model_path, args.use_safetensors)

    print("=== P2 verify: PointLLM-7B + CUDA ===")
    print("model_path:", model_path)
    print("device:", device, "dtype:", args.dtype)

    tok = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    load_kw = dict(torch_dtype=dtype, low_cpu_mem_usage=True, use_cache=True)
    if args.use_safetensors:
        load_kw["use_safetensors"] = True
    model = PointLLMLlamaForCausalLM.from_pretrained(model_path, **load_kw).to(device)
    model.initialize_tokenizer_point_backbone_config_wo_embedding(tok)
    model.eval()
    if not args.parity_keep_sdpa:
        set_eager_attention_if_supported(model)

    cfg = model.config
    nh = int(cfg.num_attention_heads)
    hd = head_dim_from_config(cfg)
    rope_theta = rope_theta_from_config(cfg)
    max_pos = max_position_embeddings_from_config(cfg)

    # 1) RoPE 自检（与 HF 实现一致）
    prompt_len = 558  # 与常见 bench 接近；仅用于 cos/sin 长度，不加载真实 prompt
    verify_decode_step_rope(
        device=device,
        num_heads=nh,
        head_dim=hd,
        rope_theta=rope_theta,
        max_position_embeddings=max_pos,
        kv_seq_len=min(prompt_len, max_pos),
        dtype=torch.float32,
    )
    print("[ok] RoPE decode-offset sanity (HF LlamaRotaryEmbedding + apply_rotary_pos_emb)")

    # 2) greedy parity
    input_ids, attention_mask = build_prompt_ids(tok, model, device)
    torch.manual_seed(42)
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
    if not torch.equal(manual, hybrid):
        diff = (manual != hybrid).nonzero(as_tuple=True)
        if diff[0].numel():
            j = int(diff[1][0].item())
            print(
                "first diff col:",
                j,
                "manual:",
                manual[0, j].item(),
                "hybrid:",
                hybrid[0, j].item(),
            )
        raise SystemExit(
            "manual vs hybrid greedy mismatch（可试去掉 --parity_keep_sdpa 或改用 --dtype float32）"
        )
    print("[ok] manual vs hybrid greedy parity")

    # 3) decode 栈 `hf` / `hf_inner` 构造
    _ = build_decode_stack("hf")
    _ = build_decode_stack("hf_inner")
    _ = build_decode_stack("torch_loop")
    print("[ok] build_decode_stack('hf' | 'hf_inner' | 'torch_loop')")

    # 3b) 单步 decode：`HF` 与 `HFInner` logits 对齐（同 prefill 语义）
    torch.manual_seed(42)
    pc = torch.randn(1, 8192, 6, device=device, dtype=dtype)
    po = hf_prefill(model, input_ids, attention_mask, point_clouds=pc)
    next_id = po.logits[:, -1, :].float().argmax(dim=-1, keepdim=True)
    mask = torch.cat([attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1)
    inp = DecodeStepInput(
        input_ids=next_id,
        attention_mask=mask,
        past_key_values=po.past_key_values,
    )
    logits_hf, _ = HFDecodeBackend().decode_step(model, inp)
    torch.manual_seed(42)
    pc2 = torch.randn(1, 8192, 6, device=device, dtype=dtype)
    po2 = hf_prefill(model, input_ids, attention_mask, point_clouds=pc2)
    inp2 = DecodeStepInput(
        input_ids=next_id,
        attention_mask=mask,
        past_key_values=po2.past_key_values,
    )
    logits_inner, _ = HFInnerDecodeBackend().decode_step(model, inp2)
    if logits_hf.shape != logits_inner.shape:
        raise SystemExit(f"hf vs hf_inner logits shape mismatch {logits_hf.shape} vs {logits_inner.shape}")
    if dtype == torch.float32:
        ok_logits = torch.allclose(logits_hf, logits_inner, rtol=0.0, atol=0.0)
    else:
        ok_logits = torch.allclose(logits_hf.float(), logits_inner.float(), rtol=0.05, atol=1e-2)
    if not ok_logits:
        d = (logits_hf.float() - logits_inner.float()).abs().max().item()
        raise SystemExit(f"hf vs hf_inner logits mismatch (max abs diff {d});可试 --dtype float32")
    print("[ok] HFDecodeBackend vs HFInnerDecodeBackend (single decode step logits)")

    # 3c) 显式 torch_loop vs HFInner（单步 logits）
    torch.manual_seed(42)
    pc_tl = torch.randn(1, 8192, 6, device=device, dtype=dtype)
    po_tl = hf_prefill(model, input_ids, attention_mask, point_clouds=pc_tl)
    next_tl = po_tl.logits[:, -1, :].float().argmax(dim=-1, keepdim=True)
    mask_tl = torch.cat([attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1)
    inp_tl = DecodeStepInput(
        input_ids=next_tl,
        attention_mask=mask_tl,
        past_key_values=po_tl.past_key_values,
    )
    logits_inner_tl, _ = HFInnerDecodeBackend().decode_step(model, inp_tl)
    torch.manual_seed(42)
    pc_tl2 = torch.randn(1, 8192, 6, device=device, dtype=dtype)
    po_tl2 = hf_prefill(model, input_ids, attention_mask, point_clouds=pc_tl2)
    inp_tl2 = DecodeStepInput(
        input_ids=next_tl,
        attention_mask=mask_tl,
        past_key_values=po_tl2.past_key_values,
    )
    try:
        logits_loop, _ = TorchLlamaExplicitLoopDecodeBackend().decode_step(model, inp_tl2)
    except Exception as e:
        raise SystemExit(f"torch_loop decode_step 失败（需较新 transformers + masking_utils）: {e}") from e
    if dtype == torch.float32:
        ok_tl = torch.allclose(logits_inner_tl, logits_loop, rtol=0.0, atol=0.0)
    else:
        ok_tl = torch.allclose(logits_inner_tl.float(), logits_loop.float(), rtol=0.05, atol=1e-2)
    if not ok_tl:
        d = (logits_inner_tl.float() - logits_loop.float()).abs().max().item()
        raise SystemExit(f"HFInner vs torch_loop logits mismatch (max abs diff {d})")
    print("[ok] HFInnerDecodeBackend vs TorchLlamaExplicitLoopDecodeBackend (single decode step)")

    # 4) 微 bench
    mean_ms = micro_bench_decode_steps(
        model,
        input_ids,
        attention_mask,
        point_clouds,
        args.decode_micro_steps,
    )
    print(f"[ok] HF decode_step mean over {args.decode_micro_steps} steps: {mean_ms:.3f} ms")

    print("=== P2 verify: ALL PASSED ===")


if __name__ == "__main__":
    main()
