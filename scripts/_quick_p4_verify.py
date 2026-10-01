#!/usr/bin/env python3
"""
临时 P4 快速验证：parity (B=1,4) + bench (B=1,4,8)
用法：
  cd /home/nano-pointllm
  PYTHONPATH=/home/PointLLM:. python scripts/_quick_p4_verify.py \
    --model_path /mnt/llm_data/pointllm_ckpt/PointLLM_7B_v1.2 \
    --device cuda:1
"""
from __future__ import annotations
import argparse, sys, time, os
import torch

_POINTLLM = os.environ.get("POINTLLM_ROOT", "/home/PointLLM")
if _POINTLLM not in sys.path:
    sys.path.insert(0, _POINTLLM)

from transformers import AutoTokenizer
from pointllm.model import PointLLMLlamaForCausalLM
from pointllm.conversation import conv_templates

from nanopointllm.parity.attn_stability import set_eager_attention_if_supported
from nanopointllm.parity.hf_manual_greedy import manual_greedy_decode
from nanopointllm.parity.model_path import validate_pretrained_local_or_hub
from nanopointllm.engine.llm_engine import PointLLMLLMEngine
from nanopointllm.sampling_params import SamplingParams


# ── helpers ──────────────────────────────────────────────────────────────────

def load_model(model_path, device, dtype):
    tok = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = PointLLMLlamaForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype, low_cpu_mem_usage=True, use_cache=True,
    ).to(device)
    model.initialize_tokenizer_point_backbone_config_wo_embedding(tok)
    model.eval()
    set_eager_attention_if_supported(model)
    return model, tok


def build_point_prompt_ids(tok, model, question="What is this?") -> list[int]:
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
    return enc["input_ids"][0].tolist()


def build_text_prompt_ids(tok, question="Describe this.") -> list[int]:
    conv = conv_templates["vicuna_v1_1"].copy()
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    enc = tok([conv.get_prompt()], return_tensors="pt")
    return enc["input_ids"][0].tolist()


def make_pc(device, dtype):
    return torch.randn(8192, 6, device=device, dtype=dtype)


def hf_greedy(model, token_ids, point_clouds, max_new_tokens, device):
    ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    mask = torch.ones_like(ids)
    pc = point_clouds.unsqueeze(0) if point_clouds is not None and point_clouds.dim() == 2 else point_clouds
    out = manual_greedy_decode(model, ids, mask, max_new_tokens, point_clouds=pc)
    return out[0, len(token_ids):].tolist()


# ── parity ───────────────────────────────────────────────────────────────────

def run_parity(model, tok, device, dtype, max_new_tokens=8):
    print("\n=== PARITY ===")
    eos = tok.eos_token_id
    pt_ids = build_point_prompt_ids(tok, model)
    txt_ids = build_text_prompt_ids(tok)

    cases = [
        ("B=1 point", 1, [pt_ids], [make_pc(device, dtype)]),
        ("B=4 point", 4, [pt_ids]*4, [make_pc(device, dtype) for _ in range(4)]),
        ("B=4 mixed (2pc+2txt)", 4,
         [pt_ids, pt_ids, txt_ids, txt_ids],
         [make_pc(device, dtype), make_pc(device, dtype), None, None]),
    ]

    all_ok = True
    for name, B, all_ids, all_pcs in cases:
        sp = SamplingParams(max_tokens=max_new_tokens)
        engine = PointLLMLLMEngine(model, eos_token_id=eos,
                                   max_num_seqs=B, max_num_batched_tokens=4096)
        requests = [
            {"token_ids": ids, "point_clouds": pc, "sampling_params": sp}
            for ids, pc in zip(all_ids, all_pcs)
        ]
        seqs = engine.generate(requests)

        ok = True
        for i, (seq, ids, pc) in enumerate(zip(seqs, all_ids, all_pcs)):
            gen = seq.token_ids[seq.num_prompt_tokens:]
            ref = hf_greedy(model, ids, pc, max_new_tokens, device)
            if gen != ref:
                print(f"  [{name}] seq {i} MISMATCH: engine={gen} ref={ref}")
                ok = False
        status = "PASS" if ok else "FAIL"
        print(f"  {name}: {status}")
        if not ok:
            all_ok = False

    return all_ok


# ── bench ────────────────────────────────────────────────────────────────────

def bench_engine(model, tok, device, dtype, decode_steps=32, warmup=1, runs=2):
    print("\n=== BENCH (PointLLMLLMEngine) ===")
    eos = tok.eos_token_id
    pt_ids = build_point_prompt_ids(tok, model)

    results = []
    for B in [1, 2, 4, 8]:
        sp = SamplingParams(max_tokens=decode_steps)
        pcs = [make_pc(device, dtype) for _ in range(B)]
        requests = [{"token_ids": pt_ids, "point_clouds": pc, "sampling_params": sp}
                    for pc in pcs]

        # warmup
        for _ in range(warmup):
            engine = PointLLMLLMEngine(model, eos_token_id=eos,
                                       max_num_seqs=B, max_num_batched_tokens=4096)
            engine.generate(requests)

        # timed
        run_decode_ms = []
        run_prefill_ms = []
        for _ in range(runs):
            engine = PointLLMLLMEngine(model, eos_token_id=eos,
                                       max_num_seqs=B, max_num_batched_tokens=4096)
            for req in requests:
                engine.add_request(**req)

            torch.cuda.synchronize()
            t_start = time.perf_counter()

            # first step = prefill
            engine.step()
            torch.cuda.synchronize()
            t_prefill = time.perf_counter()

            # remaining steps = decode
            while not engine.is_finished():
                engine.step()
            torch.cuda.synchronize()
            t_end = time.perf_counter()

            run_prefill_ms.append((t_prefill - t_start) * 1000)
            run_decode_ms.append((t_end - t_prefill) * 1000)

        prefill_mean = sum(run_prefill_ms) / len(run_prefill_ms)
        decode_total_mean = sum(run_decode_ms) / len(run_decode_ms)

        # count actual decode steps (may stop at EOS before decode_steps)
        actual_steps = max(
            sum(1 for seq in engine.scheduler.running) +
            len([s for s in [] ]),  # finished seqs' steps
            1
        )
        # simpler: use decode_steps as approximation (EOS unlikely in 32 steps for random prompt)
        decode_steps_actual = decode_steps
        decode_mean_per_step_ms = decode_total_mean / (decode_steps_actual * B)
        tokens_per_sec = (decode_steps_actual * B) / (decode_total_mean / 1000)

        results.append({
            "B": B,
            "prefill_ms": round(prefill_mean, 1),
            "decode_mean_per_step_ms": round(decode_mean_per_step_ms, 2),
            "tokens_per_sec": round(tokens_per_sec, 1),
        })
        print(f"  B={B:2d}  prefill={prefill_mean:7.1f}ms  "
              f"decode_per_step={decode_mean_per_step_ms:6.2f}ms/seq  "
              f"tps={tokens_per_sec:6.1f}")

    return results


def bench_hf_baseline(model, tok, device, dtype, decode_steps=32, warmup=1, runs=2):
    """HF single-seq baseline for comparison."""
    print("\n=== BENCH HF baseline (B=1) ===")
    from nanopointllm.engine.hf_hybrid import hf_prefill
    from nanopointllm.engine.decode_backend import HFDecodeBackend
    from nanopointllm.engine.types import DecodeStepInput

    pt_ids = build_point_prompt_ids(tok, model)
    ids = torch.tensor([pt_ids], dtype=torch.long, device=device)
    mask = torch.ones_like(ids)
    pc = make_pc(device, dtype).unsqueeze(0)
    backend = HFDecodeBackend()

    def _run():
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        po = hf_prefill(model, ids, mask, point_clouds=pc)
        torch.cuda.synchronize()
        prefill_ms = (time.perf_counter() - t0) * 1000

        nid = po.logits[:, -1, :].float().argmax(-1, keepdim=True)
        cur_mask = mask
        past = po.past_key_values
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        for _ in range(decode_steps):
            cur_mask = torch.cat([cur_mask, cur_mask.new_ones((1, 1))], dim=-1)
            logits, past = backend.decode_step(
                model, DecodeStepInput(input_ids=nid, attention_mask=cur_mask, past_key_values=past))
            nid = logits[:, -1, :].float().argmax(-1, keepdim=True)
        torch.cuda.synchronize()
        decode_ms = (time.perf_counter() - t1) * 1000
        return prefill_ms, decode_ms

    for _ in range(warmup):
        _run()

    pf_ms, dc_ms = zip(*[_run() for _ in range(runs)])
    prefill_mean = sum(pf_ms) / len(pf_ms)
    decode_mean = sum(dc_ms) / len(dc_ms)
    per_step = decode_mean / decode_steps
    tps = decode_steps / (decode_mean / 1000)
    print(f"  B= 1  prefill={prefill_mean:7.1f}ms  "
          f"decode_per_step={per_step:6.2f}ms/seq  tps={tps:6.1f}")
    return {"B": 1, "prefill_ms": round(prefill_mean, 1),
            "decode_mean_per_step_ms": round(per_step, 2),
            "tokens_per_sec": round(tps, 1)}


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--max_new_tokens", type=int, default=8)
    ap.add_argument("--decode_steps", type=int, default=32)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--skip_parity", action="store_true")
    ap.add_argument("--skip_bench", action="store_true")
    args = ap.parse_args()

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    model_path = validate_pretrained_local_or_hub(args.model_path)

    print(f"Loading model from {model_path} on {device} ({args.dtype})...")
    model, tok = load_model(model_path, device, dtype)
    print("Model loaded.")

    parity_ok = True
    if not args.skip_parity:
        parity_ok = run_parity(model, tok, device, dtype, args.max_new_tokens)

    if not args.skip_bench:
        hf_base = bench_hf_baseline(model, tok, device, dtype,
                                    args.decode_steps, args.warmup, args.runs)
        engine_results = bench_engine(model, tok, device, dtype,
                                      args.decode_steps, args.warmup, args.runs)

        print("\n=== SPEEDUP SUMMARY (vs HF B=1 decode) ===")
        hf_tps = hf_base["tokens_per_sec"]
        hf_per_step = hf_base["decode_mean_per_step_ms"]
        for r in engine_results:
            tps_ratio = r["tokens_per_sec"] / hf_tps if hf_tps else 0
            latency_ratio = hf_per_step / r["decode_mean_per_step_ms"] if r["decode_mean_per_step_ms"] else 0
            print(f"  B={r['B']:2d}  tokens/sec speedup: {tps_ratio:.2f}×  "
                  f"per-seq-step speedup: {latency_ratio:.2f}×")

    if not parity_ok:
        print("\n[FAIL] parity mismatch detected")
        sys.exit(1)
    print("\n[ok] done")


if __name__ == "__main__":
    main()
