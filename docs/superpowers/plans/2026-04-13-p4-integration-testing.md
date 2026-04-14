# P4 Integration Testing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Write production-quality parity and bench scripts that verify correctness and measure throughput of `PointLLMLLMEngine` against real `PointLLM_7B_v1.2` weights.

**Architecture:** Shared helper module `engine_test_utils.py` (model loading, prompt building, HF greedy reference, synthetic point clouds) is imported by two standalone scripts: `parity_engine_pointllm.py` (4 correctness cases, exits 1 on mismatch) and `bench_engine_pointllm.py` (batch sizes [1,2,4,8], decode latency + tokens/sec, HF B=1 baseline, JSON output). Scripts require real GPU + model; no unit tests — the scripts themselves are the integration tests.

**Tech Stack:** Python 3.10+, PyTorch ≥ 2.0, transformers ≥ 4.38, HF PointLLM (`/home/PointLLM`), `PointLLM_7B_v1.2` at `/mnt/llm_data/pointllm_ckpt/PointLLM_7B_v1.2`

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `nanopointllm/parity/engine_test_utils.py` | **Create** | Shared helpers for both scripts |
| `scripts/parity_engine_pointllm.py` | **Create** | Correctness: 4 parity cases vs HF greedy |
| `scripts/bench_engine_pointllm.py` | **Create** | Throughput: decode latency + tokens/sec by batch size |

---

## Task 1: Shared helper module (`engine_test_utils.py`)

**Files:**
- Create: `nanopointllm/parity/engine_test_utils.py`

- [ ] **Step 1: Write the file**

```python
# nanopointllm/parity/engine_test_utils.py
"""
Shared helpers for P4 integration scripts (parity + bench).
Requires real GPU + PointLLM_7B_v1.2 weights; not imported in unit tests.
"""
from __future__ import annotations

import os
import sys
from typing import Optional

import torch

_POINTLLM = os.environ.get("POINTLLM_ROOT", "/home/PointLLM")
if _POINTLLM not in sys.path:
    sys.path.insert(0, _POINTLLM)


def load_pointllm_model(
    model_path: str,
    device: torch.device,
    dtype: torch.dtype,
):
    """
    Load PointLLMLlamaForCausalLM + tokenizer, apply eager attention,
    return (model, tokenizer) ready for inference.
    """
    from pointllm.model import PointLLMLlamaForCausalLM
    from transformers import AutoTokenizer

    from nanopointllm.parity.attn_stability import set_eager_attention_if_supported
    from nanopointllm.parity.model_path import validate_pretrained_local_or_hub

    path = validate_pretrained_local_or_hub(model_path)
    tok = AutoTokenizer.from_pretrained(path, use_fast=False)
    model = PointLLMLlamaForCausalLM.from_pretrained(
        path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        use_cache=True,
    ).to(device)
    model.initialize_tokenizer_point_backbone_config_wo_embedding(tok)
    model.eval()
    set_eager_attention_if_supported(model)
    return model, tok


def build_prompt_token_ids(tokenizer, model, question: str = "What is this?") -> list[int]:
    """
    Build vicuna_v1_1 prompt with point_patch_token placeholders.
    Returns token ids as list[int] (for PointLLMLLMEngine.add_request).
    """
    from pointllm.conversation import conv_templates

    conv = conv_templates["vicuna_v1_1"].copy()
    cfg = model.get_model().point_backbone_config
    pt_len = cfg["point_token_len"]
    patch = cfg["default_point_patch_token"]
    if cfg["mm_use_point_start_end"]:
        s = cfg["default_point_start_token"]
        e = cfg["default_point_end_token"]
        qs = s + patch * pt_len + e + "\n" + question
    else:
        qs = patch * pt_len + "\n" + question
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    enc = tokenizer([conv.get_prompt()], return_tensors="pt")
    return enc["input_ids"][0].tolist()


def build_text_only_token_ids(tokenizer, question: str = "Describe this object.") -> list[int]:
    """Plain vicuna prompt with no point_patch_token placeholders (text-only sequences)."""
    from pointllm.conversation import conv_templates

    conv = conv_templates["vicuna_v1_1"].copy()
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    enc = tokenizer([conv.get_prompt()], return_tensors="pt")
    return enc["input_ids"][0].tolist()


def make_fake_point_cloud(
    device: torch.device,
    dtype: torch.dtype,
    N: int = 8192,
    C: int = 6,
) -> torch.Tensor:
    """Synthetic [N, C] point cloud tensor. No real 3D data required."""
    return torch.randn(N, C, device=device, dtype=dtype)


def hf_greedy_generate(
    model,
    token_ids: list[int],
    point_clouds: Optional[torch.Tensor],
    max_new_tokens: int,
    device: torch.device,
) -> list[int]:
    """
    Run HF manual greedy decode. Returns generated token ids only (excludes prompt).
    point_clouds: [N, C] tensor or None for text-only sequences.
    """
    from nanopointllm.parity.hf_manual_greedy import manual_greedy_decode

    ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    mask = torch.ones_like(ids)
    pc = None
    if point_clouds is not None:
        pc = point_clouds.unsqueeze(0) if point_clouds.dim() == 2 else point_clouds

    out = manual_greedy_decode(model, ids, mask, max_new_tokens, point_clouds=pc)
    # out shape: [1, prompt_len + max_new_tokens]
    return out[0, len(token_ids):].tolist()
```

- [ ] **Step 2: Verify the module imports cleanly (no GPU required)**

```bash
cd /home/nano-pointllm
PYTHONPATH=/home/PointLLM:. python -c "from nanopointllm.parity.engine_test_utils import load_pointllm_model, build_prompt_token_ids, build_text_only_token_ids, make_fake_point_cloud, hf_greedy_generate; print('imports ok')"
```

Expected output:
```
imports ok
```

- [ ] **Step 3: Commit**

```bash
git add nanopointllm/parity/engine_test_utils.py
git commit -m "feat(p4): add engine_test_utils shared helpers for parity and bench scripts"
```

---

## Task 2: Parity script (`parity_engine_pointllm.py`)

**Files:**
- Create: `scripts/parity_engine_pointllm.py`

**Context:** Four cases run sequentially. For each, `PointLLMLLMEngine.generate()` is called for a batch, then each sequence's generated tokens are compared to `hf_greedy_generate` called individually. The script exits 0 only when all cases pass.

- [ ] **Step 1: Write the script**

```python
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
    all_point_clouds: list[torch.Tensor | None],
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
    print(f"Loading model {args.model_path} on {device} ({args.dtype})…")
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
```

- [ ] **Step 2: Run the parity script against real weights**

```bash
cd /home/nano-pointllm
PYTHONPATH=/home/PointLLM:. python scripts/parity_engine_pointllm.py \
  --model_path /mnt/llm_data/pointllm_ckpt/PointLLM_7B_v1.2 \
  --device cuda:1 --dtype bfloat16 --max_new_tokens 8
```

Expected output (all four cases):
```
=== PARITY ===
  single  (B=1, 1 point cloud): PASS
  batch_4 (B=4, 4 point clouds): PASS
  batch_8 (B=8, 8 point clouds): PASS
  mixed_4 (B=4, 2 pc + 2 text-only): PASS

[ok] all parity cases passed.
```

If `batch_8` fails with mismatch (rare bf16 divergence), re-run with `--dtype float32`. If it still fails, investigate before proceeding.

- [ ] **Step 3: Commit**

```bash
git add scripts/parity_engine_pointllm.py
git commit -m "feat(p4): add parity_engine_pointllm script (4 correctness cases vs HF greedy)"
```

---

## Task 3: Bench script (`bench_engine_pointllm.py`)

**Files:**
- Create: `scripts/bench_engine_pointllm.py`

**Context:** For each batch size, the engine is run with `max_tokens=decode_steps` to ensure all sequences decode exactly `decode_steps` tokens (no early EOS on synthetic prompts). Prefill and decode times are separated by stepping the scheduler manually. HF B=1 baseline is measured inline for direct comparison. Results are printed as a table and optionally saved to JSON.

- [ ] **Step 1: Write the script**

```python
#!/usr/bin/env python3
"""
P4 bench: PointLLMLLMEngine throughput across batch sizes.

Usage:
  cd /home/nano-pointllm
  PYTHONPATH=/home/PointLLM:. python scripts/bench_engine_pointllm.py \
    --model_path /mnt/llm_data/pointllm_ckpt/PointLLM_7B_v1.2 \
    --device cuda:1 --batch_sizes 1,2,4,8 --decode_steps 32 \
    --warmup 2 --runs 3 --out_json results/p4_bench.json

Metrics:
  decode_mean_per_step_ms  — wall time per (step × sequence), matches bench_m3 naming
  tokens_per_sec           — aggregate tokens generated per second across all sequences
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

from nanopointllm.engine.decode_backend import HFDecodeBackend
from nanopointllm.engine.hf_hybrid import hf_prefill
from nanopointllm.engine.llm_engine import PointLLMLLMEngine
from nanopointllm.engine.types import DecodeStepInput
from nanopointllm.parity.engine_test_utils import (
    build_prompt_token_ids,
    load_pointllm_model,
    make_fake_point_cloud,
)
from nanopointllm.parity.model_path import validate_pretrained_local_or_hub
from nanopointllm.sampling_params import SamplingParams


# ── HF B=1 baseline ─────────────────────────────────────────────────────────

@torch.inference_mode()
def bench_hf_baseline(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    point_clouds: torch.Tensor,
    decode_steps: int,
    warmup: int,
    runs: int,
) -> dict:
    """Single-sequence HF prefill + HFDecodeBackend decode loop."""
    backend = HFDecodeBackend()
    pc = point_clouds.unsqueeze(0)  # [1, N, C]

    def _run() -> tuple[float, float]:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        po = hf_prefill(model, input_ids, attention_mask, point_clouds=pc)
        torch.cuda.synchronize()
        prefill_ms = (time.perf_counter() - t0) * 1000

        nid = po.logits[:, -1, :].float().argmax(-1, keepdim=True)
        mask = attention_mask
        past = po.past_key_values
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        for _ in range(decode_steps):
            mask = torch.cat([mask, mask.new_ones((1, 1))], dim=-1)
            logits, past = backend.decode_step(
                model,
                DecodeStepInput(input_ids=nid, attention_mask=mask, past_key_values=past),
            )
            nid = logits[:, -1, :].float().argmax(-1, keepdim=True)
        torch.cuda.synchronize()
        decode_ms = (time.perf_counter() - t1) * 1000
        return prefill_ms, decode_ms

    for _ in range(warmup):
        _run()

    pf_list, dc_list = zip(*[_run() for _ in range(runs)])
    pf_mean = sum(pf_list) / runs
    dc_mean = sum(dc_list) / runs
    return {
        "prefill_ms_mean": round(pf_mean, 2),
        "decode_mean_per_step_ms": round(dc_mean / decode_steps, 3),
        "tokens_per_sec": round(decode_steps / (dc_mean / 1000), 1),
    }


# ── Engine bench ─────────────────────────────────────────────────────────────

@torch.inference_mode()
def bench_engine_one_b(
    model,
    eos_token_id: int,
    token_ids: list[int],
    device: torch.device,
    dtype: torch.dtype,
    B: int,
    decode_steps: int,
    warmup: int,
    runs: int,
) -> dict:
    """Bench PointLLMLLMEngine at batch size B. decode_steps controls max_tokens."""
    sp = SamplingParams(max_tokens=decode_steps, ignore_eos=True)
    pcs = [make_fake_point_cloud(device, dtype) for _ in range(B)]
    requests = [
        {"token_ids": token_ids, "point_clouds": pc, "sampling_params": sp}
        for pc in pcs
    ]

    def _run() -> tuple[float, float, int]:
        engine = PointLLMLLMEngine(
            model,
            eos_token_id=eos_token_id,
            max_num_seqs=B,
            max_num_batched_tokens=4096,
        )
        for req in requests:
            engine.add_request(**req)

        # First step = prefill
        torch.cuda.synchronize()
        t_start = time.perf_counter()
        engine.step()
        torch.cuda.synchronize()
        t_prefill = time.perf_counter()

        # Remaining steps = decode
        actual_decode_steps = 0
        while not engine.is_finished():
            engine.step()
            actual_decode_steps += 1
        torch.cuda.synchronize()
        t_end = time.perf_counter()

        return (
            (t_prefill - t_start) * 1000,
            (t_end - t_prefill) * 1000,
            actual_decode_steps,
        )

    for _ in range(warmup):
        _run()

    pf_list, dc_list, steps_list = zip(*[_run() for _ in range(runs)])
    pf_mean = sum(pf_list) / runs
    dc_mean = sum(dc_list) / runs
    # Use actual decode steps from last run (should equal decode_steps with ignore_eos=True)
    actual_steps = steps_list[-1]

    decode_mean_per_step_ms = dc_mean / (actual_steps * B) if actual_steps > 0 else 0.0
    tokens_per_sec = (actual_steps * B) / (dc_mean / 1000) if dc_mean > 0 else 0.0

    return {
        "batch_size": B,
        "prefill_ms_mean": round(pf_mean, 2),
        "decode_mean_per_step_ms": round(decode_mean_per_step_ms, 3),
        "tokens_per_sec": round(tokens_per_sec, 1),
        "actual_decode_steps": actual_steps,
    }


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="P4 bench: PointLLMLLMEngine throughput")
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--batch_sizes", default="1,2,4,8")
    ap.add_argument("--decode_steps", type=int, default=32)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--out_json", default="")
    args = ap.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        sys.exit("CUDA not available")

    dtype = getattr(torch, args.dtype)
    batch_sizes = [int(x.strip()) for x in args.batch_sizes.split(",") if x.strip()]

    print(f"Loading model {args.model_path} on {device} ({args.dtype})…")
    model, tok = load_pointllm_model(args.model_path, device, dtype)
    eos = tok.eos_token_id
    pt_ids = build_prompt_token_ids(tok, model)
    print(f"Model loaded. prompt_len={len(pt_ids)}\n")

    # HF B=1 baseline
    ids_1 = torch.tensor([pt_ids], dtype=torch.long, device=device)
    mask_1 = torch.ones_like(ids_1)
    pc_1 = make_fake_point_cloud(device, dtype)
    print("Running HF B=1 baseline…")
    hf_result = bench_hf_baseline(model, ids_1, mask_1, pc_1,
                                   args.decode_steps, args.warmup, args.runs)
    print(f"  HF B=1  prefill={hf_result['prefill_ms_mean']:.1f}ms  "
          f"decode/step={hf_result['decode_mean_per_step_ms']:.3f}ms  "
          f"tps={hf_result['tokens_per_sec']:.1f}\n")

    # Engine bench per batch size
    print(f"{'B':>4}  {'prefill_ms':>10}  {'dec/step/seq_ms':>15}  {'tps':>8}")
    print("─" * 46)
    all_results = []
    for B in batch_sizes:
        try:
            r = bench_engine_one_b(
                model, eos, pt_ids, device, dtype, B,
                args.decode_steps, args.warmup, args.runs,
            )
            if B == 1:
                r["hf_b1_decode_mean_per_step_ms"] = hf_result["decode_mean_per_step_ms"]
                r["hf_b1_tokens_per_sec"] = hf_result["tokens_per_sec"]
            speedup = r["tokens_per_sec"] / hf_result["tokens_per_sec"] if hf_result["tokens_per_sec"] else 0
            print(f"  B={B:<2}  prefill={r['prefill_ms_mean']:7.1f}ms  "
                  f"dec/step={r['decode_mean_per_step_ms']:7.3f}ms  "
                  f"tps={r['tokens_per_sec']:7.1f}  ({speedup:.2f}× vs HF B=1)")
            all_results.append(r)
        except Exception as e:
            print(f"  B={B:<2}  SKIPPED ({e})")

    print()

    if args.out_json:
        out_dir = os.path.dirname(os.path.abspath(args.out_json))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        payload = {"hf_b1_baseline": hf_result, "engine_results": all_results}
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"[ok] results written to {args.out_json}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the bench script**

```bash
cd /home/nano-pointllm
PYTHONPATH=/home/PointLLM:. python scripts/bench_engine_pointllm.py \
  --model_path /mnt/llm_data/pointllm_ckpt/PointLLM_7B_v1.2 \
  --device cuda:1 --batch_sizes 1,2,4,8 --decode_steps 32 \
  --warmup 2 --runs 3 --out_json results/p4_bench.json
```

Expected output shape:
```
   B  prefill_ms  dec/step/seq_ms      tps
──────────────────────────────────────────────
  B=1   prefill= ~115ms  dec/step=~25ms  tps= ~40  (1.00× vs HF B=1)
  B=2   prefill= ~175ms  dec/step=~14ms  tps= ~72  (1.8× vs HF B=1)
  B=4   prefill= ~305ms  dec/step= ~8ms  tps=~130  (3.2× vs HF B=1)
  B=8   prefill= ~495ms  dec/step= ~5ms  tps=~196  (4.9× vs HF B=1)
```

Verify `results/p4_bench.json` was written and contains both `hf_b1_baseline` and `engine_results` keys.

- [ ] **Step 3: Commit**

```bash
git add scripts/bench_engine_pointllm.py results/p4_bench.json
git commit -m "feat(p4): add bench_engine_pointllm script and initial bench results"
```

Note: if `results/` is gitignored, commit only the script:
```bash
git add scripts/bench_engine_pointllm.py
git commit -m "feat(p4): add bench_engine_pointllm script (batch decode latency + tps)"
```

---

## Environment setup (for reference)

All scripts require:
```bash
conda activate pointllm
cd /home/nano-pointllm
export PYTHONPATH=/home/PointLLM:$PYTHONPATH
```

Model path: `/mnt/llm_data/pointllm_ckpt/PointLLM_7B_v1.2`
Recommended device: `cuda:1` (GPU 1 is typically free based on bench history)
