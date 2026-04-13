# P4 Integration Testing Design

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Verify correctness and measure throughput of the new `PointLLMLLMEngine` (multi-batch inference) against real `PointLLM_7B_v1.2` weights.

**Architecture:** Two standalone scripts (`parity_engine_pointllm.py`, `bench_engine_pointllm.py`) backed by a shared helper module (`engine_test_utils.py`). Parity verifies exact greedy token match between `PointLLMLLMEngine.generate()` and individual HF `manual_greedy_decode` calls. Bench measures `decode_mean_per_step_ms` and `tokens_per_sec` across batch sizes [1, 2, 4, 8], with an inline HF B=1 baseline for comparison.

**Tech Stack:** Python 3.10+, PyTorch ≥ 2.0, transformers ≥ 4.38, PointLLM (`/home/PointLLM`), pytest (not required for scripts), `PointLLM_7B_v1.2` weights at `/mnt/llm_data/pointllm_ckpt/PointLLM_7B_v1.2`

---

## File Structure

| File | Role |
|------|------|
| `nanopointllm/parity/engine_test_utils.py` | **NEW** Shared helpers: model load, prompt builder, HF greedy reference, synthetic point cloud fixture |
| `scripts/parity_engine_pointllm.py` | **NEW** Correctness: 4 parity cases vs HF greedy, exits 1 on mismatch |
| `scripts/bench_engine_pointllm.py` | **NEW** Throughput: batch sizes [1,2,4,8], decode latency + tokens/sec, HF B=1 baseline |

---

## Section 1: Shared Helper (`engine_test_utils.py`)

Four utilities used by both scripts:

**`load_pointllm_model(model_path, device, dtype)`** → `(model, tokenizer)`
- Same load pattern as existing scripts: `PointLLMLlamaForCausalLM.from_pretrained`, `.initialize_tokenizer_point_backbone_config_wo_embedding`, `.eval()`
- `set_eager_attention_if_supported` applied by default (avoids bf16+SDPA greedy divergence)

**`build_prompt_token_ids(tokenizer, model, question) -> list[int]`**
- Builds `vicuna_v1_1` conversation prompt with `point_patch_token` placeholders (same as existing `build_prompt_ids` but returns `list[int]` instead of tensors, for use with `PointLLMLLMEngine.add_request`)
- `mm_use_point_start_end` respected

**`build_text_only_token_ids(tokenizer, question) -> list[int]`**
- Plain prompt without point patch tokens, for mixed-batch text-only sequences

**`make_fake_point_cloud(device, dtype, N=8192, C=6) -> Tensor`**
- `torch.randn(N, C)` on the given device/dtype — no real 3D data required

**`hf_greedy_generate(model, token_ids, point_clouds, max_new_tokens) -> list[int]`**
- Wraps `manual_greedy_decode` from `nanopointllm.parity.hf_manual_greedy`
- Returns generated token ids (excluding prompt) as `list[int]`

---

## Section 2: Parity Script (`parity_engine_pointllm.py`)

### CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--model_path` | required | Path to `PointLLM_7B_v1.2` |
| `--device` | `cuda:1` | Target device |
| `--dtype` | `bfloat16` | Weight dtype; `float32` for numerical debugging |
| `--max_new_tokens` | `8` | Tokens to generate per sequence |

### Four test cases (run sequentially)

| Case | B | Point clouds | Description |
|------|---|--------------|-------------|
| `single` | 1 | 1 synthetic `[8192,6]` | Engine B=1 matches HF greedy |
| `batch_4` | 4 | 4 synthetic | Each seq independently verified vs HF greedy |
| `batch_8` | 8 | 8 synthetic | Same |
| `mixed_4` | 4 | 2 with point clouds + 2 text-only | Mixed-input path: seq 0,1 have point clouds; seq 2,3 are plain text |

### Per-case logic

```
engine = PointLLMLLMEngine(model, eos_token_id, max_num_seqs=B, max_num_batched_tokens=4096)
seqs = engine.generate(requests)
for i, seq in enumerate(seqs):
    generated = seq.token_ids[seq.num_prompt_tokens:]
    ref = hf_greedy_generate(model, requests[i]["token_ids"], requests[i]["point_clouds"], max_new_tokens)
    assert generated == ref  → PASS / FAIL
```

On first mismatch: print case name, sequence index, generated vs reference tokens, exit 1.
On all pass: print `[ok] all parity cases passed` and exit 0.

---

## Section 3: Bench Script (`bench_engine_pointllm.py`)

### CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--model_path` | required | Path to `PointLLM_7B_v1.2` |
| `--device` | `cuda:1` | Target device |
| `--dtype` | `bfloat16` | Weight dtype |
| `--batch_sizes` | `1,2,4,8` | Comma-separated list |
| `--decode_steps` | `32` | Decode steps per run |
| `--warmup` | `2` | Warmup runs per batch size |
| `--runs` | `3` | Timed runs per batch size |
| `--out_json` | `""` | Optional path to save JSON results |

### Measurement approach

For each batch size B:
1. Build B requests (all with synthetic point clouds, same prompt)
2. Warmup: run `engine.generate()` N=warmup times (discard results)
3. Timed: run `engine.generate()` N=runs times, record:
   - `prefill_ms`: wall time for first scheduler step (prefill)
   - `decode_total_ms`: wall time for remaining steps (all decode)
   - Derived: `decode_mean_per_step_ms = decode_total_ms / (decode_steps * B)` *(per-sequence cost)*
   - Derived: `tokens_per_sec = (decode_steps * B) / (decode_total_ms / 1000)`
4. Average across runs

For B=1 only: also run inline HF baseline (same `hf_prefill` + `HFDecodeBackend.decode_step` loop as `bench_m3_decode_pointllm.py`) and include `hf_b1_decode_mean_per_step_ms` in output.

### Output JSON shape

```json
[
  {
    "batch_size": 1,
    "prefill_ms_mean": 95.2,
    "decode_mean_per_step_ms": 38.4,
    "tokens_per_sec": 26.0,
    "hf_b1_decode_mean_per_step_ms": 37.1
  },
  {
    "batch_size": 4,
    "prefill_ms_mean": 180.3,
    "decode_mean_per_step_ms": 12.6,
    "tokens_per_sec": 79.4
  }
]
```

`decode_mean_per_step_ms` is the primary comparison metric (matches existing `bench_m3` naming). `tokens_per_sec` shows aggregate throughput gain.

---

## Error Handling

- Scripts exit with clear message if CUDA unavailable or model path invalid (reuse `validate_pretrained_local_or_hub`)
- Parity: if `PointLLMLLMEngine` raises during any case, print traceback + case name, exit 1
- Bench: if a batch size fails (OOM etc.), log warning, skip that size, continue

---

## Test Assumptions

- `PointLLM_7B_v1.2` weights at `/mnt/llm_data/pointllm_ckpt/PointLLM_7B_v1.2` (or overridden via `--model_path`)
- Single GPU with ≥ 16 GB VRAM (bfloat16 7B model + KV cache for B=8)
- `eager` attention mode required for exact greedy parity in bfloat16 (applied automatically)
- `PointLLMLLMEngine` uses `max_num_seqs=B`, `max_num_batched_tokens=4096` (no paged KV for P4)
