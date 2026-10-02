# Progress Log

## Session: 2026-04-16

### Phase 17: Point Encoder Reuse
- **Status:** complete
- Actions taken:
  - Added `point_features_cached`, `point_cloud_cache_key`, and `input_embed_layout_cached` to `PointLLMSequence`.
  - Extended `PointLLMWrapper` with value-stable point-cloud hashing, precomputed input splice layouts, and `prepare_inputs_embeds(..., layouts=...)`.
  - Wired both `PointLLMModelRunner` and `PagedModelRunner` to reuse projected point features across repeated requests inside a persistent runner.
  - Added parity tests for wrapper layout reuse and runner point-feature cache reuse.
  - Added `scripts/bench_point_encoder_cache.py` to measure repeated-point-cloud prefill reuse.
  - Fixed a real benchmark failure where `bfloat16` point clouds could not be hashed through `numpy()`.
- Files created/modified:
  - `nanopointllm/engine/sequence.py`
  - `nanopointllm/models/pointllm_wrapper.py`
  - `nanopointllm/engine/model_runner.py`
  - `nanopointllm/engine/paged_model_runner.py`
  - `tests/test_pointllm_wrapper.py`
  - `tests/test_model_runner.py`
  - `scripts/bench_point_encoder_cache.py`
  - `results/bench_point_encoder_cache_b4_paged.json`
  - `task_plan.md`
  - `findings.md`
  - `progress.md`

### Phase 18: Engine-Level Point Feature Cache
- **Status:** complete
- Actions taken:
  - Added explicit `PointFeatureCache` subsystem with LRU semantics and stats.
  - Changed padded and paged runners to accept an injected shared point-feature cache instead of silently owning unrelated local caches.
  - Promoted the cache to `PointLLMLLMEngine`, so one engine/session now shares point features across repeated requests.
  - Extended `PointLLMLLMEngine.add_request(...)` and `LLM.generate(...)` request dicts with `point_cloud_cache_key` and `point_features_cached`.
  - Added engine helpers for cache stats and cache clearing.
  - Added tests for cache subsystem behavior, shared engine wiring, and API passthrough.
- Files created/modified:
  - `nanopointllm/engine/point_feature_cache.py`
  - `nanopointllm/engine/model_runner.py`
  - `nanopointllm/engine/paged_model_runner.py`
  - `nanopointllm/engine/llm_engine.py`
  - `nanopointllm/engine/sequence.py`
  - `nanopointllm/llm.py`
  - `tests/test_point_feature_cache.py`
  - `tests/test_llm_engine_point_cache.py`
  - `tests/test_llm_api.py`
  - `task_plan.md`
  - `findings.md`
  - `progress.md`

### Phase 19: PointBERT Backbone Hotspot Profiling
- **Status:** complete
- Actions taken:
  - Added `scripts/profile_pointbert_backbone.py` to split PointBERT backbone execution into grouping, local encoder, bridge, per-transformer-block, and tail stages.
  - Ran real PointLLM-7B profiling on A100 with `B=1, warmup=1, runs=3`.
  - Ran the same profiler at `B=4` to confirm hotspot ordering under small batching.
  - Summed transformer-block attention/MLP totals to compare backbone-internal cost centers directly.
- Files created/modified:
  - `scripts/profile_pointbert_backbone.py`
  - `results/profile_pointbert_backbone_b1.json`
  - `results/profile_pointbert_backbone_b1_w1_r3.json`
  - `results/profile_pointbert_backbone_b4_w1_r3.json`
  - `task_plan.md`
  - `findings.md`
  - `progress.md`

### Phase 7: Decode-Only Paged Attention Kernel
- **Status:** complete
- Actions taken:
  - Inspected `PagedLlamaAttention` and confirmed the q_len=1 path can bypass per-sequence SDPA.
  - Chose a batched Triton kernel over `(sequence, query_head)` with online softmax.
  - Implemented `paged_decode_attention()` with CUDA Triton fast path and PyTorch fallback.
  - Routed `PagedLlamaAttention.forward()` to the fast path when every `seq_len` is 1.
  - Added CUDA GQA parity test against direct SDPA.
  - Ran real PointLLM-7B B=8 and B=1,2,4,8 benchmarks.
- Files created/modified:
  - `task_plan.md`
  - `findings.md`
  - `progress.md`
  - `nanopointllm/llama/paged_attention.py`
  - `tests/test_paged_attention.py`
  - `results/bench_paged_b8_d32_triton_decode.json`
  - `results/bench_paged_b1_2_4_8_d32_triton_decode.json`

### Phase 8: Mixed Decode Fast Path And Runtime Gaps
- **Status:** complete
- Actions taken:
  - Extended `PagedLlamaAttention.forward()` to use the Triton decode kernel for q_len=1 rows inside mixed prefill+decode batches.
  - Added mixed prefill+decode parity test.
  - Added `GreedySampler` with lazy CUDA `torch.compile` and routed `PointLLMModelRunner` and `PagedModelRunner` through it.
  - Confirmed `flash_attn` is not installed in `/opt/conda/envs/pointllm`.
  - Ran full tests and a continuous batching benchmark.
- Files created/modified:
  - `nanopointllm/llama/paged_attention.py`
  - `nanopointllm/engine/sampler.py`
  - `nanopointllm/engine/model_runner.py`
  - `nanopointllm/engine/paged_model_runner.py`
  - `tests/test_paged_attention.py`
  - `tests/test_sampler.py`
  - `results/bench_continuous_w4_d32_mixed_triton.json`

### Phase 9: Prefill Varlen Paged Attention Kernel
- **Status:** complete
- Actions taken:
  - Implemented q_len>1 paged varlen Triton attention kernel.
  - Added CUDA parity test covering GQA, causal prefill, and partial-prefix suffix prefill.
  - Routed CUDA q_len>1 rows in `PagedLlamaAttention.forward()` through the new varlen kernel while keeping SDPA fallback for CPU.
  - Ran full tests and PointLLM-7B benchmarks.
- Files created/modified:
  - `nanopointllm/llama/paged_attention.py`
  - `tests/test_paged_attention.py`
  - `results/bench_paged_b8_d32_varlen_prefill.json`
  - `results/bench_paged_b1_2_4_8_d32_varlen_prefill.json`

### Phase 10: Kernel Config Tuning
- **Status:** in_progress
- Actions taken:
  - Started head_dim=128 BLOCK_M/BLOCK_N tuning work.
  - Chose static config selection plus environment overrides rather than runtime search in generation.
- Files created/modified:
  - `task_plan.md`
  - `findings.md`
  - `progress.md`

### Phase 6: CUDA Benchmark
- **Status:** complete
- Actions taken:
  - Confirmed 4 idle NVIDIA A100-SXM4-40GB GPUs.
  - Found PointLLM checkpoints under `/mnt/llm_data/pointllm_ckpt`.
  - Selected existing benchmark scripts rather than creating a separate benchmark harness.
  - Default base Python could not initialize CUDA because torch was built for CUDA 13.0 while the driver is CUDA 12.2-era.
  - Switched to `/opt/conda/envs/pointllm/bin/python` with torch 2.5.1+cu121; CUDA worked.
  - Ran B=1 smoke benchmark.
  - Ran padded engine benchmark for B=1,2,4,8.
  - Ran padded vs paged benchmark for B=1,2,4,8 and continuous batching total=8/window=4.
- Files created/modified:
  - `task_plan.md`
  - `progress.md`
  - `results/bench_smoke_engine_b1.json`
  - `results/bench_engine_b1_2_4_8_d32.json`
  - `results/bench_paged_b1_2_4_8_d32.json`

### Phase 1: Discovery
- **Status:** complete
- **Started:** 2026-04-16
- Actions taken:
  - Compared the existing framework shape in `nano-pointllm` with `nano-vllm`.
  - Created persistent planning files for this multi-step change.
  - Confirmed a paged KV append off-by-one mismatch against `nano-vllm`.
- Files created/modified:
  - `task_plan.md`
  - `findings.md`
  - `progress.md`

### Phase 2-4: Design, Implementation, Verification
- **Status:** complete
- Actions taken:
  - Fixed paged KV append block-boundary semantics.
  - Added paged scheduler preemption on KV block exhaustion.
  - Made KV pool allocation GQA-aware via `num_key_value_heads`.
  - Added partial-prefix prefill mask support in paged attention.
  - Routed paged prefill through flattened varlen `run_mixed`.
  - Added `nanopointllm.LLM` top-level API wrapper.
  - Added tests for block boundary allocation, scheduler preemption, GQA KV pool shape, partial-prefix paged attention, and `LLM.generate`.
- Files created/modified:
  - `nanopointllm/engine/block_manager.py`
  - `nanopointllm/engine/scheduler.py`
  - `nanopointllm/engine/kv_pool.py`
  - `nanopointllm/engine/paged_model_runner.py`
  - `nanopointllm/llama/paged_attention.py`
  - `nanopointllm/llm.py`
  - `nanopointllm/__init__.py`
  - `tests/test_block_manager.py`
  - `tests/test_scheduler.py`
  - `tests/test_kv_pool.py`
  - `tests/test_paged_attention.py`
  - `tests/test_llm_api.py`

## Test Results
| Test | Input | Expected | Actual | Status |
|------|-------|----------|--------|--------|
| Paged framework subset | `python3 -m pytest -q tests/test_block_manager.py tests/test_scheduler.py tests/test_continuous_batching.py tests/test_varlen_paged.py tests/test_paged_attention.py` | Pass | `33 passed in 2.42s` | pass |
| Expanded framework subset | `python3 -m pytest -q tests/test_llm_api.py tests/test_kv_pool.py tests/test_paged_attention.py tests/test_block_manager.py tests/test_scheduler.py tests/test_continuous_batching.py tests/test_varlen_paged.py` | Pass | `39 passed in 2.58s` | pass |
| Full suite | `python3 -m pytest -q` | Pass | `78 passed, 1 skipped in 2.71s` | pass |
| CUDA env smoke | `/opt/conda/envs/pointllm/bin/python -c 'import torch; print(torch.cuda.is_available())'` | CUDA available | `True`, A100 visible | pass |
| B=1 smoke bench | `bench_engine_pointllm.py --batch_sizes 1 --decode_steps 4 --warmup 0 --runs 1` | Completes on PointLLM-7B | Engine B=1 `36.5 tok/s` for 4-step smoke | pass |
| Padded batch sweep | `bench_engine_pointllm.py --batch_sizes 1,2,4,8 --decode_steps 32 --warmup 1 --runs 3` | Completes | B=1/2/4/8 TPS: `23.6`, `57.0`, `137.5`, `205.3` | pass |
| Paged vs padded sweep | `bench_paged_pointllm.py --batch_sizes 1,2,4,8 --decode_steps 32 --warmup 1 --runs 3 --continuous_total 8 --continuous_window 4` | Completes | Paged TPS B=1/2/4/8: `33.3`, `60.2`, `94.1`, `127.2`; continuous TPS `76.1` | pass |
| Triton decode CUDA parity | `pytest -q tests/test_paged_attention.py::test_decode_only_triton_kernel_parity_gqa_cuda` | Pass | `1 passed in 4.93s` | pass |
| Triton related tests | `pytest -q tests/test_paged_attention.py tests/test_kv_pool.py tests/test_varlen_paged.py tests/test_block_manager.py tests/test_scheduler.py tests/test_continuous_batching.py` | Pass | `38 passed in 4.34s` | pass |
| Full suite after Triton kernel | `pytest -q` | Pass | `79 passed, 1 skipped in 4.52s` | pass |
| Triton paged B=8 bench | `bench_paged_pointllm.py --batch_sizes 8 --decode_steps 32 --warmup 1 --runs 3` | Paged improves | Padded `194.6 tok/s`; paged `237.7 tok/s` | pass |
| Triton paged batch sweep | `bench_paged_pointllm.py --batch_sizes 1,2,4,8 --decode_steps 32 --warmup 1 --runs 3` | Completes | Paged TPS B=1/2/4/8: `36.3`, `80.0`, `155.1`, `281.9` | pass |
| Mixed fast path + sampler tests | `pytest -q tests/test_sampler.py tests/test_paged_attention.py tests/test_model_runner.py tests/test_llm_engine.py tests/test_continuous_batching.py` | Pass | `23 passed in 4.14s` | pass |
| Full suite after mixed/sampler | `pytest -q` | Pass | `82 passed, 1 skipped in 4.93s` | pass |
| Continuous batching after mixed fast path | `bench_paged_pointllm.py --batch_sizes 4 --continuous_total 8 --continuous_window 4 --decode_steps 32 --warmup 1 --runs 3` | Completes | Continuous TPS `96.3`, mixed step fraction `3.2%` | pass |
| Varlen prefill CUDA parity | `pytest -q tests/test_paged_attention.py::test_varlen_prefill_triton_kernel_parity_gqa_cuda` | Pass | `1 passed in 5.01s` | pass |
| Varlen prefill related tests | `pytest -q tests/test_paged_attention.py tests/test_sampler.py tests/test_model_runner.py tests/test_llm_engine.py tests/test_continuous_batching.py` | Pass | `24 passed in 4.77s` | pass |
| Full suite after varlen prefill | `pytest -q` | Pass | `83 passed, 1 skipped in 4.70s` | pass |
| Varlen prefill B=8 bench | `bench_paged_pointllm.py --batch_sizes 8 --decode_steps 32 --warmup 1 --runs 3` | Completes | Padded `172.3 tok/s`; paged `263.6 tok/s`; paged prefill `380.8ms` | pass |
| Varlen prefill batch sweep | `bench_paged_pointllm.py --batch_sizes 1,2,4,8 --decode_steps 32 --warmup 1 --runs 3` | Completes | Paged TPS B=1/2/4/8: `36.8`, `72.8`, `138.3`, `255.6`; B=8 prefill `386.0ms` | pass |
| Kernel config targeted tests | `PYTHONPATH=. /opt/conda/envs/pointllm/bin/python -m pytest -q tests/test_paged_attention.py tests/test_sampler.py` | Pass | `9 passed in 4.36s` | pass |
| Full suite after config guards | `PYTHONPATH=. /opt/conda/envs/pointllm/bin/python -m pytest -q` | Pass | `84 passed, 1 skipped in 3.86s` | pass |
| B=8 tune default | `bench_paged_pointllm.py --batch_sizes 8 --decode_steps 32 --warmup 1 --runs 3` | Completes | `varlen M16/N64`, `decode N64`: padded `158.9 tok/s`, paged `225.9 tok/s`, prefill `380.3ms` | pass |
| B=8 tune varlen N32 | `NANOPOINTLLM_VARLEN_BLOCK_N=32 bench_paged_pointllm.py ...` | Completes | padded `139.9 tok/s`, paged `246.8 tok/s`, prefill `384.6ms` | pass |
| B=8 tune varlen N128 | `NANOPOINTLLM_VARLEN_BLOCK_N=128 bench_paged_pointllm.py ...` | Completes | padded `167.5 tok/s`, paged `190.9 tok/s`, prefill `384.9ms` | pass |
| B=8 tune decode N32 | `NANOPOINTLLM_DECODE_BLOCK_N=32 bench_paged_pointllm.py ...` | Completes | padded `166.1 tok/s`, paged `241.9 tok/s`, prefill `381.7ms` | pass |
| B=8 tune decode N128 | `NANOPOINTLLM_DECODE_BLOCK_N=128 bench_paged_pointllm.py ...` | Completes | padded `166.4 tok/s`, paged `245.7 tok/s`, prefill `381.3ms` | pass |
| B=8 tune varlen N32 + decode N128 | `NANOPOINTLLM_VARLEN_BLOCK_N=32 NANOPOINTLLM_DECODE_BLOCK_N=128 bench_paged_pointllm.py ...` | Completes | padded `167.3 tok/s`, paged `242.8 tok/s`, prefill `384.3ms` | pass |
| Framework scaffold focused tests | `pytest -q tests/test_sampler.py tests/test_metadata_staging.py tests/test_cuda_graph.py tests/test_lightweight_runner.py tests/test_block_manager.py tests/test_scheduler.py` | Pass | `31 passed in 3.01s` | pass |
| Runner integration after scaffold | `pytest -q tests/test_model_runner.py tests/test_llm_engine.py tests/test_continuous_batching.py tests/test_paged_attention.py tests/test_llm_api.py` | Pass | `25 passed in 4.74s` | pass |
| Full suite after framework scaffold | `PYTHONPATH=. /opt/conda/envs/pointllm/bin/python -m pytest -q` | Pass | `93 passed, 1 skipped in 3.95s` | pass |
| B=8 after framework scaffold | `bench_paged_pointllm.py --batch_sizes 8 --decode_steps 32 --warmup 1 --runs 3` | Completes | padded `164.5 tok/s`, paged `250.2 tok/s`, prefill `381.0ms`, decode step `3.998ms` | pass |
| Strict graph module tests | `pytest -q tests/test_sampler.py tests/test_cuda_graph.py tests/test_lightweight_runner.py tests/test_scheduler.py tests/test_metadata_staging.py` | Pass | `20 passed in 3.34s` | pass |
| Strict graph runner integration | `pytest -q tests/test_model_runner.py tests/test_llm_engine.py tests/test_continuous_batching.py tests/test_paged_attention.py tests/test_llm_api.py` | Pass | `25 passed in 4.75s` | pass |
| Full suite after strict graph/sampler changes | `PYTHONPATH=. /opt/conda/envs/pointllm/bin/python -m pytest -q` | Pass | `93 passed, 1 skipped in 4.94s` | pass |
| Strict CUDA graph B=1 smoke | Direct PointLLM-7B engine step with `NANOPOINTLLM_ENABLE_CUDA_GRAPH=1` | Completes | B=1 prefill + decode graph step completed with no fallback | pass |
| Strict CUDA graph B=8 bench | `NANOPOINTLLM_ENABLE_CUDA_GRAPH=1 bench_paged_pointllm.py --batch_sizes 8 --decode_steps 32 --warmup 1 --runs 3` | Completes | padded `187.8 tok/s`, paged `210.0 tok/s`, paged prefill `381.6ms`, decode step `4.762ms` | pass |
| Chunked long-prefill bench | `bench_paged_pointllm.py --batch_sizes 1 --target_prompt_len 2048 --max_prefill_chunk_tokens 512 --decode_steps 8 --warmup 0 --runs 1` | Completes | padded prefill `3277.7ms`, paged prefill `1774.7ms`, paged `40.0 tok/s` | pass |
| Eager B=8 after strict changes | `bench_paged_pointllm.py --batch_sizes 8 --decode_steps 32 --warmup 1 --runs 3` | Completes | padded `154.1 tok/s`, paged `268.6 tok/s`, paged prefill `471.4ms`, decode step `3.722ms` | pass |
| Replay split tests | `PYTHONPATH=. /opt/conda/envs/pointllm/bin/python -m pytest -q` | Pass | `93 passed, 1 skipped in 4.17s` | pass |
| Strict graph replay split B=8 | `NANOPOINTLLM_ENABLE_CUDA_GRAPH=1 bench_paged_pointllm.py --batch_sizes 8 --decode_steps 64 --warmup 0 --runs 1` | Completes | paged total `143.8 tok/s`, replay-only `157.5 tok/s`, first decode/capture included separately | pass |
| Eager replay split B=8 | `bench_paged_pointllm.py --batch_sizes 8 --decode_steps 64 --warmup 0 --runs 1` | Completes | paged total `187.3 tok/s`, replay-only `187.4 tok/s` | pass |
| Sampler fast graph replay B=8 | `NANOPOINTLLM_ENABLE_CUDA_GRAPH=1 bench_paged_pointllm.py --batch_sizes 8 --decode_steps 64 --warmup 0 --runs 1` | Completes | paged total `208.1 tok/s`, replay-only `238.0 tok/s` | pass |
| Triton lightweight ops graph replay B=8 | `NANOPOINTLLM_ENABLE_CUDA_GRAPH=1 NANOPOINTLLM_TRITON_LIGHTWEIGHT_OPS=1 bench_paged_pointllm.py --batch_sizes 8 --decode_steps 64 --warmup 0 --runs 1` | Completes | paged total `154.2 tok/s`, replay-only `216.2 tok/s`; slower than default layer ops | pass |
| Default ops graph replay B=8 after sampler fast | `NANOPOINTLLM_ENABLE_CUDA_GRAPH=1 bench_paged_pointllm.py --batch_sizes 8 --decode_steps 64 --warmup 0 --runs 1` | Completes | paged total `205.8 tok/s`, replay-only `235.5 tok/s` | pass |
| Fused residual RMSNorm graph replay B=8 | `NANOPOINTLLM_ENABLE_CUDA_GRAPH=1 NANOPOINTLLM_FUSED_DECODE_RMSNORM=1 bench_paged_pointllm.py --batch_sizes 8 --decode_steps 64 --warmup 0 --runs 1` | Completes | paged total `211.9 tok/s`, replay-only `244.6 tok/s` | pass |
| Full suite after sampler/layer overhead pass | `PYTHONPATH=. /opt/conda/envs/pointllm/bin/python -m pytest -q` | Pass | `93 passed, 1 skipped in 4.96s` | pass |
| Targeted tests after default fused graph residual norm | `PYTHONPATH=. /opt/conda/envs/pointllm/bin/python -m pytest -q tests/test_lightweight_runner.py tests/test_sampler.py tests/test_paged_attention.py` | Pass | `14 passed in 4.70s` | pass |
| Full suite after default fused graph residual norm | `PYTHONPATH=. /opt/conda/envs/pointllm/bin/python -m pytest -q` | Pass | `94 passed, 1 skipped in 4.93s` | pass |
| Default fused graph replay B=8 | `NANOPOINTLLM_ENABLE_CUDA_GRAPH=1 bench_paged_pointllm.py --batch_sizes 8 --decode_steps 64 --warmup 1 --runs 1` | Completes | padded total `192.7 tok/s`, replay-only `192.8 tok/s`; paged total `251.4 tok/s`, replay-only `284.7 tok/s` | pass |
| Decode QKV staging targeted tests | `PYTHONPATH=. /opt/conda/envs/pointllm/bin/python -m pytest -q tests/test_paged_attention.py tests/test_lightweight_runner.py tests/test_sampler.py` | Pass | `15 passed in 5.85s` | pass |
| Decode QKV staging graph replay B=8 | `NANOPOINTLLM_ENABLE_CUDA_GRAPH=1 NANOPOINTLLM_DECODE_QKV_INPUT_STAGING=1 bench_paged_pointllm.py --batch_sizes 8 --decode_steps 64 --warmup 1 --runs 1` | Completes | paged total `215.4 tok/s`, replay-only `258.4 tok/s`; slower than default graph path | pass |
| Targeted tests after gating decode QKV staging | `PYTHONPATH=. /opt/conda/envs/pointllm/bin/python -m pytest -q tests/test_paged_attention.py tests/test_lightweight_runner.py tests/test_sampler.py` | Pass | `15 passed in 4.55s` | pass |
| Default graph replay B=8 after gating decode QKV staging | `NANOPOINTLLM_ENABLE_CUDA_GRAPH=1 bench_paged_pointllm.py --batch_sizes 8 --decode_steps 64 --warmup 1 --runs 1` | Completes | padded total `180.1 tok/s`, replay-only `180.2 tok/s`; paged total `249.2 tok/s`, replay-only `285.7 tok/s` | pass |
| Packed QKV targeted tests | `PYTHONPATH=. /opt/conda/envs/pointllm/bin/python -m pytest -q tests/test_paged_attention.py tests/test_lightweight_runner.py tests/test_sampler.py` | Pass | `17 passed in 5.81s` | pass |
| Packed QKV graph replay B=8 | `NANOPOINTLLM_ENABLE_CUDA_GRAPH=1 NANOPOINTLLM_PACKED_QKV_DECODE=1 bench_paged_pointllm.py --batch_sizes 8 --decode_steps 64 --warmup 1 --runs 1` | Completes | paged total `246.3 tok/s`, replay-only `278.8 tok/s`; slower than default graph path | pass |
| Targeted tests after gating packed QKV | `PYTHONPATH=. /opt/conda/envs/pointllm/bin/python -m pytest -q tests/test_paged_attention.py tests/test_lightweight_runner.py tests/test_sampler.py` | Pass | `17 passed in 4.82s` | pass |
| Default graph replay B=8 after gating packed QKV | `NANOPOINTLLM_ENABLE_CUDA_GRAPH=1 bench_paged_pointllm.py --batch_sizes 8 --decode_steps 64 --warmup 1 --runs 1` | Completes | padded total `163.1 tok/s`, replay-only `162.9 tok/s`; paged total `253.8 tok/s`, replay-only `285.5 tok/s` | pass |
| Fused RMSNorm + packed QKV targeted tests | `PYTHONPATH=. /opt/conda/envs/pointllm/bin/python -m pytest -q tests/test_paged_attention.py tests/test_lightweight_runner.py tests/test_sampler.py` | Pass | `18 passed in 4.70s` | pass |
| Fused RMSNorm + packed QKV graph replay B=8 | `NANOPOINTLLM_ENABLE_CUDA_GRAPH=1 NANOPOINTLLM_FUSED_RMSNORM_PACKED_QKV=1 bench_paged_pointllm.py --batch_sizes 8 --decode_steps 64 --warmup 1 --runs 1` | Completes | paged total `47.7 tok/s`, replay-only `49.4 tok/s`; much slower than default graph path | pass |
| Full suite after fused RMSNorm + packed QKV experiment | `PYTHONPATH=. /opt/conda/envs/pointllm/bin/python -m pytest -q` | Pass | `98 passed, 1 skipped in 5.18s` | pass |
| Tiled fused kernel and staged-buffer targeted tests | `PYTHONPATH=. /opt/conda/envs/pointllm/bin/python -m pytest -q tests/test_paged_attention.py tests/test_lightweight_runner.py tests/test_sampler.py` | Pass | `19 passed in 5.90s` | pass |
| Tiled fused RMSNorm + packed QKV graph replay B=8 | `NANOPOINTLLM_ENABLE_CUDA_GRAPH=1 NANOPOINTLLM_FUSED_RMSNORM_PACKED_QKV=1 bench_paged_pointllm.py --batch_sizes 8 --decode_steps 64 --warmup 1 --runs 1` | Completes | paged total `182.6 tok/s`, replay-only `199.6 tok/s`; improved over the first fused kernel but still slower than default | pass |
| RMSNorm staged packed QKV graph replay B=8 | `NANOPOINTLLM_ENABLE_CUDA_GRAPH=1 NANOPOINTLLM_RMSNORM_STAGED_PACKED_QKV=1 bench_paged_pointllm.py --batch_sizes 8 --decode_steps 64 --warmup 1 --runs 1` | Completes | paged total `241.0 tok/s`, replay-only `292.9 tok/s`; faster than the previous default graph path | pass |
| Targeted tests after promoting staged path | `PYTHONPATH=. /opt/conda/envs/pointllm/bin/python -m pytest -q tests/test_paged_attention.py tests/test_lightweight_runner.py tests/test_sampler.py` | Pass | `20 passed in 4.90s` | pass |
| Default graph replay B=8 after promoting staged path | `NANOPOINTLLM_ENABLE_CUDA_GRAPH=1 bench_paged_pointllm.py --batch_sizes 8 --decode_steps 64 --warmup 1 --runs 1` | Completes | padded total `189.6 tok/s`, replay-only `189.8 tok/s`; paged total `259.6 tok/s`, replay-only `294.1 tok/s` | pass |
| Full suite after promoting staged path | `PYTHONPATH=. /opt/conda/envs/pointllm/bin/python -m pytest -q` | Pass | `100 passed, 1 skipped in 4.04s` | pass |
| Output projection + packed gateup targeted tests | `PYTHONPATH=. /opt/conda/envs/pointllm/bin/python -m pytest -q tests/test_paged_attention.py tests/test_lightweight_runner.py tests/test_sampler.py` | Pass | `22 passed in 5.80s` | pass |
| Default graph replay B=8 after output/MLP scheduling | `NANOPOINTLLM_ENABLE_CUDA_GRAPH=1 bench_paged_pointllm.py --batch_sizes 8 --decode_steps 64 --warmup 1 --runs 1` | Completes | padded total `195.2 tok/s`, replay-only `195.2 tok/s`; paged total `265.5 tok/s`, replay-only `298.5 tok/s` | pass |
| Full suite after output/MLP scheduling | `PYTHONPATH=. /opt/conda/envs/pointllm/bin/python -m pytest -q` | Pass | `102 passed, 1 skipped in 5.01s` | pass |

## Error Log
| Timestamp | Error | Attempt | Resolution |
|-----------|-------|---------|------------|
| 2026-04-16 | `allocate_kv_pool` read missing MagicMock config attrs as mock objects | 1 | Accept only integer `num_key_value_heads` / `head_dim`, otherwise fall back to query heads and derived head dim |
| 2026-04-16 | Base Python reported `CUDA not available`; torch 2.11.0+cu130 incompatible with driver 12020 | 1 | Used `/opt/conda/envs/pointllm/bin/python` with torch 2.5.1+cu121 |
| 2026-04-16 | `BLOCK_M=8` varlen candidate failed in Triton `tl.dot` compile on PointLLM-7B | 1 | Kept head_dim=128 default at `BLOCK_M=16` and added minimum env override guards |
| 2026-04-16 | Paged B=8 skipped after pure decode was routed to `run_decode`: invalid reshape from `[B,1]` model input | 1 | Changed decode metadata staging to flattened `[1,B]` input layout |
| 2026-04-16 | Sampler received one row for B=8 because paged decode logits were `[1,B,V]` | 1 | Sliced paged decode logits to `[B,V]` before per-request sampling |
| 2026-04-16 | Strict CUDA graph failed inside HF `create_causal_mask` during capture | 1 | Routed graph decode through `LightweightLlamaRunner` instead of HF model-level forward |
| 2026-04-16 | Strict CUDA graph failed on `context_lens.max().item()` during capture | 2 | Used static block-table upper bound for decode `MAX_CONTEXT_LEN` |
| 2026-04-16 | Chunked prefill benchmark counted remaining prefill chunks as decode time | 1 | Measured prefill until every request produced its first completion token |
| 2026-04-16 | Chunked prefill benchmark CLI parsed `--max_prefill_chunk_tokens` but did not pass it into `bench_batch` | 1 | Added the argument to the common benchmark kwargs and reran the long-prefill bench |
| 2026-04-16 | `torch.compile(lightweight_runner)` produced illegal memory access under strict graph on real PointLLM-7B | 1 | Made lightweight runner compilation explicit opt-in via `NANOPOINTLLM_COMPILE_LIGHTWEIGHT=1`; default stays strict graph without compiled runner |
| 2026-04-16 | Triton RMSNorm/SiLU-mul lightweight ops were slower than torch ops in graph replay | 1 | Kept them as explicit opt-in via `NANOPOINTLLM_TRITON_LIGHTWEIGHT_OPS=1`; default uses faster measured path |
| 2026-10-01 | Sparse clone of `PointKAN_Accel` timed out before checkout | 1 | Kept pinned Git objects and will materialize only `hw_sim` for audit |
| 2026-10-01 | Directory-level archive triggered large background promisor fetches | 2 | Stopped the session-owned processes and switched to per-file raw retrieval |
| 2026-10-01 | Geometry RTL summary over-counted final output points | 1 | Corrected the testbench's post-NBA summary; exact event/value/cycle matching was already observed |
| 2026-10-01 | Geometry Yosys gate initially used default 8 lanes instead of the formal 64-lane config | 1 | Added explicit `chparam` for all locked datapath widths |
| 2026-10-01 | GitHub raw endpoint returned 404 for PointKAN RTL | 1 | Kept source provenance pinned and switched the isolated reference build to local Git blobs |

## 2026-10-01: PointKAN Geometry RTL Audit
- Started Phase 33 to audit the pinned PointKAN `hw_sim` geometry implementation and adapt only RTL-correlatable contracts to PointLLM.
- Confirmed nano-pointllm remains clean at `3635cd7`; existing `fps` and `knn_topk` coverage remain fail-closed.
- Indexed the PointKAN `hw_sim` tree from Git objects. It mixes Python timing models, DRAMsim3, QuickFPS RTL, PointACC top-k RTL, and APU RTL rather than providing one integrated cycle-correlated geometry block.
- Audited the geometry RTL interfaces: the distance array has static packed coordinates and no memory backpressure; top-k is batch-local; the FPS primitive is min-update only. Full PointLLM FPS/KNN therefore remains unsupported.
- Added an RTL-locked 64-lane signed geometry-distance slice with ready/valid backpressure, widened arithmetic, deterministic trace lowering, Python edge model, Icarus correlation, and parameter-matched Yosys checking.
- Formal geometry artifact: `docs/results/gtsu_geometry_distance_2026-10-01/`; 12/12 cycles, 448/448 values, exact event/counter/value agreement, 64-lane synthesis gate passed.
- Reran the pinned PointKAN distance testbench: five cases passed. Its top-k testbench compiled but exceeded the bounded 50-second audit window, so it is not used as a timing oracle.
- Architecture regression passed: `34 passed` before the final corner-case addition. Final full repository regression passed: `180 passed, 1 skipped`.
- Strict coverage reports the new `geometry_distance_tile` as correlated while `fps`, `knn_topk`, and full PointLLM remain fail-closed.
- Added and passed signed INT16 extreme-value RTL correlation (`-32768` versus `32767`) to validate widened subtraction and accumulation.
- Published the implementation and formal artifacts to `origin/main` as commit `f8e1917` (`feat: correlate PointLLM geometry distance RTL`).

## 2026-10-01: Shared SIMD4 And Sequential W8A8
- Started Phase 34 to replace the geometry performance path with norm-plus-shared-dot while retaining the direct INT16 distance tile as a baseline.
- Added Phase 35 for a true sequential W8A8 PTQ pipeline; no production W8A8 claim is enabled by planning alone.

## 5-Question Reboot Check
| Question | Answer |
|----------|--------|
| Where am I? | Completed delivery |
| Where am I going? | Ready for optional CUDA/PointLLM-7B validation |
| What's the goal? | Align `nano-pointllm` closer to `nano-vllm` high-performance inference |
| What have I learned? | See `findings.md` |
| What have I done? | Created planning files and recorded initial gaps |

## 2026-09-22: Default Lightweight Eager Decode
- Started Phase 20 to make the lightweight decoder the default for ordinary paged eager decode.
- Confirmed current construction is gated by `NANOPOINTLLM_ENABLE_CUDA_GRAPH=1` or `NANOPOINTLLM_PROFILE_LAYERS=1`.
- Planned an explicit `NANOPOINTLLM_LIGHTWEIGHT_DECODE=0` opt-out for HF-path parity/debugging.
- Implemented default eager lightweight routing while preserving HF prefill/mixed-prefill execution.
- Made packed MLP decode weights lazy so the plain lightweight route shares HF weights without allocating unused packed copies.
- Added focused runner-routing and lazy-weight tests plus README usage documentation.
- Full unit suite passed: `117 passed, 1 skipped`.
- Real PointLLM-7B 4-token parity: B1 passed, while B4/B8/mixed batches diverged after the shared prefill token, generally leaving only batch row 0 correct. Phase 20 remains in progress pending a batched decode fix.
- Fixed the actual batched parity bug: explicit `position_ids=None` from PointLLM prevented ForwardContext position injection, so flattened prefill positions did not reset per sequence.
- Post-fix full suite: `118 passed, 1 skipped`.
- Post-fix real PointLLM-7B 4-token parity: B1/B4/B8/mixed all passed token-for-token.
- Same-state lightweight-vs-HF paged decode: logits exactly equal for B1/B4/B8; alternating shared-GPU speedups approximately `1.16x/1.11x/1.10x`.
- Saved noisy shared-GPU end-to-end benchmark outputs to `results/bench_lightweight_eager_default_b1_b4_b8_d32_sharedgpu.json` and `results/bench_lightweight_eager_hf_fallback_b1_b4_b8_d32_sharedgpu.json`; these are diagnostic only because all GPUs were saturated.
- Extended 20-token native-HF parity crossed the paged block boundary but showed later BF16/Triton token divergence on some synthetic point clouds. Same-state equality confirms this is outside the lightweight routing change.

## 2026-09-22: End-to-End Performance Diagnostic Suite
- Started Phase 21 to add a single reproducible benchmark for TTFT, TPOT, tokens/s, prefill/decode splits, shape sweeps, CUDA operator breakdown, KV-cache memory, and HBM diagnostics.
- Metric outputs will distinguish measured CUDA timings from analytical memory/traffic estimates.
- Implemented the unified diagnostic benchmark, pure metric helpers, profiling hooks, plots, README command, and focused tests.
- Corrected profiling semantics so layer profiling no longer selects graph-mode module optimizations for eager decode.
- Added CUDA events for final norm, LM head, and sampling; external scheduling/Python gaps are separated as `unattributed_timeline_ms` rather than mislabeled as model CUDA work.
- Full test suite passed: `122 passed, 1 skipped`; artifact validation passed for 8 results, two 32-layer profiles, JSON, and three PNG files.
- Real outputs written to `results/pointllm_diagnostics_b1_b4_i560_768_o8_32/`.
- Phase 21 completed with a shared-GPU caveat because all four A100s remained at 100% utilization.

## 2026-09-22: Isolated-GPU Benchmark Methodology
- Started Phase 22 to make the diagnostic suite suitable for empty-GPU steady-state measurement.
- Planned explicit cold-start reporting, 10/30 default warmup/run counts, percentile distributions, clock/power telemetry, and representative-only hardware-counter profiling.
- Implemented per-shape cold runs, model/engine startup timing, 10/30 defaults, and P10/P50/P90/P95 distributions.
- Added UUID-targeted `nvidia-smi` telemetry plus strict `--require_idle_gpu` validation and sensor-status warnings.
- Added `profile_pointllm_ncu.py` and `summarize_ncu_pointllm.py`; verified NCU permissions, metrics, duration-weighted CSV parsing, and push/pop NVTX filtering syntax.
- Added `cold_steady_gpu_telemetry.png` and updated all primary plots to use steady-state medians.
- Real PointLLM-7B one-shape smoke completed and emitted the expanded JSON/plots under `results/pointllm_diagnostics_phase22_smoke/`; it is marked non-publishable because the GPU was not idle.

## 2026-09-30: Native Paged Runtime And Strict Decoder Roofline
- Removed the remaining default HF model-level call from paged prefill and mixed continuous batches; all paged LLaMA execution now uses the lightweight runner unless the explicit fallback is selected.
- Added low-overhead, opt-in NVTX stage ranges around QKV, attention, O projection, MLP, LM head, and sampling.
- Added per-stage NCU orchestration, traffic/FLOP manifests, roofline fields, and conservative weight-bound evidence rules.
- Fixed the continuous parity driver so request arrivals are staggered and a real prefill+decode mixed step is mandatory.
- Corrected real PointLLM-7B mixed-step parity passed all three cases token-for-token, with observed mixed compositions `(1,1)`, `(1,3)`, and `(2,1)`.
- Full regression after the runtime/profiler changes: `133 passed, 1 skipped`.
- Added and smoke-tested automatic `stage_bottleneck.png` rendering; the synthetic smoke output was 85,262 bytes.
- Strict B1/B8 NCU collection completed on an idle A100 with context length 780 and decoder layer 0 as the representative hardware-counter scope.
- B1 confirmed QKV, O projection, MLP, and LM head as weight-bandwidth-bound; attention was KV-memory-bound/underfilled.
- B8 confirmed QKV, O projection, and MLP as weight-bandwidth-bound. LM head remained weight-stream dominated but reached only 24.17% DRAM throughput, so it did not pass the strict threshold.
- Stage-only scaled shares were B1: MLP 48.65%, attention 21.88%, QKV 20.83%, O 7.31%, LM head 1.33%; B8: MLP 40.66%, attention 34.15%, QKV 15.17%, O 6.26%, LM head 3.77%.
- Reports and plots are under `docs/results/ncu_weight_bound_2026-09-30/`; raw NCU CSV remains under ignored local `results/ncu_decode_stages/`.

## 2026-10-01: Decode MLP Block Trace
- Added a disabled-by-default observer to `LightweightMLP` and a runner-level attachment method. It captures the exact `SiLU(gate) * up` tensor in both packed and ordinary MLP paths and rejects CUDA Graph capture explicitly.
- Added `nanopointllm.analysis.mlp_block_trace` with raw trace serialization and offline metrics for contiguous neuron blocks, retained contribution, adjacent-token Jaccard, previous-token recall, same-point-cloud cross-prompt overlap, and theoretical post-gate/oracle-pre-gate/previous-token MLP weight-byte reduction.
- Normal inference behavior remains unchanged while no observer is attached.
- Real PointLLM-7B + ModelNet smoke completed for two prompts, four generated tokens each. It captured all 32 layers over three decode passes per prompt and wrote raw activations, compressed JSONL details, CSV curves, summary JSON, and two PNG diagnostics under `results/mlp_block_trace/modelnet_real_smoke/`.
- Formal BF16 ModelNet run completed for two point clouds, four prompts each, and 32 generated tokens: 7,936 layer-token records and 176.7 MB of raw activations under `results/mlp_block_trace/modelnet_n2_p4_t32_bf16/`.
- One-object Objaverse distribution check completed under `results/mlp_block_trace/objaverse_n1_p4_t32_bf16/`; contribution and temporal metrics matched ModelNet while cross-prompt overlap was lower.
- Published compact CSV/PNG evidence and interpretation under `docs/results/mlp_block_trace_modelnet_2026-10-01/`.
- Full regression passed after the trace implementation: `136 passed, 1 skipped in 6.16s`.
- Phase 24 completed. The evidence does not support direct contiguous-block MLP sparsification; neuron reordering/clustering or a learned pre-gate predictor should be validated against the measured oracle before kernel implementation.

## 2026-10-01: Neuron-Level Oracle Gate
- Added B=1/4/8/16/32 contribution-oracle analysis with Gini, entropy, normalized entropy, effective support, fixed-budget retention, and inverse target-support metrics.
- Added an explicit activation-sparsity stop rule and unit tests against uniform and single-support analytic distributions.
- ModelNet and Objaverse both triggered the stop rule. Phase 26 permutation/clustering was intentionally gated off because no layout can outperform the failed B=1 arbitrary-neuron oracle.

## 2026-10-01: Static N:M Wanda Pruning
- Added activation-calibrated Wanda and magnitude N:M mask primitives with exact group validation and unit tests.
- Added a real PointLLM quality evaluator using teacher-forced KL/top-token metrics and free greedy generation; checkpoint weights remain untouched.
- Completed 2:4 and 4:8 ModelNet evaluations. Both exact-50%-sparse variants changed free generation on two of four requests, so structured pruning is not treated as accuracy-free.
- Inspected the official SparseGPT implementation. Full-Hessian reconstruction remains a separate long-running experiment; no diagonal proxy is reported as SparseGPT.

## 2026-10-01: Precision Sensitivity And PointBERT Roofline
- Added per-output-channel W8/W4 weight-rounding sensitivity for PointBERT, projector, decoder QKV/O/MLP, and LM head, restoring the BF16 baseline after every component run.
- Real ModelNet calibration found 100% free-generation match for every W8 component in the evaluated four requests. W4 LM head failed all exact requests and decoder MLP retained only 75%; the other W4 components matched these short generations but require broader validation.
- Added strict PointBERT NCU scopes for FPS, KNN, complete local encoder, and all 12 PointTransformer QKV/attention/O/MLP stages.
- Fixed the idle methodology so the launch snapshot is taken before NCU itself allocates memory and initializes counters; both B1 and B4 formal runs started at 0% utilization.
- B1 marked-stage shares were FPS 50.35%, PT attention 17.54%, local encoder 15.60%, and PT MLP 7.16%. FPS reached only 0.29% SM throughput.
- B4 shares were local encoder 29.06%, FPS 28.79%, PT attention 22.47%, and PT MLP 8.50%. Local encoder reached 62.52% SM throughput.
- Published compact evidence under `docs/results/compression_sensitivity_2026-10-01/` and `docs/results/pointbert_ncu_roofline_2026-10-01/`.

## 2026-10-01: Architecture Simulator Kickoff
- Accepted the characterization-driven architecture direction: persistent geometry, dense tensor, elastic Split-K decode, precision-aware weight streaming, shared attention, and phase-reconfigurable SRAM.
- Began auditing `DingdongD/Compiler_Codes` plus the local PointAcc C-model/RTL projects before defining the simulator contract.

## 2026-10-01: Phase-Adaptive Architecture Estimator
- Added the standalone `architecture_simulator/` Python package with stable hardware/workload/result JSON contracts and PointLLM-7B presets.
- Implemented persistent FPS, tensor-distance/top-k KNN, dense and Split-K tensor mappings, precision-aware weight traffic/unpack, fused or materialized attention, paged-KV efficiency, vector operations, phase scheduling, traffic counters, and optional characterization hooks.
- Added resource-aware DSE across tensor dimensions, Split-K, geometry lanes, HBM, SRAM, and weight-unpack throughput; fixed clamped Split-K duplicate designs.
- Added a future C-model/RTL roadmap based on the audited Compiler_Codes and PointAcc separation patterns without copying source. No C-model or RTL was implemented in this phase.
- The initial architecture tests passed (`10 passed`). The formal `B1/S768/O128` sweep evaluated 320 unique analytical designs and archived 39 Pareto points under `docs/results/architecture_simulator_2026-10-01/`.
- Full repository regression passed after implementation and plot QA: `155 passed, 1 skipped`.

## 2026-10-01: Architecture Estimator Correctness Correction
- Audited the previously published implementation after the simulator-completeness claim was challenged. Confirmed that it was not event-level or cycle-accurate and that four apparent implementation directories were documentation-only placeholders.
- Found and fixed a decoder-attention mapping error that serialized heads and inflated the B1 decode-attention estimate by about `74.21x` over the configured-peak MAC lower bound.
- Added missing PointBERT bridge/position/vector stages, projector activation, decoder vector/sampling stages, consistent KNN HBM accounting, and correct sub-byte packing.
- Added real model-file validation and analytical conservation checks. The formal workload passes `374/374`; this is not an RTL or cycle-accuracy validation.
- Removed the empty C-model/RTL/verification/characterization source directories and consolidated future work under `architecture_simulator/docs/CMODEL_RTL_ROADMAP.md`.
- Marked CLI warnings, plots, JSON status, READMEs, and result artifacts as `exploratory_only`, `event_level_model=false`, and `rtl_correlated=false`.
- Regenerated the baseline and 320-point DSE. The old `7082.26 ms` and `1.406x` values are superseded and must not be cited.

## 2026-10-01: Cycle-Accurate Architecture Modeling Start
- Confirmed that the current repository has no PointLLM-wide RTL implementation and therefore cannot provide full-model cycle-accurate results.
- Accepted the strict requirement that unsupported operators must fail closed rather than fall back to analytical latency inside a cycle-accurate report.
- Audited the supplied implementation guidance and selected the Compiler_Codes event/resource organization as a structural reference only, not as a source of PointLLM timing constants.
- Started Phase 31 with the W8 Split-K GEMV vertical slice as the first functional/event/RTL correlation target, followed by persistent FPS.
- Confirmed that Icarus Verilog, Verilator, and Yosys are installed locally, enabling an actual RTL simulation gate.
- Inspected Compiler_Codes ready/valid, DMA, SRAM, datapath, and trace implementations. They will be used as interface/organization references only because their current request-level timing does not model the required pipelined backpressure and bank-striped traffic.
- Added the first RTL-locked W8 Split-K GEMV transaction contract and Python edge-state model.
- Added a synthesizable ready/valid FIFO and parameterized GEMV vertical-slice RTL with finite compressed, decoded, and partial-sum queues; W8 sign extension; dot accumulation; Split-K reduction; output backpressure; and event/stall counters.
- Added a deterministic SystemVerilog testbench that exposes cycle-stamped input, unpack, compute, partial, and output events. Correlation is not yet claimed until automated trace comparison passes.
- Fixed the trace parser and a time-zero testbench stimulus race exposed by the first two strict correlation attempts; both failures remain recorded in `task_plan.md`.
- Exact Python-versus-Icarus correlation now passes for three parameter configurations. The locked result has 84 identical events and zero cycle, traffic, counter, or functional mismatch.
- Added a fail-closed operator coverage registry and CLI. Requiring attention or any other uncorrelated PointLLM operator raises an error rather than using analytical timing.
- Yosys structural synthesis checks pass for the parameterized ready/valid FIFO and W8 Split-K GEMV RTL.
- Archived the compact correlation, traces, and coverage matrix under `docs/results/gtsu_cycle_splitk_rtl_2026-10-01/`.
- Audited QuickFPS as the candidate second vertical slice. It already contains stronger RTL, DMA, DRAMsim3, trace validation, and PPA infrastructure, but remains external to GTSU; integration and schema correlation are still pending.
- Reran the complete local QuickFPS functional RTL suite successfully, including point engine, bucket pipeline, ping-pong controller, SRAM, AXI reader/writer, concurrent DMA, and integrated stream subsystem.
- Added the first shared GTSU micro-op IR and deterministic lowering for `LD_W`, `UNPACK_W8`, `GEMV_SPLITK`, `PSUM_REDUCE`, and `STORE`; correlation artifacts now preserve the lowered stream.
- Final Phase 31 slice regression passed: `22` architecture tests, including three exact Icarus correlation configurations and Yosys synthesis checks; full repository regression passed `167 passed, 1 skipped`.
- Verified fail-closed CLI behavior: requesting uncorrelated `attention` coverage exits with an explicit error and does not emit a mixed-fidelity cycle report.
- Started the memory-accurate extension from the local `Compiler_Codes` commit `ad2c31a`: its LBUF contract is 16 banks x 128 bits, one independent read/write port per bank, fixed client arbitration, and a configurable 2/3-cycle external read-valid pipeline.
- Located the local QuickFPS DRAMsim3 C ABI and confirmed that it links the pinned official DRAMsim3 commit `29817593b3389f1337235d63cac515024ab8fd6e`; the selected DDR4-2400 configuration has `tCK=0.83 ns`, one channel, 16 banks, and a 64-byte transaction payload.
- Confirmed the real PointLLM checkpoint dimensions (`4096` hidden, `11008` intermediate, `32` layers, `32003` vocabulary) and safetensors index needed for checkpoint-backed lowering.
- Added a 16-bank x 128-bit, 1R1W, three-cycle-read SRAM edge model and parameterized two-client RTL. The conflict-heavy locked schedule matches exactly at 15 cycles and 21 events, including one read conflict, one write conflict, two client stalls, and a concurrent same-bank read/write check.
- Added decoder-linear lowering that reads every source shape/dtype/byte range from the real safetensors headers, assigns globally non-overlapping W8+FP16-scale addresses across all 32 layers and LM head, and emits burst-aligned Split-K tiles plus SRAM bank/row mappings.
- Added a strict local DRAMsim3 adapter with accelerator/DRAM clock conversion, tagged outstanding requests, aligned 64-byte transactions, callback validation, and bridge/config hashes. The formal 64-request q-projection sample completed in 246 accelerator cycles with 34/97/158-cycle min/median/max latency.
- Archived the SRAM traces, all decoder projection manifests, DRAMsim3 request/completion trace, coverage matrix, and fidelity boundary under `docs/results/gtsu_memory_lowering_2026-10-01/`.
- Final memory/lowering regression passed `29` architecture tests; the full repository passed `174 passed, 1 skipped` in the locked PointLLM environment. Coverage remains fail-closed for integrated PointLLM linear and full-model cycles.

## 2026-10-01: Production Decoder Linear Integration
- Integrated scale prefetch and W8 weight requests with the pinned DRAMsim3 backend, a 32-request oldest-unretired issue window, one-burst/cycle DMA ingress, finite 32-entry reorder buffer, bank-striped SRAM vector writes/reads, three-cycle SRAM response, unpack, compute issue, Split-K partials, and output retirement.
- A naive outstanding-only policy exposed a real reorder hazard: Q-projection callbacks reached a maximum 2,217 completed-but-not-in-order entries despite only 32 requests being outstanding. The final issue policy constrains sequence distance from the oldest unretired request and remains bounded at 32 entries.
- Production Q projection (`4096x4096`) matches RTL exactly at `997,858` cycles with 266,240 DMA requests, 262,144 compute bursts, 65,536 partials, 4,096 outputs, zero bank conflicts, and ROB peak 32.
- Production down projection (`4096x11008`) matches RTL exactly at `2,755,501` cycles with 715,520 DMA requests, 704,512 compute bursts, 65,536 partials, 4,096 outputs, 217,088 bank conflicts, and ROB peak 32.
- The exact small-shape CI case exercises conflict cooldown and variable controller parameters; arithmetic output values remain independently locked by the W8 Split-K GEMV datapath RTL.

## 2026-10-01: Shared Geometry-Feature Datapath Start
- Audited the existing precision sensitivity implementation and confirmed that it is isolated weight-only QDQ, not sequential W8A8.
- Audited the Split-K datapath and confirmed that its multiplication loop is embedded in the GEMV module. The next RTL change extracts one signed INT8 SIMD4 primitive and requires both Split-K and geometry-dot to instantiate it.
- Preserved the correlated INT16 direct-distance tile as the dedicated baseline; the shared geometry path will use cached norms and multiplier-free distance reconstruction.
- Added `gtsu_dot4_pe` and refactored W8 Split-K GEMV to instantiate it in four-lane groups without changing existing value/event/cycle correlation.
- Added a shared feature/geometry RTL top, deterministic mixed-mode trace, Python edge model, Icarus correlation, and Yosys hierarchy proof.
- The locked 64-PE run matches 576 lane outputs exactly in 16 cycles. The hierarchy contains 64 shared PE instances and no top-level multipliers.
- Added exact INT16 multiplication decomposition through four signed INT8 products and retained the direct INT16 tile as the dedicated baseline.
- Full repository regression passes `191 passed, 1 skipped` under the locked PointLLM Python environment. Strict coverage accepts the shared-dot and Split-K slices while full PointLLM remains fail-closed.

## 2026-10-01: Sequential W8A8 Contract Start
- Added an explicit W8A8 numerical contract with symmetric clipping, ties-to-even rounding, FP16 scale storage, INT32 linear golden accumulation, per-output W8, group-128 W8, static A8, and dynamic per-token A8.
- Added sequential activation pre-hooks and calibration observers; unit tests prove that a later layer receives the earlier layer's QDQ output rather than a restored baseline tensor.
- Added a real PointLLM evaluator with disjoint calibration/evaluation IDs for ModelNet and Objaverse, simultaneous selected-component W8, sequential A8, KL/top-1/token/exact metrics, and explicit no-speedup guardrails.
- Completed the first one-calibration/one-held-out ModelNet static-A8/per-output-W8 smoke across all 283 selected modules: KL mean `0.01443`, top-1 `1.0`, and eight-token exact match `1.0`.
- Completed the same-split group-128 comparison: lower weight RMSE but worse KL (`0.01586` mean, `0.08396` max), with exact eight-token output retained.
- Completed the matching Objaverse per-output/static-A8 smoke: KL mean `0.01168`, KL max `0.03734`, and exact eight-token output retained.
- Implemented exact SmoothQuant `x/s`, `W*s` transforms for Linear and Conv paths, with unit tests proving pre-QDQ equivalence. The alpha-0.5 ModelNet smoke improves KL mean to `0.00481`.
- Completed dynamic per-token A8 smoke: lowest KL (`0.00250`) and RMSE (`0.26934`), but one of eight teacher-forced top-1 positions differs despite identical free generation.
- Regenerated the five-way comparison figure and passed the full repository regression: `199 passed, 1 skipped`. No production W8A8 coverage or speedup claim is enabled.

## 2026-10-01: Integer And Dense-Mode Closure Start
- Accepted the remaining-gate order: true integer numerical closure, shared Dense GEMM RTL, fused FPS/KNN, geometry-feature composition, attention/vector/sampling, then one-layer and full-graph composition.
- Kept full-model and production W8A8 coverage fail-closed; QDQ smoke evidence is not reclassified as integer hardware execution.
- Added a true INT8/INT32 Linear golden and compared FP32, BF16, QDQ-FP32, QDQ-BF16, and IntegerGolden on real PointLLM q-projection, down-projection, and LM-head tensors.
- Correlated selected real q-projection rows through the shared-Dot4 dense RTL for the full `K=4096` reduction: exact accumulators, exact event trace, and zero cycle error.
- Added the shared-Dot4 dense GEMM microtile. A 64-PE `5x61x384` lock case with source/output stalls is exact against Icarus; Yosys confirms zero direct multipliers above the Dot4 instances.
- Kept production shape control, SRAM double buffering, all-output-row correlation, and generic `dense_gemm` coverage disabled.
- Passed the final repository regression: `203 passed, 1 skipped`.

## 2026-10-01: Compiler_Codes RTL Reuse Audit Start
- Confirmed that the existing Nano-pointllm quant/dequant numerical contract and shared-Dot4 RTL were independently implemented; no prior bit-exact `Compiler_Codes` requant/dequant mapping was claimed.
- Pinned the local reference to `/home/Compiler_Codes@ad2c31a` and started a synthesizable-module audit covering scale buffers, DMU, MAC/EPU post-processing, GEMM control, and SRAM interfaces.
- Added an independently implemented INT32-to-A8 requant unit informed by the reference post-processing stages. It preserves PointLLM's `[-127,127]` contract rather than the reference converter's `[-128,127]` behavior.
- Added an FP16 scale-ratio compiler, magnitude-based ties-to-even rounding golden, elastic RTL, Icarus correlation, and Yosys checks. The 12-vector lock completes in 20 cycles with exact events and one synthesized multiplier/no divider.
- Added a Dense M/N/K ping-pong controller and exact edge model. Representative Local Encoder, PointTransformer, projector, LLM-prefill, and partial-K-block suites all have zero cycle error.
- Physical 16-bank SRAM payload movement, the floating BF16 dequant path, and controller+Dot4+requant composition remain fail-closed.
- Passed the full repository regression after the new RTL slices: `209 passed, 1 skipped`.

## 2026-10-02: Physical Dense SRAM Composition Start
- Started the first physically composed Dense slice rather than another analytical bridge.
- Locked the minimal payload mapping to `N_TILE=3`: one 128-bit SRAM word carries one INT8x4 activation vector and three INT8x4 weight vectors, allowing one read port to feed three shared Dot4 PEs while the independent write port fills the alternate ping-pong K block.
- Production `N_TILE=64` will require one A word plus sixteen B words per chunk; that multiword gather is a separate gate and is not implied by this slice.
- Implemented the physical composition RTL and independent Python edge-state golden, including read-before-write SRAM semantics, fixed-latency responses, outstanding credit, FIFO backpressure, Dot4 accumulation, and parallel requant.
- The formal lock finishes in 29 cycles with zero cycle error, exact 68-event trace, exact four output vectors, and 10 true read/write overlap cycles.
- Yosys hierarchy checks pass with one SRAM/controller/FIFO/Dense tile, three Dot4 and three requant lanes, and no composition-top multiplier.
- Passed the full repository regression with the composed slice: `211 passed, 1 skipped`.

## 2026-10-02: DRAMsim3 DMA And BF16 Dequant Start
- Audited the pinned `Compiler_Codes@ad2c31a` software DDRManager, DMU, DDR RTL, and AXI master rather than treating its C-model copies as cycle timing.
- Locked the relevant reference boundary: 512-bit AXI beats, INCR bursts, optional 4-KiB splitting, 1024-bit FIFO pairing, and FIFO-driven read backpressure; the reference's fixed ID/in-order response assumption will not replace a tagged DRAMsim3 completion ROB.
- Selected a 64-byte-line to four-128-bit-word DMA contract so DRAMsim3 traffic has 100% useful payload utilization for the current physical Dense stream.
- Started an exact no-host-floating-point BF16 dequant contract after INT32 accumulation; generic production Dense coverage remains disabled.
- Completed the tagged 64-byte completion ROB, four-way 128-bit unpack, and direct physical Dense-pipeline composition. The real DRAMsim3 trace and RTL match at 62 cycles with 100% useful line utilization.
- Completed exact fused INT32/FP16/FP16-to-BF16 RNE RTL and golden correlation at 20 cycles; 1,000 additional random vectors match an independent Torch BF16 oracle.
- Archived both evidence packages under `docs/results/gtsu_dramsim_dma_pipeline_2026-10-02/` and `docs/results/gtsu_bf16_dequant_2026-10-02/`.
- Final architecture regression passed `63 passed`; full repository regression passed `217 passed, 1 skipped`.
