# Findings

## Initial State
- `nano-pointllm` already has core framework pieces: `PointLLMLLMEngine`, `Scheduler`, `PointLLMModelRunner`, `PagedModelRunner`, `BlockManager`, `KVPool`, and `PagedLlamaAttention`.
- Current framework-level tests for engine/scheduler/block/KV/paged attention passed before this task: `38 passed`.
- The public package currently exposes decode backend utilities from `nanopointllm.engine`, but no `nanopointllm.LLM` convenience entry point like `nanovllm.LLM`.

## Candidate Gaps
- Paged scheduler lacks nano-vLLM-like preemption when appending a decode token needs a new block and no block is available.
- Sampling parameters exist, but current runners use greedy `argmax`.
- Model loading/tokenizer/config are not equivalent to `nano-vllm`; `PointLLMLLMEngine` expects an already-loaded HF/PointLLM model.

## Default Lightweight Eager Decode Discovery
- `PagedModelRunner` currently constructs `LightweightLlamaRunner` only when strict CUDA Graph or layer profiling is enabled.
- Ordinary paged eager `run_decode()` therefore invokes the full PointLLM/HuggingFace model-level `forward`, even though q_len=1 paged attention and lightweight decoder layers are already available.
- The safe scope is pure paged decode only. Prefill and mixed-prefill should continue through the existing HF model path because the lightweight stack is tuned and validated primarily for q_len=1 decode.
- Plain lightweight construction previously packed every MLP gate/up matrix even when the packed path was disabled. For PointLLM-7B this would add several GiB of avoidable device memory, so packed MLP tensors must be allocated only when the packed decode mode is active.
- Real-model parity showed the plain lightweight path is not yet batch-correct: B1 passed, but B4/B8 rows after row 0 diverged at the first decode step. The first generated token matched because it came from HF prefill; subsequent divergence localizes the issue to batched q_len=1 decode rather than point encoding/prefill.
- Root cause was position injection, not the lightweight layer math or Triton attention. `PointLLMLlamaModel.forward()` explicitly forwards `position_ids=None` to `LlamaModel.forward()`, while `_install_position_ids_patch()` only injected context positions when the keyword was absent. Flattened B-way prefill therefore assigned monotonically increasing positions across sequences, making only row 0 correct; the HF decode fallback similarly used position 0. The patch must replace both absent and explicit-`None` positions.
- After the position fix, real PointLLM-7B 4-token greedy parity passed for B1, B4, B8, and mixed point/text batches.
- Same-state real-model decode comparison produced exactly identical lightweight and paged-HF logits (`maxdiff=0`, identical top-1) for B1/B4/B8.
- Alternating same-process measurements on a fully occupied shared A100 showed lightweight/HF forward speedups of about `1.16x` (B1), `1.11x` (B4), and `1.10x` (B8). Absolute latency is not a clean benchmark because all GPUs were at 100% utilization.
- A 20-token comparison against the unpatched native-HF reference diverged for some synthetic point clouds around token 14 while other cases remained exact. This is an existing paged Triton/BF16 numerical-parity limitation; it is not caused by the lightweight wrapper bypass because same-state lightweight and paged-HF logits are identical.

## Confirmed Runtime Gap
- `nano-vllm` allocates a new KV block in `may_append()` when `len(seq) % block_size == 1`, because `postprocess()` has already appended the token that will be decoded next and that token is the first slot of a new block.
- `nano-pointllm` currently uses `seq.num_tokens % block_size == 0` for `can_append()` / `may_append()`. For a sequence that grows from 4 to 5 tokens with block size 4, the next decode needs physical slot block 1 offset 0, but no block has been allocated. This is an off-by-one paged KV scheduling bug.

## Implemented
- Fixed paged KV append timing to allocate a new block when the current decode input token is at offset 0 of a new block.
- Added scheduler preemption for paged KV memory pressure so unslotted sequences are moved back to waiting instead of being sent to the runner.
- Changed KV pool allocation to use `num_key_value_heads` for GQA/Llama-style models.
- Added partial-prefix prefill masking for paged attention when prefix cache is hit and only suffix tokens are evaluated.
- Routed paged pure prefill through the varlen `run_mixed` path.
- Added `nanopointllm.LLM` plus package-level `SamplingParams` export for a `nanovllm`-like offline API around already-loaded PointLLM/HF models.

## Decode Kernel Target
- Existing paged decode loops over sequences in Python, materializes `k_full` / `v_full`, repeats KV heads, and calls SDPA per sequence.
- A first high-impact kernel can target the common pure decode case where every scheduled sequence has `q_len=1`.
- The kernel should launch over `(batch, query_head)`, walk each sequence's block table, do online softmax over the context length, and write `[B, H, D]` without materializing gathered KV.

## Decode Kernel Implemented
- Added a Triton q_len=1 paged decode kernel in `nanopointllm/llama/paged_attention.py`.
- CUDA fast path is selected only when all `ctx.seq_lens` are `1`; prefill and partial-prefix prefill still use the existing SDPA reference path.
- Kernel supports GQA by mapping query head to KV head via `head_id // num_kv_groups`.
- B=8 PointLLM-7B paged throughput improved from the prior `127.2 tok/s` measurement to `281.9 tok/s` in the full batch sweep.

## Mixed Decode Fast Path And Sampler
- Extended `PagedLlamaAttention.forward()` so q_len=1 sequences inside mixed prefill+decode batches also use the Triton paged decode kernel.
- Added mixed prefill+decode parity coverage with one q_len=2 prefill sequence and one q_len=1 decode sequence.
- Added a reusable `GreedySampler` module with lazy CUDA `torch.compile` and routed both padded and paged runners through it instead of scattered inline `argmax`.
- `flash_attn` is not installed in the current `pointllm` environment, so high-performance varlen prefill cannot be delegated to `flash_attn_varlen_func` yet.
- Continuous batching throughput improved from the earlier `76.1 tok/s` measurement to `96.3 tok/s` with the mixed decode fast path.

## Prefill Varlen Kernel
- Added a q_len>1 Triton paged varlen attention kernel that launches over `(query_block, sequence, query_head)`.
- The kernel reads K/V directly from the paged KV pool via block tables and uses online softmax with causal / partial-prefix masking.
- CUDA parity test covers GQA, variable q/context lengths, and suffix-prefill where `context_len > q_len`.
- PointLLM-7B B=8 paged prefill improved modestly from the prior `393.6ms` sweep to `386.0ms` in the full sweep and `380.8ms` in a B=8-only run.
- Current prefill kernel is correct but not yet heavily tuned; decode remains the larger win.

## Kernel Tuning Direction
- PointLLM-7B/Llama hidden layout uses `head_dim=128`, so Triton BLOCK_D is fixed at 128. The main tunables are prefill query tile `BLOCK_M` and key tile `BLOCK_N`, plus decode `BLOCK_N`.
- Use static head_dim-aware defaults and environment overrides first; runtime autotune inside generation would add compile/search overhead and make latency less predictable.

## Kernel Config Tuning Results
- `BLOCK_M=8` for varlen prefill is not a safe default for the current Triton `tl.dot(q, trans(k))` shape on PointLLM-7B; the B=8 benchmark skipped paged mode due a Triton compile error at the dot site.
- Added guarded environment overrides:
  - `NANOPOINTLLM_VARLEN_BLOCK_M`, minimum `16`
  - `NANOPOINTLLM_VARLEN_BLOCK_N`, minimum `32`
  - `NANOPOINTLLM_DECODE_BLOCK_N`, minimum `16`
- Selected conservative head_dim=128 defaults:
  - varlen prefill: `BLOCK_M=16`, `BLOCK_N=64`
  - decode: `BLOCK_N=64`
- B=8 PointLLM-7B tuning runs on A100, prompt_len=559, decode_steps=32, warmup=1, runs=3:
  - default `varlen M16/N64`, `decode N64`: paged prefill `380.3ms`, paged TPS `225.9`, padded TPS `158.9`
  - `varlen M16/N32`, `decode N64`: paged prefill `384.6ms`, paged TPS `246.8`, padded TPS `139.9`
  - `varlen M16/N128`, `decode N64`: paged prefill `384.9ms`, paged TPS `190.9`, padded TPS `167.5`
  - `varlen M16/N64`, `decode N32`: paged prefill `381.7ms`, paged TPS `241.9`, padded TPS `166.1`
  - `varlen M16/N64`, `decode N128`: paged prefill `381.3ms`, paged TPS `245.7`, padded TPS `166.4`
  - `varlen M16/N32`, `decode N128`: paged prefill `384.3ms`, paged TPS `242.8`, padded TPS `167.3`
- Interpretation: current single-run B=8 numbers have noticeable decode variance. `M16/N64` remains the best measured prefill tile and is the least risky default; `N32/N128` remain useful env-controlled decode experiments but not enough to justify changing the default.

## Framework Completion Pass
- Added reusable decode metadata staging buffers. Paged pure decode now uses flattened `[1, B]` input layout with preallocated `input_ids`, `position_ids`, `slot_mapping`, `context_lens`, and `block_tables`.
- Added decode bucket selection and an opt-in CUDA graph cache. Set `NANOPOINTLLM_ENABLE_CUDA_GRAPH=1` to attempt capture/replay; graph-incompatible HF work falls back to eager automatically.
- Added chunked prefill scheduler state. When `max_prefill_chunk_tokens` is set on `PointLLMLLMEngine`, paged prefill can process prompt chunks, return `None` for non-final chunks, and requeue the sequence without deallocating blocks.
- Extended `SamplingParams` and sampler support for per-request `do_sample`, `top_k`, `top_p`, repetition penalty, frequency penalty, presence penalty, and deterministic seed. Default remains greedy for compatibility and benchmarks.
- Added `PrefixCacheIndex` subsystem around the existing block hash mapping with hit/miss/insert stats and eviction hooks.
- Added `LightweightLlamaRunner` scaffolding. It currently wraps HF layers behind a stable execution boundary, so RMSNorm/MLP/Linear kernels can be replaced incrementally.
- B=8 PointLLM-7B after framework scaffolding: padded `164.5 tok/s`, paged `250.2 tok/s`, paged prefill `381.0ms`, decode step `3.998ms`.

## Strict CUDA Graph / Compiled Sampler Pass
- Removed CUDA graph fallback semantics. When `NANOPOINTLLM_ENABLE_CUDA_GRAPH=1`, decode graph capture/replay is strict: graph-incompatible work raises instead of silently running eager.
- Real PointLLM-7B graph capture initially failed in HF `LlamaModel.forward` because `create_causal_mask` creates a CUDA tensor during capture.
- Strict graph decode now uses `LightweightLlamaRunner` to bypass HF model-level forward and avoid HF causal-mask construction.
- Paged pure decode attention now has a graph-safe direct branch that avoids capture-time creation of `decode_rows` / `decode_q_offsets`.
- Removed capture-time `context_lens.max().item()` from decode attention; `MAX_CONTEXT_LEN` now uses the static block-table upper bound.
- B=1 strict graph decode smoke passed on real PointLLM-7B.
- B=8 strict graph benchmark completed on real PointLLM-7B: padded `187.8 tok/s`, paged `210.0 tok/s`, paged prefill `381.6ms`, decode step `4.762ms`. This includes per-run graph capture cost because the benchmark constructs a fresh engine for each run.
- Default eager B=8 after strict changes: padded `154.1 tok/s`, paged `268.6 tok/s`, paged prefill `471.4ms`, decode step `3.722ms`.
- Chunked long-prefill benchmark after fixing timing and argument wiring: B=1, prompt_len=2048, chunk=512, padded prefill `3277.7ms`, paged prefill `1774.7ms`, paged decode `40.0 tok/s`.
- Sampler no longer uses a Python per-row sampling loop. It materializes batched tensor parameters and token count matrices, then runs a single tensor sampling program with optional CUDA `torch.compile`.

## Graph Replay Benchmark Split
- Updated `scripts/bench_paged_pointllm.py` to report:
  - total decode throughput, including the first decode step
  - first decode step latency, which includes strict graph warm/capture when enabled
  - replay-only decode latency/throughput for subsequent decode steps
- PointLLM-7B B=8, decode_steps=64, strict graph enabled:
  - padded total `103.9 tok/s`, replay-only `142.0 tok/s`
  - paged total `143.8 tok/s`, replay-only `157.5 tok/s`
- PointLLM-7B B=8, decode_steps=64, eager:
  - padded total `128.7 tok/s`, replay-only `171.9 tok/s`
  - paged total `187.3 tok/s`, replay-only `187.4 tok/s`
- Interpretation: strict CUDA graph capture/replay is now measurable separately and stable, but the current graph path is slower than eager. The likely cause is that graph mode uses the new lightweight layer stack while eager still uses the optimized HF/torch execution path; the next performance target is reducing graph-path layer overhead rather than further benchmark plumbing.

## Graph Layer/Sampler Overhead Pass
- Optimized sampler staging for the dominant greedy path: when all requests use default greedy params, the sampler now calls a compiled argmax path directly and skips `[B, vocab]` token-count allocation, top-k/top-p sorting, softmax, and multinomial.
- Tried strict `torch.compile(lightweight_runner)` for graph decode. Real PointLLM-7B produced an illegal memory access, so it is not used by default. It remains a strict opt-in with `NANOPOINTLLM_COMPILE_LIGHTWEIGHT=1`.
- Added Triton RMSNorm and SiLU-multiply kernels for the lightweight runner, controlled by `NANOPOINTLLM_TRITON_LIGHTWEIGHT_OPS=1`. On B=8 strict graph replay this was slower (`216.2 tok/s`) than the default torch ops path, so it is not enabled by default.
- Best strict graph replay result after sampler fast path and default layer ops: B=8, decode_steps=64, paged total `205.8 tok/s`, replay-only `235.5 tok/s`.
- Added a narrower fused `residual + RMSNorm` Triton path for decode layers, controlled by `NANOPOINTLLM_FUSED_DECODE_RMSNORM=1`.
- B=8 strict graph replay with fused residual RMSNorm only: paged total `211.9 tok/s`, replay-only `244.6 tok/s`.
- Interpretation: the profitable part of the layer-side Triton experiment is the fused post-attention `residual + RMSNorm`, not the broader standalone RMSNorm / SiLU-mul replacement. The next fusion candidate should stay on that path: combine residual handling with surrounding decode-only layer work rather than swapping every elementwise op to Triton.
- Current gap to eager remains mostly in the graph lightweight layer execution path, not sampler staging. The next viable layer target is a fused linear/RMSNorm/MLP implementation or a graph-safe compiled layer that avoids the current illegal-memory issue.

## Default Graph Residual + RMSNorm
- Promoted the profitable post-attention `residual + RMSNorm` path into the default decode-graph lightweight runner construction instead of leaving it as an opt-in benchmark switch.
- `build_lightweight_llama_runner(..., decode_graph_mode=True)` now enables fused residual RMSNorm explicitly for graph decode, while broad Triton lightweight ops remain opt-in.
- `PagedModelRunner` now constructs the lightweight runner with `decode_graph_mode=True` whenever strict CUDA graph decode is enabled.
- Added a focused test to ensure decode-graph runner construction actually turns on fused post-attention residual normalization.
- Real PointLLM-7B, strict CUDA graph, B=8, decode_steps=64 after promoting the fused path to the default graph runner:
  - padded total `192.7 tok/s`, replay-only `192.8 tok/s`
  - paged total `251.4 tok/s`, replay-only `284.7 tok/s`
- Interpretation: making fused post-attention `residual + RMSNorm` the default graph decode path is a clear win and materially narrows the remaining graph-path gap.

## Decode QKV Input Staging Experiment
- Added a decode-only paged-attention QKV staging helper that projects from a shared contiguous `[B, hidden]` input view and bypasses the generic `[1, S, hidden] -> view -> transpose` path for pure decode batches.
- The implementation is wired behind `NANOPOINTLLM_DECODE_QKV_INPUT_STAGING=1` in `PagedLlamaAttention` rather than enabled by default.
- Unit parity for the staged Q/K/V projections passed.
- Real PointLLM-7B, strict CUDA graph, B=8, decode_steps=64 with `NANOPOINTLLM_DECODE_QKV_INPUT_STAGING=1`:
  - padded total `151.9 tok/s`, replay-only `151.7 tok/s`
  - paged total `215.4 tok/s`, replay-only `258.4 tok/s`
- The same benchmark with the staging path gated back off:
  - padded total `180.1 tok/s`, replay-only `180.2 tok/s`
  - paged total `249.2 tok/s`, replay-only `285.7 tok/s`
- Interpretation: this first QKV input staging version is correct but slower than the current default graph decode path. The likely issue is that staging plus manual per-step projection/permute work does not beat the existing linear kernels. A more promising next step is a larger fusion boundary such as fused RMSNorm plus packed QKV projection, not just input staging alone.

## Fused RMSNorm + Packed QKV Projection Experiment
- Added a packed-QKV decode helper that projects Q/K/V with a single packed linear using concatenated `[q; k; v]` weights instead of three separate projection launches.
- `PagedLlamaAttention` now supports a `precomputed_qkv` decode path so the lightweight graph runner can prepare packed Q/K/V before entering paged attention.
- `LightweightLlamaLayer` can now run a decode-only fast path of `input RMSNorm -> packed QKV projection -> paged attention`, controlled by `NANOPOINTLLM_PACKED_QKV_DECODE=1`.
- Added parity coverage for the packed decode QKV projection helper.
- Real PointLLM-7B, strict CUDA graph, B=8, decode_steps=64 with packed QKV enabled:
  - padded total `151.5 tok/s`, replay-only `151.2 tok/s`
  - paged total `246.3 tok/s`, replay-only `278.8 tok/s`
- The same benchmark with packed QKV gated back off:
  - padded total `163.1 tok/s`, replay-only `162.9 tok/s`
  - paged total `253.8 tok/s`, replay-only `285.5 tok/s`
- Interpretation: packed QKV projection is functionally correct, but this first version still loses to the current default graph path. The extra precompute path and tensor handling are not yet amortized by the launch reduction. The next profitable step is likely a tighter fusion boundary such as a single custom kernel for RMSNorm plus packed QKV, or direct integration with the attention projection/output path, rather than a Python-level packed-linear rearrangement alone.

## Custom Kernel: RMSNorm + Packed QKV
- Added a decode-only Triton kernel that computes RMSNorm and packed QKV projection in one custom path, exposed through `PagedLlamaAttention.project_qkv_decode_fused(...)`.
- `LightweightLlamaLayer` can now call this fused path under `NANOPOINTLLM_FUSED_RMSNORM_PACKED_QKV=1`, bypassing the separate input-layernorm forward plus packed-QKV projection path.
- Added CUDA parity coverage against the reference sequence of `RMSNorm -> packed linear`.
- Real PointLLM-7B, strict CUDA graph, B=8, decode_steps=64 with `NANOPOINTLLM_FUSED_RMSNORM_PACKED_QKV=1`:
  - padded total `185.4 tok/s`, replay-only `185.6 tok/s`
  - paged total `47.7 tok/s`, replay-only `49.4 tok/s`
- Interpretation: this first fused custom kernel is numerically correct but dramatically slower than the current default graph path. The likely reason is that the kernel is doing a per-row fully unrolled output sweep with repeated weight streaming, which loses badly on memory traffic and occupancy. It should remain an explicit experiment only. A better next step would be a tiled kernel over output columns with shared normalized staging reuse, or a CUTLASS/cuBLAS-backed path that reuses a fused RMSNorm staging buffer more efficiently.

## Tiled Fused Kernel vs Staged Normalized Buffer
- Reworked the custom fused kernel to launch over `(row, output_tile)` instead of having one program sweep the entire output width for a row.
- Added a second decode-only path that does:
  - Triton RMSNorm into a staged normalized buffer
  - packed QKV projection via the regular GEMM path
- The staged path is exposed via `PagedLlamaAttention.project_qkv_decode_rmsnorm_staged(...)` and `NANOPOINTLLM_RMSNORM_STAGED_PACKED_QKV=1`.
- Real PointLLM-7B, strict CUDA graph, B=8, decode_steps=64:
  - tiled fused kernel (`NANOPOINTLLM_FUSED_RMSNORM_PACKED_QKV=1`): paged total `182.6 tok/s`, replay-only `199.6 tok/s`
  - staged normalized buffer (`NANOPOINTLLM_RMSNORM_STAGED_PACKED_QKV=1`): paged total `241.0 tok/s`, replay-only `292.9 tok/s`
- Interpretation: changing the fused kernel to output-tile parallelism helped a lot versus the first version, but it is still materially slower than the staged-buffer approach. The staged normalized buffer path is the first decode-side QKV optimization in this line that beats the prior default graph baseline.

## Default Graph Decode Update
- Promoted the `RMSNorm -> staged normalized buffer -> packed GEMM` path into the default decode-graph lightweight runner construction, while keeping both packed-QKV-only and fully fused custom-kernel paths as explicit experiments.
- Real PointLLM-7B, strict CUDA graph, B=8, decode_steps=64 after promoting the staged path to the default graph runner:
  - padded total `189.6 tok/s`, replay-only `189.8 tok/s`
  - paged total `259.6 tok/s`, replay-only `294.1 tok/s`
- Interpretation: the new default graph path improves over the previous default replay-only result (`285.5 tok/s`) and is currently the best measured strict-graph decode path in this repo.

## Output Projection And MLP Scheduling
- Added a decode-only staged output projection helper for attention output, so the precomputed paged-attention result goes through a direct staged `F.linear` instead of the generic module call shape path.
- Added a packed `gate+up` projection path for the MLP decode side, reusing the same idea as packed QKV: one GEMM into a staged tensor, split into gate/up, then SiLU and down projection.
- Promoted the packed `gate+up` decode path into the default decode-graph lightweight runner construction.
- Real PointLLM-7B, strict CUDA graph, B=8, decode_steps=64 after these scheduling updates:
  - padded total `195.2 tok/s`, replay-only `195.2 tok/s`
  - paged total `265.5 tok/s`, replay-only `298.5 tok/s`
- Interpretation: the attention output scheduling plus packed MLP input-side projection buys a smaller but real additional decode improvement on top of the staged RMSNorm+QKV path, pushing the current best strict-graph paged replay result to `298.5 tok/s`.

## Point Encoder Reuse
- Added per-sequence multimodal caches beyond `inputs_embeds_cached`:
  - `point_features_cached`
  - `point_cloud_cache_key`
  - `input_embed_layout_cached`
- `PointLLMWrapper` now exposes:
  - `point_cloud_cache_key(point_cloud)` for value-stable request reuse
  - `analyze_input_layout(input_ids)` to precompute patch-token splice positions once
  - `prepare_inputs_embeds(..., layouts=...)` so repeated requests skip rescanning patch token locations
- Both padded and paged runners now keep a runner-local `_point_feature_cache` keyed by point-cloud content hash. Repeated requests with the same point cloud can reuse projected point features instead of re-running the point backbone.
- Added `scripts/bench_point_encoder_cache.py` to compare:
  - unique point clouds
  - repeated point clouds with a fresh engine
  - repeated point clouds with a persistent engine cache
- Real PointLLM-7B, paged, B=4, decode_steps=1, runs=1:
  - unique point clouds: prefill `3041.74ms`
  - repeated point clouds, fresh engine: prefill `428.37ms`
  - repeated point clouds, persistent engine reuse: prefill `226.18ms`
- Interpretation: the first worthwhile encoder-side win is reuse, not deeper kernel work. The framework now has a concrete path for multi-turn / repeated-point-cloud workloads, but encoder execution itself is still the original PointBERT/PointLLM backbone path and remains a future optimization target.

## Engine-Level Point Feature Cache
- Promoted point-feature reuse from per-runner implementation detail to an explicit `PointFeatureCache` subsystem with LRU behavior and hit/miss/insert/eviction stats.
- `PointLLMLLMEngine` now owns one shared `point_feature_cache` and passes it into padded or paged runners, so repeated point-cloud requests within the same engine/session reuse the same cached projected features.
- `PointLLMLLMEngine.add_request(...)` now accepts:
  - `point_cloud_cache_key`
  - `point_features_cached`
- `LLM.generate(...)` request dictionaries now pass through those same cache-oriented fields, making multi-turn repeated-point-cloud workflows explicit at the public API layer.
- Added engine helpers:
  - `get_point_feature_cache_stats()`
  - `clear_point_feature_cache()`
- Interpretation: the framework now has an explicit engine/session-level multimodal reuse mechanism. The next higher-value work is no longer cache plumbing, but deciding whether to add persistent cross-engine caching or to optimize the PointBERT backbone itself.

## PointBERT Backbone Hotspots
- Added `scripts/profile_pointbert_backbone.py` to split PointBERT into:
  - group stage: `FPS`, `KNN`, `gather+normalize`
  - local encoder stage: `first_conv`, `first_pool`, `expand_cat`, `second_conv`, `second_pool`
  - bridge stage: `reduce_dim`, `pos_embed`
  - transformer stage: per-block `norm1`, `attn`, `norm2`, `mlp`, `total`
  - tail: `norm+pool`
- Real PointLLM-7B, A100, `B=1`, `warmup=1`, `runs=3`:
  - backbone total: `7.51ms`
  - FPS: `3.03ms`
  - KNN: `0.50ms`
  - gather+normalize: `0.23ms`
  - local encoder total (first conv/pool + second conv/pool + expand_cat): about `1.33ms`
  - transformer block total (12 blocks): about `6.75ms`, with attention alone about `3.43ms`
- Real PointLLM-7B, A100, `B=4`, `warmup=1`, `runs=3`:
  - backbone total: `11.04ms`
  - FPS: `3.09ms`
  - KNN: `0.83ms`
  - gather+normalize: `0.14ms`
  - local encoder total: about `3.25ms`
  - transformer block total (12 blocks): about `4.81ms`, with attention alone about `3.20ms`
- Interpretation:
  - `FPS` is a persistent hotspot and does not scale much with batch size, so it is a prime candidate for optimization or reuse.
  - `KNN` is much smaller than the earlier coarse breakdown suggested once warmup noise is removed.
  - `local encoder` is meaningful, but the main steady-state PointBERT cost is the transformer stack, especially block attention.
  - `reduce_dim`, `pos_embed`, `projector`, and `splice` are all comparatively small and should not be the next optimization target.

## End-to-End Performance Diagnostics
- Added `scripts/benchmark_pointllm_diagnostics.py` with standard metric semantics:
  - TTFT includes point encoding, prefill, LM head, and first-token sampling.
  - request TPOT is decode wall time divided by decode steps and is not divided by batch size.
  - aggregate decode throughput is `batch_size * decode_steps / decode wall time`.
- CUDA profiling now covers all decoder layers plus final norm, LM head, and sampling without silently switching eager profiling to graph-optimized module settings.
- KV reporting separates physical pool reservation, block-rounded active allocation, payload, and fragmentation.
- HBM reporting is explicitly an analytical minimum-traffic estimate (one decoder-weight read plus KV reads/writes), not an Nsight hardware counter.
- Real shared-GPU matrix: B=1/4, input=560/768, output=8/32, warmup=1, runs=2.
  - At input=768/output=32: B1 TTFT `793.69ms`, TPOT `65.49ms`, decode `15.27 tok/s`; B4 TTFT `1516.12ms`, TPOT `138.03ms`, decode `28.98 tok/s`.
  - Decode component profile at input=768: B1 attention `52.77%`, MLP `17.82%`, model-other `29.04%`, LM head `0.30%`, sampling `0.06%`; B4 attention `39.38%`, MLP `38.00%`, model-other `21.62%`, LM head `0.93%`, sampling `0.07%`.
  - KV bytes/token are `524288` for PointLLM-7B BF16; input=768/output=32 uses `0.390625GiB` at B1 and `1.5625GiB` at B4.
  - Estimated minimum traffic is about 97% decoder weights at B1, supporting weight quantization as the first small-batch decode lever.
- Every GPU was at 100% utilization before and after the run. Absolute latency, batching efficiency, and effective HBM bandwidth are therefore diagnostic shared-load values, not isolated A100 peak numbers.

## Isolated-GPU Benchmark Methodology
- The diagnostic benchmark now defaults to `warmup=10` and `runs=30`; each distribution includes mean, median, P10, P90, P95, min, and max.
- Every exact `(input length, output length, batch size)` shape records one `cold_start_run` before warmup. Model load and engine initialization are separately timed.
- `--require_idle_gpu` checks the target GPU UUID before loading weights and rejects runs above configurable utilization/memory thresholds. On the currently busy host it correctly rejected GPU0 at `100%` utilization and `6823MiB` used.
- A background `nvidia-smi` process samples only across the group of measured runs, not once inside each timed function. It records P-state, SM/memory clocks, power, temperature, coarse GPU utilization, and memory-controller utilization at 200ms by default.
- Power values that remain `0W` are explicitly marked unavailable rather than interpreted as zero. GPU1 on this host exposed this sensor limitation.
- Synchronization policy:
  - one synchronization before a timed phase to drain prior work;
  - one end-event synchronization required to read each phase;
  - no telemetry-triggered synchronization inside the timed region;
  - the token-dependent sampler `.tolist()` remains an inherent autoregressive host/device dependency.
- Added an NCU-only representative decode entry point with one NVTX push/pop range. Nsight Compute 2023.2 requires the filter `pointllm_decode_profile/` for that range.
- NCU metrics are `sm__cycles_active.avg.pct_of_peak_sustained_elapsed`, `dram__throughput.avg.pct_of_peak_sustained_elapsed`, and `gpu__time_duration.sum`; the summarizer duration-weights kernel metrics and can convert DRAM percent to estimated GB/s using the supplied peak.
- NCU replay is never used for TTFT/TPOT because it perturbs latency. It is reserved for one B1 and one B8 representative step.
- Real Phase 22 smoke under shared load confirmed the schema: shape-cold TTFT `5400ms` versus one-run steady TTFT `772ms`, with automatic warnings for insufficient warmup/runs, non-idle GPU, and unavailable power. These values are functional validation only.

## Native Paged Runtime And Strict Decoder Roofline
- The paged engine now uses `LightweightLlamaRunner` for prefill, mixed prefill/decode, and pure decode. HuggingFace remains the checkpoint/module container and an explicit debug fallback, not the normal model-level execution path.
- The runtime follows the vLLM execution pattern but is locally implemented: FIFO continuous scheduler, block manager, paged KV pool, Triton paged attention, packed QKV/Gate-Up projections, CUDA Graph decode buckets, and compiled batch sampling. It does not currently link vLLM or FlashInfer.
- Decoder NCU scopes are `pointllm_qkv`, `pointllm_attention`, `pointllm_o_proj`, `pointllm_mlp`, `pointllm_lm_head`, and `pointllm_sampling`.
- Each scope has a manifest for actual runner weight bytes, activation bytes, paged K/V bytes, estimated FLOPs, and arithmetic intensity. Attention is classified from K/V traffic and is never mislabeled weight-bound.
- A strict linear-stage weight-bound result requires: idle GPU at launch, >=80% modeled compulsory traffic from weights, arithmetic intensity below the A100 roofline ridge, measured DRAM bytes >=50% of modeled traffic, DRAM throughput above the configured threshold, and DRAM pressure above SM throughput.
- Publishable B1/B8 runs completed after the idle gate observed 0% utilization and less than 40 MiB used on GPU 0. Both manifests record `idle_at_start=true`.
- QKV/O/MLP are strictly weight-bandwidth-bound at both B1 and B8. Their duration-weighted DRAM throughput is 63-75%, SM throughput is 25-31%, and measured bytes closely match modeled weight traffic.
- LM head is strictly weight-bandwidth-bound at B1 (80.29% DRAM), but B8 selects a CUTLASS GEMM that reaches only 24.17% DRAM and 13.89% SM. It remains weight-stream dominated but needs shape-specific kernel selection/autotuning rather than only more bandwidth.
- Whole-stage attention is KV-memory-bound/underfilled. The dominant tile reaches 37.16% DRAM at B1 and 71.90% at B8, while surrounding small kernels pull whole-stage throughput down to 7.92% and 34.98%, respectively.
- Scaled marked-stage shares identify MLP as the first optimization target (48.65% B1, 40.66% B8), followed by attention (21.88%, 34.15%) and QKV (20.83%, 15.17%). These are not end-to-end TPOT shares.
- The corrected real PointLLM-7B run produced mixed rounds `(1 prefill, 1 decode)`, `(1,3)`, and `(2,1)` across three request mixes; every generated token matched the unmodified HF greedy reference.
- The NCU orchestrator also writes `stage_bottleneck.png` with stage CUDA time, DRAM versus SM throughput, and arithmetic intensity versus the roofline ridge; CSV/JSON remain the source of truth.

## Decode MLP Block Trace Design
- The default lightweight decode path already materializes the exact SwiGLU intermediate `SiLU(gate) * up` inside `LightweightMLP`, including the packed gate/up path used by CUDA Graph decode. A disabled-by-default observer at this point can trace all layers without changing normal runtime behavior.
- PointLLM-7B uses 32 decoder layers. The trace should not retain full `[steps, layers, intermediate]` tensors on GPU; block scores and selected block IDs can be reduced per forward and copied to CPU.
- Existing scripts provide real PointLLM loading, prompt construction, point-cloud generation, and paged engine stepping. The new analysis should reuse those helpers while adding repeated-point-cloud / distinct-prompt grouping explicitly.
- Block importance must reflect each intermediate neuron's actual down-projection contribution, not activation magnitude alone. For block `b`, use the output-space norm of `sum_{i in b} z_i W_down[:, i]`, where `z = SiLU(gate) * up`; this captures cancellation within a block and is the correct post-gate contribution for block-level weight skipping.
- For scalable all-layer/all-token curves, the primary additive score will follow the requested neuron contribution form: `c_i = |z_i| * ||W_down[:, i]||_2`, with block mass `C_b = sum_{i in b} c_i`. This permits exact retained-mass accounting without materializing one hidden-size output vector per block. Raw `z` tensors are retained separately so alternative scores can be recomputed offline.
- The paged engine's first generated token is produced by prefill; traceable decode passes start with the second generated token. Reports must label both the decode pass index and input/output generated-token indices to avoid an off-by-one interpretation.
- The host has idle A100 GPUs and the safetensors checkpoint at `/mnt/llm_data/pointllm_ckpt/PointLLM_7B_v1.2_safetensors`. Real ModelNet/Objaverse data loaders also exist under `/home/PointLLM`; synthetic point clouds should be only a smoke-test option, not silently presented as dataset evidence.
- Real ModelNet smoke (one point cloud, two prompts, three decode passes) captured exactly 192 records (`2 * 3 * 32`) with hidden size 4096 and intermediate size 11008. At 64-neuron blocks and an actual 10.47% retained ratio, current-token top blocks retained only 14.89% of additive contribution, previous-token blocks retained 12.19%, adjacent-token Jaccard was 0.20, and same-point cross-prompt Jaccard was 0.57. This early evidence argues against assuming strong MLP block sparsity before the longer run.
- The formal ModelNet run covered two point clouds, four prompts each, 32 generated tokens, 31 decode passes, and all 32 layers (7,936 layer-token records). At 64-neuron/10.47% budget, current contribution retention was 14.63%, previous-token retention 11.63%, adjacent Jaccard 14.37%, previous recall 24.23%, and cross-prompt Jaccard 16.14%. At 50% budget current retention was only 57.17%; at 75% it was 80.24%.
- A one-point-cloud Objaverse check closely reproduced the contribution and temporal results (14.67% current retention, 11.66% previous retention, 14.61% adjacent Jaccard at 64/10%), while cross-prompt Jaccard dropped to 8.20%. This supports the conclusion that contiguous MLP block masks are neither strongly sparse nor point-cloud-static, but the Objaverse sample count is not sufficient for a dataset-level confidence interval.
- BF16-preserving and temporary FP16-storage analyses differed by less than `3.4e-10` in aggregate contribution ratios and produced identical support-overlap metrics; the published/raw canonical run preserves BF16 exactly.

## Structured Sparsity Source Check
- The NeurIPS 2021 paper "Channel Permutations for N:M Sparsity" treats channel permutation as a function-preserving layout transformation that can increase compatibility with structured N:M patterns; this supports using exact MLP neuron permutations only as an upper-bound test, not as a learned compression method. Source: https://proceedings.neurips.cc/paper_files/paper/2021/hash/6e8404c3b93a9527c8db241a1846599a-Abstract.html
- SparseGPT reports one-shot pruning without retraining and explicitly generalizes to semi-structured 2:4 and 4:8 patterns. Source: https://arxiv.org/abs/2301.00774
- Wanda ranks weights using weight magnitude and input-activation statistics, applies masks per output, and requires neither retraining nor a weight update. Its official implementation exposes unstructured, 2:4, and 4:8 modes. Sources: https://arxiv.org/abs/2306.11695 and https://github.com/locuslab/wanda

## Neuron-Level Oracle Gate
- ModelNet B=1: Gini `0.5800`, normalized entropy `0.9313`, effective-support ratio `0.5321`, top-10% contribution retention `0.4092`, and neuron ratio for 90% contribution `0.5221`.
- Objaverse B=1 closely matched: Gini `0.5813`, normalized entropy `0.9308`, effective-support ratio `0.5297`, top-10% retention `0.4109`, and ratio for 90% contribution `0.5214`.
- Grouping rapidly removes the limited neuron-level concentration. On ModelNet, B=32 has normalized entropy `0.9933`, effective-support ratio `0.9636`, top-10% retention `0.1597`, and needs `0.8519` of neurons for 90% contribution.
- Both datasets trigger the predeclared stop rule (`B=1 retention@10% < 0.50` and `ratio-for-90% > 0.50`). Activation sparsity is therefore closed, and exact permutation/clustering search is gated off: a permutation cannot exceed the arbitrary-neuron oracle.

## Static N:M Wanda Pruning
- Implemented the official Wanda metric `|W| * sqrt(E[x^2])` from real PointLLM multimodal calibration inputs and exact row-wise contiguous N:M masks across decoder QKV/O/MLP weights. The source checkpoint is never modified.
- ModelNet two-point/two-prompt/eight-token 2:4 result: KL mean `0.05982`, KL max `0.29026`, teacher-forced top-1 agreement `0.9375`, free-generation token match `0.50`, and exact-request match `0.50`.
- The matching 4:8 result: KL mean `0.05045`, KL max `0.22979`, top-1 agreement `0.90625`, free-generation token match `0.65625`, and exact-request match `0.50`.
- Both patterns create exact 50% decoder sparsity but already alter autoregressive outputs on half the requests. Dense PyTorch execution cannot realize N:M speedup, so no runtime claim is made.
- SparseGPT's official full reconstruction needs a dense input Hessian and Cholesky factor per linear layer; the 11,008-wide MLP down-projection alone requires about 485 MB for one FP32 Hessian and cubic factorization. A full 32-layer run is deferred rather than replacing it with a diagonal approximation mislabeled as SparseGPT.

## Full-Pipeline Precision Sensitivity
- Per-output-channel W8 simulation preserved all four eight-token free generations for PointBERT, projector, decoder QKV/O/MLP, and LM head; component KL means were between `0.00016` and `0.00037`.
- W4 sensitivity is strongly component-dependent. LM head produced KL mean `0.99191`, only `21.88%` free-token match, and `0%` exact-request match. Decoder MLP produced KL mean `0.00970` and `75%` exact-request match.
- PointBERT, projector, decoder QKV, and decoder O retained 100% exact-generation match in this small W4 run, but teacher-forced drift was nonzero and the sample is not large enough to certify a production precision mode.
- Unified PE guidance from current evidence: support W8 broadly; preserve at least W8 for LM head; allow independently selectable precision for decoder MLP; treat W4 frontend/QKV/O as candidates requiring a larger task-level validation rather than a default.

## Strict PointBERT NCU Roofline
- Formal B1/B4 runs profiled the real `8192 -> 512x32 -> 513x384` PointBERT and all 12 PointTransformer layers on an idle A100. NCU measured bytes/time/throughput are separated from explicitly modeled FLOPs.
- B1 total marked-stage time was `5.860 ms`: FPS `2.951 ms` (`50.35%`), PT attention `1.028 ms` (`17.54%`), local encoder `0.914 ms` (`15.60%`), and PT MLP `0.420 ms` (`7.16%`).
- B4 total marked-stage time was `10.448 ms`: local encoder `3.036 ms` (`29.06%`), FPS `3.008 ms` (`28.79%`), PT attention `2.347 ms` (`22.47%`), and PT MLP `0.888 ms` (`8.50%`).
- FPS is almost batch-invariant and reaches only `0.29%` SM throughput at B1 and `1.15%` at B4. The first frontend target is a higher-occupancy/fused FPS implementation, not bandwidth compression.
- PT QKV/O/MLP measured-byte AI is at or above the A100 ridge (`200.64 FLOP/byte`) while DRAM throughput remains `4.6-14.5%`; these 513-token GEMMs do not reproduce the decoder's small-batch weight-bandwidth bottleneck.
- PT attention remains below the ridge (`36.94` AI B1, `31.64` B4), consumes `17.54-22.47%` of marked time, and materializes eager score tensors. Fused SDPA/FlashAttention is the primary transformer software target.
- KNN measured AI is low, but radix/top-k and irregular selection dominate. Distance+selection fusion should be evaluated instead of interpreting it as a dense GEMM roofline point.

## Architecture Simulator Direction
- The simulator must represent a three-regime transition rather than a single fixed systolic dataflow: geometry/reduction, high-M dense GEMM, and low-M weight-streaming GEMV.
- First-order design variables are tensor-array rows/columns, dense tile shape, Split-K degree, output-channel parallelism, geometry lanes, reduction latency, HBM/SRAM bandwidth, weight-buffer capacity, W4/W8/BF16 mode, attention tile sizes, and unified-SRAM bank allocation.
- The first implementation should be trace-driven and analytically cycle/traffic accurate at phase level. It must expose calibration factors instead of claiming RTL cycle accuracy before C-model/RTL correlation.
- PointLLM presets should preserve measured workload differences: FPS persistent min-distance state, PointTransformer/LLM-prefill dense mode, and decoder QKV/O/MLP/LM-head weight-streaming mode with component-specific precision.
- `DingdongD/Compiler_Codes` is available at commit `ad2c31a` and separates compiler, functional simulator, cycle simulator, runtime, and RTL. Its reusable ideas are typed operator/DDR instruction parsing, YAML architecture configuration, per-engine models, explicit DDR manager, and operation timelines; the implementation itself is too broad to import as a dependency.
- `/home/PointAcc/QuickFPS` already provides a stronger geometry reference than a one-line latency equation: resident point-buffer mode, independent coordinate/min-distance DMA streams, queue/outstanding limits, cycle events/counters, optional DRAMsim3, and optional PTPX activity-to-energy mapping.
- `/home/PointAcc/Mamba_Reconfigurable_Array_v2` provides the relevant RTL handoff convention: one functional C/Python reference, parameterized reconfigurable array/PE and banked SRAM blocks, trace/profile testbenches, golden-vector checks, gate-activity tests, and PTPX summary scripts.
- Nano-pointllm should therefore use a lightweight Python phase-level model now, but standardize stable JSON contracts for configs, workload traces, counters, timelines, and characterization inputs so engines can later be replaced by C++/SystemC or RTL-correlated implementations.
- The first executable GTSU model now covers all five runtime phases and preserves the fidelity boundary: NCU values are references, while energy/area stay disabled without characterized inputs.
- In the `B1/S768/O128` W8 baseline, decode is 61.65% of modeled end-to-end cycles. Decode linear operators are unpack-bound for 76.84% of decode cycles, while decode attention accounts for the compute-bound remainder; prefill is compute-bound.
- At fixed 32x32 resources and 819.2 GB/s HBM, raising unpack throughput from 256 to 512 B/cycle improves modeled decode from 4366.21 to 2688.63 ms. At 1024 B/cycle, decode reaches 2322.55 ms and becomes HBM-dominant, identifying an explicit weight-streaming balance point rather than an unbounded width target.
- The 320-design unique DSE predicts a minimum-latency point at 32x32, Split-K 16, 256 geometry lanes, 819.2 GB/s HBM, 16 MiB SRAM, and 1024 unpack B/cycle. This is an analytical trend only; equal-area comparison remains blocked on synthesis/PTPX/CACTI characterization.

## Architecture Estimator Correctness Audit
- The implementation published in `a5650f7` was an operation-level Python estimator, not a complete simulator. The `cmodel/`, `rtl/`, `verification/`, and `characterization/` directories contained documentation only and must not be counted as implementations.
- The old decoder-attention mapping serialized 32 heads through the tensor model. For the B1 decode shape, it reported `1,011,051,776` attention compute cycles versus a configured-peak MAC lower bound of `13,623,296`, an approximately `74.21x` inflation. This invalidates the old `7082.26 ms`, `1.406x`, and old attention breakdown.
- The workload also omitted PointBERT `reduce_dim`, position MLP, transformer norm/GELU/residual/final norm, projector GELU, and decoder norm/RoPE/SiLU/residual/sampling work. KNN counted a distance matrix as HBM traffic in cycles but not in reported bytes, and sub-byte activation byte accounting used integer truncation.
- The corrected estimator parallelizes head/query rows, restores the omitted operation classes, makes KNN HBM accounting consistent, and uses packed-byte arithmetic. A validator checks real checkpoint config, PointBERT YAML, safetensors tensor headers, workload shapes, MACs, payload bytes, phase totals, traffic totals, and configured-peak lower bounds.
- The corrected B1/S768/O128 run passes `374/374` shape and analytical-conservation checks. This establishes internal consistency only; it does not validate scheduling cycles, bank conflicts, DRAM timing, area, energy, or hardware speedup.
- Corrected exploratory baseline: `6205.75 ms` end-to-end and `3460.21 ms` decode. The searched analytical minimum is `4161.77 ms` end-to-end and `1416.54 ms` decode. These are equation sensitivities and are explicitly non-publishable until event C-model and RTL correlation exist.

## Cycle-Accurate Architecture Requirement
- The user requires cycle-accurate architecture modeling, not an expanded analytical estimator. Full-model latency must therefore remain disabled until every lowered operator has an RTL-correlated implementation.
- `/home/Compiler_Codes/cyclesim/ACTransformerCycleSim` provides useful resource, DMA, local-memory, trace, and ready/valid organization, but its ACTransformer timing constants and CTC dataflow are not PointLLM golden references.
- The correct implementation order is a shared micro-op contract, functional semantics, an event-level ready/valid model, RTL using the same transaction contract, and strict functional/traffic/cycle comparison.
- The first vertical slice is W8 weight DMA -> compressed WBUF -> unpack/scale -> Split-K GEMV -> partial-sum reduction -> output. Persistent FPS is the second anchor. Dense GEMM, attention, vector, KNN/top-k, LM head, and sampling remain unsupported for cycle-accurate reporting until their own RTL slices exist.
- A cycle model is not validated merely because it advances one cycle at a time. Cycle accuracy requires a locked RTL microarchitecture and exact or bounded event/cycle correlation against that RTL configuration.
- The local toolchain includes Icarus Verilog 12.0, Verilator 4.038, and Yosys, so RTL simulation and structural synthesis checks are available in this workspace.
- Compiler_Codes `ReadyValidModule` tracks only `busy_until`; it lacks distinct latency/initiation interval, in-flight pipeline entries, and FIFO backpressure. Its DMA uses request-level `latency + ceil(bytes/bandwidth)`, and its local memory assigns an entire request to one derived bank. These are useful scaffolds but insufficient as direct GTSU cycle-accurate models.
- The new GTSU slice must model beats, not whole-operation byte counts: HBM beats enter finite compressed-weight buffers, unpack output enters a finite decoded FIFO, PE accepts according to II, and reduction/output retirement are explicit transactions.
- The first locked W8 Split-K GEMV configuration completes in exactly 32 cycles in both the Python edge-state model and Icarus RTL. All 84 events, 24 input/unpack/compute beats, 8 partial sums, 4 outputs, and output values `[1, -5, -4, 11]` match exactly.
- The parameterized RTL also passes exact correlation under FIFO depth 1 with Split-K 3 and eight lanes, plus a no-output-stall/deeper-FIFO configuration. Yosys accepts the RTL through hierarchy, process, memory, optimization, and structural checks.
- This does not cover a production PointLLM linear operation: source timing is deterministic, activation/weight SRAM addresses and banks are absent, and no DRAM backend is connected. The coverage registry therefore marks generic PointLLM Split-K linear as unsupported.
- `/home/PointAcc/QuickFPS` is substantially stronger than the generic Compiler_Codes scaffold: it has synthesizable bucket/point/DMA/ping-pong RTL, finite-credit and ready/valid contracts, analytical banked DDR plus pinned DRAMsim3 backends, RTL-cycle validation, and checked PPA/energy records.
- QuickFPS remains an external FPS architecture at commit `7008dd9`, not a GTSU-integrated block. It may serve as the second anchor only after its checked RTL validation is rerun and its event/traffic schema is adapted; its 760/395-cycle example cannot be inserted directly into PointLLM end-to-end timing.
- The QuickFPS local RTL suite was rerun successfully: the Point-Engine retained its 39-cycle contract, bucket decisions retained the II=1 window, and the integrated stream subsystem completed its smoke in 202 cycles with nonzero coordinate/distance-read/distance-write traffic.
- Compiler_Codes LBUF does not use the earlier cycle-model's whole-request-to-one-bank abstraction. The RTL exposes a 16-bank mask, 128-bit data per bank, independent read/write paths, client-ready arbitration, byte enables, and 2/3-cycle read-valid timing. GTSU SRAM must therefore schedule bank beats explicitly.
- The available local DRAMsim3 bridge submits one `AddTransaction` per address and returns tagged callbacks; transfer size is metadata on the Python side. For the local DDR4 x8 2400 configuration (`BL=8`, `bus_width=64`), the lowering must split traffic into aligned 64-byte requests and scale the 0.83 ns DRAM clock into accelerator cycles.
- PointLLM source weights are FP32 safetensors. Any W8 accelerator stream is a target-layout lowering after quantization, not the checkpoint's physical byte layout; reports must preserve both source dtype/shape and target W8 payload provenance.
- Equal Split-K element counts are incompatible with 64-byte request alignment for the `K=11008` down projection. The exact lowering partitions each row's 172 bursts as twelve 11-burst (704-element) partitions and four 10-burst (640-element) partitions; variable partition lengths must be supported by the future compute scheduler.
- The complete target W8+FP16-scale decoder address plan spans `0x100000000` through `0x28ff78000` (6,710,329,344 allocated bytes including alignment) without overlap. A single layer's seven projections plus the shared LM head contain 333,459,456 W8 payload bytes and 5,210,304 scale bytes.
- The formal q-projection sample sent 64 aligned 64-byte reads through the pinned DRAMsim3 bridge at 1 GHz with 32 outstanding credits. It completed in 246 accelerator cycles / 296 DRAM ticks; request latency was 34/97/158 cycles (min/median/max). This characterizes only the sampled request stream, not linear or decoder latency.
- The synthesizable SRAM wrapper uses behavioral arrays. Exact 16-bank production geometry is covered by Icarus event/cycle simulation; the Yosys gate is intentionally a compact default-geometry logic smoke because no foundry SRAM macro has been selected or characterized.
- DRAM outstanding count alone is not a sufficient ROB sizing rule. With 32 outstanding Q-projection requests, unconstrained DRAM scheduling produced up to 2,217 completed requests waiting behind an older tag. A bounded sequence window relative to the oldest unretired request is required for a physically finite ROB.
- The 8-word WBUF bank offset eliminates Q/O/Gate/Up bank overlap for fixed 64-byte K chunks, but the non-uniform down-projection schedule still creates 217,088 conflicting read pairs (30.81% of 704,512 bursts). Bank layout or separate activation/weight banks is therefore a concrete next memory optimization.

## PointKAN Geometry Reference Audit
- `DingdongD/PointKAN_Accel` is reachable and pinned locally at commit `aa356da232a9d30d6c784297603c2a5596c6409a`; the public web search index did not expose its `hw_sim` contents, so source-level findings must come from this pinned checkout rather than snippets from unrelated repositories.
- The first sparse clone timed out before checkout. Its Git object database is intact, but the empty worktree reports tracked files as deleted; only the requested `hw_sim` subtree will be materialized for read-only audit.
- PointKAN timing or accuracy numbers are not PointLLM evidence by themselves. Any reusable distance/FPS/KNN contract must be relowered to PointLLM's exact 3-D coordinate format, `8192 -> 512` FPS loop, 512 query centers, and `K=32`, then correlated against nano-pointllm RTL before coverage can be enabled.
- The `hw_sim` tree is heterogeneous rather than one integrated PointKAN RTL: it contains Python cycle simulators, vendored DRAMsim3 trees, copied QuickFPS point/bucket RTL, old PointACC merge/top-k RTL, and APU BSCU RTL. This is useful prior design material but does not establish an end-to-end PointKAN or PointLLM RTL timing contract.
- The repository is Apache-2.0 licensed. Any adapted source must retain provenance and changed-file notices; a clean-room contract-level implementation is preferable where the existing modules do not match PointLLM semantics.
- The primary PointKAN distance RTL is a 64-PE, three-coordinate, signed fixed-point systolic chain. It accepts all 64 database points through three static packed buses, emits one result per cycle after fill, and claims 66 compute cycles. It has no SRAM request/response interface, DMA path, finite input queue, ready/valid backpressure, or bank-conflict behavior.
- `dist_topk_engine` reduces one 64-point batch only; its own comments leave cross-batch top-k merge to an external controller. PointLLM KNN needs exact top-32 across 128 such batches for each of 512 centers, so this is not a complete KNN engine.
- `fps_min_dits2_table_128.v` is only a 128-lane persistent `min(old,new)` register table. It has no explicit operation reset, 512-iteration argmax, selected-center feedback, point streaming, or SRAM/DRAM integration, so it is a useful primitive but not an FPS RTL slice.
- PointKAN's source width defaults differ across duplicated trees (16-bit/48-bit accumulator versus 32-bit/80-bit). PointLLM must lock one quantization/overflow contract before functional and cycle correlation.
- The pinned PointKAN 64-PE distance RTL and its original testbench were compiled with Icarus and rerun in isolation. All five signed/unsigned functional cases passed, while the testbench printed approximately 71 cycles per batch rather than the source comment's 66-cycle headline; cycle boundaries are therefore not safe to reuse without a shared event definition.
- The original full-width `K_MAX=64` top-k testbench compiled but did not produce a result within a bounded 50-second audit window and was terminated. Its all-slots-wide combinational merge tree is not accepted as the PointLLM KNN implementation or timing reference.
- The new Nano-pointllm 64-lane streaming distance tile correlates exactly for 448 per-lane outputs across seven tiles: Python and Icarus both complete in 12 cycles with identical events, counters, and values. Yosys checks the same 64-lane/16-bit/36-bit parameterization (192 multipliers and 192 subtractors in generic RTLIL).
- PointLLM exact distance work is 4,194,304 evaluations for FPS and again for KNN. With 64 lanes this is 65,536 issue tiles per operator, but it is only a lower bound because FPS min/argmax barriers, KNN top-32, and memory stalls are still absent.
- Geometry bit-exactness currently applies to the locked INT16 contract only. PointLLM's FP32-to-INT16 scale/saturation, distance-order ties, FPS center equality, and KNN top-32 recall have not yet been validated on ModelNet/Objaverse, so no task-level semantic-equivalence claim is allowed.
- The signed boundary regression covers coordinate differences of 65,535 and passes exact Python/Icarus comparison, confirming that the locked 17-bit subtraction and 36-bit three-term accumulator avoid the truncation hazard present in a same-width signed subtraction.
- The next production geometry closure should keep the 49,152-byte INT16 point bank immutable and separately bank the 36,864-byte packed FPS min-state. FPS needs a read-modify-write plus deterministic argmax barrier per center; KNN should use bounded hierarchical local-top32 plus global-top32 merging rather than the audited all-wide combinational tree.

## Shared Geometry-Feature Direction
- The dedicated 64-lane INT16 distance tile remains a valid baseline but cannot support a 100%-shared arithmetic claim because it instantiates 192 geometry-only multipliers.
- The shared formulation is `||p-c||^2 = ||p||^2 + ||c||^2 - 2 p^T c`. A signed INT8 SIMD4 dot PE can execute `(x,y,z,0)` geometry dots and ordinary four-element feature/weight dots with the same multiplier instances.
- The intended claim is 100% sharing of multiplication/MAC resources, not all hardware. FPS min-state/argmax and KNN top-k necessarily remain state/comparator structures.
- A clustered shared fabric is the later system target because a monolithic time-shared array can prevent FPS/KNN from overlapping patch-local feature work. Equal-resource dedicated, fully shared, and spatially partitioned comparisons are required before an architecture winner is claimed.
- W8A8 must be treated as a new sequential PTQ experiment. Existing per-component W8 sensitivity and group-128 lowering do not establish activation quantization fidelity or cumulative 32-layer behavior.
- The current precision driver quantizes one component at a time, immediately dequantizes weights back to BF16/FP16, and never quantizes activations. Its artifacts are isolated weight-sensitivity evidence only, not a deployable W8A8 result.
- The current Split-K RTL performs signed multiplication inside `gtsu_w8_splitk_gemv.sv`; arithmetic sharing is not structurally true until Split-K and geometry instantiate one auditable dot-product primitive.
- The shared geometry path should consume cached point norms and a center norm, execute only `p dot c` on the shared PE, and reconstruct distance with add/subtract/shift. Point-norm generation can itself be scheduled as `p dot p` on the same PE before the FPS/KNN loop.
- The locked 64-PE shared slice completes in 16 cycles for nine alternating beats and matches 576 lane outputs exactly across Python and Icarus. Yosys proves zero direct top-level multipliers, four multipliers per `gtsu_dot4_pe`, and 64 shared instances.
- At 64 PEs, PointLLM's `8192 x 512` distance-dot work maps to a 65,536-cycle arithmetic issue lower bound plus 128 point-norm precompute cycles. Exact INT16-on-INT8 decomposition requires four partial products and maps to 262,144 issue cycles before control/memory overhead.

## Sequential W8A8 Contract
- The locked initial contract is symmetric `[-127, 127]` W8/A8, round-to-nearest ties-to-even, FP16-stored scales, signed INT32 accumulation, and floating-point bias after dequantization. Norm, softmax, nonlinear activation, and residual remain BF16/FP16.
- Per-output-channel W8 and group-128 W8 are separate contracts. Group-128 requires independently scaled INT32 group partials before floating-point summation; it must not be treated as one globally scaled accumulator.
- Static A8 calibration and evaluation point-cloud IDs are disjoint. All selected weights are quantized simultaneously and activation QDQ hooks operate at every selected module input, so errors propagate through PointTransformer and decoder depth rather than being reset for each component.
- The QDQ path remains dense BF16/FP16 execution and can establish fidelity only. It is not evidence of W8A8 kernel speedup or production RTL readiness.
- The first full-component ModelNet smoke quantizes 283 modules and 6,639,776,896 weight parameters simultaneously. On one disjoint held-out request it produced KL mean `0.01443`, KL max `0.05039`, teacher-forced top-1 `1.0`, and eight-token greedy exact match `1.0`; this validates the execution path but is far too small for a fidelity claim.
- On the exact same ModelNet split/request, group-128 reduces weight RMSE from `2.177e-4` to `1.556e-4` but worsens KL mean from `0.01443` to `0.01586` and KL max from `0.05039` to `0.08396`. Both retain exact eight-token output, so there is no current evidence to justify group-128's extra scale/partial-sum complexity as the production default.
- The matching Objaverse per-output/static-A8 smoke gives KL mean `0.01168`, KL max `0.03734`, top-1 `1.0`, and exact eight-token output on one held-out object. This validates cross-dataset execution only; sample size remains insufficient for fidelity certification.
- SmoothQuant alpha `0.5` improves the same ModelNet request to KL mean `0.00481`, KL max `0.01770`, and logit RMSE `0.30625`, with top-1 and greedy exact match both `1.0`. This warrants larger evaluation but does not yet select alpha.
- Dynamic per-token A8 further lowers KL mean/max to `0.00250/0.00771` and RMSE to `0.26934`, but teacher-forced top-1 falls to `0.875` while free greedy exact match remains `1.0`. Distribution averages alone are insufficient; top-token margin/argmax gates remain necessary, and dynamic scale reduction has hardware cost.

## Remaining Full-Graph Gates
- Dense GEMM now instantiates the same `gtsu_dot4_pe`: the 64-PE `5x61x384` microtile has exact value/event/cycle correlation, and Yosys finds zero top-level multipliers plus 256 INT8 multipliers inside 64 Dot4 instances. Full M/N tile sequencing and SRAM double buffering remain absent, so generic `dense_gemm` coverage stays disabled.
- QDQ fidelity and integer execution are now separate evidence paths. Real PointLLM q-projection, down-projection, and LM-head inputs were run through true INT8 x INT8 -> INT32 accumulation; integer output agrees with QDQ FP32 to at most `7.63e-6` on the representative token.
- The real q-projection payload now completes selected rows' full `K=4096` reduction in RTL. Accumulators `[30131, 18552, -2132]` match integer golden exactly in 1025 cycles; this is not evidence for all 4096 output rows or a production W8A8 kernel.
- SmoothQuant metadata is explicit: FP16 smoothing-vector storage, changed activation scale, and current runtime channelwise divide are reported. Independent q/k/v or gate/up vectors cannot all be folded into their shared producer, so zero-overhead folding is not claimed.
- FPS/KNN, attention/vector/sampling, memory closed-loop feedback, and producer-consumer composition remain independent fail-closed gates; no analytical estimate may fill these gaps.

## Compiler_Codes Quantization RTL Audit
- The existing Nano-pointllm W8A8 implementation was independently specified: Python locks symmetric `[-127,127]`, ties-to-even rounding, FP16-stored scales, INT32 accumulation, floating-point dequantization, and post-dequant bias. The current Dot4/dense RTL covers integer multiply/accumulate only; it was not copied from or bit-correlated with `Compiler_Codes` quant/dequant RTL.
- `/home/Compiler_Codes` is pinned locally at `ad2c31a`. It contains real DMU scale paths, scale buffers, MAC/EPU datapaths, and compiler quantization attributes, but their fixed-point formats and ordering must be audited before reuse. Structural similarity is not numerical compatibility.
- The closest reusable reference is `ct_cluster_ds/pp/pp_unit_ds.sv`: INT32 accumulator -> FP32 -> multiply BF16 input and weight scales -> FP32-to-BF16 RNE -> optional BF16 bias -> optional activation -> reciprocal output-scale multiply -> BF16-to-INT8 RNE/saturation. This directly motivates a staged GTSU post-accumulator pipeline.
- `epu_bf16_to_int8.sv` implements ties-to-even and saturates to `[-128,127]`, while Nano-pointllm's locked symmetric contract clips to `[-127,127]`. It cannot be reused bit-for-bit without a configurable symmetric-min policy.
- `act_bit_down.sv` provides shift-based RNE and 8/16-bit saturation, but it assumes power-of-two fixed-point scaling and has unsafe-looking `d_shift-1` combinational expressions when shift is zero. It is not the default for arbitrary FP16 per-output PointLLM scales.
- The reference LBUF uses 16 banks, 128-bit rows, byte writes, registered arbitration/muxing, and an explicit three-cycle controller contract. This matches the current GTSU bank/width/read-latency assumptions well enough to reuse the interface pattern, not the source verbatim.
- The new GTSU requant path compiles FP16-stored `sa*sw/so` into a UINT32 multiplier plus right shift, applies accumulator-domain bias, magnitude-based RNE, and symmetric `[-127,127]` saturation. Twelve edge vectors match RTL events and cycles exactly; Yosys reports one multiplier and no divider.
- M/N/K ping-pong control is now separately correlated for representative `K=8/384/2048/4096` and edge `K=132` cases. This validates control sequencing only; it still consumes abstract block-ready tokens rather than physical 16-bank SRAM payloads.
