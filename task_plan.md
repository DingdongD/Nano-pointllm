# Task Plan: nano-pointllm high-performance inference alignment

## Goal
Move `nano-pointllm` closer to the `nano-vllm` high-performance inference framework by tightening engine/scheduler/KV behavior, adding missing API surface where practical, and verifying with focused tests.

## Current Phase
Phase 24 complete

## Phases

### Phase 1: Discovery
- [x] Compare relevant `nano-vllm` runtime pieces against `nano-pointllm`
- [x] Identify a scoped implementation slice that can be completed safely now
- [x] Document findings in `findings.md`
- **Status:** complete

### Phase 2: Design
- [x] Choose concrete code changes with minimal blast radius
- [x] Record decisions and expected behavior
- **Status:** complete

### Phase 3: Implementation
- [x] Patch `nano-pointllm` runtime code
- [x] Add or update tests for new behavior
- **Status:** complete

### Phase 4: Verification
- [x] Run targeted framework tests
- [x] Fix regressions
- [x] Record test results
- **Status:** complete

### Phase 5: Delivery
- [x] Summarize what changed and remaining high-performance gaps
- **Status:** complete

### Phase 6: CUDA Benchmark
- [x] Confirm PointLLM-7B benchmark environment
- [x] Run smoke benchmark
- [x] Run different batch-size benchmarks
- [x] Record results and limitations
- **Status:** complete

### Phase 7: Decode-Only Paged Attention Kernel
- [x] Implement q_len=1 paged decode Triton kernel
- [x] Keep prefill and partial-prefix paths unchanged
- [x] Add CPU fallback / CUDA parity coverage
- [x] Run targeted tests
- **Status:** complete

### Phase 8: Mixed Decode Fast Path And Runtime Gaps
- [x] Use decode Triton kernel inside mixed prefill+decode batches
- [x] Add mixed-path parity test
- [x] Add high-performance sampler scaffold
- [x] Document unavailable flash-attn / remaining CUDA graph and HF-layer gaps
- **Status:** complete

### Phase 9: Prefill Varlen Paged Attention Kernel
- [x] Implement q_len>1 paged varlen Triton attention
- [x] Support causal prefill and partial-prefix suffix prefill
- [x] Add CUDA parity tests
- [x] Run targeted tests and PointLLM-7B benchmark
- **Status:** complete

### Phase 10: Kernel Config Tuning
- [x] Add head_dim-aware BLOCK_M/BLOCK_N config selection
- [x] Add environment overrides for controlled benchmarking
- [x] Benchmark candidate configs on PointLLM-7B
- [x] Record selected default and results
- **Status:** complete

### Phase 11: Static Metadata And Decode Buckets
- [x] Add reusable metadata staging buffers for decode/mixed paged metadata
- [x] Add decode bucket selection and CUDA graph replay scaffold
- [x] Route paged runner decode metadata through staging
- [x] Add unit tests
- **Status:** complete

### Phase 12: Chunked Prefill
- [x] Add scheduler/sequence state for chunked prompt processing
- [x] Prevent sampling until final prefill chunk
- [x] Add tests for long prompt chunk admission
- **Status:** complete

### Phase 13: Complete Sampler
- [x] Extend SamplingParams with top-k/top-p/penalties/seed
- [x] Implement batched per-request sampler
- [x] Add deterministic tests
- **Status:** complete

### Phase 14: Lightweight Llama Runner
- [x] Add lightweight runner scaffolding for non-HF layer execution
- [x] Route compatible layers incrementally
- [x] Add parity tests
- **Status:** complete

### Phase 15: Prefix Cache Subsystem
- [x] Extract prefix cache accounting/lookup interface
- [x] Track hits/misses/refcounts/eviction hooks
- [x] Add block manager tests
- **Status:** complete

### Phase 16: Integrated Benchmarks
- [x] Run focused tests
- [x] Run PointLLM-7B speed checks for new paths
- [x] Record results
- **Status:** complete

### Phase 17: Point Encoder Reuse
- [x] Add per-sequence cached point features and splice layouts
- [x] Reuse projected point features across repeated requests inside one runner
- [x] Add focused encoder-cache tests and benchmark entry point
- [x] Record results and follow-up gaps
- **Status:** complete

### Phase 18: Engine-Level Point Feature Cache
- [x] Promote point-feature reuse from runner-local dicts to an explicit cache subsystem
- [x] Share the cache at engine/session scope across runner calls
- [x] Expose request-level cache-key hooks for repeated point-cloud conversations
- [x] Add tests for engine/shared-cache wiring and API passthrough
- **Status:** complete

### Phase 19: PointBERT Backbone Hotspot Profiling
- [x] Add fine-grained PointBERT backbone profiler
- [x] Measure `FPS / KNN / gather+normalize / local encoder / transformer` on real PointLLM-7B
- [x] Compare B=1 and B=4 hotspot ordering
- [x] Record the next optimization target
- **Status:** complete

### Phase 20: Default Lightweight Eager Decode
- [x] Route ordinary paged eager decode through `LightweightLlamaRunner`
- [x] Keep prefill and mixed-prefill execution on the existing HF path
- [x] Add an explicit environment opt-out and focused routing tests
- [x] Run the full test suite and real PointLLM parity/benchmark checks where GPU availability permits
- **Status:** complete

### Phase 21: End-to-End Performance Diagnostic Suite
- [x] Audit existing benchmark/profiling hooks and define metric semantics
- [x] Implement TTFT/TPOT/throughput and input/output/batch sweeps
- [x] Add CUDA breakdown for attention/MLP/LM head/sampling
- [x] Report KV-cache memory plus measured/estimated HBM traffic
- [x] Run focused tests and a representative real PointLLM benchmark
- [x] Export JSON, plots, and bottleneck recommendations
- **Status:** complete

### Phase 22: Isolated-GPU Benchmark Methodology
- [x] Separate cold-start and steady-state measurements
- [x] Default to 10 warmups and 30 measured runs with P10/P50/P90/P95
- [x] Record clock, power, memory-controller, and GPU-utilization telemetry outside timed regions
- [x] Add representative-only Nsight Compute workflow for SM and DRAM hardware counters
- [x] Minimize and document synchronization boundaries
- [x] Add tests, documentation, and an empty-GPU execution command
- **Status:** complete

### Phase 23: Native Paged Runtime And Strict Decoder Roofline
- [x] Route paged prefill and mixed continuous batches through `LightweightLlamaRunner`
- [x] Add NCU-only NVTX ranges for QKV, attention, O projection, MLP, LM head, and sampling
- [x] Add stage traffic/FLOP manifests and strict weight-bound classification
- [x] Add true mixed-step routing tests and repair the staggered real-model parity driver
- [x] Document the native runtime boundary and reproducible B1/B8 NCU commands
- [x] Capture publishable B1/B8 NCU reports on an idle A100
- **Status:** complete

### Phase 24: Decode MLP Block Trace
- [x] Define contribution, overlap, recall, and theoretical weight-byte metrics
- [x] Capture `SiLU(gate) * up` for all 32 decoder layers and decode tokens
- [x] Support 32/64/128/256-neuron blocks and cross-prompt point-cloud grouping
- [x] Export detailed JSON, aggregate CSV/JSON, and diagnostic plots
- [x] Add unit tests and run real ModelNet and Objaverse analyses
- [x] Update documentation, findings, and progress records
- **Status:** complete

## Key Questions
1. Which `nano-vllm` features are missing but practical to implement without real PointLLM-7B weights?
2. Does paged KV scheduling handle memory pressure safely, including preemption?
3. Is there a `nano-vllm`-style public API entry point for offline inference?

## Decisions Made
| Decision | Rationale |
|----------|-----------|
| Use a scoped runtime alignment rather than wholesale porting | Full CUDA/TP/CUDA graph parity requires hardware and broader model integration; a focused patch can improve framework correctness now. |
| Fix paged KV append timing first | This is a correctness prerequisite for high-performance paged decode and directly mirrors `nano-vllm` block semantics. |
| Size KV pool by key/value heads | Llama GQA uses fewer KV heads than query heads; physical KV cache must follow `num_key_value_heads`. |
| Keep paged prefill on varlen `run_mixed` path | `PagedLlamaAttention` is designed around flattened token streams, closer to `nano-vllm` prepare-prefill semantics. |
| Add a top-level `LLM` wrapper for loaded models | This aligns the offline inference API with `nanovllm` without guessing local PointLLM checkpoint construction. |

## Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|
| `BLOCK_M=8` varlen candidate failed in Triton `tl.dot` compile on PointLLM-7B | Tried head_dim=128 default `M8/N64` | Reverted head_dim=128 default to supported `M16/N64` and added min guards for env overrides |
| Paged decode benchmark skipped after routing pure decode to `run_decode` | Used `[B, 1]` input_ids with flattened paged attention | Changed decode staging layout to `[1, bucket]`, matching mixed decode semantics |
| Sampler saw one row for B=8 flattened decode logits | Passed `[1, B, vocab]` logits directly to per-seq sampler | Sliced paged decode logits to `[B, vocab]` before sampling |
| Strict CUDA graph capture failed in HF causal-mask construction | Captured full HF/PointLLM model forward | Routed strict graph decode through `LightweightLlamaRunner` |
| Strict CUDA graph capture failed on `context_lens.max().item()` | Decode attention used runtime tensor-to-CPU max | Used static block-table upper bound for decode `MAX_CONTEXT_LEN` |
| Point-cloud cache key creation failed on `bfloat16` tensors during real benchmark | Hashed `pc.numpy()` directly on CPU `bfloat16` tensor | Normalize cache key inputs to CPU `float32` before hashing |
| Default eager lightweight real-model parity passed B1 but failed B4/B8 after the first generated token | Ran 4-token PointLLM-7B parity across point/text batches | Investigating batched lightweight decode row/KV correctness before allowing the path to remain default |
| Position injection regression test omitted required `ForwardContext` fields | First targeted test run after the position fix | Supplied explicit `None` values for slot mapping, block tables, and context lengths |
| 20-token native-HF parity diverged on some synthetic point-cloud sequences near token 14 | Extended the corrected 4-token parity beyond the basic gate | Verified same-state lightweight-vs-HF paged logits are exactly equal; classified remaining divergence as existing Triton/BF16 non-bitwise behavior rather than a lightweight routing regression |
| Phase 22 percentile unit test compared `8.549999999999999` to `8.55` exactly | First final full-suite run | Changed only the assertion to `pytest.approx`; percentile implementation was correct |
| Phase 24 progress-log patch expected a non-existent section heading | First attempt to append the MLP trace implementation log | Inspected the actual append-only log layout and appended a dated section instead |
| Phase 24 artifact inspection used unavailable `jq` and omitted explicit image detail | First smoke artifact inspection | Switched to a read-only Python JSON summary and explicit `detail=high` image inspection |
| Nested image forwarding still rejected an otherwise valid detail field | Second smoke artifact inspection | Called `view_image` one image at a time and forwarded its `image_url` explicitly |

## Notes
- Preserve user/untracked work; the repository has many untracked files and at least one modified tracked file.
- Avoid destructive git operations.
