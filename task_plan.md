# Task Plan: nano-pointllm high-performance inference alignment

## Goal
Move `nano-pointllm` closer to the `nano-vllm` high-performance inference framework by tightening engine/scheduler/KV behavior, adding missing API surface where practical, and verifying with focused tests.

## Current Phase
Phase 37 in progress

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

### Phase 25: Neuron-Level Oracle Gate
- [x] Sweep neuron groups `B=1/4/8/16/32`
- [x] Report retained contribution, Gini, normalized entropy, and effective support size
- [x] Define and apply an explicit activation-sparsity stop criterion
- [x] Export JSON/CSV/plots for ModelNet and Objaverse traces
- **Status:** complete

### Phase 26: Exact MLP Neuron Permutation/Clustering Upper Bound
- [x] Apply the Phase 25 hard gate before spending on clustering
- [x] Stop the branch because the B=1 oracle failed the concentration criterion on both datasets
- **Status:** complete (gated off; no clustering performed)

### Phase 27: Static N:M Weight Pruning
- [x] Implement calibration-based Wanda 2:4 and 4:8 masks
- [x] Verify exact per-row N:M validity and 50% decoder sparsity
- [x] Measure logits KL, top-token agreement/margin, and generation parity/quality
- [ ] Run full SparseGPT reconstruction only if a longer offline pruning window is allocated
- **Status:** complete for Wanda; SparseGPT full-Hessian run deferred

### Phase 28: Full-Pipeline Precision Sensitivity
- [x] Evaluate PointBERT, projector, decoder QKV/O, MLP, and LM head independently
- [x] Measure W8 and W4 simulated weight quantization sensitivity
- [x] Report logits KL, top-token agreement/margin, and generated-output drift
- **Status:** complete

### Phase 29: Strict PointBERT NCU Roofline
- [x] Add NVTX scopes for FPS, KNN, local encoder, and PointTransformer QKV/attention/O/MLP
- [x] Collect DRAM, SM, duration, modeled bytes/FLOPs, and arithmetic intensity
- [x] Produce strict B1/B4 representative summaries and unified roofline figure
- **Status:** complete

### Phase 30: Phase-Adaptive Architecture Estimator
- [x] Audit `DingdongD/Compiler_Codes` and local PointAcc C-model/RTL conventions
- [x] Define trace, hardware-config, phase, analytical-cycle, and traffic contracts
- [x] Implement operation-level equations for geometry/FPS, dense tensor, Split-K GEMV, weight-streaming, attention, SRAM, and serial phase accounting
- [x] Add PointLLM-7B workload presets derived from measured shapes and precision policy candidates
- [x] Add CLI, DSE sweep, JSON/CSV/plot artifacts, and NCU evidence ingestion
- [x] Validate model shapes from config/YAML/safetensors headers and add analytical conservation tests
- [x] Label every output as uncalibrated, exploratory, non-event-level, and non-RTL-correlated
- [x] Remove empty C-model/RTL/verification/characterization source placeholders
- [x] Run full repository regression
- [x] Publish to GitHub
- **Status:** complete as an analytical estimator only; the earlier complete-simulator claim is withdrawn

### Phase 31: Event-Level C-Model And RTL Correlation
- [x] Define a shared micro-op, event-trace, configuration, and unsupported-operator contract for the first vertical slice
- [ ] Implement pipelined ready/valid resources, finite FIFOs, tile issue order, arbitration, and backpressure
- [ ] Model SRAM banks/ports/conflicts and DMA latency/bandwidth/outstanding requests
  - [x] Correlate a 16-bank, 128-bit, 1R1W, 3-cycle-read SRAM/arbiter against synthesizable RTL
  - [x] Connect sampled PointLLM W8 tile bursts to the pinned local DRAMsim3 backend
  - [ ] Integrate SRAM and DRAM completion into the Split-K compute slice before enabling production-linear cycles
- [x] Complete the W8 Split-K GEMV vertical slice from functional model through cycle model and RTL
- [x] Generate functional, traffic, event, and cycle golden traces for RTL testbenches
- [x] Correlate W8 Split-K GEMV traffic exactly and cycles exactly for the locked RTL configuration
- [ ] Complete and correlate the persistent FPS vertical slice
- [ ] Add dense GEMM, attention, vector/norm/activation, KNN/top-k, LM-head, and sampling RTL slices
- [ ] Enable full PointLLM cycle-accurate reports only after every lowered operator has correlated RTL
- [ ] Import characterized area and active/idle energy before equal-area DSE claims
- [x] Lower real checkpoint-backed PointLLM decoder linears into exact tensor regions, tiles, bursts, and SRAM bank/row addresses
- **Status:** in progress; no full-model cycle-accuracy claim is allowed yet

### Phase 32: Production Decoder And Remaining Tensor Operators
- [x] Integrate real PointLLM W8 decoder-linear lowering with DRAMsim3 completion timing, finite DMA/ROB credits, banked SRAM, unpack, Split-K compute, and reduction
- [x] Correlate the integrated linear controller against RTL for reduced and production PointLLM projection shapes
- [x] Emit representative Q/O and Gate/Up/Down shape-class cycle, traffic, bank-conflict, and stall evidence; LM-head reuses the N-parameterized path
- [ ] Implement and RTL-correlate dense GEMM for PointBERT/projector/prefill shapes
- [ ] Implement and RTL-correlate attention QK/online-softmax/AV and paged-KV paths
- [ ] Implement and RTL-correlate vector norm/RoPE/activation/residual/sampling paths
- [ ] Keep whole-model reports fail-closed until every phase is covered
- **Status:** in progress; production decoder linear gate is complete, dense GEMM is next

### Phase 33: PointKAN Geometry RTL Audit And PointLLM Correlation
- [x] Pin and audit `DingdongD/PointKAN_Accel/hw_sim` without treating its constants as PointLLM evidence
- [x] Document the current nano-pointllm RTL simulation/correlation flow and its fidelity limits
- [x] Map PointKAN distance/FPS/KNN interfaces and timing assumptions to PointLLM `8192 -> 512`, `K=32`
- [x] Implement the smallest reusable distance/FPS/KNN RTL-correlated slice justified by the audit
- [x] Add functional, event, cycle, and synthesis regression gates
- [x] Keep `fps` and `knn_topk` coverage false until complete PointLLM semantics are correlated
- [x] Run the full regression, record artifacts, commit, and push
- **Status:** complete; geometry distance is correlated, while complete FPS/KNN remain fail-closed

### Phase 34: Shared SIMD4 Geometry-Feature Datapath
- [x] Extract a reusable signed INT8 SIMD4 dot PE from the Split-K arithmetic contract
- [x] Implement geometry norm-dot postprocessing without dedicated geometry multipliers
- [x] Correlate feature-dot and geometry-distance modes exactly against one shared RTL hierarchy
- [x] Preserve the existing INT16 direct-distance tile as a dedicated baseline
- [x] Add a Yosys structural proof that geometry mode introduces zero multipliers outside the shared dot PE instances
- [x] Map INT8 geometry work and INT16-on-INT8 fallback cycles without claiming semantic fidelity
- [x] Run full regression and emit formal artifacts
- [x] Commit and push
- **Status:** complete; shared arithmetic RTL is correlated and full-model coverage remains fail-closed

### Phase 35: Sequential W8A8 Contract And Fidelity Pipeline
- [x] Audit existing precision-sensitivity/calibration code and real dataset split support
- [x] Lock symmetric rounding, clipping, accumulator, scale precision, and bias/requant rules
- [x] Compare per-output-channel W8 against group-128 W8 under the same calibration/evaluation split (smoke gate; larger evaluation remains below)
- [x] Implement static A8, SmoothQuant/static A8, and dynamic per-token A8 calibration paths
- [x] Propagate QDQ outputs sequentially through PointTransformer and all 32 LLaMA layers
- [ ] Evaluate ModelNet and Objaverse with KL/top-1/token/exact/semantic metrics (dual-dataset smoke complete; larger held-out semantic run pending)
- [x] Keep production W8A8 RTL disabled until both integer-golden and model-fidelity gates pass
- **Status:** in progress; integer representative correlation is complete, while larger held-out fidelity remains pending

### Phase 36: Production W8A8 Integer Numerical Closure
- [x] Add BF16/QDQ/true-INT8xINT8-INT32 three-way comparison for real PointLLM Linear activations and weights
- [x] Make per-output scale, FP16 scale storage, bias, overflow, and output-cast boundaries explicit in artifacts
- [x] Emit SmoothQuant folding/storage/runtime-overhead analysis per module
- [x] Correlate representative real integer payloads against shared Dot4 dense-accumulator RTL values
- [x] Keep QDQ fidelity and integer execution evidence separate
- **Status:** complete for representative q/down/LM-head tensors; full production W8A8 kernel remains fail-closed

### Phase 37: Shared Dense GEMM RTL Mode
- [ ] Implement tiled M/N/K control using only `gtsu_dot4_pe` multiplier instances
- [x] Add finite microtile operand/accumulator/output ready-valid state and edge-column handling
- [ ] Correlate Local Encoder, PointTransformer QKV, projector, and LLM prefill shape classes
- [x] Prove with Yosys that the Dense GEMM microtile introduces no multiplier outside Dot4 PE
- [ ] Enable `dense_gemm` coverage only after exact value/event/cycle correlation
- **Status:** in progress; one N-tile/M-stream/K-reduction slice is exact, production M/N controller and SRAM are pending

### Phase 38: Fused FPS/KNN RTL Controller
- [ ] Integrate coordinate/norm SRAM traffic, persistent min state, deterministic argmax, and center feedback
- [ ] Implement bounded streaming tile/global Top-32 with exact distance/index tie-break semantics
- [ ] Reuse one distance stream for FPS update and KNN candidates where semantics permit
- [ ] Validate INT8 geometry center equality and KNN recall on ModelNet/Objaverse
- **Status:** pending

### Phase 39: Geometry-To-Feature Composition
- [ ] Compose center feedback, Top-32 gather, patch FIFO, local encoder, and PE-pool arbitration
- [ ] Evaluate time-shared versus 48:16, 32:32, and 16:48 spatial PE partitions
- [ ] Correlate producer-consumer stalls and overlap rather than summing operator latencies
- **Status:** pending

### Phase 40: Attention, Vector, And Sampling Closure
- [ ] Correlate shared-PE QK/AV plus online softmax and paged PointKV/TextKV behavior
- [ ] Correlate RMSNorm, RoPE, GELU/SiLU, residual, quant/requant, and scale paths
- [ ] Correlate LM-head reduction and deterministic argmax/top-k/sampling
- **Status:** pending

### Phase 41: Decoder-Layer And Full-Graph Composition
- [ ] Close one production decoder layer with numerical, event, traffic, and cycle exactness
- [ ] Replicate the validated layer schedule across all 32 layers
- [ ] Compose full PointLLM workload with no analytical fallback
- [ ] Enable full-model cycle accuracy only when every coverage and composition gate passes
- **Status:** pending

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
| Sparse clone of `PointKAN_Accel` timed out before checkout, leaving an empty worktree with tracked deletions | 1 | Preserve the pinned Git object database and restore only the required `hw_sim` subtree from `HEAD`; do not modify nano-pointllm state |
| Directory-level `git archive` on the partial clone triggered multi-gigabyte promisor fetches that survived command timeout | 2 | Terminated only the fetch/archive processes started by this session; switch to individual raw source retrieval and never archive the whole subtree |
| Initial geometry correlation summary over-counted the final output tile by one `LANES` increment | 1 | Event/value/cycle traces were exact; fixed the testbench to print the post-NBA counter without adding `LANES` again |
| Geometry simulation used 64 lanes while the initial Yosys structural gate used the RTL default of 8 lanes | 1 | Parameterize Yosys with the exact locked simulation widths before accepting the synthesis gate |
| `raw.githubusercontent.com` returned 404 for pinned PointKAN RTL files although Git remote access works | 1 | Use already fetched pinned blobs from the local partial clone and export only build inputs to `/tmp` |
| PointKAN full-width top-k RTL simulation did not complete within a bounded 50-second audit window | 1 | Terminated only the reference `vvp`; keep KNN top-k unsupported and design a bounded streaming merge before correlation |
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
| Official pruning-source clone command included a defensive `rm -rf` | First Phase 27 source-inspection attempt; command was rejected before execution | Use fresh `mktemp` directories and never delete existing paths |
| PointBERT strict idle check saw NCU's own 940 MiB/41% startup activity | First Phase 29 NCU launch | Move the unchanged idle gate before NCU attaches and pass the pre-NCU GPU snapshot into the capture manifest |
| Split-K RTL correlation initially reported all configurations as mismatches | First Phase 31 trace comparison | The parser expected eight TRACE fields while the RTL contract emits seven; corrected the parser and retained the exact-correlation gate |
| First RTL input beat carried unknown metadata/data | Second Phase 31 correlation attempt after parser repair | Removed procedural combinational stimulus with time-zero sensitivity ambiguity; generate every beat through pure functions and continuous assignments |
| Direct cycle-correlation scripts could not import `gtsu_cycle` | First standalone CLI execution | Add the architecture-simulator root to each script's import path so source-tree execution matches the documented command |
| Generated trace CSVs failed `git diff --check` because Python CSV defaulted to CRLF | First Phase 31 commit attempt | Lock the trace writer to LF, regenerate both traces, and rerun exact comparison before commit |
| Planning-log patch expected a shorter shared-direction bullet than the actual text | First Phase 34 audit update | Located the exact section and appended the audit findings without rewriting prior records |
| Full regression failed collection on local namespace-package imports when invoked through standalone `pytest` entry points | First two Phase 34 full-suite attempts (base and PointLLM env) | Use `/opt/conda/envs/pointllm/bin/python -m pytest`, which preserves the repository root on `sys.path`; all source files were present and directly discoverable |
| Dense GEMM targeted test collection found a malformed nested list comprehension | First Phase 37 test attempt | Rewrote the deterministic A/B matrix construction as explicit nested comprehensions before any RTL result was accepted |
| Dense GEMM values/events/cycles matched but output summary was one row high | Second Phase 37 test attempt | The `#1` summary samples post-NBA counters, so removed the redundant manual increment; retained all value/event/cycle checks |

## Notes
- Preserve user/untracked work; the repository has many untracked files and at least one modified tracked file.
- Avoid destructive git operations.
