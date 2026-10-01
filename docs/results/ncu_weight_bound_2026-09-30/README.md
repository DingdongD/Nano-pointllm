# PointLLM Decoder Weight-Bound Profile

## Setup

- GPU: NVIDIA A100-SXM4-40GB
- Model: PointLLM-7B v1.2, BF16
- Decode context: 780 tokens after 10 warmup decode steps
- Batches: 1 and 8
- NCU: 2023.2, kernel replay
- Hardware-counter scope: decoder layer 0 for QKV, attention, O projection, and MLP; one real LM head execution
- Full-model execution: all 32 decoder layers still run, but only the representative layer enters NCU replay

Both runs passed the idle-at-start gate. NCU replay latency is not TPOT; use these results only for stage classification and bottleneck localization.

## Strict Classification

| Stage | B1 DRAM / SM | B1 classification | B8 DRAM / SM | B8 classification |
|---|---:|---|---:|---|
| QKV | 73.93% / 25.63% | confirmed weight-bandwidth-bound | 74.87% / 30.38% | confirmed weight-bandwidth-bound |
| Attention | 7.92% / 3.52% | KV-memory-bound or underfilled | 34.98% / 13.63% | KV-memory-bound or underfilled |
| O projection | 63.36% / 24.93% | confirmed weight-bandwidth-bound | 63.25% / 24.63% | confirmed weight-bandwidth-bound |
| MLP | 73.06% / 30.03% | confirmed weight-bandwidth-bound | 74.73% / 30.58% | confirmed weight-bandwidth-bound |
| LM head | 80.29% / 39.15% | confirmed weight-bandwidth-bound | 24.17% / 13.89% | weight-stream dominated, bandwidth underfilled |

At B1 all linear stages pass the strict weight-bound rule. At B8, QKV/O/MLP still pass; LM head remains weight-traffic dominated but does not reach the 50% DRAM-throughput threshold.

The dominant paged-attention tile improves from 37.16% DRAM throughput at B1 to 71.90% at B8. The lower whole-stage values above include RoPE, K/V store, reductions, concatenation, and small elementwise kernels, exposing a fusion/launch-overhead opportunity rather than a decoder-weight bottleneck.

## Stage-Only Distribution

Representative layer times are scaled by 32 for decoder-layer stages; LM head is counted once. This distribution excludes unmarked normalization/residual/sampling/host time and must not be interpreted as end-to-end TPOT.

| Stage | B1 share | B8 share |
|---|---:|---:|
| QKV | 20.83% | 15.17% |
| Attention | 21.88% | 34.15% |
| O projection | 7.31% | 6.26% |
| MLP | 48.65% | 40.66% |
| LM head | 1.33% | 3.77% |

## Optimization Implications

1. Prioritize weight-only quantization and fused dequant GEMM/GEMV for MLP, then QKV and O projection. MLP is the largest marked stage and remains strictly bandwidth-bound at B1/B8.
2. Fuse or eliminate the small attention-side RoPE, K/V-store, concatenate, elementwise, and reduction kernels. At B8 the main tile already uses HBM well while whole-stage utilization remains much lower.
3. Add a B8-specific LM-head GEMM/GEMV policy or autotuning. Its measured weight traffic matches the model, but the selected CUTLASS kernel reaches only 24.17% DRAM and 13.89% SM throughput.
4. Treat NCU duration only as a relative stage diagnostic. Re-run the steady-state benchmark after each kernel/quantization change for TPOT and tokens/s.

## Artifacts

- `b1_summary.json`, `b8_summary.json`: machine-readable aggregate reports
- `b1_stage_bottleneck.png`, `b8_stage_bottleneck.png`: stage time, DRAM/SM throughput, and roofline plots
- Local raw CSV and per-stage reports: `/home/nano-pointllm/results/ncu_decode_stages/{b1,b8}/`
