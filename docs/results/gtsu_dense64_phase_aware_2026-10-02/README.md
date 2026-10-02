# Phase-Aware Dense64 RTL Evidence

This directory records exact Python/RTL correlation for the first four Dense64
milestones. The canonical real tensor package is stored outside the repository:

`/mnt/llm_data/nano_pointllm_quantized_tensor_packages/modelnet/layer_00_q_proj`

## Locked results

- Real layer-0 q-projection: all 4,096 ACC32 outputs match at 65,542 cycles
  with four 64-byte ingress lanes.
- Memory supply: 64/128/256 bytes per cycle require 262,150/131,078/65,542
  cycles while preserving identical transfer and arithmetic counts.
- PointTransformer high-M shape: all 590,976 ACC32 outputs match at 925,021
  cycles. K-block/M-tile reuse reduces weight bytes from 226,934,784 to
  7,520,256 (30.18x) and reaches 95.83% Dot4 issue occupancy.
- Dual output: 64 A8 lanes plus 8/16/32/64 BF16 lanes are exact. The 16-lane
  real q-projection run checks 4,096 A8 and 4,096 BF16 values in 448 adapter
  cycles using package-backed FP16 scales.

`summary.json` is the compact machine-readable index and
`phase_aware_dense_summary.png` visualizes the three principal tradeoffs.

## Claim boundary

`dense_gemm` and `dense64_dual_a8_bf16_output` are operator-level correlated by
the combined Phase 37E evidence. Individual stream-only reports deliberately
state that they do not unlock `dense_gemm` coverage on their own.
The Dense-to-output wrapper is structurally checked but not yet one composed
event trace. Request-side DRAMsim3 feedback, real PointTransformer payload,
foundry SRAM/STA/PPA, fused FPS/KNN, attention/vector, and full PointLLM cycles
remain false in the coverage matrix.
