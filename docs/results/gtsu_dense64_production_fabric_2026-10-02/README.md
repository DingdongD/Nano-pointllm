# GTSU Production-Width Dense64 Operand Fabric

This artifact replaces the proposed serial 17-word gather with a production-
width operand organization:

```text
512-bit ingress -> 4 WBUF banks/cycle -> 16-bank x 128-bit WBUF
                                             |
independent 32-bit ABUF -------------------- A4 broadcast
                                             |
                                      64 shared Dot4 PEs
                                             |
                                      ACC32 accumulation
```

Four ingress beats fill one 256-byte weight vector. A compute read accesses all
16 WBUF banks in parallel and dispatches 64 independent W4 operands plus one
broadcast A4 to the shared Dot4 array.

## Exact Edge Lock

The `M=2,N=70,K=16` case forces a 64-column tile followed by a six-column edge
tile for each row. Python and Icarus match exactly at 86 cycles across all
events, counters, and 140 output values. Yosys proves:

- one 64-column Dense tile and exactly 64 `gtsu_dot4_pe` instances;
- 16 WBUF banks and two independent ABUF ping-pong words;
- one response FIFO and zero direct multipliers in the fabric top.

Evidence is under `edge/`.

## Full Q-Projection Shape

The complete `M=1,N=4096,K=4096` control/data shape runs through a compiled
Verilator cycle driver rather than an analytical estimate:

- 262,150 exact cycles;
- 262,144 512-bit ingress lines;
- 1,048,576 individual WBUF bank writes;
- 65,536 parallel 16-bank reads and Dot4 chunks;
- 64 output tiles and all 4,096 ACC32 outputs;
- 65,535 fill/compute overlap cycles.

All 4,096 output values, counters, and total cycles match the Python edge-state
model. Evidence is under `q_proj_shape/`.

## Claim Boundary

The full-shape stimulus is deterministic production-shape data, not real
checkpoint weights or captured activations. BF16/A8 dual post-processing,
SmoothQuant metadata, request-side DRAMsim3 closed-loop, and generic
`dense_gemm` coverage remain disabled. The 256-byte WBUF read can issue in one
cycle, but the current 64-byte ingress requires four fill cycles, correctly
exposing weight supply as the throughput limiter.

Reproduce either lock:

```bash
architecture_simulator/scripts/run_dense64_fabric_correlation.py \
  --shape edge \
  --rtl_root architecture_simulator/rtl/vertical_slice \
  --output_dir docs/results/gtsu_dense64_production_fabric_2026-10-02/edge

architecture_simulator/scripts/run_dense64_fabric_correlation.py \
  --shape q_proj \
  --rtl_root architecture_simulator/rtl/vertical_slice \
  --output_dir docs/results/gtsu_dense64_production_fabric_2026-10-02/q_proj_shape
```
