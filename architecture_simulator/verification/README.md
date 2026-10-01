# Verification and Correlation Gates

The simulator uses three explicit fidelity levels:

1. `analytical`: operation cycles/traffic from the Python equations.
2. `event`: queue, ready/valid, memory request, and bank-conflict timing from the C-model.
3. `rtl-correlated`: event counters and completion cycles checked against RTL.

Golden tests must cover FPS, KNN top-k, dense GEMM, Split-K GEMV, W4/W8 unpack,
fused attention, and phase reconfiguration. Functional outputs must match
bit-exactly for integer datapaths or within a declared tolerance for BF16.

Timing promotion requires:

- exact operation count and HBM/SRAM byte count;
- exact selected dataflow and Split-K degree;
- no unexplained completion-cycle error above 5% per block;
- no end-to-end claim until all blocks contributing at least 5% modeled cycles are correlated;
- separately reported analytical, C-model, RTL, and measured-GPU values.

GPU NCU data calibrates workload and bottleneck hypotheses only. It must not be
used as evidence that an unimplemented ASIC has the same cycle count.
