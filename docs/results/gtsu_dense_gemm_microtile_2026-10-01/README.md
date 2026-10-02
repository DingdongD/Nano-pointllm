# Shared-Dot4 Dense GEMM Microtile

This directory records exact Python edge-state model versus Icarus RTL
correlation for a 64-PE output-stationary dense microtile.

- Lock case: `M=5`, valid `N=61` in a 64-column tile, `K=384`.
- Input and output stalls are enabled.
- Value, event, counter, and total-cycle traces are exact (`642` cycles).
- Yosys finds zero direct multipliers in the tile and four INT8 multipliers in
  each of 64 `gtsu_dot4_pe` instances (`256` total).

`correlation.json` is the machine-readable result. The CSV files preserve both
event traces, while `rtl_stdout.log` and `yosys_check.log` preserve raw evidence.

This is one N microtile with streamed M/K work. It does not include the full
M/N tile controller, operand SRAM double buffering, DMA, or production-shape
RTL correlation. Generic `dense_gemm` cycle coverage therefore remains off.
