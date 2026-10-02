# GTSU Requant And Compiler_Codes RTL Audit

The current GTSU quantization logic was not copied from `Compiler_Codes`.
This result records a clean, contract-specific implementation after auditing
the local reference at commit `ad2c31a`.

- Input: INT32 accumulator and optional accumulator-domain bias.
- Scale: offline-compiled UINT32 multiplier plus right shift from FP16-stored
  activation, per-output weight, and output scales.
- Rounding: round-to-nearest, ties-to-even on magnitude for both signs.
- Saturation: symmetric `[-127,127]`, intentionally different from the
  reference BF16 converter's `[-128,127]` range.
- RTL result: 12 edge vectors, elastic backpressure, 20 cycles, exact
  value/event/counter/cycle correlation.
- Yosys: one multiplier and no divider.

The scale compiler exactly represents the three recorded representative FP16
ratios in `correlation.json`. This unit is an A8 requant path; floating
INT32-to-BF16 dequantization and Dense/SRAM integration remain disabled.
