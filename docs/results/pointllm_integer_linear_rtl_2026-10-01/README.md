# Real PointLLM Integer Linear And RTL Check

This artifact separates model-fidelity QDQ evidence from true integer
execution evidence. One real ModelNet request supplies the final-token inputs
to decoder layer-0 q-projection, layer-0 down-projection, and the LM head.

- Each Linear uses explicit INT8 activations/weights and INT32 accumulation.
- FP32, BF16, QDQ-FP32, QDQ-BF16, and IntegerGolden outputs are compared.
- Scale storage, SmoothQuant runtime divide, accumulator range, worst-case
  overflow bound, bias, and output-cast behavior are recorded.
- Three q-projection output rows complete the full `K=4096` accumulation in
  dense RTL. `[30131, 18552, -2132]` matches integer golden exactly in 1025
  cycles with an exact event trace.

Integer versus QDQ-FP32 maximum absolute error is `9.54e-7` for q-projection,
`1.19e-7` for down-projection, and `7.63e-6` for the LM head. These differences
are only dequantization arithmetic order/rounding, not an integer mismatch.

The evidence covers one token and selected q-projection rows. It does not prove
all output rows, a production W8A8 kernel, end-to-end fidelity, or full-model
cycle accuracy.
