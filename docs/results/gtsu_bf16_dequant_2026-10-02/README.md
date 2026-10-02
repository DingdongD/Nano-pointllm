# GTSU BF16 Dequantization Correlation

This artifact locks a clean-room `INT32 accumulator x FP16 activation scale x
FP16 weight scale -> BF16` vertical slice. The exact integer product is rounded
once to BF16 with round-to-nearest, ties-to-even; Verilog `real`, DPI, and host
floating-point arithmetic are not used in the RTL.

## Locked Result

- Python edge model and Icarus RTL: 20 cycles, exact event and output trace.
- Yosys: two multipliers and zero divider cells.
- Numerical validation: 12 directed vectors plus 1,000 deterministic random
  vectors agree bit-for-bit with an independent Float64-to-Torch-BF16 oracle.
- Inputs include `INT32_MIN`, signed and subnormal FP16 scales, zero, and values
  that exercise normalization and ties-to-even rounding.

## Reference Boundary

`Compiler_Codes@ad2c31a` was used to audit the post-processing organization.
Its `pp_unit_ds.sv` stages INT32-to-FP32, floating multiplication, FP32-to-BF16,
and optional bias. This GTSU slice is independently implemented and deliberately
uses one fused exact-product rounding point, so bit equivalence to that staged
pipeline is not claimed. Bias, activation, and Dense-pipeline composition remain
separate gates.

The machine-readable result is in `correlation.json`.

Reproduce from the repository root:

```bash
/opt/conda/envs/pointllm/bin/python \
  architecture_simulator/scripts/run_bf16_dequant_rtl_correlation.py \
  --rtl_root architecture_simulator/rtl/vertical_slice \
  --output_dir docs/results/gtsu_bf16_dequant_2026-10-02
```
