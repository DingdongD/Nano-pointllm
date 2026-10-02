# GTSU W8 Split-K GEMV RTL correlation

This directory contains the first RTL-locked GTSU vertical slice. It is not a
full PointLLM cycle-accurate result.

## Implemented boundary

```text
deterministic weight-beat source
  -> finite compressed FIFO
  -> W8 sign-extension stage
  -> finite decoded FIFO
  -> lane dot and chunk accumulation
  -> finite partial-sum FIFO
  -> Split-K reduction
  -> backpressured output
```

The source latency and beat stream are deterministic testbench contracts, not a
DRAM timing model. Activation and weight SRAM address/bank behavior is not yet
integrated.

## Correlation result

- Python edge-state model: `32` cycles.
- Icarus SystemVerilog RTL: `32` cycles.
- Exact events: `84/84` in identical order and cycles.
- Traffic: `24` input, unpack, and compute beats; `8` partial sums; `4` outputs.
- Functional outputs: `[1, -5, -4, 11]`, exactly equal.
- Cycle error: `0`.
- Yosys `hierarchy/proc/memory/opt/check -assert`: passed.
- Additional regression configurations include FIFO depth `1`, Split-K `3`,
  eight lanes, output stalls, and a no-stall/deeper-FIFO configuration.

## Fidelity boundary

`coverage.json` is authoritative: only
`w8_splitk_gemv_vertical_slice` is marked RTL-correlated. FPS, KNN/top-k, dense
GEMM, production PointLLM Split-K linear, attention, vector/norm/activation,
LM head, and sampling remain unsupported. Requests for those operators fail
instead of falling back to analytical timing.

## Files

- `correlation.json`: exact comparison status, counters, cycles, and outputs.
- `micro_ops.json`: 84 dependency-ordered `LD_W`, `UNPACK_W8`,
  `GEMV_SPLITK`, `PSUM_REDUCE`, and `STORE` micro-ops.
- `cycle_model_trace.csv`: Python event trace.
- `rtl_trace.csv`: Icarus event trace; byte-identical to the Python trace.
- `coverage.json`: fail-closed operator coverage matrix.
- `yosys_check.log`: structural synthesis/check transcript and module statistics.

## Reproduce

```bash
cd /home/nano-pointllm/architecture_simulator
/opt/conda/envs/pointllm/bin/python scripts/run_splitk_rtl_correlation.py \
  --config configs/gtsu_splitk_rtl_lock.json \
  --rtl_root rtl/vertical_slice \
  --output_dir ../results/gtsu_cycle/splitk_rtl_lock
/opt/conda/envs/pointllm/bin/python -m pytest -q tests/test_gtsu_cycle.py
```
