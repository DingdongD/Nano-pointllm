# GTSU Memory And PointLLM Lowering Evidence

This artifact uses the local Compiler_Codes LBUF contract as the SRAM interface
reference and the pinned local DRAMsim3 bridge for sampled DDR timing.

## Verified

- SRAM Python/RTL exact correlation: `15` cycles,
  `21` events, cycle error `0`.
- PointLLM `q_proj` source shape: `{'m': 1, 'n': 4096, 'k': 4096}`, source
  `F32`, target W8 bytes `16777216`.
- DRAMsim3 sample: `64` x 64-byte requests,
  `246` accelerator cycles, latency min/median/max
  `34/97.0/158` cycles.

## Claim Boundary

The SRAM port contract is RTL-correlated, PointLLM shape/address/byte lowering is
exact, and sampled memory requests use real DRAMsim3 timing. Compute and memory
have not yet been integrated into one RTL-correlated production linear pipeline,
so full PointLLM cycle accuracy remains false and no end-to-end speedup is claimed.
