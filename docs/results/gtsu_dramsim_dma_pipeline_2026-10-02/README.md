# GTSU DRAMsim3 DMA To Dense Pipeline Correlation

This artifact composes the pinned official DRAMsim3 controller timing with a
synthesizable tagged DMA completion ROB, 64-byte to 128-bit unpack, the physical
16-bank SRAM slice, shared Dot4 compute, and INT32-to-A8 requantization.

## Locked Result

- Four aligned 64-byte DRAM reads complete at accelerator cycles 34, 39, 44,
  and 49 in this address stream.
- Each line unpacks into four 128-bit SRAM words: 256 useful bytes / 256
  transferred bytes, or 100% line utilization.
- Python and Icarus match at 62 cycles with exact event, counter, and functional
  output traces.
- The run contains 16 payload writes, 16 SRAM read issues/responses, 16 Dot4
  inputs, four output tiles, and 11 read/write overlap cycles.
- A separate controlled `1,0,3,2` completion-order test proves ordered retirement
  under out-of-order arrivals; the formal DRAM address stream happened to finish
  in order and therefore reached ROB peak occupancy one.

## Reference Boundary

`Compiler_Codes@ad2c31a` establishes a useful 512-bit AXI and FIFO-backpressure
organization, but its fixed AXI ID and ignored RID/RLAST imply an in-order active
sequence. This implementation retains DRAMsim3 tags and a finite ROB instead of
importing that assumption.

DRAMsim3 supplies controller completion timing, not DRAM PHY RTL. This artifact
does not cover the production `N_TILE=64` 17-word gather, foundry SRAM macros, or
full PointLLM cycle accuracy. See `correlation.json` and the two CSV traces.

Reproduce from the repository root:

```bash
/opt/conda/envs/pointllm/bin/python \
  architecture_simulator/scripts/run_dramsim_dma_dense_correlation.py \
  --rtl_root architecture_simulator/rtl/vertical_slice \
  --output_dir docs/results/gtsu_dramsim_dma_pipeline_2026-10-02
```
