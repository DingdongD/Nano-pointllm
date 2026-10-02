# Physical Dense SRAM, Dot4, And Requant Pipeline

This artifact is the first physically composed Dense vertical slice rather
than a sum of separately modeled operators.

```text
payload DMA interface
  -> 16-bank x 128-bit x 3-cycle behavioral SRAM
  -> outstanding-credit guard and response FIFO
  -> 3-column shared-Dot4 tile
  -> 3 parallel INT32-to-A8 requant lanes
  -> backpressured output
```

The lock shape is `M=2,N=5,K=16,N_TILE=3,K_BLOCK=8`. One SRAM word packs
one A8x4 vector and three W8x4 vectors.

- Total cycles: `29`; cycle error: `0`.
- Payload writes / reads / responses / Dot4 inputs: `16/16/16/16`.
- Output tiles: `4`, including `0x3` masks for edge-N tiles.
- Simultaneous SRAM read and alternate-buffer write: `10` cycles.
- Python versus Icarus: exact event trace, counters, output values, and cycles.
- Yosys: one SRAM, controller, FIFO, Dense tile; three Dot4 and requant lanes;
  zero multiplier in the composition top.

The lock SRAM has four rows per bank to avoid synthesizing storage capacity as
flops. Bank count, 128-bit width, 1R1W behavior, and three-cycle latency match
the separately correlated LBUF-inspired SRAM contract. This is not a foundry
SRAM macro or PPA result.

Production `N_TILE=64` requires one A word plus sixteen B words per K chunk and
a tagged multiword gather stage. DRAMsim3 DMA and floating BF16 dequantization
are also not composed, so generic Dense GEMM and full PointLLM cycle accuracy
remain disabled.
