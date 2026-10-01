# RTL Handoff

No synthesizable RTL is claimed in this revision. This directory fixes the
intended block boundary so later modules can be correlated independently.

```text
gtsu_top
|-- phase_controller
|-- geometry_engine          # FPS distance/min/argmax and KNN top-k
|-- tensor_fabric            # dense output-stationary and Split-K GEMV
|-- weight_stream_engine     # W4/W8/BF16 fetch, scale, unpack, double buffer
|-- streaming_attention      # QK, online softmax, AV, paged KV reads
|-- vector_engine            # norm, activation, pooling, reductions
`-- banked_sram_router       # phase-specific logical partitions
```

Every block must expose valid/ready command and completion channels plus named
activity counters. Memory uses tagged request/response channels so the same RTL
testbench can connect an ideal SRAM, a trace replay model, or DRAMsim3 through
the C-model boundary.

The first implementation target is `geometry_engine`, correlated against the
resident-state FPS contract and QuickFPS traces. The second is
`weight_stream_engine + tensor_fabric` for one Split-K decoder GEMV.
