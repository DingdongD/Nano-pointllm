# Unimplemented C-model and RTL Roadmap

This document is a roadmap, not source-code implementation.

The current Python estimator does not model ready/valid handshakes, queues,
resource stalls, exact bank mapping, burst timing, or producer/consumer events.
The next fidelity level should follow the separation used by the audited
`DingdongD/Compiler_Codes` cycle-model scaffold:

1. Define typed issue, memory-request, engine-start, engine-complete, and
   counter events.
2. Add ready/valid resource scoreboards for geometry, tensor, unpack, vector,
   attention, SRAM ports, and HBM channels.
3. Expand operations into tiles and preserve true PointLLM layer/token order.
4. Correlate persistent FPS and one Split-K GEMV against independent C-model or
   RTL traces before using end-to-end cycles.
5. Add DRAMsim3 and exact SRAM bank/address mapping.
6. Import synthesis, SRAM compiler/CACTI, and PTPX results with process/tool/
   corner/activity provenance.

The intended future RTL hierarchy is:

```text
gtsu_top
|-- phase_controller
|-- geometry_engine
|-- tensor_fabric
|-- weight_stream_engine
|-- streaming_attention
|-- vector_engine
`-- banked_sram_router
```

Promotion gates are exact traffic agreement, exact functional outputs where
applicable, and no unexplained block-level cycle error above 5%. GPU NCU data
may motivate the architecture but must not be used as ASIC cycle calibration.
