# C-model and RTL Coverage Roadmap

This document distinguishes implemented vertical slices from future coverage.

Implemented now:

- W8 Split-K GEMV transaction-level vertical slice;
- finite compressed/decoded/partial FIFOs and ready/valid backpressure;
- W8 sign extension, dot accumulation, Split-K reduction, and output stalls;
- Python edge-state model versus Icarus RTL exact event/cycle/output checking;
- Yosys structural synthesis check;
- a second locked slice for 16-bank, 128-bit, 1R1W SRAM arbitration and
  3-cycle reads, with exact Python/RTL event and cycle correlation;
- checkpoint-backed PointLLM decoder-linear lowering into W8 tiles, exact
  DRAM payloads/bursts, and SRAM bank/row addresses;
- sampled request timing through the local pinned official DRAMsim3 backend.
- production decoder-linear scale/weight streams with bounded issue/reorder,
  SRAM bank conflicts, registered reads, unpack, Split-K partials, and outputs;
- exact full-shape RTL-controller correlation for Q projection and non-uniform-K
  down projection using their real DRAMsim3 completion traces.

The production decoder-linear controller is now DRAM-timed and SRAM-integrated.
Arithmetic values are checked in the separate W8 GEMV datapath slice while the
production controller checks full-shape traffic and cycles. Dense GEMM,
attention, vector operations, and whole-graph scheduling remain unsupported.

The legacy Python estimator does not model ready/valid handshakes, queues,
resource stalls, exact bank mapping, burst timing, or producer/consumer events.
The next fidelity level should follow the separation used by the audited
`DingdongD/Compiler_Codes` cycle-model scaffold:

1. Define typed issue, memory-request, engine-start, engine-complete, and
   counter events.
2. Add ready/valid resource scoreboards for geometry, tensor, unpack, vector,
   attention, SRAM ports, and HBM channels.
3. Expand operations into tiles and preserve true PointLLM layer/token order.
   Decoder-linear shape/address/byte lowering is complete; whole-graph order is not.
4. Integrate the correlated Split-K slice with the implemented activation/weight
   memory plans, SRAM arbitration, and DRAMsim3 completion stream.
5. Correlate persistent FPS before using any end-to-end cycles.
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
