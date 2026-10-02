# RTL Simulation And Correlation Method

Nano-pointllm does not derive cycle accuracy from a Python latency formula.
Every covered vertical slice uses one locked transaction contract in both an
edge-state model and synthesizable SystemVerilog.

## Execution flow

1. A Python lowerer creates typed transactions from a deterministic regression
   vector or a real PointLLM tensor/address plan.
2. External-memory requests, where applicable, are submitted to the pinned
   local DRAMsim3 bridge. Completion callbacks are converted from DRAM clock
   ticks to accelerator cycles and preserve finite outstanding/reorder limits.
3. The Python model advances one rising edge at a time. Ready/valid handshakes,
   finite queues, arbitration, SRAM banks, pipeline state, and output retirement
   are updated using the same pre-edge/next-state convention as RTL.
4. The correlation driver serializes input transactions into a fixed-width hex
   trace. The Icarus testbench loads it with `$readmemh`, drives the DUT at each
   positive edge, and prints accepted events and summary counters.
5. Python parses the RTL trace and requires exact event cycle, tag, value,
   traffic counter, and completion-cycle equality. A mismatch fails instead of
   falling back to an analytical estimate.
6. Yosys runs `hierarchy`, `proc`, `memory`, `opt`, and `check -assert` on the
   synthesizable DUT. This checks structure, not timing closure, area, or power.

## Current covered slices

- W8 Split-K GEMV: exact arithmetic outputs and all ready/valid events.
- Banked SRAM: exact arbitration, conflicts, read latency, and returned data.
- Production decoder linear controller: real PointLLM shapes and DRAMsim3
  completion traces, with exact traffic/stall/cycle counters.
- Geometry distance tile: signed INT16 coordinates, widened subtraction,
  three-coordinate squared distance, elastic output backpressure, and exact
  per-lane values/events/cycles.
- Shared feature/geometry dot fabric: one signed INT8 SIMD4 PE definition used
  by both W8 Split-K GEMV and cached-norm geometry distance. Yosys additionally
  requires zero direct multipliers in the geometry top and exactly four
  multipliers in every `gtsu_dot4_pe` instance.

The behavioral SRAM is not a foundry macro. Icarus is an RTL event simulator,
not a post-layout timing simulator. Yosys acceptance is not a PPA result.
Full-model cycles remain disabled while any lowered operator lacks exact RTL
correlation.

"Exact value" for the geometry slice means bit-exact signed INT16 RTL versus
its integer golden model. PointLLM supplies floating-point coordinates, so the
coordinate scale, saturation, tie-break behavior, FPS center-index equality,
and KNN top-32 recall still require ModelNet/Objaverse validation before this
datapath can make a semantic-equivalence claim.

The shared INT8 slice has the same semantic limitation. Its 64-PE PointLLM
mapping needs 65,536 ideal distance-dot issue cycles plus 128 point-norm
precompute cycles. Exact INT16 multiplication can be decomposed into four
signed INT8 products, giving a 262,144-cycle arithmetic lower bound, but that
fallback has not yet been scheduled or RTL-correlated.

## PointKAN geometry reference

The audit is pinned to `DingdongD/PointKAN_Accel` commit
`aa356da232a9d30d6c784297603c2a5596c6409a`. Relevant references are:

- `hw_sim/rtl/dist_engine_rtl/distance_engine.v`: 64-PE fixed-point distance
  chain with static packed database coordinates;
- `hw_sim/rtl/RTL/fps_min_dits2_table_128.v`: 128-lane persistent minimum
  update primitive;
- `hw_sim/rtl/dist_engine_rtl/dist_topk_engine.v`: one 64-point batch top-k;
- Python sampling/neighbor engines under `hw_sim/cycle_simulator/`.

These files do not form a complete memory-integrated PointLLM FPS/KNN block.
The Nano-pointllm geometry tile therefore uses a clean streaming contract with
explicit backpressure rather than copying the static packed interface.

The pinned 64-PE distance testbench was rerun with Icarus: all five functional
cases passed, but its own printout counted about 71 cycles per batch while the
source header states 66. The full-width top-k testbench compiled but did not
finish within a bounded 50-second audit window. Nano-pointllm therefore treats
these modules as functional design references, not as timing or coverage proof.

For PointLLM, one FPS or KNN pass evaluates `8192 * 512 = 4,194,304`
distances. The locked 64-lane distance tile has 65,536 ideal issue tiles. That
number excludes FPS's 512 serial min/argmax barriers, KNN's cross-tile top-32
merge, and all SRAM/DRAM stalls, so it must not be reported as operator latency.
