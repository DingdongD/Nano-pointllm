# Production PointLLM Decoder Linear Correlation

Projection `down_proj`, shape `1x11008 -> 1x4096`.
The DRAMsim3 completion stream, finite 32-entry ROB, banked SRAM reads,
registered latency, unpack, compute issue, Split-K partials, and output counts
match the parameterized RTL controller exactly at `2755501` cycles.

This promotes the production decoder-linear controller only. Dense GEMM,
attention, vector operations, and full PointLLM remain fail-closed.
