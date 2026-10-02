# Production PointLLM Decoder Linear Correlation

Projection `q_proj`, shape `1x4096 -> 1x4096`.
The DRAMsim3 completion stream, finite 32-entry ROB, banked SRAM reads,
registered latency, unpack, compute issue, Split-K partials, and output counts
match the parameterized RTL controller exactly at `997858` cycles.

This promotes the production decoder-linear controller only. Dense GEMM,
attention, vector operations, and full PointLLM remain fail-closed.
