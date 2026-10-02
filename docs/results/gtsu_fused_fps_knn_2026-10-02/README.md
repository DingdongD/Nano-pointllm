# Fused FPS/KNN M5b Selector Closure

The `16`-point, `5`-center, Top-`4` tie/backpressure lock completes in `33`
cycles with exact Python/RTL values, events, counters, and cycles. The compiled
production selector completes `8192` points, `512` rounds, and Top-`32` in
`66,048` cycles with zero mismatches.

Real payload runs use deployed PointLLM center indices, CUDA-PTX-equivalent FPS
distance traces, and PointLLM KNN distances. ModelNet and Objaverse each match
all centers, next centers, `16,384` neighbor values, and Top-32 sets. Raw traces
are checksummed under `/mnt/llm_data/nano_pointllm_geometry_traces`; only compact
reports are committed.

The audit found that strict semantics require independent FPS/KNN distances and
masks. FPS skips near-origin points and uses direct FP32 FMA distance, while KNN
keeps all points and can produce small negative values from norm/dot roundoff.
The RTL's FP32 sortable-key mode handles those values exactly.

The following gates remain false:

- explicit banked SRAM macro timing for coordinates, norms, and min state;
- synthesizable IEEE FP32 MUL/FMA generation matching the deployed CUDA PTX;
- one composed arithmetic/SRAM/selector value-event-cycle trace;
- `fps`, `knn_topk`, and full-PointLLM coverage.

See `correlation.json`, `production_selector/`, `real_modelnet_sample0/`,
`real_objaverse_sample0/`, and `semantic_fidelity_real4.json` for evidence and
claim boundaries.
