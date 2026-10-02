# Fused FPS/KNN M5a RTL Lock

This directory records the first fail-closed fused selection-controller result.
The `16`-point, `5`-center, Top-`4` lock completes in `33` cycles. Python and
RTL match every center, neighbor index, neighbor distance, event, counter, and
cycle. Point 1 duplicates point 0 to exercise deterministic lower-index ties.

The synthesizable wrapper contains one shared INT8 Dot4 geometry stage and one
FPS/KNN selection controller. This is a hierarchy proof only. The following
gates remain false:

- one composed shared-Dot-to-selection value/event/cycle trace;
- production `8192`-point, `512`-round, Top-`32` compiled RTL;
- explicit banked SRAM macro timing for coordinates, norms, and min state;
- ModelNet and Objaverse INT8 center equality / KNN recall;
- `fps`, `knn_topk`, and full-PointLLM coverage.

See `correlation.json` for machine-readable evidence and claim boundaries.
