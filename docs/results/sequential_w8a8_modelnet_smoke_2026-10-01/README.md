# Sequential W8A8 ModelNet Smoke

This smoke run uses one ModelNet calibration point cloud and one disjoint
held-out point cloud, one prompt, and eight generated tokens. It simultaneously
applies per-output-channel W8 to 283 PointBERT, projector, 32-layer decoder, and
LM-head modules (6,639,776,896 weight parameters), then applies static A8 QDQ at
every selected module input so quantization error propagates sequentially.

The held-out result is KL mean `0.0144278`, KL max `0.0503888`, teacher-forced
top-1 agreement `1.0`, free-generation token match `1.0`, and exact match `1.0`.
The calibration ID is `modelnet-test-1577`; the evaluation ID is
`modelnet-test-1722`.

This is a path-validation smoke, not a fidelity certification. It uses dense
BF16 execution after QDQ, contains only one held-out request, and provides no
integer-kernel latency or energy evidence.
