# Sequential SmoothQuant W8A8 ModelNet Smoke

This run uses the same ModelNet calibration/evaluation IDs, prompt, eight-token
length, 283 modules, and per-output W8 contract as the static-A8 baseline.
SmoothQuant alpha is `0.5`; every module applies the exact `x/s`, `W*s`
transformation before static activation calibration and QDQ.

KL mean is `0.0048096`, KL max is `0.0176964`, logit RMSE is `0.306248`, and
teacher-forced top-1 plus greedy exact match are `1.0`. On this request all
three distribution metrics improve substantially over unsmoothed static A8.

The sample is too small to select alpha or certify fidelity. The run uses dense
BF16 execution after QDQ and makes no integer-kernel speed claim.
