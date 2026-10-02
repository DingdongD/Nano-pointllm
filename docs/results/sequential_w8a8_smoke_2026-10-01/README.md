# Sequential PointLLM W8A8 Smoke Comparison

`comparison.png` compares five real full-component QDQ smoke runs. Every run
quantizes 283 modules spanning PointBERT, projector, all 32 decoder layers, and
LM head. ModelNet variants share the exact calibration/evaluation object IDs,
prompt, and eight-token output length.

| Dataset | W8 granularity | A8 mode | KL mean | KL max | TF top-1 | Greedy exact |
|---|---|---|---:|---:|---:|---:|
| ModelNet | per-output | static | 0.014428 | 0.050389 | 1.000 | 1.000 |
| ModelNet | group-128 | static | 0.015862 | 0.083957 | 1.000 | 1.000 |
| ModelNet | per-output | SmoothQuant static | 0.004810 | 0.017696 | 1.000 | 1.000 |
| ModelNet | per-output | dynamic per-token | 0.002501 | 0.007714 | 0.875 | 1.000 |
| Objaverse | per-output | static | 0.011679 | 0.037336 | 1.000 | 1.000 |

These are one-calibration/one-held-out-request path-validation smokes, not
fidelity certification. All execute dense BF16 operators after QDQ.
