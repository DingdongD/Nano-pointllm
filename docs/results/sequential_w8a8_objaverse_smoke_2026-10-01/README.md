# Sequential W8A8 Objaverse Smoke

This smoke uses one Objaverse calibration object and one disjoint held-out
object from the real 8192-point validation pipeline, one prompt, eight generated
tokens, simultaneous per-output W8 across 283 modules, and sequential static A8.

The held-out result is KL mean `0.0116791`, KL max `0.0373361`, teacher-forced
top-1 agreement `1.0`, free-generation token match `1.0`, and exact match
`1.0`. Exact object IDs and module calibration ranges are preserved in
`summary.json`.

This validates execution on the second point-cloud distribution but remains a
one-request smoke. It is not an accuracy, runtime, or hardware-efficiency claim.
