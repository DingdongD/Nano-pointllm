# Sequential Dynamic-A8 ModelNet Smoke

This run uses the same held-out ModelNet request and per-output W8 weights as
the static and SmoothQuant comparisons. Linear inputs use one dynamic A8 scale
per token; convolution inputs use one scale per spatial position across input
channels.

KL mean is `0.00250105`, KL max is `0.00771366`, logit RMSE is `0.269341`,
teacher-forced top-1 agreement is `0.875`, and free greedy eight-token exact
match remains `1.0`. Lower KL/RMSE therefore does not guarantee identical
teacher-forced argmax at every low-margin position.

Dynamic scale reduction and scale delivery add hardware work. Larger semantic
evaluation and a cycle-correlated reduction path are required before selecting
this mode.
