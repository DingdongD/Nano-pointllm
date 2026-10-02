# Sequential Group-128 W8A8 ModelNet Smoke

This run uses exactly the same calibration ID, held-out ID, prompt, token count,
283 modules, and static A8 contract as the per-output W8 smoke. Only weight
granularity changes to group-128.

Group-128 lowers parameter-weighted weight RMSE from `0.000217721` to
`0.000155587`, but held-out KL mean changes from `0.0144278` to `0.0158621` and
KL max from `0.0503888` to `0.0839571`. Both schemes preserve teacher-forced
top-1 and eight-token greedy exact match on this single request.

This smoke provides no evidence that group-128 is preferable to the simpler
per-output scale contract. Larger disjoint evaluation is required, and any
accuracy result must also be weighed against group-scale storage and
group-partial dequantization cost.
