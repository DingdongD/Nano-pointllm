# Shared INT8 Dot4 Feature/Geometry Correlation

The locked 64-PE configuration alternates ordinary four-element feature dots
and geometry dots in one RTL hierarchy. Python and Icarus complete in 16 cycles
with exact agreement for 576 lane outputs, all events, all counters, signed
dot values, and reconstructed squared distances.

Yosys reports zero multipliers directly in `gtsu_shared_dot_geometry`, four
signed INT8 multipliers per `gtsu_dot4_pe`, 64 PE instances, and 256 shared
INT8 multipliers total. The existing W8 Split-K GEMV now instantiates the same
PE definition.

For PointLLM `8192 x 512`, the arithmetic-only mapping is 65,536 distance-dot
issue cycles and 128 point-norm precompute cycles at 64 PEs. A mathematically
exact INT16 fallback maps each multiplication to four INT8 partial products,
or 262,144 distance-dot issue cycles. Neither number includes SRAM stalls, FPS
min/argmax barriers, or KNN top-k.

This artifact does not establish INT8 point-cloud semantic fidelity, a complete
FPS/KNN implementation, PPA, timing closure, or full PointLLM cycle accuracy.
