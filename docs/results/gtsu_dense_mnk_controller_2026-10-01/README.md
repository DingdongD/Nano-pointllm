# Dense M/N/K Ping-Pong Controller

This suite correlates the standalone Dense GEMM tile controller against Icarus
RTL for representative PointLLM K classes and edge tiles.

| Case | Representative shape | RTL cycles | Cycle error |
|---|---:|---:|---:|
| Local Encoder | `2x70x8` | 11 | 0 |
| PointTransformer | `2x70x384` | 481 | 0 |
| Projector | `1x70x2048` | 1281 | 0 |
| LLM prefill | `1x70x4096` | 2561 | 0 |
| Partial K block | `3x70x132` | 250 | 0 |

All event and counter traces are exact. The controller covers M/N/K counters,
edge-N masks, partial final K blocks, backpressure, and ping-pong buffer
lifecycle. Each subdirectory preserves its RTL log, synthesis log, and JSON.

The load interface currently represents an abstract "K block is resident"
token. It does not yet move payload through the physical 16-bank SRAM or DMA,
and these representative rows are not full production-shape traces.
