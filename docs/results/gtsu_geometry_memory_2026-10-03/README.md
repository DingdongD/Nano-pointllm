# Geometry SRAM And DRAM Request Closure

This directory records a clean-room memory behavior contract informed by the
externally visible architecture of `/home/Compiler_Codes` at commit `ad2c31a`.
No proprietary RTL source is copied.

The SRAM fabric maps one 128-bit `xyz FP32 + norm FP32` word per point across
four groups of 16 banks. A fifth 16-bank SRAM packs four FP32 persistent-min
values per word. Production logical capacity is 160 KiB: 128 KiB point/norm
plus 32 KiB min state. Byte enables, independent read/write behavior, and
three-cycle reads are correlated in `sram/correlation.json`.

The DRAM path follows the 512-bit AXI contract: 64-byte beats, a 256-beat
maximum, no burst crossing a 4-KiB boundary when enabled, finite outstanding
credits, and DRAMsim3 completion timing.
The 8,192-point preload contains 2,048 lines in 32 bursts and completes in 9,343
accelerator cycles for the locked DDR4 configuration.

`dma_sram/correlation.json` is a reduced 128-point payload lock. It sends 32
DRAMsim3-timed lines through the tagged completion ROB, unpacks four 128-bit
words per line, writes the point SRAM, and reads every point back exactly.

These results do not instantiate a foundry SRAM macro, provide SRAM STA/PPA,
compose production 8,192-point payload RTL, or close FP32 arithmetic and the
FPS/KNN selector in one pipeline. Complete FPS/KNN and full-model coverage
therefore remain disabled.
