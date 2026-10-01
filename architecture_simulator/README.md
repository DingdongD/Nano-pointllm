# PointLLM Phase-Adaptive Architecture Simulator

This subproject models the proposed **Geometry-Tensor-Streaming Unified (GTSU)** accelerator. It consumes explicit PointLLM operation traces and reports phase cycles, HBM/SRAM traffic, tensor utilization, dataflow selection, and bottlenecks.

It is a **trace-driven phase-level analytical model**, not an RTL cycle-accuracy claim. Cross-operation scheduling is conservative and serial. Within an operation, compute, HBM, unpack, and SRAM overlap only when the modeled buffer contract permits it. Energy and area remain disabled until characterized RTL/PTPX/CACTI values are supplied.

## Why a separate subproject

The structure follows useful boundaries from the locally audited references:

- `/home/Compiler_Codes` at `ad2c31a`: compiler, functional C-model, cycle model, runtime, and RTL are separate contracts; architecture configuration and operation traces are explicit.
- `/home/PointAcc/QuickFPS`: persistent state, event/counter output, DMA/memory backend boundaries, and RTL-correlated geometry timing.
- `/home/PointAcc/Mamba_Reconfigurable_Array_v2`: functional reference, parameterized RTL, golden trace, gate-activity testbench, and PTPX handoff.

No source is copied from those projects. This directory keeps a small, PointLLM-specific JSON contract that can later drive a C++/SystemC implementation or RTL testbench.

## Modeled architecture

```text
                         HBM
                          |
        +-----------------+------------------+
        |                                    |
 Persistent Geometry                 Weight Stream Engine
 distance/min/argmax/top-k       W4/W8/BF16 + scale + double buffer
        |                                    |
        +-----------------+------------------+
                          |
              Reconfigurable Tensor Fabric
          dense output-stationary | Split-K GEMV
                          |
          +---------------+----------------+
          |                                |
    Vector/fusion engine          Streaming attention
  norm/activation/pool/reduce   dense/causal/paged online softmax
                          |
               Phase-partitioned banked SRAM
```

The PointLLM preset is generated from the checked-in workload evidence:

- PointBERT: `8192 -> 512x32 -> 513x384`, 12 layers, 6 heads.
- Projector: `384 -> 1024 -> 2048 -> 4096`.
- LLaMA: hidden 4096, intermediate 11008, 32 layers/heads, vocabulary 32003.
- Precision policy defaults to W8 for all weights; component overrides support W4/W8/BF16 studies.

## Quick start

```bash
cd /home/nano-pointllm/architecture_simulator

# Create an inspectable operation trace.
python -m pointllm_archsim preset \
  --batch_size 1 --input_tokens 768 --output_tokens 128 \
  --output workloads/pointllm_7b_b1_s768_o128.json

# Simulate one architecture.
python -m pointllm_archsim simulate \
  --config configs/gtsu_baseline.json \
  --workload workloads/pointllm_7b_b1_s768_o128.json \
  --output_dir ../results/architecture_simulator/baseline --plot

# Compare mixed precision without changing the model trace format.
python -m pointllm_archsim simulate \
  --config configs/gtsu_baseline.json \
  --batch_size 1 --input_tokens 768 --output_tokens 128 \
  --precision_policy decoder_qkv=4,decoder_o=4 \
  --output_dir ../results/architecture_simulator/mixed_w4_qkvo --plot

# Sweep array, Split-K, FPS lanes, HBM, and SRAM.
python -m pointllm_archsim sweep \
  --config configs/gtsu_baseline.json \
  --sweep configs/sweep_quick.json \
  --batch_size 1 --input_tokens 768 --output_tokens 128 \
  --output_dir ../results/architecture_simulator/sweep_quick --plot

# Normalize measured NCU evidence; this does not fit ASIC cycles to A100 time.
python -m pointllm_archsim extract-evidence \
  --repo_root /home/nano-pointllm \
  --output ../results/architecture_simulator/ncu_evidence.json
```

## Output contract

`simulation.json` contains:

- total and per-phase cycles/seconds;
- per-operation compute/HBM/SRAM/unpack cycles;
- HBM reads/writes and SRAM traffic;
- tensor utilization, selected dataflow, Split-K degree, and bottleneck;
- start/end cycle timeline and links to measured references;
- optional energy/area only when characterization fields are complete.

The DSE emits `sweep.csv`, a resource-aware `pareto.csv`, `summary.json`, and an optional plot. Raw results belong under the repository-level ignored `results/` directory.

The quick sweep treats tensor PEs, SRAM capacity, HBM bandwidth, geometry lanes,
and weight-unpack bytes/cycle as separate resource costs. This matters because a
precision-aware decoder can move from unpack-bound to HBM-bound before adding
more tensor PEs helps.

## Subproject layout

```text
pointllm_archsim/   executable Python analytical model and stable schemas
configs/            architecture and DSE parameters
cmodel/             event-level C++/SystemC handoff contract (planned)
rtl/                synthesizable block/interface contract (planned)
verification/       golden trace and correlation gates
characterization/   area/energy parameter provenance and templates
tests/              analytical invariants and CLI-facing contracts
```

The planned directories deliberately contain interface documentation rather
than placeholder implementations. A block is promoted from Python to C-model
or RTL only after its input/output events, counters, and correlation threshold
are fixed in `verification/`.

## Fidelity roadmap

1. Correlate persistent FPS against the QuickFPS RTL/C-model event contract.
2. Add tile-level producer/consumer overlap and bank-conflict traces.
3. Replace analytical HBM with an optional DRAMsim3 backend.
4. Generate tensor/Split-K/attention golden vectors for RTL testbenches.
5. Import synthesized area and PTPX active/idle energy by stable block name.
6. Compare shared GTSU against a disaggregated PointISA-like frontend plus conventional LLM accelerator under equal area/HBM constraints.
