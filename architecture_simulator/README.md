# PointLLM Phase-Adaptive Architecture Estimator

This subproject is an executable analytical estimator for the proposed **Geometry-Tensor-Streaming Unified (GTSU)** accelerator. It consumes explicit PointLLM operation traces and reports estimated phase cycles, HBM/SRAM traffic, tensor utilization, dataflow selection, and bottlenecks.

It is **not a complete cycle simulator, C-model, or RTL implementation**. Cross-operation scheduling is conservative and serial; there is no ready/valid event scheduler, bank-conflict model, DRAM timing backend, or RTL correlation yet. Within an operation, compute, HBM, unpack, and SRAM overlap only according to analytical equations. Energy and area remain disabled until characterized RTL/PTPX/CACTI values are supplied.

## Implementation status

| Capability | Status |
|---|---|
| PointLLM operation trace and shape contract | Implemented and checked against config/YAML/safetensors headers |
| Analytical MAC, traffic, tensor mapping, and phase estimates | Implemented with conservation tests |
| DSE and JSON/CSV/plot output | Implemented |
| Event-level resource occupancy and stalls | Not implemented |
| C++/SystemC C-model | Not implemented |
| Synthesizable RTL | Not implemented |
| RTL cycle correlation | Not implemented |
| Characterized area/energy | Not implemented |

Every result is tagged `result_status=exploratory_only`,
`event_level_model=false`, and `rtl_correlated=false`. Passing the validator
means shapes and analytical conservation are internally consistent; it does
not make the latency prediction cycle-accurate or publishable.

## Why a separate subproject

The structure follows useful boundaries from the locally audited references:

- `/home/Compiler_Codes` at `ad2c31a`: compiler, functional C-model, cycle model, runtime, and RTL are separate contracts; architecture configuration and operation traces are explicit.
- `/home/PointAcc/QuickFPS`: persistent state, event/counter output, DMA/memory backend boundaries, and RTL-correlated geometry timing.
- `/home/PointAcc/Mamba_Reconfigurable_Array_v2`: functional reference, parameterized RTL, golden trace, gate-activity testbench, and PTPX handoff.

No source is copied from those projects. Their unimplemented handoff plan is documented in [`docs/CMODEL_RTL_ROADMAP.md`](docs/CMODEL_RTL_ROADMAP.md).

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

# Check the workload against the real model files and verify conservation.
python -m pointllm_archsim validate \
  --config configs/gtsu_baseline.json \
  --workload workloads/pointllm_7b_b1_s768_o128.json \
  --checkpoint_config /mnt/llm_data/pointllm_ckpt/PointLLM_7B_v1.2_safetensors/config.json \
  --pointbert_config /home/PointLLM/pointllm/model/pointbert/PointTransformer_8192point_2layer.yaml \
  --output ../results/architecture_simulator/validation.json
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
pointllm_archsim/   executable Python estimator, validation, and schemas
configs/            architecture and DSE parameters
docs/               explicit unimplemented C-model/RTL roadmap
tests/              shape, conservation, mapping, and CLI contracts
```

Only executable code is listed as implemented. The future C-model and RTL work
is intentionally kept under `docs/` rather than represented by empty source
directories.

## Fidelity roadmap

1. Correlate persistent FPS against the QuickFPS RTL/C-model event contract.
2. Add tile-level producer/consumer overlap and bank-conflict traces.
3. Replace analytical HBM with an optional DRAMsim3 backend.
4. Generate tensor/Split-K/attention golden vectors for RTL testbenches.
5. Import synthesized area and PTPX active/idle energy by stable block name.
6. Compare shared GTSU against a disaggregated PointISA-like frontend plus conventional LLM accelerator under equal area/HBM constraints.
