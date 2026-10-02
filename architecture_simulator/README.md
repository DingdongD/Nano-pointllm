# PointLLM Phase-Adaptive Architecture Estimator

This subproject is an executable analytical estimator for the proposed **Geometry-Tensor-Streaming Unified (GTSU)** accelerator. It consumes explicit PointLLM operation traces and reports estimated phase cycles, HBM/SRAM traffic, tensor utilization, dataflow selection, and bottlenecks.

It is **not a complete PointLLM cycle simulator or RTL implementation**. The
legacy `pointllm_archsim` path remains an uncalibrated operation-level
estimator. A separate `gtsu_cycle` path now contains the first RTL-locked
vertical slice, but full-model cycle accuracy remains disabled. Energy and area
remain disabled until characterized RTL/PTPX/CACTI values are supplied.

## Implementation status

| Capability | Status |
|---|---|
| PointLLM operation trace and shape contract | Implemented and checked against config/YAML/safetensors headers |
| Analytical MAC, traffic, tensor mapping, and phase estimates | Implemented with conservation tests |
| DSE and JSON/CSV/plot output | Implemented |
| W8 Split-K GEMV slice: finite FIFO/backpressure event model | Implemented |
| W8 Split-K GEMV slice: synthesizable RTL | Implemented; Yosys checked |
| W8 Split-K GEMV slice: exact event/cycle/output correlation | Implemented for locked test configurations |
| 16-bank x 128-bit 1R1W SRAM: conflicts, arbitration, 3-cycle reads | Python/RTL exact correlation for locked two-client schedule |
| PointLLM decoder linear lowering | Real safetensors shapes to W8 tiles, DRAM bursts, and SRAM bank/row addresses |
| DRAM timing | Local pinned DRAMsim3 backend for explicit sampled requests |
| Production decoder linear controller | Q and non-uniform-K down projection exactly RTL-correlated with DRAMsim3 traces |
| Geometry distance tile | Signed INT16 3-D distance, elastic ready/valid, exact Python/Icarus event/value/cycle correlation |
| Shared-Dot4 dense GEMM microtile | Output-stationary M/K stream, edge-column mask, elastic output, exact Python/Icarus value/event/cycle correlation |
| Dense M/N/K ping-pong controller | Exact counter, edge-tile, K-block, and two-buffer lifecycle correlation for representative PointLLM K classes |
| INT32-to-A8 requant | Offline FP16-scale compiler, INT32 bias, multiplier/shift RNE, symmetric saturation, exact Python/Icarus correlation |
| Physical Dense SRAM composition | 16-bank/128-bit/3-cycle SRAM payload, ping-pong 1R1W, response FIFO, 3-column Dot4 plus requant; exact value/event/cycle correlation |
| Full PointLLM event-level resource occupancy and stalls | Not implemented |
| C++/SystemC C-model | Not implemented |
| Full M/N/K GEMM controller and SRAM double buffering | Not implemented; microtile only |
| FPS min/argmax, KNN global top-k, attention, vector, sampling RTL | Not implemented |
| Full PointLLM RTL cycle correlation | Not implemented; fail-closed |
| Characterized area/energy | Not implemented |

Every legacy estimator result is tagged `result_status=exploratory_only`,
`event_level_model=false`, and `rtl_correlated=false`. Passing the validator
means shapes and analytical conservation are internally consistent; it does
not make the latency prediction cycle-accurate or publishable.

The cycle path has an explicit coverage registry. Requesting an unsupported
operator raises an error instead of falling back to analytical latency.

## Why a separate subproject

The structure follows useful boundaries from the locally audited references:

- `/home/Compiler_Codes` at `ad2c31a`: compiler, functional C-model, cycle model, runtime, and RTL are separate contracts; architecture configuration and operation traces are explicit.
- `/home/PointAcc/QuickFPS`: persistent state, event/counter output, DMA/memory backend boundaries, and RTL-correlated geometry timing.
- `/home/PointAcc/Mamba_Reconfigurable_Array_v2`: functional reference, parameterized RTL, golden trace, gate-activity testbench, and PTPX handoff.

No source is copied from those projects. Their unimplemented handoff plan is documented in [`docs/CMODEL_RTL_ROADMAP.md`](docs/CMODEL_RTL_ROADMAP.md).
The executable RTL simulation and exact-correlation procedure is documented in
[`docs/RTL_SIMULATION_METHOD.md`](docs/RTL_SIMULATION_METHOD.md).

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

# Run exact Python-event-model versus Icarus RTL correlation for the first slice.
python scripts/run_splitk_rtl_correlation.py \
  --config configs/gtsu_splitk_rtl_lock.json \
  --rtl_root rtl/vertical_slice \
  --output_dir ../results/gtsu_cycle/splitk_rtl_lock

# Inspect strict coverage. Requiring attention currently fails by design.
python scripts/check_cycle_coverage.py

# Correlate SRAM RTL, lower real PointLLM tensors, and time sampled requests
# with the pinned local DRAMsim3 bridge.
python scripts/run_pointllm_memory_correlation.py \
  --output-dir ../docs/results/gtsu_memory_lowering_2026-10-01 \
  --projection q_proj --layer 0 --sample-bursts 64

# Correlate the 64-lane signed geometry distance tile. This is the shared
# distance front-end, not a complete FPS or KNN cycle result.
python scripts/run_geometry_rtl_correlation.py \
  --config configs/gtsu_geometry_rtl_lock.json \
  --rtl_root rtl/vertical_slice \
  --output_dir ../docs/results/gtsu_geometry_distance_2026-10-01

# Correlate alternating feature and cached-norm geometry operations on the
# same signed INT8 SIMD4 PE hierarchy.
python scripts/run_shared_dot_rtl_correlation.py \
  --config configs/gtsu_shared_dot_rtl_lock.json \
  --rtl_root rtl/vertical_slice \
  --output_dir ../docs/results/gtsu_shared_dot_geometry_2026-10-01

# Correlate a 64-PE dense microtile at the PointTransformer K=384 shape.
# This covers one N tile; production M/N tiling and SRAM are still fail-closed.
python scripts/run_dense_gemm_rtl_correlation.py \
  --rows 5 --columns 61 --n_tile 64 --k 384 \
  --rtl_root rtl/vertical_slice \
  --output_dir ../docs/results/gtsu_dense_gemm_microtile_2026-10-01

# Correlate Local Encoder, PointTransformer, projector, LLM-prefill, and
# non-divisible K-block controller cases. Payload SRAM remains a separate gate.
python scripts/run_dense_controller_shape_suite.py \
  --rtl_root rtl/vertical_slice \
  --output_dir ../docs/results/gtsu_dense_mnk_controller_2026-10-01

# Correlate the Compiler_Codes-informed, PointLLM-contract-specific requant unit.
python scripts/run_requant_rtl_correlation.py \
  --rtl_root rtl/vertical_slice \
  --output_dir ../docs/results/gtsu_requant_compilercodes_audit_2026-10-01

# Run the first physically composed SRAM -> Dot4 -> requant slice.
python scripts/run_dense_sram_pipeline_rtl_correlation.py \
  --rtl_root rtl/vertical_slice \
  --output_dir ../docs/results/gtsu_dense_sram_requant_pipeline_2026-10-02

# Sweep 64/128/256-byte-per-cycle production q-projection supply.
for lanes in 1 2 4; do
  python scripts/run_dense64_fabric_correlation.py \
    --rtl_root rtl/vertical_slice --shape q_proj --memory_lanes "$lanes" \
    --output_dir "../docs/results/dense64_qproj_lane${lanes}"
done

# Correlate K-block-resident high-M mode at the full PointTransformer QKV shape.
python scripts/run_pointtransformer_weight_reuse.py \
  --rtl_root rtl/vertical_slice \
  --output_dir ../docs/results/dense64_pointtransformer_reuse

# Sweep 64-way A8 plus configurable BF16 post-processing.
python scripts/run_dense64_dual_output_sweep.py \
  --rtl_root rtl/vertical_slice \
  --output_dir ../docs/results/dense64_dual_output
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
gtsu_cycle/         RTL-locked event models and fail-closed coverage registry
configs/            architecture and DSE parameters
rtl/vertical_slice/ synthesizable RTL and testbench for correlated slices
scripts/            RTL correlation and coverage entry points
docs/               fidelity roadmap and boundaries
tests/              shape, conservation, mapping, and CLI contracts
```

The shared geometry path uses
`||p-c||^2 = ||p||^2 + ||c||^2 - 2 p^T c`. Point norms are precomputed on the
same dot fabric and stored as 16-bit values; the steady-state distance path
reads norms and uses only the shared dot, add, subtract, and shift. This is a
claim of shared multiplication resources only. FPS min/argmax and KNN top-k
remain specialized structures.

The matching fidelity-side W8A8 contract is implemented in
`nanopointllm/compression/w8a8.py`. A small sequential-QDQ run can be launched
from the repository root with:

```bash
python scripts/evaluate_sequential_w8a8_pointllm.py \
  --dataset modelnet --num_calibration_samples 2 --num_eval_samples 2 \
  --max_new_tokens 8 --weight_granularity per_output \
  --activation_mode static --output_dir results/w8a8_modelnet
```

This executes dense BF16/FP16 operators after QDQ and measures fidelity only;
it does not benchmark an integer kernel.

Production Dense64 now has two exact dataflows: a weight-stream mode for decode
and a K-block-resident mode with tagged partial sums for high-M GEMM. The real
layer-0 q-projection package drives all 4096 ACC32 outputs and package-backed
A8/BF16 post-processing. The full PointTransformer QKV shape checks all 590,976
outputs. Request-side DRAMsim3 feedback, foundry SRAM/timing, geometry,
attention, and end-to-end PointLLM latency remain fail-closed.

## Fidelity roadmap

1. Connect the geometry tile to resident point/min-distance SRAM, then correlate
   the persistent FPS min/argmax loop and KNN cross-tile top-32 merge.
2. Connect the implemented tile/address lowering, correlated SRAM ports, and
   DRAMsim3 completions to the Split-K RTL producer/consumer pipeline.
3. Add production-size multi-client SRAM scheduling and DMA outstanding-credit
   correlation; replace behavioral arrays with characterized SRAM macros.
4. Generate tensor/Split-K/attention golden vectors for RTL testbenches.
5. Import synthesized area and PTPX active/idle energy by stable block name.
6. Compare shared GTSU against a disaggregated PointISA-like frontend plus conventional LLM accelerator under equal area/HBM constraints.
