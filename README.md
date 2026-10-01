# nano-pointllm（`nanopointllm`）

独立于 [`nano-vllm`](https://github.com/GeeeekExplorer/nano-vllm) 主线的 **Llama / PointLLM decode** 实验仓库：包名为 **`nanopointllm`**，避免与 PyPI / 本地的 `nanovllm` 冲突。

与 **PointLLM** 的对接说明见：  
[`/home/PointLLM/docs/fork_nano_decode_plan.md`](file:///home/PointLLM/docs/fork_nano_decode_plan.md)

**参考 nano-vllm 做性能上限**：移植与阶段划分见 **`docs/nanovllm_port_roadmap.md`**；**Llama 主干先行**，**终验环境：单卡 CUDA + PointLLM-7B**；代码落在 **`nanopointllm/llama/`**（占位包，随 P2 填充）。

## 状态

| 里程碑 | 状态 |
|--------|------|
| **M0** | 可 `pip install -e .`，含类型与 HF 参考 decode 循环 |
| **M1** | **`manual_greedy_decode`** 与 HF **`generate`** 的 parity（`scripts/parity_greedy_*.py`） |
| **M2** | **`hf_prefill` / `hf_decode_step` / `hybrid_greedy_decode`**（`nanopointllm.engine`）：显式 prefill + decode 边界，仍用 HF `forward`，与整段 greedy parity（`scripts/parity_hybrid_prefill_decode_*.py`） |
| **M3** | **`DecodeBackend` + 单步 decode 栈**（`decode_stack.py`：`hf`、`hf|compile`…）+ **`bench_m3_decode_pointllm.py`**（`--decode_stack`）与 Phase1 **同名指标**；默认 HF 为基线，**compile 等外层为可选实验** |
| **M4+** | 已实现 lightweight LLaMA、paged KV、Triton paged attention、continuous batching、decode CUDA Graph 与 batch sampling；HF 仅作为权重/模块容器和 parity fallback |

**P2 自定义层（进行中）**：`nanopointllm.llama` 内 **`LlamaRMSNormRef` / `LlamaSwiGLUMLPRef`** 与 HF 数值对齐（pytest 见下）；decode 栈根 **`hf_inner`** 见 `decode_stack`。

## 当前运行时结构

```text
Point cloud
    |
FPS / KNN / PointBERT -> Projector -> 513 point embeddings
                                      |
                                      v
                         Lightweight LLaMA prefill
                                      |
                    Block manager + paged KV pool
                                      |
             Continuous-batching scheduler (prefill/decode)
                                      |
      Lightweight LLaMA decode: packed QKV -> Triton paged attention
                    -> O projection -> packed MLP -> LM head
                                      |
                  CUDA Graph buckets -> compiled sampling
```

带 `num_kvcache_blocks` 的 `PointLLMLLMEngine` 是当前 inference runtime：PointBERT/Projector 只在请求 prefill 时执行，LLaMA prefill、mixed batch 和 decode 默认全部走 `LightweightLlamaRunner`，不进入 HuggingFace 模型级 `forward`。线性层复用 PyTorch/cuBLAS 的 GEMM/GEMV，并通过 packed QKV、packed Gate/Up 和连续布局减少 launch/读写；attention/KV 管理由本仓库 Triton kernel 和分页池实现。该实现继承 vLLM 的执行范式，但当前**不是** vLLM 或 FlashInfer 的直接依赖。

## 安装

**请始终在 PointLLM 文档中的同一 conda 环境里操作**（例如 [`setup_inference.md`](file:///home/PointLLM/docs/setup_inference.md) 里的 `conda create -n pointllm` / `conda activate pointllm`），与推理、评测共用一套 `torch` / `transformers`，避免版本分叉和重复下载 PyPI。

```bash
conda activate pointllm
cd /home/nano-pointllm
# 环境已含 torch/transformers 时推荐加 --no-deps，避免 pip 再拉一遍 torch
pip install -e . --no-deps
# 跑 pytest 单测（未装 pytest 时）：pip install pytest 或 pip install -e ".[dev]"
```

若要与 **PointLLM 代码与权重** 做 parity，还需：

```bash
export PYTHONPATH=/home/PointLLM:$PYTHONPATH
```

## 脚本

```bash
# 标准 Llama：手动 greedy 与 model.generate 是否一致（需本地或 Hub 权重路径）
python scripts/parity_greedy_hf_llama.py --model_path /path/to/Llama-2-7b-hf --max_new_tokens 8

# PointLLM（需 PYTHONPATH 含 /home/PointLLM；`--model_path` 须为真实目录，勿写 `...`）
python scripts/parity_greedy_pointllm.py \
  --model_path /home/PointLLM/checkpoints/PointLLM_7B_v1.2 \
  --device cuda:1 \
  --max_new_tokens 8

# M2：prefill + 分步 decode（hybrid） vs 整段 manual greedy
python scripts/parity_hybrid_prefill_decode_llama.py --model_path /path/to/Llama-2-7b-hf --max_new_tokens 8
python scripts/parity_hybrid_prefill_decode_pointllm.py \
  --model_path /mnt/llm_data/pointllm_ckpt/PointLLM_7B_v1.2 \
  --device cuda:1 \
  --max_new_tokens 8

# M3：经 engine 的 prefill + HF decode 后端计时（与 PointLLM phase1 脚本指标对齐）
python scripts/bench_m3_decode_pointllm.py \
  --model_path /mnt/llm_data/pointllm_ckpt/PointLLM_7B_v1.2 \
  --device cuda:1 \
  --decode_steps 32 \
  --warmup 2

# 单步 decode 栈：内→外用 `|` 连接；例如 HF 外再包 torch.compile（适当增大 warmup）
python scripts/bench_m3_decode_pointllm.py \
  --model_path /mnt/llm_data/pointllm_ckpt/PointLLM_7B_v1.2 \
  --device cuda:1 \
  --decode_stack "hf|compile" \
  --compile_mode reduce-overhead \
  --warmup 5 \
  --decode_steps 32

# 完整诊断：TTFT/TPOT/吞吐、长度与 batch sweep、CUDA breakdown、KV/HBM
python scripts/benchmark_pointllm_diagnostics.py \
  --model_path /mnt/llm_data/pointllm_ckpt/PointLLM_7B_v1.2_safetensors \
  --device cuda:0 \
  --batch_sizes 1,4,8 \
  --input_lengths 560,768,1024 \
  --output_lengths 16,64,256 \
  --warmup 10 \
  --runs 30 \
  --profile_batch_sizes 1,8 \
  --require_idle_gpu \
  --output_dir results/pointllm_diagnostics_full
```

**旧版 HF decode 基线栈**（`nanopointllm.engine.decode_stack`）：`build_decode_stack("hf|compile")` 只用于 M3 对照与回归，不是 paged runtime 的默认执行路径。  
**完整诊断指标口径**：TTFT 是请求提交至首 token，包含点云编码、prefill、LM head 与首 token sampling；TPOT 是首 token 后的 decode 总耗时除以 decode step 数，**不再除以 batch size**；`tokens/s` 是 batch 聚合吞吐。每个 shape 先记录一次 `cold_start_run`，再 warmup，最后以独立的 steady-state runs 报告 mean、median、P10、P90、P95、min/max；正式对比以 median 和 P10/P90 为主。脚本将性能 sweep 与少量代表配置的 CUDA event profiling 分开运行，输出 `metrics.json`、`summary.json`、`latency_throughput.png`、`decode_cuda_breakdown.png` 和 `kv_cache_hbm.png`。HBM 图使用“单次 decoder 权重读取 + KV 读写”的解析流量下界计算有效带宽，并非 Nsight Compute 硬件计数器。PointLLM 的完整 point-token span 决定最短有效输入长度，请勿用截断 point tokens 的方式伪造短输入。  
**稳态环境控制**：`--require_idle_gpu` 会在加载权重前检查目标 GPU 利用率和显存占用，不满足阈值时直接退出。每个正式配置期间以默认 200ms 周期记录 P-state、SM/显存时钟、功耗、温度、GPU utilization 和 memory-controller utilization；后者只是控制器忙碌度，不能当作 DRAM GB/s。若有管理员权限，可在实验前通过 `nvidia-smi -pm 1` 开启持久模式，并用 `nvidia-smi -lgc <min,max>` 固定 graphics clock；不要在 benchmark 脚本内部切换频率。无权限时保留 telemetry，并检查 SM clock 的 P10/P90 是否漂移。脚本只在阶段开始前清空旧 GPU 工作，以及在读取阶段 CUDA event 时同步；不会为 telemetry 增加计时区间内的同步。自回归 token 回传本身仍有必要的 host/device 依赖。  
**Nsight Compute 分段验证**：NCU replay 会改变延迟，因此只用于 QKV、attention、O projection、MLP 和 LM head 的硬件计数器，不与 TTFT/TPOT 混用。脚本会禁用 CUDA Graph 以隔离 NVTX stage，但保持相同的 packed projection、paged attention 与 lightweight runner。默认以 `--profile_layer 0` 采同构 decoder 的代表层计数器，完整 32 层仍会执行，只是不进入 NCU replay；这可避免为 7B 模型逐层保存近整卡显存。空卡时分别运行 B1/B8：

```bash
cd /home/nano-pointllm
for B in 1 8; do
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/home/PointLLM:. \
  /opt/conda/envs/pointllm/bin/python scripts/run_pointllm_ncu_stages.py \
  --model_path /mnt/llm_data/pointllm_ckpt/PointLLM_7B_v1.2_safetensors \
  --device cuda:0 --batch_size "$B" --input_length 768 \
  --warmup_decode_steps 10 --profile_steps 1 \
  --peak_hbm_gbps 1555 --peak_compute_tflops 312 \
  --output_dir "results/ncu_decode_stages/b${B}"
done
```

每个目录会生成一份完整 `all_stages.csv`、五组按 NVTX 拆分的 CSV、manifest、stage summary、`stage_bottleneck_summary.json` 和图片。只有空卡启动、权重占理论强制流量至少 80%、算术强度低于 roofline ridge、实测 DRAM 字节达到理论流量的 50%、DRAM throughput 达阈值且高于 SM throughput 时，线性 stage 才标为 `confirmed_weight_bandwidth_bound`；attention 单独按 paged K/V 流量判断，不能称为 weight-bound。

2026-09-30 的 A100 B1/B8 正式结果、图和结论见 [`docs/results/ncu_weight_bound_2026-09-30/`](docs/results/ncu_weight_bound_2026-09-30/README.md)。

### Decode MLP block liveness

`scripts/analyze_decode_mlp_blocks.py` 在普通 eager paged decode 上记录全部 32 层的 `z = SiLU(gate) * up`，并按连续 32/64/128/256 neurons 分块。默认用真实 ModelNet 点云、同一点云的四种 prompt 和 32-token greedy 输出，生成 retained contribution/block-ratio 曲线、相邻 token Jaccard、previous-token recall、同点云跨 prompt overlap，以及三种理论 MLP weight-byte reduction：

```bash
cd /home/nano-pointllm
PYTHONPATH=/home/PointLLM:. /opt/conda/envs/pointllm/bin/python \
  scripts/analyze_decode_mlp_blocks.py \
  --model_path /mnt/llm_data/pointllm_ckpt/PointLLM_7B_v1.2_safetensors \
  --device cuda:0 --dataset modelnet --num_point_clouds 2 \
  --max_new_tokens 32 \
  --output_dir results/mlp_block_trace/modelnet_n2_p4_t32_bf16
```

原始 BF16 activation 保存在 `mlp_swiglu_activations.pt`；`summary.json`、压缩 JSONL、全局/逐层/逐 token CSV 和两张 PNG 位于同一目录。第一个生成 token 来自 prefill，所以 32-token 输出对应 31 个 decode trace。贡献分数定义为 `|z_i| * ||W_down[:, i]||_2`，block 分数为块内求和。post-gate 只能跳过未选 block 的 `down_proj`；oracle-pre-gate 假定提前知道当前 token support，是不可因果实现的上界；previous-token 使用上一 decode token support，属于可因果预测，其贡献保持率与 overlap 才是是否值得设计稀疏 kernel 的关键。理论 weight bytes 只计算 gate/up/down 权重，不含 activation、索引和元数据流量。

2026-10-01 的 2-point-cloud、4-prompt、32-token ModelNet 结果见 [`docs/results/mlp_block_trace_modelnet_2026-10-01/`](docs/results/mlp_block_trace_modelnet_2026-10-01/README.md)。

### Compression sensitivity and PointBERT roofline

Neuron-level oracle、Wanda 2:4/4:8 与 PointBERT/projector/decoder/LM-head W8/W4 sensitivity 的脚本分别为：

```bash
python scripts/analyze_mlp_neuron_oracle.py --help
python scripts/evaluate_nm_pruning_pointllm.py --help
python scripts/evaluate_precision_sensitivity_pointllm.py --help
```

当前 neuron oracle 已触发 activation-sparsity stop gate，因此没有继续做 permutation/clustering；Wanda 50% N:M 在小规模真实生成中产生明显漂移；逐组件 sensitivity 表明 W8 是共同起点，而 W4 LM head 明显不可接受。完整结果和限制见 [`docs/results/compression_sensitivity_2026-10-01/`](docs/results/compression_sensitivity_2026-10-01/README.md)。

PointBERT 的严格 NCU B1/B4 profile 使用与 decoder 相同的空卡门禁和 duration-weighted 计数口径：

```bash
PYTHONPATH=/home/PointLLM:. /opt/conda/envs/pointllm/bin/python \
  scripts/run_pointbert_ncu_stages.py \
  --model_path /mnt/llm_data/pointllm_ckpt/PointLLM_7B_v1.2_safetensors \
  --device cuda:0 --batch_sizes 1,4 --warmup 5 \
  --peak_hbm_gbps 1555 --peak_compute_tflops 312 \
  --output_dir results/ncu_pointbert_strict_b1_b4
```

该流程分别输出 FPS、KNN、local encoder、完整 12 层 PointTransformer QKV/attention/O/MLP 的 NCU DRAM bytes、DRAM/SM throughput、显式 FLOP model、measured-byte AI 和统一 roofline 图。正式结果见 [`docs/results/pointbert_ncu_roofline_2026-10-01/`](docs/results/pointbert_ncu_roofline_2026-10-01/README.md)。

### Phase-adaptive architecture estimator

[`architecture_simulator/`](architecture_simulator/README.md) 提供独立的
PointLLM operation-level 解析估算器，覆盖 persistent FPS/KNN、dense tensor、
Split-K decode、W4/W8/BF16 weight streaming、fused/paged attention、分阶段
SRAM 和资源约束 DSE。它不是完整 cycle simulator、C-model 或 RTL；当前
没有 ready/valid、stall、bank conflict、DRAM timing 或 RTL correlation，
因此输出只用于探索，不能作为可发表的性能/面积/能耗结论。

```bash
cd /home/nano-pointllm/architecture_simulator
/opt/conda/envs/pointllm/bin/python -m pointllm_archsim simulate \
  --config configs/gtsu_baseline.json \
  --batch_size 1 --input_tokens 768 --output_tokens 128 \
  --output_dir ../results/architecture_simulator/baseline --plot
```

快速 DSE 会联合扫描 tensor shape、Split-K、geometry lanes、HBM、SRAM 与
weight-unpack 吞吐；`validate` 命令负责核对真实 config/YAML/safetensors
shape 和解析守恒。C-model/RTL 仍是明确标注的 future work。修正后的
`B1/S768/O128` 结果与限制见
[`docs/results/architecture_simulator_2026-10-01/`](docs/results/architecture_simulator_2026-10-01/README.md)。

**Paged runtime 默认轻量路径**：构造带 `num_kvcache_blocks` 的 `PointLLMLLMEngine` 后，prefill、mixed step 和 eager decode 默认均经 `LightweightLlamaRunner` 绕过 HuggingFace 模型级 `forward`。设置 `NANOPOINTLLM_LIGHTWEIGHT_DECODE=0` 才恢复显式 HF fallback，用于 parity/debug；`NANOPOINTLLM_ENABLE_CUDA_GRAPH=1` 启用固定 shape bucket 的 decode graph 并强制轻量路径。  
分页执行会从 `ForwardContext` 注入每个逻辑请求自己的 position ids，包括 PointLLM 显式向基类传递 `position_ids=None` 的情况；修改运行时代码后需重启已有 Python 服务，确保类级 patch 重新安装。  
**`torch_loop`**：需 **`transformers.cache_utils`**（含 `DynamicCache`）；若环境缺该模块，运行时会 **自动退回 `hf_inner`** 并打 log，升级 `transformers` 后可走显式逐层路径。  
**transformers≥5.5 且 PyTorch 低于 2.6 时加载 `.bin` 权重**：会触发 **CVE-2025-32434** 相关校验报错；请 **`pip install -U 'torch>=2.6'`**（与驱动匹配），或对含 **safetensors** 分片的目录使用脚本 **`--use_safetensors`**。详见 **`PointLLM/docs/setup_inference.md`** 常见问题表。  
**切勿**只升 `torch` 而忽略 **与 CUDA 索引配套的 `torchvision`/`torchaudio`**；若出现 **`ncclCommWindowDeregister` undefined symbol**，见 **`setup_inference.md` §6** 与 **§6.1**（成套重装 + 与 conda CUDA 避免混装）。

**P2：Llama RMSNorm / MLP 与 HF 对齐（无 GPU，可本地 CI）**：

```bash
python3 -m pip install -q pytest && cd /home/nano-pointllm && python3 -m pytest -q tests/test_llama_hf_config.py tests/test_llama_norm_mlp_parity.py tests/test_eager_attention_kv.py tests/test_torch_loop_past_normalize.py tests/test_build_decode_stack.py
```

**P2 验证（单卡 CUDA + PointLLM-7B）**：

```bash
conda activate pointllm
export PYTHONPATH=/home/PointLLM:$PYTHONPATH
cd /home/nano-pointllm
python3 -m pip install -q pytest && python3 -m pytest -q tests/test_llama_hf_config.py tests/test_llama_norm_mlp_parity.py tests/test_eager_attention_kv.py tests/test_torch_loop_past_normalize.py tests/test_build_decode_stack.py
python scripts/verify_p2_pointllm_cuda.py \
  --model_path /mnt/llm_data/pointllm_ckpt/PointLLM_7B_v1.2 \
  --device cuda:1
```

## 致谢

实现思路参考 **nano-vllm**（轻量调度、KV 块等）；本仓库 **不复用**其 Qwen3 模型代码路径，以便独立演进。
