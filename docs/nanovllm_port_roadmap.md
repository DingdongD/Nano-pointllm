# 参考 nano-vllm 提升 nano-pointllm 性能：路线图

**原则**：`/home/nano-vllm` 保持 **只读参考**；在 **`/home/nano-pointllm`**（`nanopointllm`）内 **按需移植或重写**，避免与上游 **Qwen3 / ASR** 主线耦合。性能收益主要来自 **替换 HF decode 路径**（自定义 KV 布局 + 融合注意力 + 调度），而不是在同一 `forward` 上叠 `torch.compile`。

### 默认目标环境（团队约定）

| 项 | 约定 |
|----|------|
| **硬件** | **单卡 CUDA**（`cuda:0` / `cuda:1` 按机器占用选择；bench / parity 与 PointLLM 文档一致即可） |
| **主权重** | **PointLLM-7B v1.2**（例：`/mnt/llm_data/pointllm_ckpt/PointLLM_7B_v1.2` 或 `checkpoints/PointLLM_7B_v1.2` 软链） |
| **dtype** | 与现网一致：**`bfloat16`**（parity 需要时可临时 `float32` 缩小数值差） |
| **验收优先级** | **P2b 以 PointLLM 7B 为准**：HF prefill + 自定义 decode vs 全 HF；**P2a** 可用标准 Llama HF 做小步验证（可选，非替代 7B 终验） |

---

## 0. 策略：Llama 模块先行（推荐）

**同意**：应 **优先做 Llama 主干** 与 nano-vllm **思想/层实现** 的适配；**暂缓**把 Qwen3 整条栈或 PointBERT 绑进第一版自定义 decode。

| 顺序 | 做什么 | 为什么 |
|------|--------|--------|
| **L0** | 继续用 **HF `LlamaForCausalLM`** 做 **decode 单步 parity**（已有 `DecodeBackend` / `hybrid` / bench） | 基线不动 |
| **L1** | 在 **`nanopointllm/llama/`**（新建包）里，按 **HF `LlamaConfig`** 从 nano-vllm **只移植与 Llama 同构** 的片段：`rotary_embedding`、`RMSNorm`、`SiLU MLP` 等；**不要**抄 `Qwen3Attention` 的 QKV 合并方式，改参照 **`transformers.models.llama`** 的张量形状 | 先保证 **与 HF Llama 数值一致**，再谈 Triton |
| **L2** | （**可选**）在 **标准 Llama HF** 上实现 **`DecodeBackend`** 与 **`HFDecodeBackend` greedy parity**（`parity_greedy_hf_llama.py`），单卡 CUDA 上排错更快 | **不替代**终验；主权重仍以 **PointLLM 7B** 为准 |
| **L3** | **必做**：**PointLLM-7B**，单卡 CUDA — 仅替换 **无 `point_clouds` 的 decode 步**（prefill 仍 HF）；主干为 **同一套 Llama 适配** | **终验与 bench**（`parity_hybrid_prefill_decode_pointllm.py`、`bench_m3_decode_pointllm.py`）均以此配置为主 |

**代码落点建议**：新建 **`nanopointllm/llama/`** 子包（`kernels/`、`blocks/` 可按需再拆），与 **`nanopointllm/engine/`** 通过 **`DecodeBackend`** 对接；**不要**在 `nanopointllm/models/` 里混放 Qwen 残留命名。

---

## 1. nano-vllm 模块 → 可借鉴点（按优先级）

| nano-vllm 路径 | 作用摘要 | nano-pointllm 侧建议 |
|----------------|----------|----------------------|
| `nanovllm/engine/block_manager.py` | 块级 KV、prefix hash 复用 | **P3**：在 Llama 头维/层数与 **HF `past` 布局** 对齐后，移植或简化版 `BlockManager`；先单序列再多序列 |
| `nanovllm/engine/scheduler.py` | 调度、与 block 联动 | **P3**：decode 批量化时再接；初期可跳过 |
| `nanovllm/engine/sequence.py` | 序列元数据、block 表 | **P3**：与上同步 |
| `nanovllm/engine/model_runner.py` | 模型加载、KV 池、`capture_cudagraph` | **P2–P4**：抽象「**仅 decode**」的 `ModelRunner` 子集；**不要**照搬 `build_model` 的 Qwen3/Whisper 分支 |
| `nanovllm/layers/attention.py` | Triton `store_kvcache`、Flash 路径 | **P2（Llama 化之后）**：在 **Llama 头维 / GQA 与 HF 一致** 的前提下再迁；**勿**直接接 Qwen3 的 `Attention` 类到 Llama |
| `nanovllm/layers/linear.py` / `layernorm.py` / `activation.py` | 低层算子 | **P2**：与 HF 数值 parity 后再换 |
| `nanovllm/layers/rotary_embedding.py` | RoPE | **P2**：与 `transformers` 同版本 Llama 对齐 |
| `nanovllm/layers/sampler.py` | 采样 | **P4**：greedy 可先 `argmax`；采样后接 |
| `nanovllm/models/qwen3.py` 等 | Qwen 权重与结构 | **勿直接搬**；仅作「如何组织 forward」参考 |

---

## 2. 推荐阶段划分（与现有 M 里程碑衔接）

### P1 — KV 契约（已完成大半）

- 用 **`hybrid_greedy_decode` + `HFDecodeBackend`** 与 **`manual_greedy_decode` / `generate`** 做 **greedy parity**（你已跑通 PointLLM）。
- 明确 **HF `past_key_values` 每层 `(k, v)` 的 shape/dtype/device**，为后续「零拷贝接入」或「边界 convert」写单测。

### P2 — Llama 模块适配（**终验：PointLLM-7B + 单卡 CUDA**）

**P2a — 标准 Llama HF（可选加速器）**

- 新建 **`nanopointllm/llama/`**，从 nano-vllm 只拿 **与 Qwen 解耦、且与 Llama 同语义** 的组件（RoPE、RMSNorm、SwiGLU 等），**按 `transformers` Llama 的 shape 与公式** 重写或薄封装，避免继承 `Qwen3Attention`。
- **验收（可选）**：在 **HF `LlamaForCausalLM` + 单卡 CUDA** 上，`DecodeBackend` 与 **`HFDecodeBackend`** greedy 对齐（`parity_greedy_hf_llama.py` 或专用脚本）。**无点云**，便于拆 kernel 问题。

**P2b — PointLLM-7B，仅 decode（主路径）**

- **环境**：**单卡 CUDA** + **`PointLLM_7B_v1.2`**（与 `setup_inference.md` / 现有 bench 命令一致）。
- 在 **prefill 仍走 HF** 的前提下，**decode 步** 换用 P2a 的 **`DecodeBackend`**（`point_clouds=None`；Llama 主干与 7B 配置一致）。
- **验收（必做）**：`parity_hybrid_prefill_decode_pointllm.py` — **HF prefill + 新 decode 栈** vs 全 HF greedy。
- **bench（必做）**：`bench_m3_decode_pointllm.py` 注册新段（如 **`hf|llama_triton`**，名称待定），与 **`hf`** / **`hf|compile`** 对比 **`decode_mean_per_step_ms`**。
- **已加（P2b 挂点）**：根栈 **`hf_inner`**（`get_model()` / `model.model` + **`lm_head`**），与 **`hf`** 单步 logits 在 `verify_p2_pointllm_cuda.py` 中对齐；后续自定义层可替换 `HFInnerDecodeBackend` 内部实现而保持栈名。
- **已落地（P2 第一步）**：`scripts/verify_p2_pointllm_cuda.py` — 单卡上跑 **RoPE（HF 实现）自检**、**manual/hybrid greedy parity**、**`HFDecodeBackend` decode 微 bench**；`nanopointllm/llama/hf_config.py`、`rope_sanity.py` 供后续自定义层读配置与对齐 RoPE。
- **P2 本步（RMSNorm + SwiGLU MLP）**：`nanopointllm/llama/rmsnorm_ref.py`（`LlamaRMSNormRef`）、`mlp_ref.py`（`LlamaSwiGLUMLPRef`），与 HF `LlamaRMSNorm` / `LlamaMLP` 同公式与同 `state_dict` 键名；CPU 单测 `tests/test_llama_norm_mlp_parity.py`（`LlamaMLP` 构造在单测内按 `inspect` 兼容 **config 单参** 与 **三整数** 两版 API）。
- **P2 本步（Attention eager + 显式逐层 decode）**：`nanopointllm/llama/eager_attention_kv.py`（`repeat_kv` / `eager_attention_forward_ref`，与 HF `eager_attention_forward` 对齐）；`torch_llama_decode_backend.py`（`TorchLlamaExplicitLoopDecodeBackend`：复刻 ``LlamaModel.forward`` 的 embed / `create_causal_mask` / RoPE / **逐层** `LlamaDecoderLayer` / norm + `lm_head`）；decode 栈根 **`torch_loop`**（可 `torch_loop|compile`）；**legacy tuple** `past_key_values` 经 `normalize_past_key_values_for_torch_loop` → `DynamicCache.from_legacy_cache`；单测 `tests/test_eager_attention_kv.py`、`tests/test_torch_loop_past_normalize.py`；`verify_p2_pointllm_cuda.py` 增加 **HFInner vs torch_loop** 单步 logits。下一步：在 `eager_attention_forward_ref` 上接 Triton 或替换 `LlamaAttention` 内调用，再做 hybrid 全序列 parity。

### P2′（可选并行）— 目录与命名

- 推荐包路径：**`nanopointllm.llama`**（与 `pointllm.model.pointllm` 的 Llama 子类语义一致，便于检索）。

### P3 — Paged KV + 调度（nano-vllm 核心收益区）

- 移植 **`BlockManager` + `Sequence`** 思想；KV 物理块与 **HF  tuple cache** 之间二选一或做 **import/export**（有拷贝成本需 bench）。
- 多请求时再上 **`scheduler`**；单请求 PointLLM 可先只做 **固定块池 + 单序列** 验证延迟。

### P4 — PointLLM 产品化接入

- `PointLLM` 评测 **`--decode_engine`** 或环境变量切到 `nanopointllm` 新后端。
- **prefill** 仍 HF（或后续 **prefill 栈**）；与 `fork_nano_decode_plan.md` 一致。

---

## 3. 风险与核对清单

- **GQA / MQA**：Llama-2/3 与 Qwen3 头数不同，**不可**直接复用 Qwen 的 head 划分。
- **RMSNorm eps、SwiGLU 顺序**：与 `transformers` 当前 Llama 实现 **逐层对齐**再谈加速。
- **CUDA Graph**：自回归 KV 更新与 capture 冲突（你已在 `TorchCompileDecodeWrapper` 遇到）；nano-vllm 用 **slot + 静态 KV 池** 规避 —— 移植时必须 **整体采纳其 KV 契约**，不能半套 HF、半套 graph。
- **版权**：nano-vllm 衍生代码保留原仓库许可说明（见 nano-vllm 仓库 `LICENSE`）。

---

## 4. 与当前代码的挂钩

- **入口**：新引擎实现 **`DecodeBackend.decode_step`**，经 **`build_decode_stack("hf|your_backend")`** 或 **`hybrid_greedy_decode(..., decode_backend=...)`** 接入。
- **基线**：继续用 **`bench_m3_decode_pointllm.py`** 与 Phase1 的 **`decode_mean_per_step_ms`** 对比。

---

## 5. 小结

- **可以且应该**参考 nano-vllm 的 **块 KV、Triton attention、调度** 来拉高 nano-pointllm 的上限；当前 **增益有限** 是因为 **仍在 HF 路径上**，尚未执行 **P2–P3**。
- **对齐**指的是 **设计思想与模块边界** 对齐，而不是把 **`nanovllm` 包名或 Qwen3 模型** 并进 PointLLM 环境。
