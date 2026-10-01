# Multi-Batch Inference Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 nano-pointllm 上实现支持多请求并发的批量推理引擎，涵盖点云批量编码、批量 prefill、批量 decode（padded KV）和调度器，最终支持分页 KV cache。

**Architecture:** PointLLM 是 decoder-only 模型（LLaMA）+ prefill 阶段嵌入级点云融合，对应 nano-vllm 的 Qwen3-ASR 模式而非 Whisper。核心链路：`PointLLMWrapper.prepare_inputs_embeds`（批量编码点云并融合到 inputs_embeds）→ `Scheduler` 调度 → `PointLLMModelRunner`（批量 prefill + padded-KV 批量 decode）→ `PointLLMLLMEngine` 顶层 API。Tasks 1-5 形成完整可测试的多 batch 推理系统；Task 6-7 追加分页 KV。

**Tech Stack:** Python 3.10+, PyTorch ≥ 2.0, transformers ≥ 4.38（DynamicCache），HF PointLLM（lazy import，运行时依赖），hashlib（无额外依赖替代 xxhash），pytest

---

## 文件结构

| 文件 | 职责 |
|------|------|
| `nanopointllm/sampling_params.py` | **NEW** SamplingParams 数据类 |
| `nanopointllm/models/pointllm_wrapper.py` | **NEW** PointLLMWrapper：抽离 `prepare_inputs_embeds`，批量点云编码 |
| `nanopointllm/engine/sequence.py` | **NEW** PointLLMSequence 序列状态 |
| `nanopointllm/engine/scheduler.py` | **NEW** Scheduler：FIFO 调度，waiting/running 队列 |
| `nanopointllm/engine/model_runner.py` | **NEW** PointLLMModelRunner：批量 prefill + padded-KV 批量 decode |
| `nanopointllm/engine/llm_engine.py` | **NEW** PointLLMLLMEngine：顶层 API |
| `nanopointllm/engine/block_manager.py` | **NEW** BlockManager：分页 KV 元数据 + prefix hash 缓存 |
| `nanopointllm/engine/kv_cache.py` | **NEW** PagedKVCache：全局 KV Tensor 池 + slot 管理 |
| `tests/test_pointllm_wrapper.py` | **NEW** PointLLMWrapper 单测（mock 模型） |
| `tests/test_sequence.py` | **NEW** PointLLMSequence 单测 |
| `tests/test_scheduler.py` | **NEW** Scheduler 单测 |
| `tests/test_model_runner.py` | **NEW** ModelRunner 批量 prefill/decode 单测（mock 模型） |
| `tests/test_block_manager.py` | **NEW** BlockManager 单测 |

---

## Task 1: SamplingParams + PointLLMSequence

**Files:**
- Create: `nanopointllm/sampling_params.py`
- Create: `nanopointllm/engine/sequence.py`
- Test: `tests/test_sequence.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_sequence.py
import pytest
from nanopointllm.sampling_params import SamplingParams
from nanopointllm.engine.sequence import PointLLMSequence, SequenceStatus


def test_sampling_params_defaults():
    p = SamplingParams()
    assert p.temperature == 1.0
    assert p.max_tokens == 128
    assert p.ignore_eos is False


def test_sequence_init():
    seq = PointLLMSequence(token_ids=[1, 2, 3])
    assert seq.num_tokens == 3
    assert seq.num_prompt_tokens == 3
    assert seq.num_completion_tokens == 0
    assert seq.status == SequenceStatus.WAITING
    assert seq.last_token == 3


def test_sequence_append_token():
    seq = PointLLMSequence(token_ids=[1, 2, 3])
    seq.append_token(99)
    assert seq.num_tokens == 4
    assert seq.last_token == 99
    assert seq.num_completion_tokens == 1


def test_sequence_status_transition():
    seq = PointLLMSequence(token_ids=[1, 2, 3])
    assert not seq.is_finished
    seq.status = SequenceStatus.FINISHED
    assert seq.is_finished


def test_sequence_with_point_clouds():
    import torch
    pc = torch.randn(1024, 3)
    seq = PointLLMSequence(token_ids=[1, 2, 3], point_clouds=pc)
    assert seq.point_clouds is not None
    assert seq.inputs_embeds_cached is None  # not encoded yet
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/nano-pointllm && python -m pytest tests/test_sequence.py -v 2>&1 | head -30
```

Expected: `ModuleNotFoundError` or `ImportError` (files don't exist yet)

- [ ] **Step 3: Implement SamplingParams**

```python
# nanopointllm/sampling_params.py
from dataclasses import dataclass


@dataclass
class SamplingParams:
    temperature: float = 1.0
    max_tokens: int = 128
    ignore_eos: bool = False
```

- [ ] **Step 4: Implement PointLLMSequence**

```python
# nanopointllm/engine/sequence.py
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional

import torch

from nanopointllm.sampling_params import SamplingParams


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class PointLLMSequence:
    """单个推理请求的完整状态。"""

    def __init__(
        self,
        token_ids: list[int],
        point_clouds: Optional[Any] = None,
        sampling_params: Optional[SamplingParams] = None,
    ):
        self.token_ids: list[int] = list(token_ids)
        self.point_clouds = point_clouds          # 原始点云，prefill 时编码一次
        self.sampling_params = sampling_params or SamplingParams()

        # ── 引擎运行时填充 ──
        self.status = SequenceStatus.WAITING
        self.num_prompt_tokens: int = len(token_ids)

        # 点云编码后的 inputs_embeds（encode 一次后缓存，decode 阶段不再用点云）
        self.inputs_embeds_cached: Optional[torch.Tensor] = None

        # HF DynamicCache（prefill 后持有；batch decode 时临时合并再拆分）
        self.past_key_values: Optional[Any] = None

        # 分页 KV 元数据（Task 6 启用）
        self.block_table: list[int] = []
        self.num_cached_tokens: int = 0

    # ── 属性 ──

    @property
    def num_tokens(self) -> int:
        return len(self.token_ids)

    @property
    def num_completion_tokens(self) -> int:
        return self.num_tokens - self.num_prompt_tokens

    @property
    def last_token(self) -> int:
        return self.token_ids[-1]

    @property
    def is_finished(self) -> bool:
        return self.status == SequenceStatus.FINISHED

    @property
    def max_tokens(self) -> int:
        return self.sampling_params.max_tokens

    @property
    def temperature(self) -> float:
        return self.sampling_params.temperature

    @property
    def ignore_eos(self) -> bool:
        return self.sampling_params.ignore_eos

    # ── 突变 ──

    def append_token(self, token_id: int) -> None:
        self.token_ids.append(token_id)

    def kv_seq_len(self) -> int:
        """当前 KV cache 对应的序列长度（prefill 后 = prompt_len，每 decode 步 +1）。"""
        if self.past_key_values is None:
            return 0
        cache = self.past_key_values
        if hasattr(cache, "get_seq_length"):
            return int(cache.get_seq_length())
        # legacy tuple: each layer is (k, v) with shape [batch, heads, seq, dim]
        if isinstance(cache, (tuple, list)) and len(cache) > 0:
            k0 = cache[0][0]
            return int(k0.shape[-2])
        return 0
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd /home/nano-pointllm && python -m pytest tests/test_sequence.py -v
```

Expected: 5 passed

- [ ] **Step 6: Commit**

```bash
cd /home/nano-pointllm && git add nanopointllm/sampling_params.py nanopointllm/engine/sequence.py tests/test_sequence.py && git commit -m "feat(p3): add SamplingParams and PointLLMSequence state types

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 2: PointLLMWrapper — prepare_inputs_embeds 分离

**Files:**
- Create: `nanopointllm/models/pointllm_wrapper.py`
- Test: `tests/test_pointllm_wrapper.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_pointllm_wrapper.py
"""
使用 mock 模型测试 PointLLMWrapper，无需实际加载 PointLLM-7B 权重。
"""
import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch
from nanopointllm.models.pointllm_wrapper import PointLLMWrapper


def _make_mock_hf_model(vocab_size=32, hidden_size=64, num_patch_tokens=4):
    """构造一个最小化的 mock PointLLMLlamaForCausalLM。"""
    PATCH_TOKEN_ID = 30

    # inner model（模拟 PointLLMLlamaModel）
    inner = MagicMock()
    inner.embed_tokens = nn.Embedding(vocab_size, hidden_size)
    inner.point_backbone_config = {
        "point_patch_token": PATCH_TOKEN_ID,
        "point_token_len": num_patch_tokens,
        "mm_use_point_start_end": False,
        "backbone_output_dim": 32,
        "project_output_dim": hidden_size,
    }

    # point_backbone: 返回 [batch, num_patch_tokens, backbone_dim]
    def _backbone(point_clouds):
        B = point_clouds.shape[0] if point_clouds.dim() == 3 else 1
        return torch.zeros(B, num_patch_tokens, 32)

    inner.point_backbone = _backbone
    inner.point_proj = nn.Linear(32, hidden_size)

    # hf_model
    hf_model = MagicMock()
    hf_model.model = inner
    hf_model.get_model = MagicMock(return_value=inner)
    hf_model.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    return hf_model, PATCH_TOKEN_ID, num_patch_tokens, hidden_size


def test_wrapper_construction():
    hf_model, *_ = _make_mock_hf_model()
    wrapper = PointLLMWrapper(hf_model)
    assert wrapper.inner is hf_model.model


def test_encode_point_clouds_list():
    hf_model, PATCH_TOKEN_ID, num_patch_tokens, hidden_size = _make_mock_hf_model()
    wrapper = PointLLMWrapper(hf_model)

    pc1 = torch.randn(512, 3)
    pc2 = torch.randn(512, 3)
    features = wrapper.encode_point_clouds([pc1, pc2])
    assert len(features) == 2
    for f in features:
        assert f.shape == (num_patch_tokens, hidden_size)


def test_encode_point_clouds_batch_tensor():
    hf_model, PATCH_TOKEN_ID, num_patch_tokens, hidden_size = _make_mock_hf_model()
    wrapper = PointLLMWrapper(hf_model)

    pcs = torch.randn(3, 512, 3)  # batch of 3
    features = wrapper.encode_point_clouds(pcs)
    assert len(features) == 3
    for f in features:
        assert f.shape == (num_patch_tokens, hidden_size)


def test_prepare_inputs_embeds_fuses_patch_tokens():
    hf_model, PATCH_TOKEN_ID, num_patch_tokens, hidden_size = _make_mock_hf_model()
    wrapper = PointLLMWrapper(hf_model)

    # input_ids: [BOS, PATCH*4, EOS] per sample, batch=2
    L = 1 + num_patch_tokens + 1
    input_ids = torch.tensor([
        [1] + [PATCH_TOKEN_ID] * num_patch_tokens + [2],
        [1] + [PATCH_TOKEN_ID] * num_patch_tokens + [2],
    ])
    point_features = [torch.randn(num_patch_tokens, hidden_size) for _ in range(2)]
    embeds = wrapper.prepare_inputs_embeds(input_ids, point_features)
    assert embeds.shape == (2, L, hidden_size)


def test_prepare_inputs_embeds_no_point_clouds():
    """无点云时退化为纯文本嵌入。"""
    hf_model, PATCH_TOKEN_ID, num_patch_tokens, hidden_size = _make_mock_hf_model()
    wrapper = PointLLMWrapper(hf_model)

    input_ids = torch.tensor([[1, 5, 6, 2]])
    embeds = wrapper.prepare_inputs_embeds(input_ids, point_features=None)
    assert embeds.shape == (1, 4, hidden_size)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/nano-pointllm && python -m pytest tests/test_pointllm_wrapper.py -v 2>&1 | head -20
```

Expected: `ImportError` — `pointllm_wrapper` not found

- [ ] **Step 3: Implement PointLLMWrapper**

```python
# nanopointllm/models/pointllm_wrapper.py
"""
PointLLMWrapper：把 PointLLMLlamaForCausalLM 包装成
  * prepare_inputs_embeds(input_ids, point_features) → inputs_embeds
  * encode_point_clouds(point_clouds) → list[Tensor]
两步接口，便于批量推理引擎分阶段调用（点云编码 batch 化 + LLM decode 解耦）。

PointLLM 的点云融合逻辑原始位置：
  /home/PointLLM/pointllm/model/pointllm.py  PointLLMLlamaModel.forward L112-L171
"""
from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn


class PointLLMWrapper:
    """
    轻量封装层；hf_model 须为 PointLLMLlamaForCausalLM（或 duck-type 兼容对象）。
    不改变 hf_model 的权重或状态。
    """

    def __init__(self, hf_model: nn.Module) -> None:
        self.hf_model = hf_model

    @property
    def inner(self) -> nn.Module:
        """PointLLMLlamaModel（backbone）。"""
        if hasattr(self.hf_model, "get_model") and self.hf_model.get_model() is not None:
            return self.hf_model.get_model()
        return self.hf_model.model

    @property
    def lm_head(self) -> nn.Module:
        return self.hf_model.lm_head

    def get_input_embeddings(self) -> nn.Embedding:
        return self.inner.embed_tokens

    # ──────────────────────────────────────────────────────────────
    # 点云编码
    # ──────────────────────────────────────────────────────────────

    @torch.inference_mode()
    def encode_point_clouds(
        self,
        point_clouds: Any,
    ) -> list[torch.Tensor]:
        """
        批量编码点云，返回投影后的 patch token 特征列表（每个元素对应一个样本）。

        Args:
            point_clouds: Tensor [B, N, C] 或 list of Tensor [N, C]

        Returns:
            list of Tensor, 每个 shape = (point_token_len, hidden_size)
        """
        inner = self.inner
        backbone = inner.point_backbone
        proj = inner.point_proj

        if isinstance(point_clouds, list):
            raw_features = []
            for pc in point_clouds:
                pc_3d = pc.unsqueeze(0) if pc.dim() == 2 else pc  # [1, N, C]
                feat = backbone(pc_3d)                              # [1, token_len, backbone_dim]
                raw_features.append(feat[0])                        # [token_len, backbone_dim]
        else:
            # Tensor [B, N, C] 或 [N, C]
            if point_clouds.dim() == 2:
                point_clouds = point_clouds.unsqueeze(0)
            raw = backbone(point_clouds)                           # [B, token_len, backbone_dim]
            raw_features = [raw[i] for i in range(raw.shape[0])]

        projected = [proj(f) for f in raw_features]              # list of [token_len, hidden_size]
        return projected

    # ──────────────────────────────────────────────────────────────
    # inputs_embeds 融合
    # ──────────────────────────────────────────────────────────────

    def prepare_inputs_embeds(
        self,
        input_ids: torch.Tensor,
        point_features: Optional[list[torch.Tensor]],
    ) -> torch.Tensor:
        """
        将文本 token embeddings 与点云 patch features 融合，返回 inputs_embeds。

        逻辑来自 PointLLMLlamaModel.forward（L112-L171），但可独立调用：
          - 无点云时退化为纯文本嵌入（decode 阶段调用路径）。
          - 有点云时用 point_features 替换 input_ids 中的 patch 占位 token。

        Args:
            input_ids: [B, L]  LongTensor
            point_features: list of [point_token_len, hidden_size]，长度=B；
                            或 None（decode 步 / 纯文本样本）

        Returns:
            inputs_embeds: [B, L, hidden_size]
        """
        inner = self.inner
        inputs_embeds = inner.embed_tokens(input_ids)   # [B, L, H]

        if point_features is None:
            return inputs_embeds

        cfg = inner.point_backbone_config
        use_start_end: bool = cfg.get("mm_use_point_start_end", False)
        patch_token_id: int = cfg["point_patch_token"]

        new_embeds = []
        for i, (cur_ids, cur_emb) in enumerate(zip(input_ids, inputs_embeds)):
            cur_feat = point_features[i].to(device=cur_emb.device, dtype=cur_emb.dtype)
            num_patches = cur_feat.shape[0]

            if (cur_ids == patch_token_id).sum() == 0:
                # 纯文本样本：无点云占位符，直接沿用文本嵌入
                new_embeds.append(cur_emb)
                continue

            if use_start_end:
                start_token_id: int = cfg["point_start_token"]
                end_token_id: int = cfg["point_end_token"]
                start_positions = torch.where(cur_ids == start_token_id)[0]
                # 每个 point_start_token 后跟 num_patches 个 patch token 再跟 point_end_token
                cur_new_emb = cur_emb
                for pos in reversed(start_positions):  # 从后往前替换避免偏移
                    cur_new_emb = torch.cat([
                        cur_new_emb[:pos + 1],
                        cur_feat,
                        cur_new_emb[pos + num_patches + 1:],
                    ])
                new_embeds.append(cur_new_emb)
            else:
                # 连续 patch token 直接替换
                masked_indices = torch.where(cur_ids == patch_token_id)[0]
                start_idx = masked_indices[0]
                new_embeds.append(torch.cat([
                    cur_emb[:start_idx],
                    cur_feat,
                    cur_emb[start_idx + num_patches:],
                ]))

        return torch.stack(new_embeds)   # [B, L, H]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /home/nano-pointllm && python -m pytest tests/test_pointllm_wrapper.py -v
```

Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
cd /home/nano-pointllm && git add nanopointllm/models/pointllm_wrapper.py tests/test_pointllm_wrapper.py && git commit -m "feat(p3): add PointLLMWrapper with prepare_inputs_embeds and batch point cloud encoding

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 3: Scheduler — FIFO 请求调度

**Files:**
- Create: `nanopointllm/engine/scheduler.py`
- Test: `tests/test_scheduler.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_scheduler.py
import pytest
from nanopointllm.engine.sequence import PointLLMSequence, SequenceStatus
from nanopointllm.engine.scheduler import Scheduler
from nanopointllm.sampling_params import SamplingParams


def _seq(length: int, max_tokens: int = 10) -> PointLLMSequence:
    return PointLLMSequence(
        token_ids=list(range(length)),
        sampling_params=SamplingParams(max_tokens=max_tokens),
    )


def test_empty_scheduler_is_finished():
    s = Scheduler(max_num_seqs=4, max_num_batched_tokens=512, eos_token_id=2)
    assert s.is_finished()


def test_add_and_schedule_prefill():
    s = Scheduler(max_num_seqs=4, max_num_batched_tokens=512, eos_token_id=2)
    seq = _seq(10)
    s.add(seq)
    seqs, is_prefill = s.schedule()
    assert is_prefill is True
    assert seq in seqs
    assert seq.status == SequenceStatus.RUNNING


def test_schedule_decode_after_prefill():
    s = Scheduler(max_num_seqs=4, max_num_batched_tokens=512, eos_token_id=2)
    seq = _seq(10)
    s.add(seq)
    s.schedule()                        # prefill moves seq to running
    seqs, is_prefill = s.schedule()     # decode: no new requests
    assert is_prefill is False
    assert seq in seqs


def test_max_num_seqs_respected():
    s = Scheduler(max_num_seqs=2, max_num_batched_tokens=512, eos_token_id=2)
    for _ in range(5):
        s.add(_seq(10))
    seqs, is_prefill = s.schedule()
    assert is_prefill is True
    assert len(seqs) <= 2


def test_max_num_batched_tokens_respected():
    s = Scheduler(max_num_seqs=8, max_num_batched_tokens=20, eos_token_id=2)
    s.add(_seq(15))
    s.add(_seq(15))     # total=30 > 20; only first fits
    seqs, _ = s.schedule()
    assert len(seqs) == 1


def test_postprocess_eos_finishes_sequence():
    s = Scheduler(max_num_seqs=4, max_num_batched_tokens=512, eos_token_id=2)
    seq = _seq(5, max_tokens=100)
    s.add(seq)
    seqs, _ = s.schedule()
    s.postprocess(seqs, [2])            # EOS token
    assert seq.is_finished
    assert s.is_finished()


def test_postprocess_max_tokens_finishes_sequence():
    s = Scheduler(max_num_seqs=4, max_num_batched_tokens=512, eos_token_id=2)
    seq = _seq(5, max_tokens=1)
    s.add(seq)
    seqs, _ = s.schedule()
    s.postprocess(seqs, [99])           # non-EOS but max_tokens reached
    assert seq.is_finished


def test_postprocess_appends_token():
    s = Scheduler(max_num_seqs=4, max_num_batched_tokens=512, eos_token_id=2)
    seq = _seq(5, max_tokens=10)
    s.add(seq)
    seqs, _ = s.schedule()
    s.postprocess(seqs, [77])
    assert seq.last_token == 77
    assert not seq.is_finished
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/nano-pointllm && python -m pytest tests/test_scheduler.py -v 2>&1 | head -20
```

Expected: `ImportError` — scheduler not found

- [ ] **Step 3: Implement Scheduler**

```python
# nanopointllm/engine/scheduler.py
"""
FIFO 两队列调度器：prefill 优先，prefill 队列空时切 decode。
不依赖分页 KV（Task 6 之前用 per-sequence DynamicCache）。
"""
from __future__ import annotations

from collections import deque

from nanopointllm.engine.sequence import PointLLMSequence, SequenceStatus


class Scheduler:

    def __init__(
        self,
        max_num_seqs: int,
        max_num_batched_tokens: int,
        eos_token_id: int,
    ) -> None:
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.eos_token_id = eos_token_id
        self.waiting: deque[PointLLMSequence] = deque()
        self.running: deque[PointLLMSequence] = deque()

    def is_finished(self) -> bool:
        return not self.waiting and not self.running

    def add(self, seq: PointLLMSequence) -> None:
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[PointLLMSequence], bool]:
        """
        返回 (scheduled_seqs, is_prefill)。
        is_prefill=True  → 这批序列需要走 prefill（含点云编码）。
        is_prefill=False → 这批序列走 decode 步。
        """
        # ── Prefill 阶段：从 waiting 拉入新请求 ──
        prefill_seqs: list[PointLLMSequence] = []
        total_tokens = 0
        while self.waiting and len(prefill_seqs) < self.max_num_seqs:
            seq = self.waiting[0]
            if total_tokens + seq.num_tokens > self.max_num_batched_tokens:
                break
            total_tokens += seq.num_tokens
            self.waiting.popleft()
            seq.status = SequenceStatus.RUNNING
            self.running.append(seq)
            prefill_seqs.append(seq)

        if prefill_seqs:
            return prefill_seqs, True

        # ── Decode 阶段：处理所有 running 序列 ──
        assert self.running, "schedule() called on empty engine"
        decode_seqs = list(self.running)
        return decode_seqs, False

    def postprocess(
        self,
        seqs: list[PointLLMSequence],
        token_ids: list[int],
    ) -> None:
        """将新 token 追加到序列并检查终止条件。"""
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)
            finished = (
                (not seq.ignore_eos and token_id == self.eos_token_id)
                or seq.num_completion_tokens >= seq.max_tokens
            )
            if finished:
                seq.status = SequenceStatus.FINISHED
                self.running.remove(seq)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /home/nano-pointllm && python -m pytest tests/test_scheduler.py -v
```

Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
cd /home/nano-pointllm && git add nanopointllm/engine/scheduler.py tests/test_scheduler.py && git commit -m "feat(p3): add Scheduler with FIFO prefill/decode queue management

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 4: PointLLMModelRunner — 批量 prefill + padded-KV 批量 decode

**Files:**
- Create: `nanopointllm/engine/model_runner.py`
- Test: `tests/test_model_runner.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_model_runner.py
"""
用 mock 模型测试 ModelRunner 的批量 prefill 和批量 decode 逻辑，
无需加载 PointLLM-7B 权重。
"""
import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch

from nanopointllm.engine.sequence import PointLLMSequence, SequenceStatus
from nanopointllm.engine.model_runner import PointLLMModelRunner
from nanopointllm.models.pointllm_wrapper import PointLLMWrapper
from nanopointllm.sampling_params import SamplingParams


# ────── mock helpers ──────

VOCAB = 64
HIDDEN = 32
PATCH_TOKEN_ID = 55
NUM_PATCH = 4
NUM_LAYERS = 2
NUM_HEADS = 4
HEAD_DIM = HIDDEN // NUM_HEADS


def _make_mock_hf_model():
    """返回能跑 forward 的最小 mock PointLLMLlamaForCausalLM。"""
    inner = MagicMock()
    inner.embed_tokens = nn.Embedding(VOCAB, HIDDEN)
    inner.point_backbone_config = {
        "point_patch_token": PATCH_TOKEN_ID,
        "point_token_len": NUM_PATCH,
        "mm_use_point_start_end": False,
        "backbone_output_dim": 16,
        "project_output_dim": HIDDEN,
    }

    def _backbone(pcs):
        B = pcs.shape[0]
        return torch.zeros(B, NUM_PATCH, 16)

    inner.point_backbone = _backbone
    inner.point_proj = nn.Linear(16, HIDDEN)

    hf_model = MagicMock()
    hf_model.model = inner
    hf_model.get_model = MagicMock(return_value=inner)
    hf_model.lm_head = nn.Linear(HIDDEN, VOCAB, bias=False)
    return hf_model


def _build_runner():
    hf_model = _make_mock_hf_model()
    wrapper = PointLLMWrapper(hf_model)
    runner = PointLLMModelRunner(hf_model=hf_model)
    return runner


def _seq_with_point_cloud(prompt_len: int = 8) -> PointLLMSequence:
    token_ids = [1] + [PATCH_TOKEN_ID] * NUM_PATCH + [2] * (prompt_len - NUM_PATCH - 1)
    seq = PointLLMSequence(
        token_ids=token_ids,
        point_clouds=torch.randn(512, 3),
        sampling_params=SamplingParams(max_tokens=5),
    )
    return seq


def test_runner_construction():
    runner = _build_runner()
    assert runner.wrapper is not None


def test_encode_point_clouds_fills_cache(monkeypatch):
    runner = _build_runner()
    seq = _seq_with_point_cloud()
    assert seq.inputs_embeds_cached is None
    runner._encode_point_clouds_batch([seq])
    assert seq.inputs_embeds_cached is not None
    assert seq.inputs_embeds_cached.shape[-1] == HIDDEN


def test_encode_point_clouds_skips_already_cached(monkeypatch):
    runner = _build_runner()
    seq = _seq_with_point_cloud()
    dummy = torch.zeros(1, 10, HIDDEN)
    seq.inputs_embeds_cached = dummy
    runner._encode_point_clouds_batch([seq])
    # 应保持不变（不重新编码）
    assert seq.inputs_embeds_cached is dummy


def test_run_prefill_sets_past_key_values(monkeypatch):
    """prefill 后每个 seq 应有 past_key_values。"""
    runner = _build_runner()

    # 用一个返回 fake CausalLMOutputWithPast 的 mock 替换 hf_prefill
    from types import SimpleNamespace
    from transformers.cache_utils import DynamicCache

    def fake_prefill(model, input_ids, attention_mask, point_clouds=None, **kw):
        B, L = input_ids.shape
        logits = torch.randn(B, 1, VOCAB)
        cache = DynamicCache()
        # 注入假 KV 数据（2 层，每层 k/v shape [B, NUM_HEADS, L, HEAD_DIM]）
        for _ in range(NUM_LAYERS):
            k = torch.zeros(B, NUM_HEADS, L, HEAD_DIM)
            v = torch.zeros(B, NUM_HEADS, L, HEAD_DIM)
            cache.key_cache.append(k)
            cache.value_cache.append(v)
        from nanopointllm.engine.types import PrefillOutput
        return PrefillOutput(logits=logits, past_key_values=cache, prompt_len=L)

    monkeypatch.setattr("nanopointllm.engine.model_runner.hf_prefill", fake_prefill)

    seqs = [_seq_with_point_cloud(8), _seq_with_point_cloud(8)]
    tokens = runner.run_prefill(seqs)
    assert len(tokens) == 2
    for seq in seqs:
        assert seq.past_key_values is not None
        assert seq.kv_seq_len() > 0


def test_run_decode_returns_next_tokens(monkeypatch):
    """decode 步应为每个 seq 返回一个 token。"""
    from transformers.cache_utils import DynamicCache

    runner = _build_runner()

    # 先填好 past_key_values（模拟 prefill 后状态）
    seqs = [_seq_with_point_cloud(8), _seq_with_point_cloud(8)]
    for seq in seqs:
        cache = DynamicCache()
        for _ in range(NUM_LAYERS):
            cache.key_cache.append(torch.zeros(1, NUM_HEADS, 8, HEAD_DIM))
            cache.value_cache.append(torch.zeros(1, NUM_HEADS, 8, HEAD_DIM))
        seq.past_key_values = cache

    # mock decode_step
    call_count = [0]
    orig_logits = torch.zeros(len(seqs), 1, VOCAB)
    orig_logits[0, 0, 10] = 100.0  # seq0 → token 10
    orig_logits[1, 0, 20] = 100.0  # seq1 → token 20

    def fake_batch_decode(model, input_ids, attention_mask, past_key_values, **kw):
        call_count[0] += 1
        from types import SimpleNamespace
        # 返回 fake CausalLMOutput with updated cache
        new_cache = DynamicCache()
        for layer in range(NUM_LAYERS):
            new_cache.key_cache.append(torch.zeros(len(seqs), NUM_HEADS, 9, HEAD_DIM))
            new_cache.value_cache.append(torch.zeros(len(seqs), NUM_HEADS, 9, HEAD_DIM))
        return SimpleNamespace(
            logits=orig_logits,
            past_key_values=new_cache,
        )

    monkeypatch.setattr("nanopointllm.engine.model_runner._batch_hf_decode", fake_batch_decode)

    tokens = runner.run_decode(seqs)
    assert tokens == [10, 20]
    assert call_count[0] == 1          # single batched forward
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/nano-pointllm && python -m pytest tests/test_model_runner.py -v 2>&1 | head -20
```

Expected: `ImportError`

- [ ] **Step 3: Implement PointLLMModelRunner**

```python
# nanopointllm/engine/model_runner.py
"""
PointLLMModelRunner：批量 prefill（点云编码 + HF forward）+ padded-KV 批量 decode。

批量 decode 策略（无分页 KV）：
  1. 取所有 running 序列的 KV cache，left-pad 到统一长度。
  2. 一次 hf forward 得到 [B, 1, vocab] logits 与更新后的 KV。
  3. 把新 KV 按序列维切回各自独立 cache。

分页 KV（Task 6）上线后，本文件里的 padded-KV 路径将被替换为 slot/block_table 路径。
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from nanopointllm.engine.hf_hybrid import hf_prefill
from nanopointllm.engine.sequence import PointLLMSequence
from nanopointllm.models.pointllm_wrapper import PointLLMWrapper


# ──────────────────────────────────────────────────────────────────────────────
# 内部工具：padded KV 合并 / 拆分
# ──────────────────────────────────────────────────────────────────────────────

def _build_batched_cache(seqs: list[PointLLMSequence]):
    """
    把多个序列的 DynamicCache（各自 kv_len 可不同）left-pad 成 [B, heads, max_kv_len, dim]
    并返回合并后的 DynamicCache 与 attention_mask。

    Left-pad 使实际 token 靠右，配合 causal mask 可正确 attend。
    """
    from transformers.cache_utils import DynamicCache

    kv_lens = [seq.kv_seq_len() for seq in seqs]
    max_kv_len = max(kv_lens)
    num_layers = len(seqs[0].past_key_values.key_cache)
    B = len(seqs)

    batched_cache = DynamicCache()
    for layer_idx in range(num_layers):
        k_list, v_list = [], []
        for seq in seqs:
            k = seq.past_key_values.key_cache[layer_idx]   # [1, H, seq_len, D]
            v = seq.past_key_values.value_cache[layer_idx]
            seq_len = k.shape[-2]
            pad_len = max_kv_len - seq_len
            if pad_len > 0:
                pad_k = k.new_zeros(1, k.shape[1], pad_len, k.shape[-1])
                pad_v = v.new_zeros(1, v.shape[1], pad_len, v.shape[-1])
                k = torch.cat([pad_k, k], dim=-2)
                v = torch.cat([pad_v, v], dim=-2)
            k_list.append(k)
            v_list.append(v)
        batched_cache.key_cache.append(torch.cat(k_list, dim=0))   # [B, H, max_kv_len, D]
        batched_cache.value_cache.append(torch.cat(v_list, dim=0))

    # attention_mask: 1 = valid，0 = pad；形状 [B, max_kv_len + 1]（+1 for new token）
    mask_rows = []
    for kv_len in kv_lens:
        row = torch.zeros(max_kv_len + 1, dtype=torch.long)
        row[-(kv_len + 1):] = 1   # 最后 kv_len+1 个位置有效（含新 token 本身）
        mask_rows.append(row)
    attention_mask = torch.stack(mask_rows)                        # [B, max_kv_len+1]

    return batched_cache, attention_mask, max_kv_len


def _split_batched_cache(
    batched_cache,
    seqs: list[PointLLMSequence],
    orig_kv_lens: list[int],
):
    """
    从 [B, heads, max_kv_len+1, dim] 的合并 cache 中切出每个序列更新后的 slice。
    """
    from transformers.cache_utils import DynamicCache

    num_layers = len(batched_cache.key_cache)
    for i, (seq, kv_len) in enumerate(zip(seqs, orig_kv_lens)):
        new_cache = DynamicCache()
        for layer_idx in range(num_layers):
            # 新序列长度 = 原 kv_len + 1（新 token 写入 KV）
            k = batched_cache.key_cache[layer_idx][i:i+1, :, -(kv_len+1):, :]
            v = batched_cache.value_cache[layer_idx][i:i+1, :, -(kv_len+1):, :]
            new_cache.key_cache.append(k.contiguous())
            new_cache.value_cache.append(v.contiguous())
        seq.past_key_values = new_cache


def _batch_hf_decode(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    past_key_values,
    **forward_kwargs,
):
    """单次 HF forward（decode 模式，无点云）。供测试 monkeypatch。"""
    return model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        use_cache=True,
        return_dict=True,
        point_clouds=None,
        **{k: v for k, v in forward_kwargs.items()
           if k in _batch_hf_decode_allowed_kwargs(model)},
    )


def _batch_hf_decode_allowed_kwargs(model: nn.Module) -> set[str]:
    import inspect
    return set(inspect.signature(model.forward).parameters.keys())


# ──────────────────────────────────────────────────────────────────────────────
# ModelRunner
# ──────────────────────────────────────────────────────────────────────────────

class PointLLMModelRunner:

    def __init__(self, hf_model: nn.Module) -> None:
        self.hf_model = hf_model
        self.wrapper = PointLLMWrapper(hf_model)

    # ── 点云批量编码 ──

    def _encode_point_clouds_batch(self, seqs: list[PointLLMSequence]) -> None:
        """
        对所有尚未编码的序列批量调用 PointBERT + projection，
        将结果（inputs_embeds）缓存进 seq.inputs_embeds_cached。
        """
        pending = [
            seq for seq in seqs
            if seq.point_clouds is not None and seq.inputs_embeds_cached is None
        ]
        if not pending:
            return

        for seq in pending:
            # 逐个编码（point cloud 尺寸可变；批量拼接可在尺寸统一时优化）
            features = self.wrapper.encode_point_clouds(seq.point_clouds)
            input_ids = torch.tensor([seq.token_ids], dtype=torch.long)
            if input_ids.device != next(self.wrapper.inner.parameters()).device:
                input_ids = input_ids.to(next(self.wrapper.inner.parameters()).device)
            embeds = self.wrapper.prepare_inputs_embeds(input_ids, features)  # [1, L, H]
            seq.inputs_embeds_cached = embeds.squeeze(0)   # [L, H]

    # ── Prefill ──

    @torch.inference_mode()
    def run_prefill(self, seqs: list[PointLLMSequence]) -> list[int]:
        """
        批量 prefill：
          1. 批量点云编码（填 seq.inputs_embeds_cached）
          2. pad 到统一长度，组成 inputs_embeds [B, max_L, H]
          3. HF forward（use_cache=True），每个 seq 拿到自己的 KV cache
          4. greedy 采样首个生成 token

        Returns:
            list[int]: 每个 seq 的第一个生成 token id
        """
        self._encode_point_clouds_batch(seqs)

        # ── 组 batch（left-pad） ──
        max_len = max(seq.num_tokens for seq in seqs)
        B = len(seqs)

        embeds_list = []
        attn_masks = []
        for seq in seqs:
            L = seq.num_tokens
            if seq.inputs_embeds_cached is not None:
                emb = seq.inputs_embeds_cached                            # [L, H]
            else:
                ids = torch.tensor([seq.token_ids], dtype=torch.long)
                emb = self.wrapper.get_input_embeddings()(ids).squeeze(0) # [L, H]

            pad_len = max_len - L
            if pad_len > 0:
                device = emb.device
                pad = emb.new_zeros(pad_len, emb.shape[-1])
                emb = torch.cat([pad, emb], dim=0)
                mask = torch.cat([torch.zeros(pad_len, dtype=torch.long, device=device),
                                   torch.ones(L, dtype=torch.long, device=device)])
            else:
                mask = torch.ones(L, dtype=torch.long, device=emb.device)

            embeds_list.append(emb)
            attn_masks.append(mask)

        inputs_embeds = torch.stack(embeds_list)     # [B, max_len, H]
        attention_mask = torch.stack(attn_masks)     # [B, max_len]

        # HF prefill（input_ids=None，传 inputs_embeds）
        # 用假 input_ids（全零）占位，point_clouds=None（已融入 embeds）
        dummy_ids = torch.zeros(B, max_len, dtype=torch.long, device=inputs_embeds.device)
        prefill_out = hf_prefill(
            self.hf_model,
            dummy_ids,
            attention_mask,
            point_clouds=None,       # 已融入 inputs_embeds，不再传原始点云
            inputs_embeds=inputs_embeds,
        )

        # 分配 per-sequence KV cache（从 batch cache 切片）
        _split_batched_cache(
            prefill_out.past_key_values,
            seqs,
            orig_kv_lens=[0] * B,    # prefill 前 kv_len 均为 0
        )

        # Greedy 采样
        logits = prefill_out.logits              # [B, 1, V]
        next_tokens = logits[:, -1, :].float().argmax(dim=-1).tolist()  # list[int]
        return next_tokens

    # ── Decode ──

    @torch.inference_mode()
    def run_decode(self, seqs: list[PointLLMSequence]) -> list[int]:
        """
        批量 decode 步（padded KV 合并策略）：
          1. 合并所有 seq 的 KV cache（left-pad 至 max_kv_len）
          2. 拼成 [B, 1] input_ids（每 seq 最后一个 token）
          3. 单次 HF forward
          4. 拆分 KV cache 回 per-seq，greedy 采样

        Returns:
            list[int]: 每个 seq 的新生成 token id
        """
        B = len(seqs)
        orig_kv_lens = [seq.kv_seq_len() for seq in seqs]

        batched_cache, attention_mask, max_kv_len = _build_batched_cache(seqs)

        input_ids = torch.tensor(
            [[seq.last_token] for seq in seqs],
            dtype=torch.long,
            device=batched_cache.key_cache[0].device,
        )   # [B, 1]

        attention_mask = attention_mask.to(device=input_ids.device)

        out = _batch_hf_decode(
            self.hf_model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=batched_cache,
        )

        # 更新 per-seq KV cache
        _split_batched_cache(out.past_key_values, seqs, orig_kv_lens)

        # Greedy 采样
        logits = out.logits          # [B, 1, V] 或 [B, V]
        if logits.dim() == 3:
            logits = logits[:, -1, :]
        next_tokens = logits.float().argmax(dim=-1).tolist()
        return next_tokens

    # ── 统一入口 ──

    def run(self, seqs: list[PointLLMSequence], is_prefill: bool) -> list[int]:
        if is_prefill:
            return self.run_prefill(seqs)
        return self.run_decode(seqs)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /home/nano-pointllm && python -m pytest tests/test_model_runner.py -v
```

Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
cd /home/nano-pointllm && git add nanopointllm/engine/model_runner.py tests/test_model_runner.py && git commit -m "feat(p3): add PointLLMModelRunner with batch prefill and padded-KV batch decode

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 5: PointLLMLLMEngine — 顶层推理 API

**Files:**
- Create: `nanopointllm/engine/llm_engine.py`
- Test: `tests/test_llm_engine.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_llm_engine.py
"""
PointLLMLLMEngine 集成测试（mock 模型，不加载权重）。
验证完整的 add_request → step → 生成完成 生命周期。
"""
import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch, call
from transformers.cache_utils import DynamicCache

from nanopointllm.engine.llm_engine import PointLLMLLMEngine
from nanopointllm.engine.sequence import SequenceStatus
from nanopointllm.sampling_params import SamplingParams

VOCAB = 64
HIDDEN = 32
NUM_PATCH = 4
PATCH_TOKEN_ID = 55
NUM_LAYERS = 2
NUM_HEADS = 4
HEAD_DIM = HIDDEN // NUM_HEADS


def _make_mock_hf_model():
    inner = MagicMock()
    inner.embed_tokens = nn.Embedding(VOCAB, HIDDEN)
    inner.point_backbone_config = {
        "point_patch_token": PATCH_TOKEN_ID,
        "point_token_len": NUM_PATCH,
        "mm_use_point_start_end": False,
        "backbone_output_dim": 16,
        "project_output_dim": HIDDEN,
    }

    def _backbone(pcs):
        B = pcs.shape[0]
        return torch.zeros(B, NUM_PATCH, 16)

    inner.point_backbone = _backbone
    inner.point_proj = nn.Linear(16, HIDDEN)

    hf_model = MagicMock()
    hf_model.model = inner
    hf_model.get_model = MagicMock(return_value=inner)
    hf_model.lm_head = nn.Linear(HIDDEN, VOCAB, bias=False)
    return hf_model


def _make_engine() -> PointLLMLLMEngine:
    hf_model = _make_mock_hf_model()
    return PointLLMLLMEngine(
        hf_model=hf_model,
        eos_token_id=2,
        max_num_seqs=4,
        max_num_batched_tokens=256,
    )


def _fake_run_prefill(seqs, _orig=None):
    """fake prefill: 给 seq 挂上 cache 并返回 token=5。"""
    for seq in seqs:
        cache = DynamicCache()
        for _ in range(NUM_LAYERS):
            L = seq.num_tokens
            cache.key_cache.append(torch.zeros(1, NUM_HEADS, L, HEAD_DIM))
            cache.value_cache.append(torch.zeros(1, NUM_HEADS, L, HEAD_DIM))
        seq.past_key_values = cache
    return [5] * len(seqs)


def _fake_run_decode(seqs, step_counter=[0]):
    """fake decode: 第一步 token=9，第二步 token=EOS(2)。"""
    step_counter[0] += 1
    if step_counter[0] == 1:
        return [9] * len(seqs)
    return [2] * len(seqs)   # EOS


def test_engine_construction():
    engine = _make_engine()
    assert engine.scheduler is not None
    assert engine.runner is not None


def test_engine_is_finished_when_empty():
    engine = _make_engine()
    assert engine.is_finished()


def test_engine_full_lifecycle(monkeypatch):
    """add_request → step loop → all seqs finished。"""
    step_counter = [0]

    def fake_run(seqs, is_prefill):
        if is_prefill:
            return _fake_run_prefill(seqs)
        step_counter[0] += 1
        if step_counter[0] == 1:
            return [9] * len(seqs)
        return [2] * len(seqs)  # EOS

    engine = _make_engine()
    monkeypatch.setattr(engine.runner, "run", fake_run)

    tokens = [1] + [PATCH_TOKEN_ID] * NUM_PATCH + [3]
    seq1 = engine.add_request(token_ids=tokens, point_clouds=torch.randn(512, 3),
                               sampling_params=SamplingParams(max_tokens=10))
    seq2 = engine.add_request(token_ids=tokens,
                               sampling_params=SamplingParams(max_tokens=10))

    assert not engine.is_finished()

    # step 1: prefill
    engine.step()
    assert seq1.status == SequenceStatus.RUNNING
    assert seq1.last_token == 5

    # step 2: decode → token 9
    engine.step()
    assert seq1.last_token == 9

    # step 3: decode → EOS
    engine.step()
    assert seq1.is_finished
    assert seq2.is_finished
    assert engine.is_finished()


def test_engine_generate_returns_completions(monkeypatch):
    step_counter = [0]

    def fake_run(seqs, is_prefill):
        if is_prefill:
            return _fake_run_prefill(seqs)
        step_counter[0] += 1
        return [2] * len(seqs)  # immediate EOS

    engine = _make_engine()
    monkeypatch.setattr(engine.runner, "run", fake_run)

    tokens = [1, 3, 4, 5]
    results = engine.generate([
        {"token_ids": tokens, "sampling_params": SamplingParams(max_tokens=5)},
        {"token_ids": tokens, "sampling_params": SamplingParams(max_tokens=5)},
    ])
    assert len(results) == 2
    for seq in results:
        assert seq.is_finished
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/nano-pointllm && python -m pytest tests/test_llm_engine.py -v 2>&1 | head -20
```

Expected: `ImportError`

- [ ] **Step 3: Implement PointLLMLLMEngine**

```python
# nanopointllm/engine/llm_engine.py
"""
PointLLMLLMEngine：把 Scheduler + PointLLMModelRunner 串成完整推理循环。

典型用法（离线 batch）::

    engine = PointLLMLLMEngine(hf_model=model, eos_token_id=2)
    seqs = engine.generate([
        {"token_ids": prompt1, "point_clouds": pc1},
        {"token_ids": prompt2},
    ])
    for seq in seqs:
        print(seq.token_ids[seq.num_prompt_tokens:])
"""
from __future__ import annotations

from typing import Any, Optional

import torch.nn as nn

from nanopointllm.engine.model_runner import PointLLMModelRunner
from nanopointllm.engine.scheduler import Scheduler
from nanopointllm.engine.sequence import PointLLMSequence
from nanopointllm.sampling_params import SamplingParams


class PointLLMLLMEngine:

    def __init__(
        self,
        hf_model: nn.Module,
        eos_token_id: int,
        max_num_seqs: int = 8,
        max_num_batched_tokens: int = 4096,
    ) -> None:
        self.runner = PointLLMModelRunner(hf_model=hf_model)
        self.scheduler = Scheduler(
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
            eos_token_id=eos_token_id,
        )

    def is_finished(self) -> bool:
        return self.scheduler.is_finished()

    def add_request(
        self,
        token_ids: list[int],
        point_clouds: Optional[Any] = None,
        sampling_params: Optional[SamplingParams] = None,
    ) -> PointLLMSequence:
        seq = PointLLMSequence(
            token_ids=token_ids,
            point_clouds=point_clouds,
            sampling_params=sampling_params or SamplingParams(),
        )
        self.scheduler.add(seq)
        return seq

    def step(self) -> None:
        """执行一轮调度（prefill 或 decode）并更新序列状态。"""
        seqs, is_prefill = self.scheduler.schedule()
        token_ids = self.runner.run(seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids)

    def generate(
        self,
        requests: list[dict],
    ) -> list[PointLLMSequence]:
        """
        批量推理接口：提交所有请求后循环 step 直到全部完成。

        Args:
            requests: list of dict，每个 dict 传给 add_request：
                      {"token_ids": [...], "point_clouds": Tensor, "sampling_params": ...}

        Returns:
            list[PointLLMSequence]，与 requests 顺序对应，seq.is_finished==True
        """
        seqs = [self.add_request(**req) for req in requests]
        while not self.is_finished():
            self.step()
        return seqs
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /home/nano-pointllm && python -m pytest tests/test_llm_engine.py -v
```

Expected: 5 passed

- [ ] **Step 5: 运行全套测试确保无回归**

```bash
cd /home/nano-pointllm && python -m pytest tests/ -v
```

Expected: 所有已有测试 + 新测试均通过

- [ ] **Step 6: Commit**

```bash
cd /home/nano-pointllm && git add nanopointllm/engine/llm_engine.py tests/test_llm_engine.py && git commit -m "feat(p3): add PointLLMLLMEngine top-level inference API completing multi-batch engine

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 6: BlockManager — 分页 KV 元数据 + prefix hash 缓存

**Files:**
- Create: `nanopointllm/engine/block_manager.py`
- Test: `tests/test_block_manager.py`

> **前置：** Tasks 1-5 已可独立运行多 batch 推理（per-seq DynamicCache + padded batch decode）。Task 6-7 是内存效率优化，不影响正确性。

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_block_manager.py
import pytest
from nanopointllm.engine.sequence import PointLLMSequence
from nanopointllm.engine.block_manager import BlockManager
from nanopointllm.sampling_params import SamplingParams


def _seq(length: int) -> PointLLMSequence:
    return PointLLMSequence(token_ids=list(range(length)))


def test_can_allocate_fits():
    bm = BlockManager(num_blocks=10, block_size=4)
    seq = _seq(8)   # needs 2 blocks
    assert bm.can_allocate(seq)


def test_can_allocate_too_large():
    bm = BlockManager(num_blocks=3, block_size=4)
    seq = _seq(16)  # needs 4 blocks, only 3 free
    assert not bm.can_allocate(seq)


def test_allocate_fills_block_table():
    bm = BlockManager(num_blocks=10, block_size=4)
    seq = _seq(8)   # 2 full blocks
    bm.allocate(seq)
    assert len(seq.block_table) == 2


def test_deallocate_frees_blocks():
    bm = BlockManager(num_blocks=4, block_size=4)
    seq = _seq(8)
    bm.allocate(seq)
    assert len(bm.free_block_ids) == 2
    bm.deallocate(seq)
    assert len(bm.free_block_ids) == 4


def test_can_append_same_block():
    bm = BlockManager(num_blocks=10, block_size=4)
    seq = _seq(5)   # 2 blocks: [4 tokens] + [1 token], last block not full
    bm.allocate(seq)
    # appending one more token stays in current block → no new block needed
    assert bm.can_append(seq)


def test_can_append_needs_new_block():
    bm = BlockManager(num_blocks=2, block_size=4)
    seq = _seq(4)   # exactly 1 full block
    bm.allocate(seq)
    # appending → crosses block boundary, but only 1 free block left
    assert bm.can_append(seq)  # 1 free block is enough


def test_can_append_no_free_blocks():
    bm = BlockManager(num_blocks=1, block_size=4)
    seq = _seq(4)   # uses the only block
    bm.allocate(seq)
    # next append would cross boundary but no free blocks
    assert not bm.can_append(seq)


def test_may_append_allocates_new_block():
    bm = BlockManager(num_blocks=4, block_size=4)
    seq = _seq(4)   # full block
    bm.allocate(seq)
    assert len(seq.block_table) == 1
    seq.token_ids.append(99)    # simulate new token pushing to next block
    bm.may_append(seq)
    assert len(seq.block_table) == 2


def test_prefix_cache_reuse():
    """相同 prompt prefix 的两个请求应复用 cache block。"""
    bm = BlockManager(num_blocks=10, block_size=4)
    tokens = list(range(8))
    seq1 = PointLLMSequence(token_ids=list(tokens))
    seq2 = PointLLMSequence(token_ids=list(tokens))

    bm.allocate(seq1)
    bm.allocate(seq2)

    # 两个序列的完整 block 都相同，应共享 block（ref_count=2）
    for bid in seq1.block_table[:-1]:   # 排除最后不满 block
        block = bm.blocks[bid]
        if block.hash != -1:
            assert block.ref_count >= 2
            break
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/nano-pointllm && python -m pytest tests/test_block_manager.py -v 2>&1 | head -20
```

Expected: `ImportError`

- [ ] **Step 3: Implement BlockManager（用 hashlib 替代 xxhash）**

```python
# nanopointllm/engine/block_manager.py
"""
分页 KV cache 的 block 元数据管理（参考 nano-vllm block_manager.py）。
用 hashlib.shake_128 代替 xxhash（无额外依赖）；语义保持一致。

与 nano-vllm 的主要差异：
  - 不依赖 xxhash，使用 hashlib
  - 适配 PointLLMSequence（无 whisper_cache_slot 字段）
"""
from __future__ import annotations

import hashlib
import struct
from collections import deque

import numpy as np

from nanopointllm.engine.sequence import PointLLMSequence


class Block:

    def __init__(self, block_id: int) -> None:
        self.block_id = block_id
        self.ref_count = 0
        self.hash: int = -1          # -1 = 不可复用
        self.token_ids: list[int] = []

    def update(self, hash_val: int, token_ids: list[int]) -> None:
        self.hash = hash_val
        self.token_ids = list(token_ids)

    def reset(self) -> None:
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:

    def __init__(self, num_blocks: int, block_size: int) -> None:
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = {}
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()

    # ── hash 工具 ──

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1) -> int:
        h = hashlib.shake_128()
        if prefix != -1:
            h.update(struct.pack("<q", prefix))
        h.update(np.array(token_ids, dtype=np.int64).tobytes())
        # shake_128 可输出任意长度；取 8 字节作 int64
        return int.from_bytes(h.digest(8), "little")

    # ── block 分配 / 释放 ──

    def _allocate_block(self, block_id: int) -> Block:
        block = self.blocks[block_id]
        assert block.ref_count == 0
        block.reset()
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return block

    def _deallocate_block(self, block_id: int) -> None:
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.discard(block_id)
        self.free_block_ids.append(block_id)

    # ── 序列级接口 ──

    def _num_blocks_for(self, seq: PointLLMSequence) -> int:
        return (seq.num_tokens + self.block_size - 1) // self.block_size

    def can_allocate(self, seq: PointLLMSequence) -> bool:
        return len(self.free_block_ids) >= self._num_blocks_for(seq)

    def allocate(self, seq: PointLLMSequence) -> None:
        """Prefill 调度时调用；尝试 prefix cache 命中，否则新分配。"""
        assert not seq.block_table
        n_blocks = self._num_blocks_for(seq)
        h = -1
        cache_miss = False
        for i in range(n_blocks):
            start = i * self.block_size
            end = min(start + self.block_size, seq.num_tokens)
            chunk = seq.token_ids[start:end]
            is_full = len(chunk) == self.block_size
            h_new = self.compute_hash(chunk, h) if is_full else -1

            block_id_cached = self.hash_to_block_id.get(h_new, -1)
            if (
                not cache_miss
                and h_new != -1
                and block_id_cached != -1
                and self.blocks[block_id_cached].token_ids == chunk
            ):
                # Prefix cache 命中：复用
                seq.num_cached_tokens += self.block_size
                bid = block_id_cached
                if bid in self.used_block_ids:
                    self.blocks[bid].ref_count += 1
                else:
                    self._allocate_block(bid)
            else:
                cache_miss = True
                bid = self.free_block_ids[0]
                block = self._allocate_block(bid)
                if h_new != -1:
                    block.update(h_new, chunk)
                    self.hash_to_block_id[h_new] = bid

            h = h_new
            seq.block_table.append(bid)

    def deallocate(self, seq: PointLLMSequence) -> None:
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: PointLLMSequence) -> bool:
        """Decode 前检查：新 token 是否需要新 block，以及是否有空闲 block。"""
        needs_new_block = (seq.num_tokens % self.block_size) == 0
        return not needs_new_block or len(self.free_block_ids) >= 1

    def may_append(self, seq: PointLLMSequence) -> None:
        """
        Decode 前元数据维护：
          - 上一个 block 刚被填满 → 计算 hash 写入 prefix cache
          - 当前 token 是新 block 的第一格 → 分配新物理 block
        """
        n_tokens = seq.num_tokens
        n_blocks_needed = (n_tokens + self.block_size - 1) // self.block_size
        if n_blocks_needed > len(seq.block_table):
            # 需要新 block
            bid = self.free_block_ids[0]
            self._allocate_block(bid)
            seq.block_table.append(bid)
        # 如果前一个 block 刚好满了，写入 hash（供后续 prefix cache）
        last_idx = len(seq.block_table) - 2
        if last_idx >= 0 and (n_tokens - 1) % self.block_size == 0:
            prev_bid = seq.block_table[last_idx]
            block = self.blocks[prev_bid]
            if block.hash == -1:
                chunk_start = last_idx * self.block_size
                chunk = seq.token_ids[chunk_start: chunk_start + self.block_size]
                prev_hash = self.blocks[seq.block_table[last_idx - 1]].hash if last_idx > 0 else -1
                h = self.compute_hash(chunk, prev_hash)
                block.update(h, chunk)
                self.hash_to_block_id[h] = prev_bid
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /home/nano-pointllm && python -m pytest tests/test_block_manager.py -v
```

Expected: 9 passed

- [ ] **Step 5: Scheduler 集成 BlockManager（替换纯 token 计数）**

修改 `nanopointllm/engine/scheduler.py`，在构造函数接受可选 `block_manager` 参数；有 BlockManager 时在 `schedule`/`postprocess` 中调用 `allocate`/`may_append`/`deallocate`（向后兼容，无 BlockManager 时行为不变）：

```python
# nanopointllm/engine/scheduler.py  (替换整个文件)
from __future__ import annotations

from collections import deque
from typing import Optional

from nanopointllm.engine.sequence import PointLLMSequence, SequenceStatus


class Scheduler:

    def __init__(
        self,
        max_num_seqs: int,
        max_num_batched_tokens: int,
        eos_token_id: int,
        block_manager=None,   # Optional[BlockManager]；None = 不做分页 KV 管理
    ) -> None:
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.eos_token_id = eos_token_id
        self.block_manager = block_manager
        self.waiting: deque[PointLLMSequence] = deque()
        self.running: deque[PointLLMSequence] = deque()

    def is_finished(self) -> bool:
        return not self.waiting and not self.running

    def add(self, seq: PointLLMSequence) -> None:
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[PointLLMSequence], bool]:
        bm = self.block_manager

        # ── Prefill ──
        prefill_seqs: list[PointLLMSequence] = []
        total_tokens = 0
        while self.waiting and len(prefill_seqs) < self.max_num_seqs:
            seq = self.waiting[0]
            if bm is not None and not bm.can_allocate(seq):
                break
            uncached = seq.num_tokens - seq.num_cached_tokens
            if total_tokens + uncached > self.max_num_batched_tokens:
                break
            total_tokens += uncached
            self.waiting.popleft()
            if bm is not None:
                bm.allocate(seq)
            seq.status = SequenceStatus.RUNNING
            self.running.append(seq)
            prefill_seqs.append(seq)

        if prefill_seqs:
            return prefill_seqs, True

        # ── Decode ──
        assert self.running
        decode_seqs: list[PointLLMSequence] = []
        tmp = list(self.running)
        for seq in tmp:
            if bm is not None:
                while not bm.can_append(seq):
                    if tmp:
                        victim = tmp.pop()
                        self._preempt(victim)
                    else:
                        self._preempt(seq)
                        break
                else:
                    bm.may_append(seq)
                    decode_seqs.append(seq)
            else:
                decode_seqs.append(seq)

        return decode_seqs, False

    def _preempt(self, seq: PointLLMSequence) -> None:
        seq.status = SequenceStatus.WAITING
        seq.past_key_values = None
        if self.block_manager is not None:
            self.block_manager.deallocate(seq)
        self.running.discard(seq) if hasattr(self.running, "discard") else None
        if seq in self.running:
            self.running.remove(seq)
        self.waiting.appendleft(seq)

    def postprocess(
        self,
        seqs: list[PointLLMSequence],
        token_ids: list[int],
    ) -> None:
        bm = self.block_manager
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)
            finished = (
                (not seq.ignore_eos and token_id == self.eos_token_id)
                or seq.num_completion_tokens >= seq.max_tokens
            )
            if finished:
                seq.status = SequenceStatus.FINISHED
                if bm is not None:
                    bm.deallocate(seq)
                if seq in self.running:
                    self.running.remove(seq)
```

- [ ] **Step 6: 确认所有测试仍然通过**

```bash
cd /home/nano-pointllm && python -m pytest tests/ -v
```

Expected: 全部通过（scheduler 原有测试仍兼容，因 block_manager=None）

- [ ] **Step 7: Commit**

```bash
cd /home/nano-pointllm && git add nanopointllm/engine/block_manager.py nanopointllm/engine/scheduler.py tests/test_block_manager.py && git commit -m "feat(p3): add BlockManager with prefix hash cache, integrate into Scheduler

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 7: PagedKV 集成 — PointLLMLLMEngine 接入 BlockManager

**Files:**
- Modify: `nanopointllm/engine/llm_engine.py`（添加 BlockManager 参数）
- Modify: `nanopointllm/engine/model_runner.py`（预留 num_kvcache_blocks 分配接口）

> **说明：** 真正的 paged KV（全局 tensor + slot_mapping attention）需要替换 HF LlamaAttention，属于 P4 工作。本 Task 仅将 BlockManager 的元数据管理织入 Engine，为未来 slot-based attention 准备好接口；现阶段 KV 数据仍存在 per-seq DynamicCache 中。

- [ ] **Step 1: 更新 PointLLMLLMEngine 接受可选 BlockManager**

```python
# nanopointllm/engine/llm_engine.py  （完整替换）
from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn

from nanopointllm.engine.block_manager import BlockManager
from nanopointllm.engine.model_runner import PointLLMModelRunner
from nanopointllm.engine.scheduler import Scheduler
from nanopointllm.engine.sequence import PointLLMSequence
from nanopointllm.sampling_params import SamplingParams


class PointLLMLLMEngine:

    def __init__(
        self,
        hf_model: nn.Module,
        eos_token_id: int,
        max_num_seqs: int = 8,
        max_num_batched_tokens: int = 4096,
        # 分页 KV 参数（可选；不传则退回 per-seq DynamicCache）
        num_kvcache_blocks: Optional[int] = None,
        kvcache_block_size: int = 16,
    ) -> None:
        self.runner = PointLLMModelRunner(hf_model=hf_model)

        block_manager = None
        if num_kvcache_blocks is not None:
            block_manager = BlockManager(
                num_blocks=num_kvcache_blocks,
                block_size=kvcache_block_size,
            )

        self.scheduler = Scheduler(
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
            eos_token_id=eos_token_id,
            block_manager=block_manager,
        )

    def is_finished(self) -> bool:
        return self.scheduler.is_finished()

    def add_request(
        self,
        token_ids: list[int],
        point_clouds: Optional[Any] = None,
        sampling_params: Optional[SamplingParams] = None,
    ) -> PointLLMSequence:
        seq = PointLLMSequence(
            token_ids=token_ids,
            point_clouds=point_clouds,
            sampling_params=sampling_params or SamplingParams(),
        )
        self.scheduler.add(seq)
        return seq

    def step(self) -> None:
        seqs, is_prefill = self.scheduler.schedule()
        token_ids = self.runner.run(seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids)

    def generate(
        self,
        requests: list[dict],
    ) -> list[PointLLMSequence]:
        seqs = [self.add_request(**req) for req in requests]
        while not self.is_finished():
            self.step()
        return seqs
```

- [ ] **Step 2: Run all tests**

```bash
cd /home/nano-pointllm && python -m pytest tests/ -v
```

Expected: 全部通过

- [ ] **Step 3: Commit**

```bash
cd /home/nano-pointllm && git add nanopointllm/engine/llm_engine.py && git commit -m "feat(p3): integrate BlockManager into PointLLMLLMEngine as optional paged KV metadata

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## 自我检查

**Spec coverage（参照分析报告）：**

| 缺口 | 对应 Task |
|------|-----------|
| prepare_inputs_embeds 分离 | Task 2 ✓ |
| 批量点云编码 | Task 2 + Task 4 ✓ |
| PointLLMSequence 状态管理 | Task 1 ✓ |
| Scheduler FIFO 调度 | Task 3 ✓ |
| 批量 prefill（含 padding） | Task 4 ✓ |
| 批量 decode（padded KV） | Task 4 ✓ |
| 顶层引擎 API | Task 5 ✓ |
| BlockManager + prefix cache | Task 6 ✓ |
| Engine 集成 BlockManager | Task 7 ✓ |

**Placeholder scan：** 无 TBD / TODO / "implement later"。

**Type consistency：**
- `PointLLMSequence.kv_seq_len()` 在 Task 1 定义，Task 4 `_build_batched_cache` 中使用 ✓
- `PointLLMWrapper.prepare_inputs_embeds(input_ids, point_features)` 参数名在 Task 2 定义和测试中一致 ✓
- `Scheduler(block_manager=None)` 在 Task 3 定义，Task 6 修改为带 BlockManager，向后兼容 ✓
- `_batch_hf_decode` 在 Task 4 定义，测试中 monkeypatch 同名函数 ✓
