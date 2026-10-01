"""
M3：decode 步可插拔后端。自定义 CUDA / Triton 实现应保持
`decode_step(model, DecodeStepInput, **forward_kwargs) -> (logits, past)` 语义与 HF 一致。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn

from nanopointllm.engine.types import DecodeStepInput
from nanopointllm.parity.forward_kwargs import filter_forward_kwargs


class DecodeBackend(ABC):
    """单 token（或当前步 `input_ids` 形状）decode：`past` 进、`past` 出 + 本步 logits。"""

    @abstractmethod
    def decode_step(
        self,
        model: nn.Module,
        inp: DecodeStepInput,
        **forward_kwargs: Any,
    ) -> tuple[torch.Tensor, Any]:
        """
        Returns:
            logits: `(B, L_step, V)`，与 HF `CausalLMOutputWithPast.logits` 一致；常规 decode 下 `L_step==1`。
            past_key_values: 更新后的 KV cache。
        """


class HFDecodeBackend(DecodeBackend):
    """默认：HuggingFace `model.forward`（`point_clouds=None`）。"""

    @torch.inference_mode()
    def decode_step(
        self,
        model: nn.Module,
        inp: DecodeStepInput,
        **forward_kwargs: Any,
    ) -> tuple[torch.Tensor, Any]:
        call_kw = filter_forward_kwargs(
            model,
            input_ids=inp.input_ids,
            attention_mask=inp.attention_mask,
            past_key_values=inp.past_key_values,
            use_cache=True,
            return_dict=True,
            point_clouds=None,
            **forward_kwargs,
        )
        out = model(**call_kw)
        return out.logits, out.past_key_values


def _decoder_base_model(model: nn.Module) -> nn.Module:
    """`LlamaForCausalLM` / `PointLLMLlamaForCausalLM` 的 backbone（`model` 子模块）。"""
    if hasattr(model, "get_model"):
        inner = model.get_model()
        if inner is not None:
            return inner
    inner = getattr(model, "model", None)
    if inner is None:
        raise TypeError(
            "HFInnerDecodeBackend 需要 `model.get_model()` 或属性 `model.model`（CausalLM 主干）"
        )
    return inner


class HFInnerDecodeBackend(DecodeBackend):
    """
    仅走 **decoder backbone**（如 `PointLLMLlamaModel` / `LlamaModel`）再经 **`lm_head`**，
    与整段 `CausalLM.forward` 在 decode 步（无 `point_clouds`）语义对齐；便于后续把 backbone 换成自定义实现。

    要求：`model` 含 `lm_head`，且 backbone 的 `forward` 接受与 HF Llama 解码一致的参数。
    """

    @torch.inference_mode()
    def decode_step(
        self,
        model: nn.Module,
        inp: DecodeStepInput,
        **forward_kwargs: Any,
    ) -> tuple[torch.Tensor, Any]:
        inner = _decoder_base_model(model)
        call_kw = filter_forward_kwargs(
            inner,
            input_ids=inp.input_ids,
            attention_mask=inp.attention_mask,
            past_key_values=inp.past_key_values,
            use_cache=True,
            return_dict=True,
            point_clouds=None,
            **forward_kwargs,
        )
        out = inner(**call_kw)
        hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        lm = getattr(model, "lm_head", None)
        if lm is None:
            raise TypeError("HFInnerDecodeBackend 需要 `model.lm_head`")
        logits = lm(hidden)
        return logits, out.past_key_values


# 模块级默认实例，避免 `hybrid_greedy_decode` 每次构造
_DEFAULT_HF_BACKEND = HFDecodeBackend()


def default_hf_decode_backend() -> HFDecodeBackend:
    return _DEFAULT_HF_BACKEND
