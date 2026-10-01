"""
M2：显式拆分 **prefill** 与 **decode 步**，仍调用 HF `forward`（与自定义 CUDA 引擎的边界对齐）。

首步可传 `point_clouds`（PointLLM）；decode 步固定 `point_clouds=None`。
"""
from __future__ import annotations

from typing import Any, Optional

import torch

from nanopointllm.engine.decode_backend import DecodeBackend, default_hf_decode_backend
from nanopointllm.parity.forward_kwargs import filter_forward_kwargs

from nanopointllm.engine.types import DecodeStepInput, PrefillOutput


def _greedy_next_token_like_manual(logits: torch.Tensor) -> torch.Tensor:
    """与 `manual_greedy_decode` 相同：`logits` 为 `(B, L, V)` 且 `L>=1`。"""
    return logits[:, -1, :].float().argmax(dim=-1, keepdim=True)


@torch.inference_mode()
def hf_prefill(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    point_clouds: Optional[torch.Tensor] = None,
    **forward_kwargs: Any,
) -> PrefillOutput:
    """
    整段 prompt 一次 forward（含点云融合）。返回 **最后一个位置** 的 logits 切片 `(B, 1, V)` 与 KV cache。
    """
    call_kw = filter_forward_kwargs(
        model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        past_key_values=None,
        use_cache=True,
        return_dict=True,
        point_clouds=point_clouds,
        **forward_kwargs,
    )
    out = model(**call_kw)
    L = int(input_ids.shape[1])
    last_logits = out.logits[:, -1:, :].contiguous()
    return PrefillOutput(
        logits=last_logits,
        past_key_values=out.past_key_values,
        prompt_len=L,
    )


@torch.inference_mode()
def hf_decode_step(
    model: torch.nn.Module,
    inp: DecodeStepInput,
    **forward_kwargs: Any,
) -> tuple[torch.Tensor, Any]:
    """
    单 token decode（HF）。返回 `(logits[:, -1:, :], new_past_key_values)`。
    """
    logits, past = default_hf_decode_backend().decode_step(model, inp, **forward_kwargs)
    return logits[:, -1:, :].contiguous(), past


@torch.inference_mode()
def hybrid_greedy_decode(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    max_new_tokens: int,
    *,
    point_clouds: Optional[torch.Tensor] = None,
    decode_backend: Optional[DecodeBackend] = None,
    **forward_kwargs: Any,
) -> torch.Tensor:
    """
    HF prefill 一次 + decode 步循环（默认 `HFDecodeBackend`）；greedy 与 `manual_greedy_decode` 对齐。

    `decode_backend`：M3 起可换自定义实现，须返回与 HF 一致的 `(logits, past_key_values)`。
    """
    if max_new_tokens <= 0:
        return input_ids

    po = hf_prefill(
        model,
        input_ids,
        attention_mask,
        point_clouds=point_clouds,
        **forward_kwargs,
    )
    # 与 manual 一致：用整段 logits 的 `[:, -1, :]`，避免 (B,1,V) 上 argmax 维数陷阱
    next_t = _greedy_next_token_like_manual(po.logits)
    cur_ids = torch.cat([input_ids, next_t], dim=-1)
    mask = torch.cat(
        [
            attention_mask,
            attention_mask.new_ones((attention_mask.shape[0], 1)),
        ],
        dim=-1,
    )
    past = po.past_key_values
    backend = decode_backend if decode_backend is not None else default_hf_decode_backend()

    for _ in range(max_new_tokens - 1):
        step_in = DecodeStepInput(
            input_ids=cur_ids[:, -1:],
            attention_mask=mask,
            past_key_values=past,
        )
        logits, past = backend.decode_step(model, step_in, **forward_kwargs)
        next_t = _greedy_next_token_like_manual(logits)
        cur_ids = torch.cat([cur_ids, next_t], dim=-1)
        mask = torch.cat([mask, mask.new_ones((mask.shape[0], 1))], dim=-1)
    return cur_ids
