"""
HF 参考：手写 greedy decode 循环（便于与日后自定义引擎逐 token 对比）。

首步可传 `point_clouds`（PointLLM）；后续步自动置 `None`。
"""
from __future__ import annotations

from typing import Any, Optional

import torch

from nanopointllm.parity.forward_kwargs import filter_forward_kwargs


@torch.inference_mode()
def manual_greedy_decode(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    max_new_tokens: int,
    *,
    point_clouds: Optional[torch.Tensor] = None,
    **forward_kwargs: Any,
) -> torch.Tensor:
    """
    Returns full token ids (prompt + generated), same layout as extending from prompt.

    `model` 需支持 `forward(..., use_cache=True, return_dict=True)` 且返回 `past_key_values`。
    """
    cur_ids = input_ids
    mask = attention_mask
    past = None

    for step in range(max_new_tokens):
        inp = cur_ids if past is None else cur_ids[:, -1:]
        pc = point_clouds if past is None else None
        call_kw = filter_forward_kwargs(
            model,
            input_ids=inp,
            attention_mask=mask,
            past_key_values=past,
            use_cache=True,
            return_dict=True,
            point_clouds=pc,
            **forward_kwargs,
        )
        out = model(**call_kw)
        past = out.past_key_values
        next_t = out.logits[:, -1, :].float().argmax(dim=-1, keepdim=True)
        cur_ids = torch.cat([cur_ids, next_t], dim=-1)
        mask = torch.cat(
            [mask, mask.new_ones((mask.shape[0], 1))],
            dim=-1,
        )
    return cur_ids
