from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class ForwardContext:
    is_prefill: bool
    slot_mapping: torch.Tensor          # [total_new_tokens] int32; -1 = skip (padding)
    block_tables: Optional[torch.Tensor]  # [B, max_num_blocks] int32; decode only
    context_lens: Optional[torch.Tensor]  # [B] int32; decode only
    position_ids: Optional[torch.Tensor] = None  # [B, 1] int64; decode only


_ctx: threading.local = threading.local()


def set_forward_context(ctx: Optional[ForwardContext]) -> None:
    _ctx.value = ctx


def get_forward_context() -> Optional[ForwardContext]:
    return getattr(_ctx, "value", None)


def clear_forward_context() -> None:
    _ctx.value = None
