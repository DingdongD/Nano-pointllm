from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class ForwardContext:
    slot_mapping:  torch.Tensor            # [total_tokens] int32; -1 = skip (cached prefix)
    block_tables:  Optional[torch.Tensor]  # [total_seqs, max_blocks] int32
    context_lens:  Optional[torch.Tensor]  # [total_seqs] int32 — full KV length per seq
    position_ids:  Optional[torch.Tensor] = None  # [1, total_tokens] int64
    seq_lens:      list[int] = field(default_factory=list)  # new query tokens per seq; consumed by varlen SDPA loop (Task 4/run_mixed)


_ctx: threading.local = threading.local()


def set_forward_context(ctx: Optional[ForwardContext]) -> None:
    _ctx.value = ctx


def get_forward_context() -> Optional[ForwardContext]:
    return getattr(_ctx, "value", None)


def clear_forward_context() -> None:
    _ctx.value = None
