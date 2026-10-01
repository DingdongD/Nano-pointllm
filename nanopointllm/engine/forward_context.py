from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class ForwardContext:
    slot_mapping:       torch.Tensor            # [total_new_tokens] int32; -1 = skip
    block_tables:       Optional[torch.Tensor]  # [total_seqs, max_blocks] int32
    context_lens:       Optional[torch.Tensor]  # [total_seqs] int32 — full KV length per seq
    position_ids:       Optional[torch.Tensor] = None   # [1, total_new_tokens] int64
    seq_lens:           list[int] = field(default_factory=list)  # new-token count per seq

    # nano-vllm style fields: populated by run_mixed for the prefill portion
    num_prefill_tokens: int = 0                         # total new tokens from prefill seqs
    num_prefill_seqs:   int = 0                         # number of prefill sequences
    cu_seqlens_q:       Optional[torch.Tensor] = None   # [num_pf_seqs + 1] int32
    cu_seqlens_k:       Optional[torch.Tensor] = None   # [num_pf_seqs + 1] int32
    max_seqlen_q:       int = 0
    max_seqlen_k:       int = 0


_ctx: threading.local = threading.local()


def set_forward_context(ctx: Optional[ForwardContext]) -> None:
    _ctx.value = ctx


def get_forward_context() -> Optional[ForwardContext]:
    return getattr(_ctx, "value", None)


def clear_forward_context() -> None:
    _ctx.value = None
