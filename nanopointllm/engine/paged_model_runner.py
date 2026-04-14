"""
PagedModelRunner: prefill + paged-KV decode using ForwardContext.
Replaces PointLLMModelRunner when the engine is constructed with num_kvcache_blocks.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from nanopointllm.engine.forward_context import (
    ForwardContext,
    clear_forward_context,
    get_forward_context,
    set_forward_context,
)
from nanopointllm.engine.kv_pool import KVPool
from nanopointllm.engine.sequence import PointLLMSequence
from nanopointllm.models.pointllm_wrapper import PointLLMWrapper


def _install_position_ids_patch() -> None:
    """
    PointLLMLlamaModel.forward() calls super().forward() without position_ids,
    so LlamaModel always defaults to position 0 for decode tokens.  We patch
    LlamaModel.forward (class-level, once) to read position_ids from the
    active ForwardContext and inject them transparently.
    """
    try:
        from transformers.models.llama.modeling_llama import LlamaModel
    except ImportError:
        return  # non-Llama model; no-op

    if getattr(LlamaModel, "_paged_position_ids_patched", False):
        return  # already installed

    _orig = LlamaModel.forward

    def _forward_with_ctx_positions(self, *args, **kwargs):
        ctx = get_forward_context()
        if (
            ctx is not None
            and ctx.position_ids is not None
            and "position_ids" not in kwargs
        ):
            kwargs["position_ids"] = ctx.position_ids
        return _orig(self, *args, **kwargs)

    LlamaModel.forward = _forward_with_ctx_positions
    LlamaModel._paged_position_ids_patched = True


class PagedModelRunner:

    def __init__(self, hf_model: nn.Module, kv_pool: KVPool, block_manager) -> None:
        self.hf_model      = hf_model
        self.kv_pool       = kv_pool
        self.block_manager = block_manager
        self.wrapper       = PointLLMWrapper(hf_model)
        self.block_size    = kv_pool.block_size
        _install_position_ids_patch()

    # ── Point cloud encoding (mirrors PointLLMModelRunner) ──────────────────

    def _encode_point_clouds_batch(self, seqs: list[PointLLMSequence]) -> None:
        pending = [
            seq for seq in seqs
            if seq.point_clouds is not None and seq.inputs_embeds_cached is None
        ]
        if not pending:
            return

        try:
            device = next(self.wrapper.inner.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

        def _to_2d(pc: torch.Tensor) -> torch.Tensor:
            return pc.squeeze(0) if pc.dim() == 3 else pc

        pcs_2d = [_to_2d(seq.point_clouds).to(device) for seq in pending]
        shapes = {pc.shape for pc in pcs_2d}

        if len(shapes) == 1:
            stacked = torch.stack(pcs_2d)
            all_features = self.wrapper.encode_point_clouds(stacked)
        else:
            all_features = [self.wrapper.encode_point_clouds(pc)[0] for pc in pcs_2d]

        for seq, feat in zip(pending, all_features):
            ids    = torch.tensor([seq.token_ids], dtype=torch.long, device=device)
            embeds = self.wrapper.prepare_inputs_embeds(ids, [feat])
            seq.inputs_embeds_cached = embeds.squeeze(0)

    # ── Slot-mapping helpers ─────────────────────────────────────────────────

    def _prefill_slot_mapping(
        self,
        seq: PointLLMSequence,
        max_len: int,
        pad_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Returns [max_len] int32 tensor.
        Padding positions → -1.
        Real token at logical_pos → block_table[blk] * block_size + offset.
        """
        slots = torch.full((max_len,), -1, dtype=torch.int32, device=device)
        for pos in range(seq.num_tokens):
            blk = pos // self.block_size
            off = pos  % self.block_size
            slots[pad_len + pos] = seq.block_table[blk] * self.block_size + off
        return slots

    def _decode_slot(self, seq: PointLLMSequence) -> int:
        """Physical slot for the newly written token (after append_token)."""
        pos = seq.num_tokens - 1
        blk = pos // self.block_size
        off = pos  % self.block_size
        if blk >= len(seq.block_table):
            raise RuntimeError(
                f"KV pool exhausted: seq needs block {blk} but block_table has "
                f"{len(seq.block_table)} blocks (num_tokens={seq.num_tokens}, "
                f"block_size={self.block_size})"
            )
        return seq.block_table[blk] * self.block_size + off

    # ── Prefill ──────────────────────────────────────────────────────────────

    @torch.inference_mode()
    def run_prefill(self, seqs: list[PointLLMSequence]) -> list[int]:
        self._encode_point_clouds_batch(seqs)

        B       = len(seqs)
        max_len = max(seq.num_tokens for seq in seqs)

        try:
            device = next(self.hf_model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

        embeds_list, attn_masks, slot_maps, pos_ids_list = [], [], [], []
        for seq in seqs:
            L       = seq.num_tokens
            pad_len = max_len - L

            if seq.inputs_embeds_cached is not None:
                emb = seq.inputs_embeds_cached
            else:
                ids = torch.tensor([seq.token_ids], dtype=torch.long, device=device)
                emb = self.wrapper.get_input_embeddings()(ids).squeeze(0)

            if pad_len > 0:
                pad = emb.new_zeros(pad_len, emb.shape[-1])
                emb  = torch.cat([pad, emb], dim=0)
                mask = torch.cat([
                    torch.zeros(pad_len, dtype=torch.long, device=device),
                    torch.ones(L,        dtype=torch.long, device=device),
                ])
                # Left-padding: real tokens must be encoded at positions [0..L-1],
                # not at [pad_len..pad_len+L-1] (LlamaModel default without explicit ids).
                pos = torch.cat([
                    torch.zeros(pad_len, dtype=torch.long, device=device),
                    torch.arange(L,      dtype=torch.long, device=device),
                ])
            else:
                mask = torch.ones(L, dtype=torch.long, device=device)
                pos  = torch.arange(L, dtype=torch.long, device=device)

            embeds_list.append(emb)
            attn_masks.append(mask)
            slot_maps.append(self._prefill_slot_mapping(seq, max_len, pad_len, device))
            pos_ids_list.append(pos)

        inputs_embeds  = torch.stack(embeds_list)
        attention_mask = torch.stack(attn_masks)
        slot_mapping   = torch.stack(slot_maps).reshape(-1)
        position_ids   = torch.stack(pos_ids_list)  # [B, max_len]

        dummy_ids = torch.zeros(B, max_len, dtype=torch.long, device=device)

        seq_lens_list = [seq.num_tokens for seq in seqs]
        set_forward_context(ForwardContext(
            slot_mapping=slot_mapping,
            block_tables=None,
            context_lens=None,
            position_ids=position_ids,
            seq_lens=seq_lens_list,
        ))
        try:
            out = self.hf_model(
                input_ids=dummy_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                point_clouds=None,
                use_cache=False,
                return_dict=True,
            )
        finally:
            clear_forward_context()

        logits = out.logits
        return logits[:, -1, :].float().argmax(dim=-1).tolist()

    # ── Decode ───────────────────────────────────────────────────────────────

    @torch.inference_mode()
    def run_decode(self, seqs: list[PointLLMSequence]) -> list[int]:
        B = len(seqs)
        try:
            device = next(self.hf_model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

        input_ids = torch.tensor(
            [[seq.last_token] for seq in seqs], dtype=torch.long, device=device,
        )
        positions = torch.tensor(
            [[seq.num_tokens - 1] for seq in seqs], dtype=torch.long, device=device,
        )
        slot_mapping = torch.tensor(
            [self._decode_slot(seq) for seq in seqs], dtype=torch.int32, device=device,
        )
        context_lens = torch.tensor(
            [seq.num_tokens for seq in seqs], dtype=torch.int32, device=device,
        )
        max_blocks = max(len(seq.block_table) for seq in seqs)
        block_tables = torch.tensor(
            [seq.block_table + [-1] * (max_blocks - len(seq.block_table)) for seq in seqs],
            dtype=torch.int32, device=device,
        )

        set_forward_context(ForwardContext(
            slot_mapping=slot_mapping,
            block_tables=block_tables,
            context_lens=context_lens,
            position_ids=positions,
            seq_lens=[1] * B,
        ))
        try:
            out = self.hf_model(
                input_ids=input_ids,
                attention_mask=None,
                use_cache=False,
                return_dict=True,
            )
        finally:
            clear_forward_context()

        logits = out.logits
        if logits.dim() == 3:
            logits = logits[:, -1, :]
        return logits.float().argmax(dim=-1).tolist()

    def run(self, seqs: list[PointLLMSequence], is_prefill: bool) -> list[int]:
        if is_prefill:
            return self.run_prefill(seqs)
        return self.run_decode(seqs)
