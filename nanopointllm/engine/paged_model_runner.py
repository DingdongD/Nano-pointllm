"""
PagedModelRunner: prefill + paged-KV decode using ForwardContext.
Replaces PointLLMModelRunner when the engine is constructed with num_kvcache_blocks.
"""
from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn as nn

from nanopointllm.engine.cuda_graph import DecodeGraphCache, DecodeGraphKey
from nanopointllm.engine.forward_context import (
    ForwardContext,
    clear_forward_context,
    get_forward_context,
    set_forward_context,
)
from nanopointllm.engine.kv_pool import KVPool
from nanopointllm.engine.metadata_staging import DecodeMetadataStagingBuffer, next_bucket_size
from nanopointllm.engine.point_feature_cache import PointFeatureCache
from nanopointllm.engine.sampler import build_sampler
from nanopointllm.engine.sequence import PointLLMSequence
from nanopointllm.llama.lightweight_runner import build_lightweight_llama_runner
from nanopointllm.models.pointllm_wrapper import PointLLMWrapper
from nanopointllm.profiling import nvtx_stage


def _install_position_ids_patch() -> None:
    """
    PointLLMLlamaModel.forward() can pass position_ids=None to super().forward(),
    so LlamaModel would derive positions for the flattened token stream instead
    of each logical request. Patch LlamaModel.forward (class-level, once) to
    inject positions from the active ForwardContext when the value is missing.
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
            and kwargs.get("position_ids") is None
        ):
            kwargs["position_ids"] = ctx.position_ids
        return _orig(self, *args, **kwargs)

    LlamaModel.forward = _forward_with_ctx_positions
    LlamaModel._paged_position_ids_patched = True


class PagedModelRunner:

    def __init__(
        self,
        hf_model: nn.Module,
        kv_pool: KVPool,
        block_manager,
        point_feature_cache: PointFeatureCache | None = None,
    ) -> None:
        self.hf_model      = hf_model
        self.kv_pool       = kv_pool
        self.block_manager = block_manager
        self.wrapper       = PointLLMWrapper(hf_model)
        self.block_size    = kv_pool.block_size
        self.sampler       = build_sampler(compile_greedy_cuda=True)
        if point_feature_cache is None:
            point_feature_cache = PointFeatureCache(
                max_entries=int(os.environ.get("NANOPOINTLLM_POINT_FEATURE_CACHE_SIZE", "128"))
            )
        self.point_feature_cache = point_feature_cache
        self._decode_metadata: DecodeMetadataStagingBuffer | None = None
        self._decode_graphs = DecodeGraphCache(
            enabled=os.environ.get("NANOPOINTLLM_ENABLE_CUDA_GRAPH", "0") == "1",
        )
        lightweight_requested = os.environ.get(
            "NANOPOINTLLM_LIGHTWEIGHT_DECODE", "1"
        ) != "0"
        profile_layers = os.environ.get("NANOPOINTLLM_PROFILE_LAYERS", "0") == "1"
        # Graph capture requires the lightweight stack. Profiling also requires
        # that stack for layer events, but must not silently switch eager decode
        # to the graph-optimized module configuration.
        lightweight_required = self._decode_graphs.enabled or profile_layers
        self.lightweight_decode_enabled = lightweight_requested or lightweight_required
        self.lightweight_runner = None
        if self.lightweight_decode_enabled:
            self.lightweight_runner = build_lightweight_llama_runner(
                hf_model,
                decode_graph_mode=self._decode_graphs.enabled,
            )
            if os.environ.get("NANOPOINTLLM_COMPILE_LIGHTWEIGHT", "0") == "1":
                self.lightweight_runner = torch.compile(
                    self.lightweight_runner,
                    mode="reduce-overhead",
                    fullgraph=False,
                )
        _install_position_ids_patch()
        self.profile_runtime = profile_layers
        self.last_cuda_profile: dict[str, float] = {}

    def _sample_logits(
        self,
        logits: torch.Tensor,
        seqs: list[PointLLMSequence],
    ) -> torch.Tensor:
        if not self.profile_runtime or not logits.is_cuda:
            with nvtx_stage("pointllm_sampling", logits):
                return self.sampler(logits, seqs)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with nvtx_stage("pointllm_sampling", logits):
            sampled = self.sampler(logits, seqs)
        end.record()
        end.synchronize()
        self.last_cuda_profile = {"sampling_ms": start.elapsed_time(end)}
        return sampled

    def _get_decode_metadata_buffer(
        self,
        *,
        device: torch.device,
        batch_size: int,
        max_blocks: int,
    ) -> DecodeMetadataStagingBuffer:
        bucket_size = next_bucket_size(batch_size)
        if (
            self._decode_metadata is None
            or self._decode_metadata.device != device
            or not self._decode_metadata.can_fit(bucket_size, max_blocks)
        ):
            self._decode_metadata = DecodeMetadataStagingBuffer(
                device=device,
                max_batch_size=bucket_size,
                max_blocks_per_seq=max_blocks,
            )
        return self._decode_metadata

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

        uncached_seqs: list[PointLLMSequence] = []
        uncached_pcs: list[torch.Tensor] = []
        for seq in pending:
            if seq.point_features_cached is not None:
                continue
            key = seq.point_cloud_cache_key
            if key is None:
                key = self.wrapper.point_cloud_cache_key(seq.point_clouds)
                seq.point_cloud_cache_key = key
            feat = self.point_feature_cache.get(key)
            if feat is not None:
                seq.point_features_cached = feat
                continue
            pc_2d = _to_2d(seq.point_clouds).to(device)
            uncached_seqs.append(seq)
            uncached_pcs.append(pc_2d)

        if uncached_seqs:
            shapes = {pc.shape for pc in uncached_pcs}
            if len(shapes) == 1:
                stacked = torch.stack(uncached_pcs)
                all_features = self.wrapper.encode_point_clouds(stacked)
            else:
                all_features = [self.wrapper.encode_point_clouds(pc)[0] for pc in uncached_pcs]

            for seq, feat in zip(uncached_seqs, all_features):
                seq.point_features_cached = feat
                assert seq.point_cloud_cache_key is not None
                self.point_feature_cache.put(seq.point_cloud_cache_key, feat)

        for seq in pending:
            ids    = torch.tensor([seq.token_ids], dtype=torch.long, device=device)
            if seq.input_embed_layout_cached is None:
                seq.input_embed_layout_cached = self.wrapper.analyze_input_layout(ids[0])
            embeds = self.wrapper.prepare_inputs_embeds(
                ids,
                [seq.point_features_cached],
                layouts=[seq.input_embed_layout_cached],
            )
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
        # Keep paged prefill on the same varlen path as mixed prefill+decode.
        # PagedLlamaAttention expects a flattened [1, total_tokens, hidden]
        # stream, which run_mixed prepares for both pure and mixed batches.
        return self.run_mixed(seqs, [])

    @torch.inference_mode()
    def _run_prefill_padded_legacy(self, seqs: list[PointLLMSequence]) -> list[int]:
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
        context_lens = torch.tensor(seq_lens_list, dtype=torch.int32, device=device)
        max_blocks = max(len(seq.block_table) for seq in seqs)
        block_tables = torch.tensor(
            [seq.block_table + [-1] * (max_blocks - len(seq.block_table)) for seq in seqs],
            dtype=torch.int32, device=device,
        )
        set_forward_context(ForwardContext(
            slot_mapping=slot_mapping,
            block_tables=block_tables,
            context_lens=context_lens,
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

        return self.sampler(out.logits, seqs).tolist()

    # ── Decode ───────────────────────────────────────────────────────────────

    @torch.inference_mode()
    def run_decode(self, seqs: list[PointLLMSequence]) -> list[int]:
        B = len(seqs)
        try:
            device = next(self.hf_model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

        max_blocks = max(len(seq.block_table) for seq in seqs)
        metadata = self._get_decode_metadata_buffer(
            device=device,
            batch_size=B,
            max_blocks=max_blocks,
        ).prepare_decode(seqs, block_size=self.block_size)

        def _forward_logits(input_ids: torch.Tensor, ctx: ForwardContext) -> torch.Tensor:
            set_forward_context(ctx)
            try:
                if self.lightweight_runner is not None:
                    return self.lightweight_runner(
                        input_ids=input_ids,
                        position_ids=ctx.position_ids,
                        attention_mask=None,
                    )
                return self.hf_model(
                    input_ids=input_ids,
                    attention_mask=None,
                    use_cache=False,
                    return_dict=True,
                ).logits
            finally:
                clear_forward_context()

        graph_key = DecodeGraphKey(
            bucket_size=metadata.bucket_size,
            max_blocks_per_seq=metadata.bucket_context.block_tables.shape[1],
            vocab_size=int(getattr(getattr(self.hf_model, "config", None), "vocab_size", 0)),
            dtype=next(self.hf_model.parameters()).dtype,
            device=device,
        )
        if self._decode_graphs.enabled:
            logits = self._decode_graphs.run(
                graph_key,
                lambda: _forward_logits(metadata.bucket_input_ids, metadata.bucket_context),
            )
            logits = logits[:B]
        else:
            logits = _forward_logits(metadata.input_ids, metadata.context)
        if logits.dim() == 3 and logits.shape[0] == 1:
            logits = logits[0, :B, :]

        return self._sample_logits(logits, seqs).tolist()

    @torch.inference_mode()
    def run_mixed(
        self,
        prefill_seqs: list[PointLLMSequence],
        decode_seqs: list[PointLLMSequence],
    ) -> list[Optional[int]]:
        """
        Run one lightweight LLaMA pass over prefill new-tokens + decode last-tokens.

        The HuggingFace model-level ``forward`` is used only when the explicit
        lightweight-runtime fallback is disabled.
        Returns one next token_id per sequence (prefill_seqs first, then decode_seqs).
        """
        if not prefill_seqs and decode_seqs:
            return self.run_decode(decode_seqs)

        all_seqs = prefill_seqs + decode_seqs
        if not all_seqs:
            return []

        self._encode_point_clouds_batch(prefill_seqs)
        try:
            device = next(self.hf_model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

        emb_parts, pos_parts, slot_parts = [], [], []
        seq_lens_list, context_lens_list, block_tables_list = [], [], []
        prefill_is_final: list[bool] = []

        for seq in prefill_seqs:
            new_start = seq.num_cached_tokens
            chunk_end = seq.prefill_chunk_end or seq.num_tokens
            new_len   = chunk_end - new_start
            seq_lens_list.append(new_len)
            context_lens_list.append(chunk_end)
            block_tables_list.append(seq.block_table)
            prefill_is_final.append(chunk_end >= seq.num_tokens)

            # Embedding: only new tokens (prefix already in pool)
            if seq.inputs_embeds_cached is not None:
                emb = seq.inputs_embeds_cached[new_start:chunk_end]
            else:
                ids = torch.tensor(seq.token_ids[new_start:chunk_end], dtype=torch.long, device=device)
                emb = self.wrapper.get_input_embeddings()(ids.unsqueeze(0)).squeeze(0)
            emb_parts.append(emb)

            # Positions: [new_start .. chunk_end-1]
            pos_parts.append(torch.arange(new_start, chunk_end,
                                          dtype=torch.long, device=device))

            # Slots: new positions only → physical slot
            slots = []
            for abs_pos in range(new_start, chunk_end):
                blk = abs_pos // self.block_size
                off = abs_pos % self.block_size
                slots.append(seq.block_table[blk] * self.block_size + off)
            slot_parts.append(torch.tensor(slots, dtype=torch.int32, device=device))

        for seq in decode_seqs:
            seq_lens_list.append(1)
            context_lens_list.append(seq.num_tokens)
            block_tables_list.append(seq.block_table)

            ids = torch.tensor([seq.last_token], dtype=torch.long, device=device)
            emb_parts.append(self.wrapper.get_input_embeddings()(ids.unsqueeze(0)).squeeze(0))

            pos = seq.num_tokens - 1
            slot = seq.block_table[pos // self.block_size] * self.block_size + pos % self.block_size
            pos_parts.append(torch.tensor([pos],  dtype=torch.long,  device=device))
            slot_parts.append(torch.tensor([slot], dtype=torch.int32, device=device))

        total_tokens  = sum(seq_lens_list)
        inputs_embeds = torch.cat(emb_parts).unsqueeze(0)            # [1, T, H]
        slot_mapping  = torch.cat(slot_parts)                        # [T]
        position_ids  = torch.cat(pos_parts).unsqueeze(0)            # [1, T]
        max_blocks    = max(len(bt) for bt in block_tables_list)
        block_tables  = torch.tensor(
            [bt + [-1] * (max_blocks - len(bt)) for bt in block_tables_list],
            dtype=torch.int32, device=device,
        )
        context_lens  = torch.tensor(context_lens_list, dtype=torch.int32, device=device)

        set_forward_context(ForwardContext(
            slot_mapping=slot_mapping,
            block_tables=block_tables,
            context_lens=context_lens,
            position_ids=position_ids,
            seq_lens=seq_lens_list,
        ))
        try:
            if self.lightweight_runner is not None:
                logits = self.lightweight_runner(
                    inputs_embeds=inputs_embeds,
                    position_ids=position_ids,
                    attention_mask=None,
                )
            else:
                dummy_ids = torch.zeros(1, total_tokens, dtype=torch.long, device=device)
                logits = self.hf_model(
                    input_ids=dummy_ids,
                    inputs_embeds=inputs_embeds,
                    attention_mask=None,
                    point_clouds=None,
                    use_cache=False,
                    return_dict=True,
                ).logits
        finally:
            clear_forward_context()

        # [1, total_tokens, vocab] for both lightweight and explicit HF fallback.
        last_positions = [sum(seq_lens_list[:i+1]) - 1 for i in range(len(all_seqs))]
        idx = torch.tensor(last_positions, dtype=torch.long, device=device)
        sampled = self._sample_logits(logits[0, idx, :], all_seqs).tolist()
        results: list[Optional[int]] = []
        for seq, token_id, is_final in zip(prefill_seqs, sampled[:len(prefill_seqs)], prefill_is_final):
            seq.num_cached_tokens = seq.prefill_chunk_end or seq.num_tokens
            results.append(token_id if is_final else None)
        results.extend(sampled[len(prefill_seqs):])
        return results

    def run(self, seqs: list[PointLLMSequence], is_prefill: bool) -> list[int]:
        if is_prefill:
            return self.run_prefill(seqs)
        return self.run_decode(seqs)
