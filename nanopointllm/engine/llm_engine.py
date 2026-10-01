"""
PointLLMLLMEngine：把 Scheduler + ModelRunner 串成完整推理循环。

当 num_kvcache_blocks 未指定时使用 PointLLMModelRunner（padded KV，原有行为）。
当 num_kvcache_blocks 指定时使用 PagedModelRunner（物理 KV 池，分页 decode）。

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

from nanopointllm.engine.point_feature_cache import PointFeatureCache
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
        num_kvcache_blocks: Optional[int] = None,
        kvcache_block_size: int = 16,
        max_prefill_chunk_tokens: Optional[int] = None,
        point_feature_cache_size: int = 128,
    ) -> None:
        self.point_feature_cache = PointFeatureCache(max_entries=point_feature_cache_size)
        if num_kvcache_blocks is not None:
            import torch
            from nanopointllm.engine.block_manager import BlockManager
            from nanopointllm.engine.kv_pool import allocate_kv_pool
            from nanopointllm.engine.paged_model_runner import PagedModelRunner
            from nanopointllm.llama.inject_paged import inject_paged_attention

            device = next(hf_model.parameters()).device
            dtype  = next(hf_model.parameters()).dtype
            block_manager = BlockManager(
                num_blocks=num_kvcache_blocks,
                block_size=kvcache_block_size,
            )
            kv_pool = allocate_kv_pool(
                hf_model, num_kvcache_blocks, kvcache_block_size, device, dtype,
            )
            inject_paged_attention(hf_model, kv_pool)
            self.runner = PagedModelRunner(
                hf_model,
                kv_pool,
                block_manager,
                point_feature_cache=self.point_feature_cache,
            )
        else:
            from nanopointllm.engine.model_runner import PointLLMModelRunner
            block_manager = None
            self.runner = PointLLMModelRunner(
                hf_model=hf_model,
                point_feature_cache=self.point_feature_cache,
            )

        self.scheduler = Scheduler(
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
            eos_token_id=eos_token_id,
            block_manager=block_manager,
            max_prefill_chunk_tokens=max_prefill_chunk_tokens,
        )

    def is_finished(self) -> bool:
        return self.scheduler.is_finished()

    def add_request(
        self,
        token_ids: list[int],
        point_clouds: Optional[Any] = None,
        point_cloud_cache_key: Optional[str] = None,
        point_features_cached: Optional[Any] = None,
        sampling_params: Optional[SamplingParams] = None,
    ) -> PointLLMSequence:
        seq = PointLLMSequence(
            token_ids=token_ids,
            point_clouds=point_clouds,
            point_cloud_cache_key=point_cloud_cache_key,
            point_features_cached=point_features_cached,
            sampling_params=sampling_params or SamplingParams(),
        )
        self.scheduler.add(seq)
        return seq

    def get_point_feature_cache_stats(self) -> dict[str, int]:
        stats = self.point_feature_cache.stats
        return {
            "hits": stats.hits,
            "misses": stats.misses,
            "inserts": stats.inserts,
            "evictions": stats.evictions,
            "size": len(self.point_feature_cache),
        }

    def clear_point_feature_cache(self) -> None:
        self.point_feature_cache.clear()

    def step(self) -> None:
        """Execute one scheduling round (prefill or decode) and update sequences."""
        prefill_seqs, decode_seqs = self.scheduler.schedule()
        if not prefill_seqs and not decode_seqs:
            return
        token_ids = self.runner.run_mixed(prefill_seqs, decode_seqs)
        self.scheduler.postprocess(prefill_seqs + decode_seqs, token_ids)

    def generate(
        self,
        requests: list[dict],
    ) -> list[PointLLMSequence]:
        """
        Batch inference: submit all requests then loop step() until done.

        Args:
            requests: list of dict passed to add_request:
                      {"token_ids": [...], "point_clouds": Tensor, "sampling_params": ...}

        Returns:
            list[PointLLMSequence] in request order, all is_finished==True
        """
        seqs = [self.add_request(**req) for req in requests]
        while not self.is_finished():
            self.step()
        return seqs
