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
    ) -> None:
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
            self.runner = PagedModelRunner(hf_model, kv_pool, block_manager)
        else:
            from nanopointllm.engine.model_runner import PointLLMModelRunner
            block_manager = None
            self.runner = PointLLMModelRunner(hf_model=hf_model)

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
        """Execute one scheduling round (prefill or decode) and update sequences."""
        seqs, is_prefill = self.scheduler.schedule()
        token_ids = self.runner.run(seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids)

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
