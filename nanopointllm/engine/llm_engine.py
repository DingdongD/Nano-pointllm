"""
PointLLMLLMEngine：把 Scheduler + PointLLMModelRunner 串成完整推理循环。

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

from nanopointllm.engine.model_runner import PointLLMModelRunner
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
        # 分页 KV 参数（可选；Task 7 接入 BlockManager 时使用）
        num_kvcache_blocks: Optional[int] = None,
        kvcache_block_size: int = 16,
    ) -> None:
        self.runner = PointLLMModelRunner(hf_model=hf_model)

        block_manager = None
        if num_kvcache_blocks is not None:
            from nanopointllm.engine.block_manager import BlockManager
            block_manager = BlockManager(
                num_blocks=num_kvcache_blocks,
                block_size=kvcache_block_size,
            )

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
        """执行一轮调度（prefill 或 decode）并更新序列状态。"""
        seqs, is_prefill = self.scheduler.schedule()
        token_ids = self.runner.run(seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids)

    def generate(
        self,
        requests: list[dict],
    ) -> list[PointLLMSequence]:
        """
        批量推理接口：提交所有请求后循环 step 直到全部完成。

        Args:
            requests: list of dict，每个 dict 传给 add_request：
                      {"token_ids": [...], "point_clouds": Tensor, "sampling_params": ...}

        Returns:
            list[PointLLMSequence]，与 requests 顺序对应，seq.is_finished==True
        """
        seqs = [self.add_request(**req) for req in requests]
        while not self.is_finished():
            self.step()
        return seqs
