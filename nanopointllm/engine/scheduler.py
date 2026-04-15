"""
FIFO 两队列调度器：prefill 优先，prefill 队列空时切 decode。
不依赖分页 KV（Task 6 之前用 per-sequence DynamicCache）。
"""
from __future__ import annotations

from collections import deque
from typing import Optional

from nanopointllm.engine.sequence import PointLLMSequence, SequenceStatus


class Scheduler:

    def __init__(
        self,
        max_num_seqs: int,
        max_num_batched_tokens: int,
        eos_token_id: int = 0,
        block_manager=None,   # Optional[BlockManager]；None = 不做分页 KV 管理（Task 6 前）
    ) -> None:
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.eos_token_id = eos_token_id
        self.block_manager = block_manager
        self.waiting: deque[PointLLMSequence] = deque()
        self.running: list[PointLLMSequence] = []

    def is_finished(self) -> bool:
        return not self.waiting and not self.running

    def add(self, seq: PointLLMSequence) -> None:
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[PointLLMSequence], list[PointLLMSequence]]:
        """
        Returns (prefill_seqs, decode_seqs).

        Prefill seqs: admitted from waiting this step.
          - block_manager.can_allocate(seq) must pass
          - token budget: sum(seq.num_tokens - seq.num_cached_tokens) <= max_num_batched_tokens
          - count: total running seqs <= max_num_seqs
        Decode seqs: all currently running seqs not being prefilled this step.
        Both lists may be empty simultaneously only when engine is finished.
        """
        bm = self.block_manager
        prefill_seqs: list[PointLLMSequence] = []
        token_budget = 0

        while self.waiting:
            seq = self.waiting[0]
            if len(self.running) >= self.max_num_seqs:
                break
            if bm is not None and not bm.can_allocate(seq):
                break
            new_tokens = seq.num_tokens - seq.num_cached_tokens
            if token_budget + new_tokens > self.max_num_batched_tokens:
                break
            token_budget += new_tokens
            self.waiting.popleft()
            if bm is not None:
                bm.allocate(seq)
            seq.status = SequenceStatus.RUNNING
            self.running.append(seq)
            prefill_seqs.append(seq)

        prefill_set = set(id(s) for s in prefill_seqs)
        decode_seqs = [s for s in self.running if id(s) not in prefill_set]

        if bm is not None:
            for seq in decode_seqs:
                if bm.can_append(seq):
                    bm.may_append(seq)

        return prefill_seqs, decode_seqs

    def postprocess(
        self,
        seqs: list[PointLLMSequence],
        token_ids: list[int],
    ) -> None:
        """将新 token 追加到序列并检查终止条件。"""
        bm = self.block_manager
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)
            finished = (
                (not seq.ignore_eos and token_id == self.eos_token_id)
                or seq.num_completion_tokens >= seq.max_tokens
            )
            if finished:
                seq.status = SequenceStatus.FINISHED
                if bm is not None:
                    bm.deallocate(seq)
                if seq in self.running:
                    self.running.remove(seq)
