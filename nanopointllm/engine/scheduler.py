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
        eos_token_id: int,
        block_manager=None,   # Optional[BlockManager]；None = 不做分页 KV 管理（Task 6 前）
    ) -> None:
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.eos_token_id = eos_token_id
        self.block_manager = block_manager
        self.waiting: deque[PointLLMSequence] = deque()
        self.running: deque[PointLLMSequence] = deque()

    def is_finished(self) -> bool:
        return not self.waiting and not self.running

    def add(self, seq: PointLLMSequence) -> None:
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[PointLLMSequence], bool]:
        """
        返回 (scheduled_seqs, is_prefill)。
        is_prefill=True  → 这批序列需要走 prefill（含点云编码）。
        is_prefill=False → 这批序列走 decode 步。
        """
        bm = self.block_manager

        # ── Prefill 阶段：从 waiting 拉入新请求 ──
        prefill_seqs: list[PointLLMSequence] = []
        total_tokens = 0
        while self.waiting and len(prefill_seqs) < self.max_num_seqs:
            seq = self.waiting[0]
            if bm is not None and not bm.can_allocate(seq):
                break
            uncached = seq.num_tokens - seq.num_cached_tokens
            if total_tokens + uncached > self.max_num_batched_tokens:
                break
            total_tokens += uncached
            self.waiting.popleft()
            if bm is not None:
                bm.allocate(seq)
            seq.status = SequenceStatus.RUNNING
            self.running.append(seq)
            prefill_seqs.append(seq)

        if prefill_seqs:
            return prefill_seqs, True

        # ── Decode 阶段：处理所有 running 序列 ──
        assert self.running, "schedule() called on empty engine"
        decode_seqs = list(self.running)

        if bm is not None:
            for seq in decode_seqs:
                if bm.can_append(seq):
                    bm.may_append(seq)

        return decode_seqs, False

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
