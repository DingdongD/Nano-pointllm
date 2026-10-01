from __future__ import annotations

from dataclasses import dataclass

import torch

from nanopointllm.engine.forward_context import ForwardContext
from nanopointllm.engine.sequence import PointLLMSequence


def next_bucket_size(n: int, buckets: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64)) -> int:
    for bucket in buckets:
        if n <= bucket:
            return bucket
    return n


@dataclass
class DecodeMetadataView:
    input_ids: torch.Tensor
    position_ids: torch.Tensor
    context: ForwardContext
    bucket_input_ids: torch.Tensor
    bucket_position_ids: torch.Tensor
    bucket_context: ForwardContext
    batch_size: int
    bucket_size: int


class DecodeMetadataStagingBuffer:
    """Reusable device buffers for paged decode metadata.

    The active views are sliced to the real batch size for normal execution.
    The full bucket-sized tensors are retained so a future CUDA graph path can
    capture fixed-shape decode buckets without changing scheduler output.
    """

    def __init__(
        self,
        *,
        device: torch.device,
        max_batch_size: int,
        max_blocks_per_seq: int,
    ) -> None:
        self.device = device
        self.max_batch_size = max_batch_size
        self.max_blocks_per_seq = max_blocks_per_seq
        self.input_ids = torch.zeros(1, max_batch_size, dtype=torch.long, device=device)
        self.position_ids = torch.zeros(1, max_batch_size, dtype=torch.long, device=device)
        self.slot_mapping = torch.full((max_batch_size,), -1, dtype=torch.int32, device=device)
        self.context_lens = torch.zeros(max_batch_size, dtype=torch.int32, device=device)
        self.block_tables = torch.full(
            (max_batch_size, max_blocks_per_seq),
            -1,
            dtype=torch.int32,
            device=device,
        )

    def can_fit(self, batch_size: int, max_blocks_per_seq: int) -> bool:
        return batch_size <= self.max_batch_size and max_blocks_per_seq <= self.max_blocks_per_seq

    def prepare_decode(
        self,
        seqs: list[PointLLMSequence],
        *,
        block_size: int,
    ) -> DecodeMetadataView:
        batch_size = len(seqs)
        if batch_size == 0:
            raise ValueError("prepare_decode requires at least one sequence")
        max_blocks = max(len(seq.block_table) for seq in seqs)
        if not self.can_fit(batch_size, max_blocks):
            raise ValueError(
                f"decode metadata buffer too small: batch={batch_size}, blocks={max_blocks}, "
                f"capacity=({self.max_batch_size}, {self.max_blocks_per_seq})"
            )

        active_blocks = self.block_tables[:batch_size, :max_blocks]
        self.input_ids[:self.max_batch_size].zero_()
        self.position_ids[:self.max_batch_size].zero_()
        self.slot_mapping[:self.max_batch_size].fill_(-1)
        self.context_lens[:self.max_batch_size].zero_()
        active_blocks.fill_(-1)
        for i, seq in enumerate(seqs):
            pos = seq.num_tokens - 1
            blk = pos // block_size
            off = pos % block_size
            if blk >= len(seq.block_table):
                raise RuntimeError(
                    f"sequence needs block {blk}, but block_table has {len(seq.block_table)}"
                )
            self.input_ids[0, i] = seq.last_token
            self.position_ids[0, i] = pos
            self.slot_mapping[i] = seq.block_table[blk] * block_size + off
            self.context_lens[i] = seq.num_tokens
            active_blocks[i, :len(seq.block_table)] = torch.tensor(
                seq.block_table,
                dtype=torch.int32,
                device=self.device,
            )

        ctx = ForwardContext(
            slot_mapping=self.slot_mapping[:batch_size],
            block_tables=active_blocks,
            context_lens=self.context_lens[:batch_size],
            position_ids=self.position_ids[:, :batch_size],
            seq_lens=[1] * batch_size,
        )
        bucket_ctx = ForwardContext(
            slot_mapping=self.slot_mapping[:self.max_batch_size],
            block_tables=self.block_tables[:self.max_batch_size, :max_blocks],
            context_lens=self.context_lens[:self.max_batch_size],
            position_ids=self.position_ids[:, :self.max_batch_size],
            seq_lens=[1] * self.max_batch_size,
        )
        return DecodeMetadataView(
            input_ids=self.input_ids[:, :batch_size],
            position_ids=self.position_ids[:, :batch_size],
            context=ctx,
            bucket_input_ids=self.input_ids[:, :self.max_batch_size],
            bucket_position_ids=self.position_ids[:, :self.max_batch_size],
            bucket_context=bucket_ctx,
            batch_size=batch_size,
            bucket_size=self.max_batch_size,
        )
