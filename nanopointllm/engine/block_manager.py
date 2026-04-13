"""
分页 KV cache 的 block 元数据管理（参考 nano-vllm block_manager.py）。
用 hashlib.shake_128 代替 xxhash（无额外依赖）；语义保持一致。
"""
from __future__ import annotations

import hashlib
import struct
from collections import deque

import numpy as np

from nanopointllm.engine.sequence import PointLLMSequence


class Block:

    def __init__(self, block_id: int) -> None:
        self.block_id = block_id
        self.ref_count = 0
        self.hash: int = -1      # -1 = 不可复用
        self.token_ids: list[int] = []

    def update(self, hash_val: int, token_ids: list[int]) -> None:
        self.hash = hash_val
        self.token_ids = list(token_ids)

    def reset(self) -> None:
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:

    def __init__(self, num_blocks: int, block_size: int) -> None:
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = {}
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1) -> int:
        h = hashlib.shake_128()
        if prefix != -1:
            h.update(struct.pack("<q", prefix))
        h.update(np.array(token_ids, dtype=np.int64).tobytes())
        return int.from_bytes(h.digest(8), "little")

    def _allocate_block(self, block_id: int) -> Block:
        block = self.blocks[block_id]
        assert block.ref_count == 0
        block.reset()
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return block

    def _deallocate_block(self, block_id: int) -> None:
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.discard(block_id)
        self.free_block_ids.append(block_id)

    def _num_blocks_for(self, seq: PointLLMSequence) -> int:
        return (seq.num_tokens + self.block_size - 1) // self.block_size

    def can_allocate(self, seq: PointLLMSequence) -> bool:
        return len(self.free_block_ids) >= self._num_blocks_for(seq)

    def allocate(self, seq: PointLLMSequence) -> None:
        """Prefill 调度时调用；尝试 prefix cache 命中，否则新分配。"""
        assert not seq.block_table
        n_blocks = self._num_blocks_for(seq)
        h = -1
        cache_miss = False
        for i in range(n_blocks):
            start = i * self.block_size
            end = min(start + self.block_size, seq.num_tokens)
            chunk = seq.token_ids[start:end]
            is_full = len(chunk) == self.block_size
            h_new = self.compute_hash(chunk, h) if is_full else -1

            block_id_cached = self.hash_to_block_id.get(h_new, -1)
            if (
                not cache_miss
                and h_new != -1
                and block_id_cached != -1
                and self.blocks[block_id_cached].token_ids == chunk
            ):
                # Prefix cache 命中：复用
                seq.num_cached_tokens += self.block_size
                bid = block_id_cached
                if bid in self.used_block_ids:
                    self.blocks[bid].ref_count += 1
                else:
                    self._allocate_block(bid)
            else:
                cache_miss = True
                bid = self.free_block_ids[0]
                block = self._allocate_block(bid)
                if h_new != -1:
                    block.update(h_new, chunk)
                    self.hash_to_block_id[h_new] = bid

            h = h_new
            seq.block_table.append(bid)

    def deallocate(self, seq: PointLLMSequence) -> None:
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: PointLLMSequence) -> bool:
        """新 token 是否需要新 block，以及是否有空闲 block。"""
        needs_new_block = (seq.num_tokens % self.block_size) == 0
        return not needs_new_block or len(self.free_block_ids) >= 1

    def may_append(self, seq: PointLLMSequence) -> None:
        """
        Decode 前元数据维护：
        - 若需要新 block（token 数是 block_size 的整倍数），分配之
        - 若前一个 block 刚好填满，写入 hash 供 prefix cache
        """
        n_tokens = seq.num_tokens
        n_blocks_needed = (n_tokens + self.block_size - 1) // self.block_size
        if n_blocks_needed > len(seq.block_table):
            bid = self.free_block_ids[0]
            self._allocate_block(bid)
            seq.block_table.append(bid)
        # 如果前一个 block 刚好满了，写入 hash
        last_idx = len(seq.block_table) - 2
        if last_idx >= 0 and (n_tokens - 1) % self.block_size == 0:
            prev_bid = seq.block_table[last_idx]
            block = self.blocks[prev_bid]
            if block.hash == -1:
                chunk_start = last_idx * self.block_size
                chunk = seq.token_ids[chunk_start: chunk_start + self.block_size]
                prev_hash = self.blocks[seq.block_table[last_idx - 1]].hash if last_idx > 0 else -1
                h = self.compute_hash(chunk, prev_hash)
                block.update(h, chunk)
                self.hash_to_block_id[h] = prev_bid
