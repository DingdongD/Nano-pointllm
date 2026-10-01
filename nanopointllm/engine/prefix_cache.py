from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PrefixCacheStats:
    lookups: int = 0
    hits: int = 0
    misses: int = 0
    inserts: int = 0


class PrefixCacheIndex:
    """Hash -> block id index with lightweight stats and eviction hooks."""

    def __init__(self, mapping: dict[int, int] | None = None) -> None:
        self.mapping = mapping if mapping is not None else {}
        self.stats = PrefixCacheStats()

    def lookup(self, hash_value: int) -> int:
        self.stats.lookups += 1
        block_id = self.mapping.get(hash_value, -1)
        if block_id == -1:
            self.stats.misses += 1
        else:
            self.stats.hits += 1
        return block_id

    def insert(self, hash_value: int, block_id: int) -> None:
        if hash_value == -1:
            return
        self.mapping[hash_value] = block_id
        self.stats.inserts += 1

    def evict(self, hash_value: int) -> None:
        self.mapping.pop(hash_value, None)

    def clear(self) -> None:
        self.mapping.clear()
