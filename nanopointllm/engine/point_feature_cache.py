from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import torch


@dataclass
class PointFeatureCacheStats:
    hits: int = 0
    misses: int = 0
    inserts: int = 0
    evictions: int = 0


class PointFeatureCache:
    def __init__(self, max_entries: int = 128) -> None:
        self.max_entries = max(1, int(max_entries))
        self._store: OrderedDict[str, torch.Tensor] = OrderedDict()
        self.stats = PointFeatureCacheStats()

    def get(self, key: str) -> torch.Tensor | None:
        value = self._store.get(key)
        if value is None:
            self.stats.misses += 1
            return None
        self._store.move_to_end(key)
        self.stats.hits += 1
        return value

    def put(self, key: str, value: torch.Tensor) -> None:
        if key in self._store:
            self._store[key] = value
            self._store.move_to_end(key)
            return
        if len(self._store) >= self.max_entries:
            self._store.popitem(last=False)
            self.stats.evictions += 1
        self._store[key] = value
        self.stats.inserts += 1

    def __contains__(self, key: str) -> bool:
        return key in self._store

    def __len__(self) -> int:
        return len(self._store)

    def clear(self) -> None:
        self._store.clear()
