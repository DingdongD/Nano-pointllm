from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch


@dataclass(frozen=True)
class DecodeGraphKey:
    bucket_size: int
    max_blocks_per_seq: int
    vocab_size: int
    dtype: torch.dtype
    device: torch.device


class DecodeGraphBucket:
    """Strict CUDA graph replay wrapper for fixed-shape decode buckets."""

    def __init__(self, key: DecodeGraphKey) -> None:
        self.key = key
        self.graph: torch.cuda.CUDAGraph | None = None
        self.static_logits: torch.Tensor | None = None

    @property
    def captured(self) -> bool:
        return self.graph is not None and self.static_logits is not None

    def capture(self, forward: Callable[[], torch.Tensor]) -> torch.Tensor:
        if self.key.device.type != "cuda":
            raise RuntimeError("strict CUDA graph decode requires a CUDA device")
        if not torch.cuda.is_available():
            raise RuntimeError("strict CUDA graph decode requires CUDA")

        # Compile/warm all kernels before capture.  This is strict: errors are
        # allowed to propagate instead of falling back to eager execution.
        for _ in range(2):
            forward()
        torch.cuda.synchronize(self.key.device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            logits = forward()
        self.graph = graph
        self.static_logits = logits
        return logits

    def replay(self) -> torch.Tensor:
        if not self.captured:
            raise RuntimeError("decode graph bucket has not been captured")
        assert self.graph is not None
        assert self.static_logits is not None
        self.graph.replay()
        return self.static_logits


class DecodeGraphCache:
    """Strict decode graph cache keyed by static bucket shape."""

    def __init__(self, *, enabled: bool = False) -> None:
        self.enabled = enabled
        self._buckets: dict[DecodeGraphKey, DecodeGraphBucket] = {}

    def get(self, key: DecodeGraphKey) -> DecodeGraphBucket:
        bucket = self._buckets.get(key)
        if bucket is None:
            bucket = DecodeGraphBucket(key)
            self._buckets[key] = bucket
        return bucket

    def run(self, key: DecodeGraphKey, forward: Callable[[], torch.Tensor]) -> torch.Tensor:
        if not self.enabled:
            raise RuntimeError("decode CUDA graph cache is disabled")
        bucket = self.get(key)
        if bucket.captured:
            return bucket.replay()
        return bucket.capture(forward)
