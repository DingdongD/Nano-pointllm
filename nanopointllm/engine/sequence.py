from __future__ import annotations

from enum import Enum, auto
from typing import Any, Optional

import torch

from nanopointllm.sampling_params import SamplingParams


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class PointLLMSequence:
    """单个推理请求的完整状态。"""

    def __init__(
        self,
        token_ids: list[int],
        point_clouds: Optional[Any] = None,
        point_cloud_cache_key: Optional[str] = None,
        point_features_cached: Optional[torch.Tensor] = None,
        sampling_params: Optional[SamplingParams] = None,
    ):
        self.token_ids: list[int] = list(token_ids)
        self.point_clouds = point_clouds
        self.sampling_params = sampling_params or SamplingParams()

        self.status = SequenceStatus.WAITING
        self.num_prompt_tokens: int = len(token_ids)

        # 点云编码后缓存（prefill 阶段填充，decode 阶段不再用点云）
        self.point_features_cached: Optional[torch.Tensor] = point_features_cached
        self.point_cloud_cache_key: Optional[str] = point_cloud_cache_key
        self.input_embed_layout_cached: Optional[Any] = None
        self.inputs_embeds_cached: Optional[torch.Tensor] = None

        # HF DynamicCache（prefill 后持有）
        self.past_key_values: Optional[Any] = None

        # 分页 KV 元数据（Task 6 启用）
        self.block_table: list[int] = []
        self.num_cached_tokens: int = 0
        self.prefill_chunk_end: Optional[int] = None

        # Multimodal seqs must not share KV blocks via prefix cache:
        # same token_ids can produce different embeddings (different point clouds).
        self.disable_prefix_cache: bool = point_clouds is not None

    @property
    def num_tokens(self) -> int:
        return len(self.token_ids)

    @property
    def num_completion_tokens(self) -> int:
        return self.num_tokens - self.num_prompt_tokens

    @property
    def last_token(self) -> int:
        assert self.token_ids, "PointLLMSequence.last_token called on empty token_ids"
        return self.token_ids[-1]

    @property
    def is_finished(self) -> bool:
        return self.status == SequenceStatus.FINISHED

    @property
    def max_tokens(self) -> int:
        return self.sampling_params.max_tokens

    @property
    def temperature(self) -> float:
        return self.sampling_params.temperature

    @property
    def ignore_eos(self) -> bool:
        return self.sampling_params.ignore_eos

    def append_token(self, token_id: int) -> None:
        self.token_ids.append(token_id)

    @property
    def kv_seq_len(self) -> int:
        """当前 KV cache 对应的序列长度。"""
        if self.past_key_values is None:
            return 0
        cache = self.past_key_values
        if hasattr(cache, "get_seq_length"):
            seq_len = int(cache.get_seq_length())
            if seq_len > 0:
                return seq_len
        # Fallback: support legacy key_cache / value_cache list attribute
        # (used by tests and older transformers versions)
        if hasattr(cache, "key_cache") and cache.key_cache:
            return int(cache.key_cache[0].shape[-2])
        if isinstance(cache, (tuple, list)) and len(cache) > 0:
            k0 = cache[0][0]
            return int(k0.shape[-2])
        return 0
