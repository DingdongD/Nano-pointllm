"""
PointLLMWrapper：把 PointLLMLlamaForCausalLM 包装成
  * prepare_inputs_embeds(input_ids, point_features) → inputs_embeds
  * encode_point_clouds(point_clouds) → list[Tensor]
两步接口，供多 batch 推理引擎分阶段调用。

逻辑来源：/home/PointLLM/pointllm/model/pointllm.py PointLLMLlamaModel.forward L112-L171
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn


@dataclass(frozen=True)
class PointInputLayout:
    has_point_tokens: bool
    use_start_end: bool
    start_idx: int = -1
    start_positions: tuple[int, ...] = ()


class PointLLMWrapper:
    """
    轻量封装层；hf_model 须为 PointLLMLlamaForCausalLM（或 duck-type 兼容对象）。
    不改变 hf_model 的权重或状态。
    """

    def __init__(self, hf_model: nn.Module) -> None:
        self.hf_model = hf_model

    @property
    def inner(self) -> nn.Module:
        """PointLLMLlamaModel（backbone）。"""
        if hasattr(self.hf_model, "get_model") and self.hf_model.get_model() is not None:
            return self.hf_model.get_model()
        return self.hf_model.model

    @property
    def lm_head(self) -> nn.Module:
        return self.hf_model.lm_head

    def get_input_embeddings(self) -> nn.Embedding:
        return self.inner.embed_tokens

    @staticmethod
    def point_cloud_cache_key(point_cloud: torch.Tensor) -> str:
        pc = point_cloud.detach()
        if pc.dim() == 3 and pc.shape[0] == 1:
            pc = pc.squeeze(0)
        pc = pc.contiguous().to(device="cpu", dtype=torch.float32)
        digest = hashlib.sha1(pc.numpy().tobytes()).hexdigest()
        return f"{tuple(pc.shape)}:{str(pc.dtype)}:{digest}"

    def analyze_input_layout(self, input_ids: torch.Tensor) -> PointInputLayout:
        inner = self.inner
        cfg = inner.point_backbone_config
        use_start_end: bool = cfg.get("mm_use_point_start_end", False)
        patch_token_id: int = cfg["point_patch_token"]

        if input_ids.dim() != 1:
            raise ValueError(f"analyze_input_layout expects 1D token ids, got {tuple(input_ids.shape)}")

        if use_start_end:
            start_token_id: int = cfg["point_start_token"]
            starts = tuple(torch.where(input_ids == start_token_id)[0].tolist())
            return PointInputLayout(
                has_point_tokens=len(starts) > 0,
                use_start_end=True,
                start_positions=starts,
            )

        masked = torch.where(input_ids == patch_token_id)[0]
        if masked.numel() == 0:
            return PointInputLayout(has_point_tokens=False, use_start_end=False)
        return PointInputLayout(
            has_point_tokens=True,
            use_start_end=False,
            start_idx=int(masked[0]),
        )

    @torch.inference_mode()
    def encode_point_clouds(self, point_clouds: Any) -> list[torch.Tensor]:
        """
        批量编码点云，返回投影后的 patch token 特征列表。

        Args:
            point_clouds: Tensor [B, N, C] 或 list of Tensor [N, C]

        Returns:
            list of Tensor，每个 shape = (point_token_len, hidden_size)
        """
        inner = self.inner
        backbone = inner.point_backbone
        proj = inner.point_proj

        if isinstance(point_clouds, list):
            raw_features = []
            for pc in point_clouds:
                pc_3d = pc.unsqueeze(0) if pc.dim() == 2 else pc   # [1, N, C]
                feat = backbone(pc_3d)                               # [1, token_len, backbone_dim]
                raw_features.append(feat[0])                         # [token_len, backbone_dim]
        else:
            if point_clouds.dim() == 2:
                point_clouds = point_clouds.unsqueeze(0)
            raw = backbone(point_clouds)                             # [B, token_len, backbone_dim]
            raw_features = [raw[i] for i in range(raw.shape[0])]

        return [proj(f) for f in raw_features]                       # list of [token_len, hidden_size]

    def prepare_inputs_embeds(
        self,
        input_ids: torch.Tensor,
        point_features: Optional[list[torch.Tensor]],
        layouts: Optional[list[PointInputLayout]] = None,
    ) -> torch.Tensor:
        """
        将文本 token embeddings 与点云 patch features 融合，返回 inputs_embeds。

        Args:
            input_ids: [B, L]  LongTensor
            point_features: list of [point_token_len, hidden_size]，长度=B；
                            或 None（decode 步 / 纯文本样本）

        Returns:
            inputs_embeds: [B, L, hidden_size]
        """
        inner = self.inner
        inputs_embeds = inner.embed_tokens(input_ids)   # [B, L, H]

        if point_features is None:
            return inputs_embeds

        cfg = inner.point_backbone_config
        if layouts is None:
            layouts = [self.analyze_input_layout(row) for row in input_ids]
        if len(layouts) != input_ids.shape[0]:
            raise ValueError("layouts length must match batch size")

        new_embeds = []
        for i, (cur_ids, cur_emb, layout) in enumerate(zip(input_ids, inputs_embeds, layouts)):
            cur_feat = point_features[i].to(device=cur_emb.device, dtype=cur_emb.dtype)
            new_embeds.append(self.splice_point_features(cur_emb, cur_feat, layout))

        return torch.stack(new_embeds)   # [B, L, H]

    def splice_point_features(
        self,
        token_embeds: torch.Tensor,
        point_features: torch.Tensor,
        layout: PointInputLayout,
    ) -> torch.Tensor:
        num_patches = point_features.shape[0]
        if not layout.has_point_tokens:
            return token_embeds
        if layout.use_start_end:
            cur_new = token_embeds
            for pos in reversed(layout.start_positions):
                cur_new = torch.cat([
                    cur_new[:pos + 1],
                    point_features,
                    cur_new[pos + num_patches + 1:],
                ])
            return cur_new
        return torch.cat([
            token_embeds[:layout.start_idx],
            point_features,
            token_embeds[layout.start_idx + num_patches:],
        ])
