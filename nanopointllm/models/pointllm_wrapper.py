"""
PointLLMWrapper：把 PointLLMLlamaForCausalLM 包装成
  * prepare_inputs_embeds(input_ids, point_features) → inputs_embeds
  * encode_point_clouds(point_clouds) → list[Tensor]
两步接口，供多 batch 推理引擎分阶段调用。

逻辑来源：/home/PointLLM/pointllm/model/pointllm.py PointLLMLlamaModel.forward L112-L171
"""
from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn


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
        use_start_end: bool = cfg.get("mm_use_point_start_end", False)
        patch_token_id: int = cfg["point_patch_token"]

        new_embeds = []
        for i, (cur_ids, cur_emb) in enumerate(zip(input_ids, inputs_embeds)):
            cur_feat = point_features[i].to(device=cur_emb.device, dtype=cur_emb.dtype)
            num_patches = cur_feat.shape[0]

            if (cur_ids == patch_token_id).sum() == 0:
                # 纯文本样本：无点云占位符
                new_embeds.append(cur_emb)
                continue

            if use_start_end:
                start_token_id: int = cfg["point_start_token"]
                start_positions = torch.where(cur_ids == start_token_id)[0]
                cur_new_emb = cur_emb
                for pos in reversed(start_positions):
                    cur_new_emb = torch.cat([
                        cur_new_emb[:pos + 1],
                        cur_feat,
                        cur_new_emb[pos + num_patches + 1:],
                    ])
                new_embeds.append(cur_new_emb)
            else:
                masked_indices = torch.where(cur_ids == patch_token_id)[0]
                start_idx = masked_indices[0]
                new_embeds.append(torch.cat([
                    cur_emb[:start_idx],
                    cur_feat,
                    cur_emb[start_idx + num_patches:],
                ]))

        return torch.stack(new_embeds)   # [B, L, H]
