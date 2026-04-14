"""
PointLLMModelRunner：批量 prefill（点云编码 + HF forward）+ padded-KV 批量 decode。

批量 decode 策略（无分页 KV）：
  1. 取所有 running 序列的 KV cache，left-pad 到统一长度。
  2. 一次 HF forward 得到 [B, 1, vocab] logits 与更新后的 KV。
  3. 把新 KV 按序列维切回各自独立 cache。
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from nanopointllm.engine.hf_hybrid import hf_prefill
from nanopointllm.engine.sequence import PointLLMSequence
from nanopointllm.models.pointllm_wrapper import PointLLMWrapper


# ──────────────────────────────────────────────────────────────────────────────
# 内部工具：padded KV 合并 / 拆分
# ──────────────────────────────────────────────────────────────────────────────

def _cache_num_layers(cache) -> int:
    """transformers ≥5.5: DynamicCache.layers; <5.5: DynamicCache.key_cache."""
    if hasattr(cache, "key_cache"):
        return len(cache.key_cache)
    return len(cache.layers)


def _cache_get_kv(cache, layer_idx):
    """Return (key, value) tensors for layer_idx."""
    if hasattr(cache, "key_cache"):
        return cache.key_cache[layer_idx], cache.value_cache[layer_idx]
    layer = cache.layers[layer_idx]
    return layer.keys, layer.values


def _cache_append_layer(cache, k: torch.Tensor, v: torch.Tensor) -> None:
    """Append a (k, v) pair as a new layer to cache."""
    if hasattr(cache, "key_cache"):
        cache.key_cache.append(k)
        cache.value_cache.append(v)
        return
    from transformers.cache_utils import DynamicLayer
    layer = DynamicLayer()
    layer.lazy_initialization(k, v)
    layer.keys = k
    layer.values = v
    cache.layers.append(layer)


def _build_batched_cache(seqs: list[PointLLMSequence]):
    """
    把多个序列的 DynamicCache（各自 kv_len 可不同）left-pad 成
    [B, heads, max_kv_len, dim] 并返回合并后的 DynamicCache 与 attention_mask。

    Left-pad 使实际 token 靠右，配合 attention_mask 可正确 attend。
    """
    from transformers.cache_utils import DynamicCache

    kv_lens = [seq.kv_seq_len for seq in seqs]
    max_kv_len = max(kv_lens)
    num_layers = _cache_num_layers(seqs[0].past_key_values)
    B = len(seqs)

    batched_cache = DynamicCache()
    for layer_idx in range(num_layers):
        k_list, v_list = [], []
        for seq in seqs:
            k, v = _cache_get_kv(seq.past_key_values, layer_idx)  # [1, H, kv_len, D]
            seq_len = k.shape[-2]
            pad_len = max_kv_len - seq_len
            if pad_len > 0:
                pad_k = k.new_zeros(1, k.shape[1], pad_len, k.shape[-1])
                pad_v = v.new_zeros(1, v.shape[1], pad_len, v.shape[-1])
                k = torch.cat([pad_k, k], dim=-2)
                v = torch.cat([pad_v, v], dim=-2)
            k_list.append(k)
            v_list.append(v)
        _cache_append_layer(batched_cache, torch.cat(k_list, dim=0), torch.cat(v_list, dim=0))

    # attention_mask: [B, max_kv_len + 1]（+1 for new token）
    # 1 = valid，0 = padding
    mask_rows = []
    for kv_len in kv_lens:
        row = torch.zeros(max_kv_len + 1, dtype=torch.long)
        row[-(kv_len + 1):] = 1
        mask_rows.append(row)
    attention_mask = torch.stack(mask_rows)

    return batched_cache, attention_mask, max_kv_len


def _split_batched_cache(
    batched_cache,
    seqs: list[PointLLMSequence],
    orig_kv_lens: list[int],
) -> None:
    """
    从 [B, heads, max_kv_len+1, dim] 的合并 cache 中切出每个序列更新后的 slice。
    新序列 KV 长度 = 原 kv_len + 1（新 token 的 KV 已写入）。
    """
    from transformers.cache_utils import DynamicCache

    num_layers = _cache_num_layers(batched_cache)
    for i, (seq, kv_len) in enumerate(zip(seqs, orig_kv_lens)):
        new_cache = DynamicCache()
        for layer_idx in range(num_layers):
            k, v = _cache_get_kv(batched_cache, layer_idx)
            # 取 right side: 新 kv_len = kv_len + 1
            _cache_append_layer(new_cache,
                                k[i:i+1, :, -(kv_len+1):, :].contiguous(),
                                v[i:i+1, :, -(kv_len+1):, :].contiguous())
        seq.past_key_values = new_cache


def _split_prefill_cache(prefill_cache, seqs: list[PointLLMSequence]) -> None:
    """
    从 prefill 的 batched DynamicCache [B, H, max_len, D] 中
    切出每个序列的 KV（右侧 seq.num_tokens 行，因为使用了 left-padding）。
    """
    from transformers.cache_utils import DynamicCache
    num_layers = _cache_num_layers(prefill_cache)
    for i, seq in enumerate(seqs):
        L = seq.num_tokens
        new_cache = DynamicCache()
        for layer_idx in range(num_layers):
            k, v = _cache_get_kv(prefill_cache, layer_idx)
            _cache_append_layer(new_cache,
                                k[i:i+1, :, -L:, :].contiguous(),
                                v[i:i+1, :, -L:, :].contiguous())
        seq.past_key_values = new_cache


def _batch_hf_decode(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    past_key_values,
    **forward_kwargs,
):
    """单次 HF forward（decode 模式，无点云）。供测试 monkeypatch。"""
    import inspect
    allowed = set(inspect.signature(model.forward).parameters.keys())
    filtered = {k: v for k, v in forward_kwargs.items() if k in allowed}
    return model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        use_cache=True,
        return_dict=True,
        point_clouds=None,
        **filtered,
    )


# ──────────────────────────────────────────────────────────────────────────────
# ModelRunner
# ──────────────────────────────────────────────────────────────────────────────

class PointLLMModelRunner:

    def __init__(self, hf_model: nn.Module) -> None:
        self.hf_model = hf_model
        self.wrapper = PointLLMWrapper(hf_model)

    def _encode_point_clouds_batch(self, seqs: list[PointLLMSequence]) -> None:
        """
        对所有尚未编码的序列编码点云，缓存 inputs_embeds 到 seq.inputs_embeds_cached。

        若所有待编码序列的点云形状相同，则将其堆叠为 [B, N, C] 一次性送入 backbone，
        避免串行循环中重复执行 FPS 的 Python-level for 循环（512 次迭代 × B）。
        形状不同时退化为逐条编码。
        """
        pending = [
            seq for seq in seqs
            if seq.point_clouds is not None and seq.inputs_embeds_cached is None
        ]
        if not pending:
            return

        try:
            param = next(self.wrapper.inner.parameters())
            device = param.device if isinstance(param, torch.Tensor) else torch.device("cpu")
        except (StopIteration, TypeError, AttributeError):
            device = torch.device("cpu")

        # 归一化为 [N, C]，检查是否可以批量堆叠
        def _to_2d(pc: torch.Tensor) -> torch.Tensor:
            return pc.squeeze(0) if pc.dim() == 3 else pc

        pcs_2d = [_to_2d(seq.point_clouds).to(device) for seq in pending]
        shapes = {pc.shape for pc in pcs_2d}

        if len(shapes) == 1:
            # 全部相同形状 → 单次 backbone forward [B, N, C]
            stacked = torch.stack(pcs_2d)          # [B, N, C]
            all_features = self.wrapper.encode_point_clouds(stacked)  # list[B × Tensor]
        else:
            # 形状不一致 → 逐条编码（fallback）
            all_features = [
                self.wrapper.encode_point_clouds(pc)[0] for pc in pcs_2d
            ]

        for seq, feat in zip(pending, all_features):
            ids = torch.tensor([seq.token_ids], dtype=torch.long, device=device)
            embeds = self.wrapper.prepare_inputs_embeds(ids, [feat])  # [1, L, H]
            seq.inputs_embeds_cached = embeds.squeeze(0)              # [L, H]

    @torch.inference_mode()
    def run_prefill(self, seqs: list[PointLLMSequence]) -> list[int]:
        """
        批量 prefill：
          1. 批量点云编码
          2. left-pad 到统一长度，组成 inputs_embeds [B, max_L, H]
          3. HF forward（use_cache=True），每个 seq 拿到自己的 KV cache
          4. greedy 采样首个生成 token
        """
        self._encode_point_clouds_batch(seqs)

        max_len = max(seq.num_tokens for seq in seqs)
        B = len(seqs)

        embeds_list = []
        attn_masks = []
        for seq in seqs:
            L = seq.num_tokens
            if seq.inputs_embeds_cached is not None:
                emb = seq.inputs_embeds_cached                            # [L, H]
            else:
                try:
                    device = next(self.wrapper.inner.parameters()).device
                except StopIteration:
                    device = torch.device("cpu")
                ids = torch.tensor([seq.token_ids], dtype=torch.long, device=device)
                emb = self.wrapper.get_input_embeddings()(ids).squeeze(0)

            pad_len = max_len - L
            if pad_len > 0:
                pad = emb.new_zeros(pad_len, emb.shape[-1])
                emb = torch.cat([pad, emb], dim=0)
                mask = torch.cat([
                    torch.zeros(pad_len, dtype=torch.long, device=emb.device),
                    torch.ones(L, dtype=torch.long, device=emb.device),
                ])
            else:
                mask = torch.ones(L, dtype=torch.long, device=emb.device)

            embeds_list.append(emb)
            attn_masks.append(mask)

        inputs_embeds = torch.stack(embeds_list)     # [B, max_len, H]
        attention_mask = torch.stack(attn_masks)     # [B, max_len]

        # HF prefill：input_ids 全零占位（点云已融入 inputs_embeds），point_clouds=None
        dummy_ids = torch.zeros(B, max_len, dtype=torch.long, device=inputs_embeds.device)
        prefill_out = hf_prefill(
            self.hf_model,
            dummy_ids,
            attention_mask,
            point_clouds=None,
            inputs_embeds=inputs_embeds,
        )

        # 分配 per-sequence KV cache（从 batch DynamicCache 切片）
        _split_prefill_cache(prefill_out.past_key_values, seqs)

        # Greedy 采样
        logits = prefill_out.logits   # [B, 1, V]
        return logits[:, -1, :].float().argmax(dim=-1).tolist()

    @torch.inference_mode()
    def run_decode(self, seqs: list[PointLLMSequence]) -> list[int]:
        """
        批量 decode 步（padded KV 合并策略）：
          1. 合并所有 seq 的 KV cache（left-pad 至 max_kv_len）
          2. 拼成 [B, 1] input_ids（每 seq 最后一个 token）
          3. 单次 HF forward
          4. 拆分 KV cache 回 per-seq，greedy 采样
        """
        orig_kv_lens = [seq.kv_seq_len for seq in seqs]
        batched_cache, attention_mask, _ = _build_batched_cache(seqs)

        input_ids = torch.tensor(
            [[seq.last_token] for seq in seqs],
            dtype=torch.long,
            device=_cache_get_kv(batched_cache, 0)[0].device,
        )   # [B, 1]
        attention_mask = attention_mask.to(device=input_ids.device)

        out = _batch_hf_decode(
            self.hf_model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=batched_cache,
        )

        _split_batched_cache(out.past_key_values, seqs, orig_kv_lens)

        logits = out.logits   # [B, 1, V] 或 [B, V]
        if logits.dim() == 3:
            logits = logits[:, -1, :]
        return logits.float().argmax(dim=-1).tolist()

    def run(self, seqs: list[PointLLMSequence], is_prefill: bool) -> list[int]:
        if is_prefill:
            return self.run_prefill(seqs)
        return self.run_decode(seqs)
