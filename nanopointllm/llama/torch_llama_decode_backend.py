"""
显式 **逐层** 调用 HF ``LlamaModel`` 子模块完成 decode 步（embed → RoPE → layers → final norm），
再走 ``lm_head``；与 ``HFInnerDecodeBackend`` 数值对齐，便于后续把单层替换为自定义 attention/KV。

依赖 ``transformers.masking_utils.create_causal_mask`` 与 ``transformers.cache_utils.DynamicCache``（与当前 Llama 主线一致）。
若 ``hf_prefill`` 仍返回 **legacy tuple** ``past_key_values``，入口会先 ``DynamicCache.from_legacy_cache`` 再算 ``cache_position`` / mask。
若当前环境 **无** ``transformers.cache_utils``（旧版或残缺安装），``decode_step`` 会 **退化为** ``HFInnerDecodeBackend`` 并打 warning；请 ``pip install -U 'transformers>=4.38'`` 后再用显式 ``torch_loop``。
"""
from __future__ import annotations

import inspect
import logging
from typing import Any

import torch
import torch.nn as nn

from nanopointllm.engine.decode_backend import DecodeBackend, _decoder_base_model
from nanopointllm.engine.types import DecodeStepInput

logger = logging.getLogger(__name__)


def dynamic_cache_cls() -> type | None:
    """``DynamicCache`` 所在子模块因 transformers 版本而异；缺失时返回 None。"""
    try:
        from transformers.cache_utils import DynamicCache

        return DynamicCache
    except ImportError:
        return None


def normalize_past_key_values_for_torch_loop(
    past_key_values: Any, config: Any, *, DynamicCache: type | None = None
) -> Any:
    """
    ``hf_prefill`` / 部分 HF 版本仍返回 **legacy tuple** ``((k,v), ...)``；
    ``create_causal_mask`` 与 ``get_seq_length`` 需要 ``Cache`` 接口，故转为 ``DynamicCache``。
    """
    if past_key_values is None:
        return None
    if hasattr(past_key_values, "get_seq_length"):
        return past_key_values
    if isinstance(past_key_values, (tuple, list)) and len(past_key_values) > 0:
        first = past_key_values[0]
        if isinstance(first, (tuple, list)) and len(first) == 2:
            DC = DynamicCache if DynamicCache is not None else dynamic_cache_cls()
            if DC is None:
                raise ImportError(
                    "无法 import transformers.cache_utils.DynamicCache，"
                    "无法将 legacy tuple 转为 Cache；请升级 transformers。"
                )
            convert = getattr(DC, "from_legacy_cache", None)
            if convert is not None:
                return convert(past_key_values)
    raise TypeError(
        "torch_loop 需要 HF Cache 或 legacy tuple past_key_values；收到 "
        f"{type(past_key_values).__name__}"
    )


def _past_seq_length(past_key_values: Any) -> int:
    if past_key_values is None:
        return 0
    if hasattr(past_key_values, "get_seq_length"):
        return int(past_key_values.get_seq_length())
    if isinstance(past_key_values, (tuple, list)) and len(past_key_values) > 0:
        k0 = past_key_values[0][0]
        return int(k0.shape[-2])
    return 0


def _create_causal_mask_compat(
    inner: nn.Module,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor | None,
    cache_position: torch.Tensor,
    past_key_values: Any,
    position_ids: torch.Tensor,
):
    from transformers.masking_utils import create_causal_mask

    # HF 参数名为 ``input_embeds``（与变量名 ``inputs_embeds`` 区分）
    return create_causal_mask(
        config=inner.config,
        input_embeds=inputs_embeds,
        attention_mask=attention_mask,
        cache_position=cache_position,
        past_key_values=past_key_values,
        position_ids=position_ids,
    )


def _call_decoder_layer(decoder_layer: nn.Module, forward_kwargs: dict[str, Any], **fixed: Any) -> torch.Tensor:
    sig = inspect.signature(decoder_layer.forward)
    call: dict[str, Any] = {}
    for k, v in fixed.items():
        if k in sig.parameters:
            call[k] = v
    for k, v in forward_kwargs.items():
        if k in sig.parameters:
            call[k] = v
    return decoder_layer(**call)


class TorchLlamaExplicitLoopDecodeBackend(DecodeBackend):
    """
    在 **backbone** 上复刻 ``LlamaModel.forward`` 的 decode 路径（单段 ``input_ids`` + ``past``），
    不调用 ``inner.forward`` 整段入口，从而固定为「显式 layer loop + KV 更新」结构。
    """

    @torch.inference_mode()
    def decode_step(
        self,
        model: nn.Module,
        inp: DecodeStepInput,
        **forward_kwargs: Any,
    ) -> tuple[torch.Tensor, Any]:
        inner = _decoder_base_model(model)
        if not hasattr(inner, "embed_tokens") or not hasattr(inner, "layers"):
            raise TypeError("TorchLlamaExplicitLoopDecodeBackend 需要 backbone 含 embed_tokens 与 layers（HF LlamaModel）")

        DC = dynamic_cache_cls()
        if DC is None:
            logger.warning(
                "未找到 transformers.cache_utils.DynamicCache；torch_loop 退化为 HFInnerDecodeBackend。"
                "升级 transformers 后可使用显式逐层路径。"
            )
            from nanopointllm.engine.decode_backend import HFInnerDecodeBackend

            return HFInnerDecodeBackend().decode_step(model, inp, **forward_kwargs)

        input_ids = inp.input_ids
        attention_mask = inp.attention_mask
        past_key_values = inp.past_key_values
        use_cache = True

        inputs_embeds = inner.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DC(config=inner.config)
        else:
            past_key_values = normalize_past_key_values_for_torch_loop(
                past_key_values, inner.config, DynamicCache=DC
            )

        past_seen = _past_seq_length(past_key_values)
        cache_position = torch.arange(
            past_seen,
            past_seen + inputs_embeds.shape[1],
            device=inputs_embeds.device,
            dtype=torch.long,
        )
        position_ids = cache_position.unsqueeze(0)

        try:
            causal_mask = _create_causal_mask_compat(
                inner,
                inputs_embeds,
                attention_mask,
                cache_position,
                past_key_values,
                position_ids,
            )
        except Exception as e:
            logger.warning("create_causal_mask 失败，TorchLlamaExplicitLoopDecodeBackend 不可用: %s", e)
            raise

        hidden_states = inputs_embeds
        position_embeddings = inner.rotary_emb(hidden_states, position_ids)

        kwargs = forward_kwargs
        for decoder_layer in inner.layers[: inner.config.num_hidden_layers]:
            hidden_states = _call_decoder_layer(
                decoder_layer,
                kwargs,
                hidden_states=hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                use_cache=use_cache,
            )

        hidden_states = inner.norm(hidden_states)
        lm = getattr(model, "lm_head", None)
        if lm is None:
            raise TypeError("TorchLlamaExplicitLoopDecodeBackend 需要 model.lm_head")
        logits = lm(hidden_states)
        return logits, past_key_values
