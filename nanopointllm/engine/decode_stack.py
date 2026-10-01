"""
单步 **decode 栈**：由内到外串联若干层（当前支持 `hf` 根 + 可选 `torch.compile` 外壳）。
后续可在此注册 Triton/CUDA、同步/打点等中间层；**prefill 栈**另开模块时再接。

语法（`build_decode_stack`）::

    hf                  → 仅 HuggingFace `CausalLM.forward` 单步
    hf_inner            → 仅 backbone + `lm_head`（P2b 自定义 decode 的挂点）
    torch_loop          → 显式逐层 backbone + `lm_head`（与 ``LlamaModel.forward`` 对齐，便于替换 attention）
    hf|compile          → 外层 `torch.compile` 包裹 HF（无额外 `forward_kwargs` 时生效）
    hf_inner|compile    → 同上，根为 `hf_inner`
    hf|torch_compile    → 同 `compile`
"""
from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

from nanopointllm.engine.decode_backend import (
    DecodeBackend,
    HFDecodeBackend,
    HFInnerDecodeBackend,
)
from nanopointllm.engine.types import DecodeStepInput

logger = logging.getLogger(__name__)


class DecodeWrapper(DecodeBackend):
    """包一层内层 `DecodeBackend`，便于链式组合。"""

    def __init__(self, inner: DecodeBackend):
        self.inner = inner


class TorchCompileDecodeWrapper(DecodeWrapper):
    """
    将 `inner.decode_step` 包进 `torch.compile`（仅当 `forward_kwargs` 为空时走编译路径，
    与 bench / greedy 默认用法一致）。编译或运行失败时自动回退 eager。

    **CUDA Graph**：自回归 decode 会原地更新 ``past_key_values``；``reduce-overhead`` 下
    Inductor 可能启用 Triton CUDAGraph，导致
    ``accessing tensor output of CUDAGraphs that has been overwritten``。
    默认在 ``torch.compile(..., options={"triton.cudagraphs": False})`` 关闭；
    若显式 ``allow_triton_cudagraphs=True``，则不在 compile 里关闭，并在**每次**调用前执行
    ``torch.compiler.cudagraph_mark_step_begin()``（需较新 PyTorch）。
    """

    def __init__(
        self,
        inner: DecodeBackend,
        *,
        mode: str = "reduce-overhead",
        fullgraph: bool = False,
        allow_triton_cudagraphs: bool = False,
    ):
        super().__init__(inner)
        self.mode = mode
        self.fullgraph = fullgraph
        self.allow_triton_cudagraphs = allow_triton_cudagraphs
        self._compiled: Any | None = None
        self._compile_failed = False

    @torch.inference_mode()
    def decode_step(
        self,
        model: nn.Module,
        inp: DecodeStepInput,
        **forward_kwargs: Any,
    ) -> tuple[torch.Tensor, Any]:
        if self._compile_failed or forward_kwargs:
            return self.inner.decode_step(model, inp, **forward_kwargs)

        if self._compiled is None:
            inner = self.inner

            def _eager_step(
                m: nn.Module,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                past_key_values: Any,
            ) -> tuple[torch.Tensor, Any]:
                inner_in = DecodeStepInput(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                )
                return inner.decode_step(m, inner_in)

            compile_kw: dict[str, Any] = {"mode": self.mode, "fullgraph": self.fullgraph}
            try:
                if not self.allow_triton_cudagraphs:
                    # 避免 KV cache 原地更新与 CUDAGraph 捕获的别名冲突（见 PyTorch 报错提示）
                    try:
                        self._compiled = torch.compile(
                            _eager_step,
                            options={"triton.cudagraphs": False},
                            **compile_kw,
                        )
                    except TypeError:
                        self._compiled = torch.compile(_eager_step, **compile_kw)
                else:
                    self._compiled = torch.compile(_eager_step, **compile_kw)
            except Exception as e:
                logger.warning("torch.compile 构造失败，decode 栈回退 eager: %s", e)
                self._compile_failed = True
                return self.inner.decode_step(model, inp, **forward_kwargs)

        try:
            if self.allow_triton_cudagraphs:
                mark = getattr(getattr(torch, "compiler", None), "cudagraph_mark_step_begin", None)
                if callable(mark):
                    mark()
            return self._compiled(
                model,
                inp.input_ids,
                inp.attention_mask,
                inp.past_key_values,
            )
        except Exception as e:
            logger.warning("torch.compile decode_step 运行失败，回退 eager: %s", e)
            self._compile_failed = True
            self._compiled = None
            return self.inner.decode_step(model, inp, **forward_kwargs)


def build_decode_stack(
    spec: str,
    *,
    compile_mode: str = "reduce-overhead",
    compile_fullgraph: bool = False,
    compile_allow_triton_cudagraphs: bool = False,
) -> DecodeBackend:
    """
    解析栈描述字符串，返回最外层 `DecodeBackend`。

    - 段从左到右为 **内 → 外**：第一段为根，须为 `hf` 或 `hf_inner`。
    - 允许写法：`hf`、`hf_inner`、`torch_loop`、`hf|compile`、`hf_inner|compile`、`torch_loop|compile`、`hf|torch_compile`。
    """
    parts = [p.strip().lower() for p in spec.split("|") if p.strip()]
    if not parts:
        raise ValueError("decode stack spec 为空")
    if parts[0] == "hf":
        root: DecodeBackend = HFDecodeBackend()
    elif parts[0] == "hf_inner":
        root = HFInnerDecodeBackend()
    elif parts[0] == "torch_loop":
        from nanopointllm.llama.torch_llama_decode_backend import TorchLlamaExplicitLoopDecodeBackend

        root = TorchLlamaExplicitLoopDecodeBackend()
    else:
        raise ValueError(
            f"decode stack 根层须为 'hf'、'hf_inner' 或 'torch_loop'，收到: {spec!r}"
        )
    for layer in parts[1:]:
        if layer in ("compile", "torch_compile"):
            root = TorchCompileDecodeWrapper(
                root,
                mode=compile_mode,
                fullgraph=compile_fullgraph,
                allow_triton_cudagraphs=compile_allow_triton_cudagraphs,
            )
        else:
            raise ValueError(
                f"未知 decode 栈层 {layer!r}；支持: compile, torch_compile"
            )
    return root
