"""
与 HuggingFace `LlamaMLP`（SwiGLU：`down(silu(gate(x)) * up(x))`）同结构、同 `state_dict` 键名。

兼容两类 HF 构造：旧版 ``LlamaMLP(config)`` 与新版 ``LlamaMLP(hidden_size, intermediate_size, hidden_act)``；
单测用 ``inspect`` 分支构造 HF 层；本模块的 ``LlamaSwiGLUMLPRef`` 始终采用三整数签名的显式形状。

用于 P2：先与 HF 数值对齐，再逐步替换为自定义算子 / Triton。
"""
from __future__ import annotations

import torch.nn as nn
from transformers import LlamaConfig
from transformers.activations import ACT2FN


class LlamaSwiGLUMLPRef(nn.Module):
    """对齐 `transformers.models.llama.modeling_llama.LlamaMLP`（与 HF 相同 Linear 声明顺序）。"""

    def __init__(self, hidden_size: int, intermediate_size: int, hidden_act: str):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.act_fn = ACT2FN[hidden_act]

    @classmethod
    def from_llama_config(cls, config: LlamaConfig) -> LlamaSwiGLUMLPRef:
        return cls(config.hidden_size, config.intermediate_size, config.hidden_act)

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
