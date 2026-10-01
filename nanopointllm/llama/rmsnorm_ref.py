"""
与 HuggingFace `LlamaRMSNorm` 同公式、同 `state_dict` 键名，便于从已加载 HF 层拷贝权重或做 parity。

后续可在此文件替换为融合 / Triton 实现，保持 `forward` 形状与语义不变。
"""
from __future__ import annotations

import torch
import torch.nn as nn


class LlamaRMSNormRef(nn.Module):
    """对齐 `transformers.models.llama.modeling_llama.LlamaRMSNorm`。"""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self) -> str:
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"
