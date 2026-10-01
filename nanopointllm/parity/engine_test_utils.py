# nanopointllm/parity/engine_test_utils.py
"""
Shared helpers for P4 integration scripts (parity + bench).
Requires real GPU + PointLLM_7B_v1.2 weights; not imported in unit tests.
"""
from __future__ import annotations

import os
import sys
from typing import Optional

import torch

_POINTLLM = os.environ.get("POINTLLM_ROOT", "/home/PointLLM")
if _POINTLLM not in sys.path:
    sys.path.insert(0, _POINTLLM)


def load_pointllm_model(
    model_path: str,
    device: torch.device,
    dtype: torch.dtype,
):
    """
    Load PointLLMLlamaForCausalLM + tokenizer, apply eager attention,
    return (model, tokenizer) ready for inference.
    """
    from pointllm.model import PointLLMLlamaForCausalLM
    from transformers import AutoTokenizer

    from nanopointllm.parity.attn_stability import set_eager_attention_if_supported
    from nanopointllm.parity.model_path import validate_pretrained_local_or_hub

    path = validate_pretrained_local_or_hub(model_path)
    tok = AutoTokenizer.from_pretrained(path, use_fast=False)
    model = PointLLMLlamaForCausalLM.from_pretrained(
        path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        use_cache=True,
    ).to(device)
    model.initialize_tokenizer_point_backbone_config_wo_embedding(tok)
    model.eval()
    set_eager_attention_if_supported(model)
    return model, tok


def build_prompt_token_ids(tokenizer, model, question: str = "What is this?") -> list[int]:
    """
    Build vicuna_v1_1 prompt with point_patch_token placeholders.
    Returns token ids as list[int] (for PointLLMLLMEngine.add_request).
    """
    from pointllm.conversation import conv_templates

    conv = conv_templates["vicuna_v1_1"].copy()
    cfg = model.get_model().point_backbone_config
    pt_len = cfg["point_token_len"]
    patch = cfg["default_point_patch_token"]
    if cfg["mm_use_point_start_end"]:
        s = cfg["default_point_start_token"]
        e = cfg["default_point_end_token"]
        qs = s + patch * pt_len + e + "\n" + question
    else:
        qs = patch * pt_len + "\n" + question
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    enc = tokenizer([conv.get_prompt()], return_tensors="pt")
    return enc["input_ids"][0].tolist()


def build_text_only_token_ids(tokenizer, question: str = "Describe this object.") -> list[int]:
    """Plain vicuna prompt with no point_patch_token placeholders (text-only sequences)."""
    from pointllm.conversation import conv_templates

    conv = conv_templates["vicuna_v1_1"].copy()
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    enc = tokenizer([conv.get_prompt()], return_tensors="pt")
    return enc["input_ids"][0].tolist()


def make_fake_point_cloud(
    device: torch.device,
    dtype: torch.dtype,
    N: int = 8192,
    C: int = 6,
) -> torch.Tensor:
    """Synthetic [N, C] point cloud tensor. No real 3D data required."""
    return torch.randn(N, C, device=device, dtype=dtype)


def hf_greedy_generate(
    model,
    token_ids: list[int],
    point_clouds: Optional[torch.Tensor],
    max_new_tokens: int,
    device: torch.device,
) -> list[int]:
    """
    Run HF manual greedy decode. Returns generated token ids only (excludes prompt).
    point_clouds: [N, C] tensor or None for text-only sequences.
    """
    from nanopointllm.parity.hf_manual_greedy import manual_greedy_decode

    ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    mask = torch.ones_like(ids)
    pc = None
    if point_clouds is not None:
        pc = point_clouds.unsqueeze(0) if point_clouds.dim() == 2 else point_clouds

    out = manual_greedy_decode(model, ids, mask, max_new_tokens, point_clouds=pc)
    # out shape: [1, prompt_len + max_new_tokens]
    return out[0, len(token_ids):].tolist()
