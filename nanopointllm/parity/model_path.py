"""校验 `from_pretrained` 的路径 / Hub id，避免把文档里的 `...` 误传给 HuggingFace。"""
from __future__ import annotations

import os
import re

# 与 huggingface_hub 对 repo id 的规则大致对齐（两段式 org/model）
_HUB_REPO_RE = re.compile(
    r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,94}/[a-zA-Z0-9][a-zA-Z0-9._-]{0,94}$"
)


def validate_pretrained_local_or_hub(model_path: str) -> str:
    s = (model_path or "").strip()
    if not s or s in ("...", ".", "..") or set(s) <= {"."}:
        raise SystemExit(
            "无效的 --model_path：请替换成本机权重目录或 Hub 上的 org/model，"
            "不要把文档里的省略号 ... 原样粘贴。\n"
            "示例: --model_path /mnt/llm_data/pointllm_ckpt/PointLLM_7B_v1.2"
        )
    expanded = os.path.expanduser(s)
    if os.path.isdir(expanded):
        return expanded
    if _HUB_REPO_RE.match(s):
        return s
    raise SystemExit(
        f"未找到本地权重目录，且参数也不像有效的 Hub repo id: {model_path!r}\n"
        "请确认本机路径存在（目录内应有 config.json 等），或使用 org/model 形式的 Hub id。"
    )


def validate_safetensors_weights_present(model_dir: str) -> None:
    """
    ``from_pretrained(..., use_safetensors=True)`` 要求目录内存在 ``*.safetensors``（或 HF 识别的分片命名）。
    若仅有 ``pytorch_model*.bin``，应先升级 PyTorch 或完成格式转换，勿单独加 ``--use_safetensors``。
    """
    if not os.path.isdir(model_dir):
        return
    if any(name.endswith(".safetensors") for name in os.listdir(model_dir)):
        return
    raise SystemExit(
        "已指定 --use_safetensors，但该目录下没有找到任何 *.safetensors 文件。\n"
        f"目录: {model_dir}\n"
        "若权重仅有 pytorch_model*.bin，请任选其一：\n"
        "  (1) 去掉 --use_safetensors，并确保 PyTorch>=2.6（满足 transformers 5.5+ 对 .bin 的加载策略）；\n"
        "  (2) 将 checkpoint 转为 safetensors 后再放入该目录。\n"
        "详见 PointLLM/docs/setup_inference.md §6。"
    )


def _semver_tuple_from_version_string(version: str) -> tuple[int, int, int]:
    raw = version.split("+", 1)[0]
    parts: list[int] = []
    for seg in raw.split(".")[:3]:
        digits = ""
        for ch in seg:
            if ch.isdigit():
                digits += ch
            else:
                break
        parts.append(int(digits) if digits else 0)
    while len(parts) < 3:
        parts.append(0)
    return parts[0], parts[1], parts[2]


def validate_torch_version_for_bin_weights(model_dir: str, use_safetensors: bool) -> None:
    """
    transformers 5.5+ 在加载 ``pytorch_model*.bin`` 时会调用 ``check_torch_load_is_safe()``，
    要求 **PyTorch>=2.6**（CVE-2025-32434）；使用目录内已有 ``*.safetensors`` 并 ``use_safetensors=True`` 时不受此条限制。

    仅对**可枚举的本地目录**做预检；Hub ``org/model`` 无法判断分片格式，故跳过。
    """
    if not model_dir or not os.path.isdir(model_dir):
        return
    if use_safetensors:
        return
    names = os.listdir(model_dir)
    if any(n.endswith(".safetensors") for n in names):
        return
    if not any(n.endswith(".bin") for n in names):
        return
    import torch

    ver = torch.__version__
    if _semver_tuple_from_version_string(ver) >= (2, 6, 0):
        return

    raise SystemExit(
        "检测到权重目录内仅有 *.bin、且未使用 --use_safetensors；当前 transformers 会拒绝在 PyTorch<2.6 下用 torch.load 加载此类文件（CVE-2025-32434）。\n"
        f"torch.__version__ = {ver}\n"
        f"目录: {model_dir}\n"
        "请任选其一：\n"
        "  (1) 升级 PyTorch 至 2.6+（需与 CUDA 驱动匹配的官方 wheel/conda 包）；\n"
        "  (2) 将 checkpoint 转为 safetensors 放入该目录后，使用 --use_safetensors 加载。\n"
        "详见 PointLLM/docs/setup_inference.md §6。"
    )
