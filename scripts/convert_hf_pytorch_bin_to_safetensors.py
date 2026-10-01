#!/usr/bin/env python3
"""
将 HuggingFace 风格的 ``pytorch_model*.bin`` 权重目录转为 ``*.safetensors``（含分片与 ``model.safetensors.index.json``），
便于在 **torch<2.6** 且 **transformers≥5.5** 下用 ``--use_safetensors`` 加载（绕开对 ``.bin`` 的 ``torch.load`` 限制）。

不依赖 ``transformers.from_pretrained`` 读权重，仅用 ``torch.load`` + ``huggingface_hub.save_torch_state_dict``。

示例::

  conda activate pointllm
  cd /home/nano-pointllm
  python scripts/convert_hf_pytorch_bin_to_safetensors.py \\
    --model_path /mnt/llm_data/pointllm_ckpt/PointLLM_7B_v1.2

  # 输出到子目录（自动复制 config/tokenizer 等非 pytorch_model*.bin 文件；原目录 .bin 不动）：
  python scripts/convert_hf_pytorch_bin_to_safetensors.py \\
    --model_path /path/to/ckpt --output_dir /path/to/ckpt_safetensors
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
from huggingface_hub import save_torch_state_dict

from nanopointllm.parity.model_path import validate_pretrained_local_or_hub


def _torch_load_state_dict_shard(path: str) -> dict:
    kw: dict = {"map_location": "cpu"}
    try:
        return torch.load(path, **kw, weights_only=True)
    except TypeError:
        return torch.load(path, **kw)


def load_hf_pytorch_state_dict(model_dir: str) -> dict[str, torch.Tensor]:
    index_path = os.path.join(model_dir, "pytorch_model.bin.index.json")
    single_path = os.path.join(model_dir, "pytorch_model.bin")
    if os.path.isfile(index_path):
        with open(index_path, encoding="utf-8") as f:
            index = json.load(f)
        weight_map: dict[str, str] = index["weight_map"]
        shard_files = sorted(set(weight_map.values()))
        state: dict[str, torch.Tensor] = {}
        for fname in shard_files:
            path = os.path.join(model_dir, fname)
            if not os.path.isfile(path):
                raise SystemExit(f"index 指向的文件不存在: {path}")
            part = _torch_load_state_dict_shard(path)
            state.update(part)
        return state
    if os.path.isfile(single_path):
        out = _torch_load_state_dict_shard(single_path)
        if not isinstance(out, dict):
            raise SystemExit("pytorch_model.bin 内容不是 state_dict 字典。")
        return out
    raise SystemExit(
        f"未找到 {index_path!r} 或 {single_path!r}。\n"
        "请确认目录为 HuggingFace 预训练格式（含 pytorch_model.bin 或分片 + index）。"
    )


def _dir_has_safetensors(d: str) -> bool:
    return any(name.endswith(".safetensors") for name in os.listdir(d))


def _is_pytorch_weight_artifact(name: str) -> bool:
    if name == "pytorch_model.bin.index.json":
        return True
    return name.startswith("pytorch_model") and name.endswith(".bin")


def copy_hf_sidecar_files(src_dir: str, dst_dir: str) -> None:
    """将 config / tokenizer 等非 pytorch 分片文件复制到输出目录（``dst_dir`` 可与 ``src_dir`` 不同）。"""
    for name in os.listdir(src_dir):
        if _is_pytorch_weight_artifact(name):
            continue
        s, d = os.path.join(src_dir, name), os.path.join(dst_dir, name)
        if os.path.isdir(s):
            if os.path.isdir(d):
                shutil.rmtree(d)
            shutil.copytree(s, d)
        else:
            shutil.copy2(s, d)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model_path", type=str, required=True, help="含 pytorch_model*.bin 的权重目录")
    ap.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="写入 safetensors 的目录；默认与 --model_path 相同（就地增加 *.safetensors）",
    )
    ap.add_argument(
        "--max_shard_size",
        type=str,
        default="5GB",
        help="传给 huggingface_hub.save_torch_state_dict(max_shard_size=...)",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="若 output_dir 已存在 *.safetensors 仍继续写入（可能产生混杂旧分片，建议输出到空目录）",
    )
    ap.add_argument("--dry_run", action="store_true", help="只检查路径与是否已有 safetensors，不写盘")
    args = ap.parse_args()

    model_dir = validate_pretrained_local_or_hub(args.model_path)
    out_dir = os.path.abspath((args.output_dir or model_dir).strip() or model_dir)
    os.makedirs(out_dir, exist_ok=True)

    if _dir_has_safetensors(out_dir) and not args.overwrite:
        raise SystemExit(
            f"目标目录已存在 *.safetensors：{out_dir}\n"
            "若确需覆盖写入，请加 --overwrite；更推荐 --output_dir 指向新的空目录。"
        )

    if args.dry_run:
        print("[dry-run] model_dir:", model_dir)
        print("[dry-run] output_dir:", out_dir)
        print("[dry-run] 将加载 pytorch 分片并写入 safetensors（未执行）")
        if os.path.normpath(out_dir) != os.path.normpath(os.path.abspath(model_dir)):
            print("[dry-run] 非就地转换时会复制 config/tokenizer 等非 .bin 文件到 output_dir")
        return

    if os.path.normpath(out_dir) != os.path.normpath(os.path.abspath(model_dir)):
        print("复制 config / tokenizer 等到输出目录…", flush=True)
        copy_hf_sidecar_files(model_dir, out_dir)

    print("加载 pytorch 权重（CPU map_location）…", flush=True)
    state = load_hf_pytorch_state_dict(model_dir)
    n = len(state)
    print(f"共 {n} 个张量键，写入 {out_dir!r} …", flush=True)
    save_torch_state_dict(
        state,
        out_dir,
        safe_serialization=True,
        max_shard_size=args.max_shard_size,
        force_contiguous=True,
    )
    print("完成。请在该目录使用 from_pretrained(..., use_safetensors=True) 或 nano-pointllm 脚本的 --use_safetensors。")
    if os.path.normpath(out_dir) == os.path.normpath(os.path.abspath(model_dir)):
        print("提示：原 .bin 仍在目录中；加载时 HF 会优先 safetensors。若需节省磁盘可手动删除 pytorch_model*.bin 与对应 index。")


if __name__ == "__main__":
    main()
