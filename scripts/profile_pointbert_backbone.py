#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time

import torch

from pointllm.model.pointbert import misc as pb_misc
from pointllm.model.pointbert.dvae import knn_point

from nanopointllm.parity.engine_test_utils import load_pointllm_model, make_fake_point_cloud


def cuda_ms(fn, warmup: int, runs: int) -> tuple[float, list[float]]:
    for _ in range(warmup):
        fn()
    vals = []
    for _ in range(runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        vals.append((time.perf_counter() - t0) * 1000)
    return sum(vals) / len(vals), vals


@torch.inference_mode()
def main() -> None:
    ap = argparse.ArgumentParser(description="Fine-grained PointBERT backbone profiler")
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--out_json", default="")
    args = ap.parse_args()

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    print(f"Loading model {args.model_path} on {device} ({args.dtype})...")
    model, _ = load_pointllm_model(args.model_path, device=device, dtype=dtype)
    inner = model.get_model()
    backbone = inner.point_backbone
    gd = backbone.group_divider
    encoder = backbone.encoder
    print("Model loaded.")

    pc = torch.stack([make_fake_point_cloud(device, dtype) for _ in range(args.batch_size)])
    xyz = pc[:, :, :3].contiguous()

    fps_mean, fps_runs = cuda_ms(lambda: pb_misc.fps(xyz, gd.num_group), args.warmup, args.runs)
    centers = pb_misc.fps(xyz, gd.num_group)

    knn_mean, knn_runs = cuda_ms(lambda: knn_point(gd.group_size, xyz, centers), args.warmup, args.runs)
    idx = knn_point(gd.group_size, xyz, centers)

    def _gather_normalize():
        batch_size, num_points, _ = xyz.shape
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        flat_idx = (idx + idx_base).view(-1)
        neighborhood_xyz = xyz.view(batch_size * num_points, -1)[flat_idx, :]
        neighborhood_xyz = neighborhood_xyz.view(batch_size, gd.num_group, gd.group_size, 3).contiguous()
        rgb = pc[:, :, 3:]
        neighborhood_rgb = rgb.view(batch_size * num_points, -1)[flat_idx, :]
        neighborhood_rgb = neighborhood_rgb.view(batch_size, gd.num_group, gd.group_size, -1).contiguous()
        neighborhood_xyz = neighborhood_xyz - centers.unsqueeze(2)
        return torch.cat((neighborhood_xyz, neighborhood_rgb), dim=-1)

    gather_mean, gather_runs = cuda_ms(_gather_normalize, args.warmup, args.runs)
    neighborhood = _gather_normalize()

    first_conv_mean, first_conv_runs = cuda_ms(
        lambda: encoder.first_conv(neighborhood.reshape(args.batch_size * gd.num_group, gd.group_size, neighborhood.shape[-1]).transpose(2, 1)),
        args.warmup,
        args.runs,
    )
    point_groups = neighborhood.reshape(args.batch_size * gd.num_group, gd.group_size, neighborhood.shape[-1])
    first_conv = encoder.first_conv(point_groups.transpose(2, 1))

    first_pool_mean, first_pool_runs = cuda_ms(
        lambda: torch.max(first_conv, dim=2, keepdim=True)[0],
        args.warmup,
        args.runs,
    )
    feature_global = torch.max(first_conv, dim=2, keepdim=True)[0]

    expand_cat_mean, expand_cat_runs = cuda_ms(
        lambda: torch.cat([feature_global.expand(-1, -1, gd.group_size), first_conv], dim=1),
        args.warmup,
        args.runs,
    )
    second_in = torch.cat([feature_global.expand(-1, -1, gd.group_size), first_conv], dim=1)

    second_conv_mean, second_conv_runs = cuda_ms(lambda: encoder.second_conv(second_in), args.warmup, args.runs)
    second_conv = encoder.second_conv(second_in)

    second_pool_mean, second_pool_runs = cuda_ms(
        lambda: torch.max(second_conv, dim=2, keepdim=False)[0],
        args.warmup,
        args.runs,
    )
    group_tokens = torch.max(second_conv, dim=2, keepdim=False)[0].reshape(args.batch_size, gd.num_group, encoder.encoder_channel)

    reduce_mean, reduce_runs = cuda_ms(lambda: backbone.reduce_dim(group_tokens), args.warmup, args.runs)
    reduced = backbone.reduce_dim(group_tokens)
    pos_mean, pos_runs = cuda_ms(lambda: backbone.pos_embed(centers), args.warmup, args.runs)
    pos = backbone.pos_embed(centers)

    cls_tokens = backbone.cls_token.expand(reduced.size(0), -1, -1)
    cls_pos = backbone.cls_pos.expand(reduced.size(0), -1, -1)
    x = torch.cat((cls_tokens, reduced), dim=1)
    pos_full = torch.cat((cls_pos, pos), dim=1)

    block_rows = []
    x_cur = x
    for i, block in enumerate(backbone.blocks.blocks):
        block_input = x_cur + pos_full
        norm1_mean, _ = cuda_ms(lambda b=block, bi=block_input: b.norm1(bi), args.warmup, args.runs)
        attn_in = block.norm1(block_input)
        attn_mean, _ = cuda_ms(lambda b=block, ai=attn_in: b.attn(ai), args.warmup, args.runs)
        attn_out = block.attn(attn_in)
        norm2_base = x_cur + block.drop_path(attn_out)
        norm2_mean, _ = cuda_ms(lambda b=block, n2=norm2_base: b.norm2(n2), args.warmup, args.runs)
        mlp_in = block.norm2(norm2_base)
        mlp_mean, _ = cuda_ms(lambda b=block, mi=mlp_in: b.mlp(mi), args.warmup, args.runs)

        total_mean, _ = cuda_ms(lambda b=block, xc=x_cur, pf=pos_full: b(xc + pf), args.warmup, args.runs)
        x_cur = block(block_input)
        block_rows.append(
            {
                "block": i,
                "norm1_ms": round(norm1_mean, 3),
                "attn_ms": round(attn_mean, 3),
                "norm2_ms": round(norm2_mean, 3),
                "mlp_ms": round(mlp_mean, 3),
                "total_ms": round(total_mean, 3),
            }
        )

    norm_pool_mean, norm_pool_runs = cuda_ms(
        lambda: torch.cat([backbone.norm(x_cur)[:, 0], backbone.norm(x_cur)[:, 1:].max(1)[0]], dim=-1).unsqueeze(1),
        args.warmup,
        args.runs,
    )

    full_backbone_mean, full_backbone_runs = cuda_ms(lambda: backbone(pc), args.warmup, args.runs)

    payload = {
        "batch_size": args.batch_size,
        "group": {
            "num_group": gd.num_group,
            "group_size": gd.group_size,
            "fps_ms_mean": round(fps_mean, 3),
            "fps_ms_runs": [round(v, 3) for v in fps_runs],
            "knn_ms_mean": round(knn_mean, 3),
            "knn_ms_runs": [round(v, 3) for v in knn_runs],
            "gather_normalize_ms_mean": round(gather_mean, 3),
            "gather_normalize_ms_runs": [round(v, 3) for v in gather_runs],
        },
        "encoder": {
            "first_conv_ms_mean": round(first_conv_mean, 3),
            "first_pool_ms_mean": round(first_pool_mean, 3),
            "expand_cat_ms_mean": round(expand_cat_mean, 3),
            "second_conv_ms_mean": round(second_conv_mean, 3),
            "second_pool_ms_mean": round(second_pool_mean, 3),
        },
        "bridge": {
            "reduce_dim_ms_mean": round(reduce_mean, 3),
            "pos_embed_ms_mean": round(pos_mean, 3),
        },
        "transformer_blocks": block_rows,
        "tail": {
            "norm_pool_ms_mean": round(norm_pool_mean, 3),
            "norm_pool_ms_runs": [round(v, 3) for v in norm_pool_runs],
        },
        "backbone_total_ms_mean": round(full_backbone_mean, 3),
        "backbone_total_ms_runs": [round(v, 3) for v in full_backbone_runs],
    }

    print(json.dumps(payload, indent=2))
    if args.out_json:
        out_dir = os.path.dirname(os.path.abspath(args.out_json))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)


if __name__ == "__main__":
    main()
