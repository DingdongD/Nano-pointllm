#!/usr/bin/env python3
"""Run warmed, stage-isolated PointBERT work for external NCU capture."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from pointllm.model.pointbert import misc as pb_misc
from pointllm.model.pointbert.dvae import knn_point

from nanopointllm.gpu_telemetry import query_gpu_state
from nanopointllm.parity.engine_test_utils import load_pointllm_model, make_fake_point_cloud


STAGE_RANGES = {
    "fps": "pointbert_fps",
    "knn": "pointbert_knn",
    "local_encoder": "pointbert_local_encoder",
    "pt_qkv": "pointbert_pt_qkv",
    "pt_attention": "pointbert_pt_attention",
    "pt_o_proj": "pointbert_pt_o_proj",
    "pt_mlp": "pointbert_pt_mlp",
}


def _tensor_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def _parameter_bytes(module: nn.Module) -> int:
    return sum(_tensor_bytes(parameter) for parameter in module.parameters())


def _gather_neighborhood(
    point_cloud: torch.Tensor,
    xyz: torch.Tensor,
    centers: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    batch_size, num_points, _ = xyz.shape
    num_group, group_size = indices.shape[1:]
    flat_indices = (
        indices
        + torch.arange(batch_size, device=xyz.device).view(-1, 1, 1) * num_points
    ).reshape(-1)
    neighborhood_xyz = xyz.reshape(batch_size * num_points, -1)[flat_indices]
    neighborhood_xyz = neighborhood_xyz.reshape(batch_size, num_group, group_size, 3)
    neighborhood_xyz = neighborhood_xyz - centers.unsqueeze(2)
    if point_cloud.shape[-1] == 3:
        return neighborhood_xyz.contiguous()
    extra = point_cloud[:, :, 3:].reshape(batch_size * num_points, -1)[flat_indices]
    extra = extra.reshape(batch_size, num_group, group_size, -1)
    return torch.cat((neighborhood_xyz, extra), dim=-1).contiguous()


def _attention_core(attention: nn.Module, qkv_tensor: torch.Tensor) -> torch.Tensor:
    batch_size, tokens, channels3 = qkv_tensor.shape
    channels = channels3 // 3
    qkv = qkv_tensor.reshape(
        batch_size, tokens, 3, attention.num_heads, channels // attention.num_heads
    ).permute(2, 0, 3, 1, 4)
    query, key, value = qkv[0], qkv[1], qkv[2]
    scores = (query @ key.transpose(-2, -1)) * attention.scale
    scores = attention.attn_drop(scores.softmax(dim=-1))
    return (scores @ value).transpose(1, 2).reshape(batch_size, tokens, channels)


def _prepare_transformer_inputs(backbone: nn.Module, neighborhood: torch.Tensor, centers: torch.Tensor):
    group_tokens = backbone.encoder(neighborhood)
    reduced = backbone.reduce_dim(group_tokens)
    pos = backbone.pos_embed(centers)
    cls_tokens = backbone.cls_token.expand(reduced.shape[0], -1, -1)
    cls_pos = backbone.cls_pos.expand(reduced.shape[0], -1, -1)
    hidden = torch.cat((cls_tokens, reduced), dim=1)
    pos = torch.cat((cls_pos, pos), dim=1)

    attn_inputs = []
    qkv_tensors = []
    attention_outputs = []
    mlp_inputs = []
    for block in backbone.blocks.blocks:
        block_input = hidden + pos
        attn_input = block.norm1(block_input)
        qkv_tensor = block.attn.qkv(attn_input)
        attention_output = _attention_core(block.attn, qkv_tensor)
        projected = block.attn.proj(attention_output)
        after_attention = block_input + block.drop_path(projected)
        mlp_input = block.norm2(after_attention)
        hidden = after_attention + block.drop_path(block.mlp(mlp_input))
        attn_inputs.append(attn_input)
        qkv_tensors.append(qkv_tensor)
        attention_outputs.append(attention_output)
        mlp_inputs.append(mlp_input)
    return attn_inputs, qkv_tensors, attention_outputs, mlp_inputs


def _linear_stage_cost(
    modules: list[nn.Linear], inputs: list[torch.Tensor], outputs: list[torch.Tensor], iterations: int
) -> dict:
    weight_bytes = sum(_parameter_bytes(module) for module in modules)
    activation_bytes = sum(_tensor_bytes(tensor) for tensor in inputs + outputs)
    flops = sum(
        2 * input_tensor.numel() // input_tensor.shape[-1] * module.in_features * module.out_features
        for module, input_tensor in zip(modules, inputs)
    )
    return {
        "estimated_flops": flops * iterations,
        "weight_bytes": weight_bytes * iterations,
        "activation_bytes_lower_bound": activation_bytes * iterations,
        "compulsory_bytes_lower_bound": (weight_bytes + activation_bytes) * iterations,
        "work_model_note": "Linear MACs count as two FLOPs; compulsory bytes include weights and stage input/output only.",
    }


def build_manifest(
    backbone: nn.Module,
    *,
    point_cloud: torch.Tensor,
    xyz: torch.Tensor,
    centers: torch.Tensor,
    indices: torch.Tensor,
    neighborhood: torch.Tensor,
    attn_inputs: list[torch.Tensor],
    qkv_tensors: list[torch.Tensor],
    attention_outputs: list[torch.Tensor],
    mlp_inputs: list[torch.Tensor],
    iterations: int,
) -> dict:
    batch_size, num_points, _ = xyz.shape
    num_groups = centers.shape[1]
    group_size = indices.shape[2]
    blocks = list(backbone.blocks.blocks)
    dtype_bytes = neighborhood.element_size()
    index_bytes = indices.element_size()

    fps_flops = batch_size * num_groups * num_points * 8
    fps_bytes = _tensor_bytes(xyz) + _tensor_bytes(centers) + batch_size * num_groups * 8
    knn_flops = batch_size * num_groups * num_points * 9
    knn_bytes = _tensor_bytes(xyz) + _tensor_bytes(centers) + _tensor_bytes(indices)

    conv_modules = [module for module in backbone.encoder.modules() if isinstance(module, nn.Conv1d)]
    positions = batch_size * num_groups * group_size
    local_encoder_flops = sum(2 * positions * module.weight.numel() for module in conv_modules)
    local_encoder_output_elements = batch_size * num_groups * backbone.encoder.encoder_channel
    local_encoder_output_bytes = local_encoder_output_elements * dtype_bytes
    local_encoder_weight_bytes = _parameter_bytes(backbone.encoder)

    qkv = _linear_stage_cost(
        [block.attn.qkv for block in blocks],
        attn_inputs,
        qkv_tensors,
        iterations,
    )
    o_proj = _linear_stage_cost(
        [block.attn.proj for block in blocks],
        attention_outputs,
        [torch.empty_like(tensor) for tensor in attention_outputs],
        iterations,
    )
    mlp_outputs = [torch.empty_like(tensor) for tensor in mlp_inputs]
    mlp_weight_bytes = sum(_parameter_bytes(block.mlp) for block in blocks)
    mlp_activation_bytes = sum(
        _tensor_bytes(input_tensor) + _tensor_bytes(output_tensor)
        for input_tensor, output_tensor in zip(mlp_inputs, mlp_outputs)
    )
    mlp_flops = sum(
        2 * input_tensor.numel() // input_tensor.shape[-1]
        * (block.mlp.fc1.in_features * block.mlp.fc1.out_features
           + block.mlp.fc2.in_features * block.mlp.fc2.out_features)
        for block, input_tensor in zip(blocks, mlp_inputs)
    )

    tokens = attn_inputs[0].shape[1]
    channels = attn_inputs[0].shape[2]
    heads = blocks[0].attn.num_heads
    attention_flops = len(blocks) * (
        4 * batch_size * heads * tokens * tokens * (channels // heads)
        + 5 * batch_size * heads * tokens * tokens
    )
    score_bytes = len(blocks) * batch_size * heads * tokens * tokens * dtype_bytes
    attention_io_bytes = sum(
        _tensor_bytes(qkv_tensor) + _tensor_bytes(output)
        for qkv_tensor, output in zip(qkv_tensors, attention_outputs)
    )

    stages = {
        "fps": {
            "estimated_flops": fps_flops * iterations,
            "compulsory_bytes_lower_bound": fps_bytes * iterations,
            "work_model_note": "Eight distance FLOPs per point/centroid; comparison, index, and gather operations are not counted as FLOPs.",
        },
        "knn": {
            "estimated_flops": knn_flops * iterations,
            "compulsory_bytes_lower_bound": knn_bytes * iterations,
            "index_element_size": index_bytes,
            "work_model_note": "Square-distance matmul and elementwise arithmetic are counted; top-k comparisons are not FLOPs.",
        },
        "local_encoder": {
            "estimated_flops": local_encoder_flops * iterations,
            "weight_bytes": local_encoder_weight_bytes * iterations,
            "compulsory_bytes_lower_bound": (
                _tensor_bytes(neighborhood) + local_encoder_output_bytes + local_encoder_weight_bytes
            ) * iterations,
            "work_model_note": "Conv1d MACs count as two FLOPs; BN, ReLU, concatenation, and reductions are excluded from estimated FLOPs but remain in NCU bytes/time.",
        },
        "pt_qkv": qkv,
        "pt_attention": {
            "estimated_flops": attention_flops * iterations,
            "compulsory_bytes_lower_bound": (attention_io_bytes + score_bytes) * iterations,
            "work_model_note": "QK and AV matmuls plus an approximate five FLOPs per softmax element; eager score-tensor traffic is included once as a lower bound.",
        },
        "pt_o_proj": o_proj,
        "pt_mlp": {
            "estimated_flops": mlp_flops * iterations,
            "weight_bytes": mlp_weight_bytes * iterations,
            "activation_bytes_lower_bound": mlp_activation_bytes * iterations,
            "compulsory_bytes_lower_bound": (mlp_weight_bytes + mlp_activation_bytes) * iterations,
            "work_model_note": "FC1/FC2 MACs count as two FLOPs; GELU and intermediate activation traffic are excluded from the lower-bound traffic model.",
        },
    }
    return {
        "runtime": "PointLLM PointBERT eager PyTorch/CUDA kernels",
        "batch_size": batch_size,
        "num_points": num_points,
        "num_groups": num_groups,
        "group_size": group_size,
        "point_transformer_tokens": tokens,
        "point_transformer_channels": channels,
        "point_transformer_heads": heads,
        "point_transformer_layers": len(blocks),
        "profile_iterations": iterations,
        "dtype": str(neighborhood.dtype),
        "stage_ranges": STAGE_RANGES,
        "stages": stages,
        "scope_note": (
            "NCU measured DRAM bytes are the source of truth. Modeled bytes are conservative compulsory lower bounds; "
            "modeled FLOPs omit comparisons, indexing, reductions, normalization, and nonlinearities where stated."
        ),
    }


def _run_stage(name: str, fn, iterations: int):
    torch.cuda.nvtx.range_push(STAGE_RANGES[name])
    result = None
    for _ in range(iterations):
        result = fn()
    torch.cuda.nvtx.range_pop()
    return result


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16"))
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--profile_iterations", type=int, default=1)
    parser.add_argument("--manifest_output", required=True)
    parser.add_argument(
        "--gpu_precheck_json",
        default="",
        help="Idle-GPU snapshot captured by the orchestrator before NCU attaches.",
    )
    parser.add_argument("--require_idle_gpu", action="store_true")
    parser.add_argument("--idle_utilization_threshold", type=float, default=5.0)
    parser.add_argument("--idle_memory_threshold_mb", type=float, default=1024.0)
    args = parser.parse_args()
    if args.batch_size <= 0 or args.warmup < 0 or args.profile_iterations <= 0:
        parser.error("batch_size/profile_iterations must be positive and warmup non-negative")

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    properties = torch.cuda.get_device_properties(device)
    if args.gpu_precheck_json:
        gpu_before = json.loads(Path(args.gpu_precheck_json).read_text(encoding="utf-8"))
    else:
        gpu_before = query_gpu_state(f"GPU-{properties.uuid}")
    idle_at_start = (
        gpu_before["gpu_utilization_pct"] <= args.idle_utilization_threshold
        and gpu_before["memory_used_mb"] <= args.idle_memory_threshold_mb
    )
    if args.require_idle_gpu and not idle_at_start:
        parser.error(
            f"target GPU is not idle: util={gpu_before['gpu_utilization_pct']:.0f}%, "
            f"memory={gpu_before['memory_used_mb']:.0f}MiB"
        )

    model, _ = load_pointllm_model(args.model_path, device=device, dtype=dtype)
    backbone = model.get_model().point_backbone
    group_divider = backbone.group_divider
    point_cloud = torch.stack([
        make_fake_point_cloud(device, dtype) for _ in range(args.batch_size)
    ])
    xyz = point_cloud[:, :, :3].contiguous()

    centers = pb_misc.fps(xyz, group_divider.num_group)
    indices = knn_point(group_divider.group_size, xyz, centers)
    neighborhood = _gather_neighborhood(point_cloud, xyz, centers, indices)
    prepared = _prepare_transformer_inputs(backbone, neighborhood, centers)
    attn_inputs, qkv_tensors, attention_outputs, mlp_inputs = prepared
    blocks = list(backbone.blocks.blocks)

    def run_qkv():
        return [block.attn.qkv(value) for block, value in zip(blocks, attn_inputs)]

    def run_attention():
        return [_attention_core(block.attn, value) for block, value in zip(blocks, qkv_tensors)]

    def run_o_proj():
        return [block.attn.proj(value) for block, value in zip(blocks, attention_outputs)]

    def run_mlp():
        return [block.mlp(value) for block, value in zip(blocks, mlp_inputs)]

    warmup_functions = (
        lambda: pb_misc.fps(xyz, group_divider.num_group),
        lambda: knn_point(group_divider.group_size, xyz, centers),
        lambda: backbone.encoder(neighborhood),
        run_qkv,
        run_attention,
        run_o_proj,
        run_mlp,
    )
    for _ in range(args.warmup):
        for function in warmup_functions:
            function()
    torch.cuda.synchronize(device)

    manifest = build_manifest(
        backbone,
        point_cloud=point_cloud,
        xyz=xyz,
        centers=centers,
        indices=indices,
        neighborhood=neighborhood,
        attn_inputs=attn_inputs,
        qkv_tensors=qkv_tensors,
        attention_outputs=attention_outputs,
        mlp_inputs=mlp_inputs,
        iterations=args.profile_iterations,
    )
    manifest["gpu_before"] = gpu_before
    manifest["idle_at_start"] = idle_at_start
    manifest["idle_check_scope"] = (
        "orchestrator_before_ncu_attach" if args.gpu_precheck_json else "profile_process_start"
    )
    output = Path(args.manifest_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    torch.cuda.nvtx.range_push("pointbert_profile")
    _run_stage("fps", lambda: pb_misc.fps(xyz, group_divider.num_group), args.profile_iterations)
    _run_stage("knn", lambda: knn_point(group_divider.group_size, xyz, centers), args.profile_iterations)
    _run_stage("local_encoder", lambda: backbone.encoder(neighborhood), args.profile_iterations)
    _run_stage("pt_qkv", run_qkv, args.profile_iterations)
    _run_stage("pt_attention", run_attention, args.profile_iterations)
    _run_stage("pt_o_proj", run_o_proj, args.profile_iterations)
    _run_stage("pt_mlp", run_mlp, args.profile_iterations)
    torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize(device)
    print(json.dumps({
        "manifest_output": str(output),
        "stage_ranges": STAGE_RANGES,
        "idle_at_start": idle_at_start,
        "gpu_before": gpu_before,
    }, indent=2))


if __name__ == "__main__":
    main()
