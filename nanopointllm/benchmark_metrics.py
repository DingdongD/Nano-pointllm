from __future__ import annotations

import math
from typing import Iterable


GIB = 1024 ** 3


def resize_point_prompt(token_ids: list[int], target_len: int) -> list[int]:
    """Extend a valid multimodal prompt without truncating its point-token span."""
    if target_len <= 0 or target_len == len(token_ids):
        return list(token_ids)
    if target_len < len(token_ids):
        raise ValueError(
            f"PointLLM prompt cannot be shortened from {len(token_ids)} to {target_len}; "
            "the base prompt contains the complete point-token span"
        )
    tail = token_ids[-min(len(token_ids), 128):]
    out = list(token_ids)
    while len(out) < target_len:
        out.extend(tail[:target_len - len(out)])
    return out


def kv_bytes_per_token(
    *,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    dtype_bytes: int,
) -> int:
    return 2 * num_layers * num_kv_heads * head_dim * dtype_bytes


def kv_cache_metrics(
    *,
    num_layers: int,
    num_blocks: int,
    block_size: int,
    num_kv_heads: int,
    head_dim: int,
    dtype_bytes: int,
    batch_size: int,
    prompt_tokens: int,
    output_tokens: int,
) -> dict[str, float | int]:
    per_token = kv_bytes_per_token(
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        dtype_bytes=dtype_bytes,
    )
    per_block = per_token * block_size
    blocks_per_seq = math.ceil((prompt_tokens + output_tokens) / block_size)
    active_blocks = batch_size * blocks_per_seq
    payload_bytes = batch_size * (prompt_tokens + output_tokens) * per_token
    active_bytes = active_blocks * per_block
    reserved_bytes = num_blocks * per_block
    return {
        "bytes_per_token": per_token,
        "bytes_per_block": per_block,
        "blocks_per_sequence_peak": blocks_per_seq,
        "active_blocks_peak": active_blocks,
        "payload_peak_gib": payload_bytes / GIB,
        "active_allocated_peak_gib": active_bytes / GIB,
        "reserved_pool_gib": reserved_bytes / GIB,
        "block_fragmentation_gib": (active_bytes - payload_bytes) / GIB,
        "pool_utilization_peak": active_blocks / num_blocks,
    }


def estimate_decode_hbm(
    *,
    decoder_weight_bytes: int,
    kv_bytes_token: int,
    batch_size: int,
    average_context_tokens: float,
    batch_step_ms: float,
    peak_hbm_gbps: float | None,
) -> dict[str, float | None]:
    # kv_bytes_token already includes all layers and both K/V tensors.
    kv_read_bytes = kv_bytes_token * batch_size * average_context_tokens
    kv_write_bytes = kv_bytes_token * batch_size
    total_bytes = decoder_weight_bytes + kv_read_bytes + kv_write_bytes
    effective_gbps = (
        total_bytes / (batch_step_ms / 1000.0) / 1e9
        if batch_step_ms > 0
        else 0.0
    )
    return {
        "estimated_decoder_weight_read_gb_per_step": decoder_weight_bytes / 1e9,
        "estimated_kv_read_gb_per_step": kv_read_bytes / 1e9,
        "estimated_kv_write_gb_per_step": kv_write_bytes / 1e9,
        "estimated_minimum_hbm_traffic_gb_per_step": total_bytes / 1e9,
        "estimated_effective_hbm_gbps": effective_gbps,
        "estimated_peak_hbm_utilization": (
            effective_gbps / peak_hbm_gbps if peak_hbm_gbps else None
        ),
    }


def mean(values: Iterable[float]) -> float:
    vals = list(values)
    return sum(vals) / len(vals) if vals else 0.0


def identify_peak_hbm_gbps(device_name: str) -> tuple[float | None, str]:
    name = device_name.lower()
    known = (
        ("a100-sxm4-40gb", 1555.0),
        ("a100-sxm4-80gb", 2039.0),
        ("a100-pcie-40gb", 1555.0),
        ("a100-pcie-80gb", 1935.0),
        ("h100 80gb hbm3", 3350.0),
        ("h100 sxm", 3350.0),
        ("h100 pcie", 2000.0),
    )
    compact = name.replace("nvidia ", "")
    for needle, value in known:
        if needle in compact:
            return value, "device-name lookup"
    return None, "unknown; pass --hbm_peak_gbps"
