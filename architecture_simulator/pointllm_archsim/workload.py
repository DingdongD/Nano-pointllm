"""PointLLM workload presets derived from the checked-in model and NCU shapes."""
from __future__ import annotations

from .schema import Operation, Workload


def build_pointllm_7b_workload(
    *,
    batch_size: int = 1,
    input_tokens: int = 768,
    output_tokens: int = 128,
    precision_policy: dict[str, int] | None = None,
) -> Workload:
    if batch_size <= 0 or input_tokens <= 0 or output_tokens <= 0:
        raise ValueError("batch/input/output sizes must be positive")
    policy = {
        "pointbert": 8,
        "projector": 8,
        "decoder_qkv": 8,
        "decoder_o": 8,
        "decoder_mlp": 8,
        "lm_head": 8,
    }
    if precision_policy:
        policy.update({key: int(value) for key, value in precision_policy.items()})

    point_tokens = 513
    point_hidden = 384
    point_intermediate = 1536
    point_layers = 12
    llama_hidden = 4096
    llama_intermediate = 11008
    llama_layers = 32
    llama_heads = 32
    vocab = 32003
    group_count = 512
    group_size = 32
    grouped_points = batch_size * group_count * group_size
    operations: list[Operation] = []

    operations.extend([
        Operation(
            "fps_8192_to_512",
            "geometry",
            "fps",
            batch=batch_size,
            weight_bits=16,
            attributes={
                "points": 8192,
                "samples": group_count,
                "coordinate_dims": 3,
                "coordinate_bytes": 2,
                "distance_bytes": 2,
            },
            reference={"ncu_b1_ms": 2.95088, "ncu_b4_ms": 3.0078},
        ),
        Operation(
            "knn_512x8192_top32",
            "geometry",
            "knn",
            batch=batch_size,
            m=group_count,
            n=8192,
            k=3,
            weight_bits=16,
            attributes={"neighbors": group_size, "coordinate_bytes": 2},
            reference={"ncu_b1_ms": 0.29072, "ncu_b4_ms": 0.7462},
        ),
    ])

    for name, in_features, out_features in (
        ("local_conv_6_128", 6, 128),
        ("local_conv_128_256", 128, 256),
        ("local_conv_512_512", 512, 512),
        ("local_conv_512_256", 512, 256),
    ):
        operations.append(Operation(
            name,
            "point_encoder",
            "linear",
            m=grouped_points,
            n=out_features,
            k=in_features,
            weight_bits=policy["pointbert"],
            attributes={"mapping": "dense", "operator": "conv1x1"},
        ))
    operations.append(Operation(
        "local_encoder_bn_relu_pool",
        "point_encoder",
        "vector",
        elements=batch_size * group_count * group_size * (256 + 256),
        weight_bits=policy["pointbert"],
        attributes={"operations_per_element": 3, "fusion_candidate": True},
        reference={"ncu_b1_stage_ms": 0.9141, "ncu_b4_stage_ms": 3.0358},
    ))

    pt_m = batch_size * point_tokens
    operations.extend([
        Operation(
            "point_transformer_qkv",
            "point_encoder",
            "linear",
            repeat=point_layers,
            m=pt_m,
            n=3 * point_hidden,
            k=point_hidden,
            weight_bits=policy["pointbert"],
            attributes={"mapping": "dense", "layers": point_layers},
            reference={"ncu_b1_stage_ms": 0.1378, "ncu_b4_stage_ms": 0.2667},
        ),
        Operation(
            "point_transformer_attention",
            "point_encoder",
            "attention",
            repeat=point_layers,
            batch=batch_size,
            q_len=point_tokens,
            kv_len=point_tokens,
            heads=6,
            hidden=point_hidden,
            weight_bits=policy["pointbert"],
            attributes={"causal": False, "paged": False},
            reference={"ncu_b1_stage_ms": 1.0280, "ncu_b4_stage_ms": 2.3473},
        ),
        Operation(
            "point_transformer_o",
            "point_encoder",
            "linear",
            repeat=point_layers,
            m=pt_m,
            n=point_hidden,
            k=point_hidden,
            weight_bits=policy["pointbert"],
            attributes={"mapping": "dense", "layers": point_layers},
        ),
        Operation(
            "point_transformer_mlp_up",
            "point_encoder",
            "linear",
            repeat=point_layers,
            m=pt_m,
            n=point_intermediate,
            k=point_hidden,
            weight_bits=policy["pointbert"],
            attributes={"mapping": "dense", "layers": point_layers},
        ),
        Operation(
            "point_transformer_mlp_down",
            "point_encoder",
            "linear",
            repeat=point_layers,
            m=pt_m,
            n=point_hidden,
            k=point_intermediate,
            weight_bits=policy["pointbert"],
            attributes={"mapping": "dense", "layers": point_layers},
            reference={"ncu_b1_combined_stage_ms": 0.4196, "ncu_b4_combined_stage_ms": 0.8883},
        ),
    ])

    projector_dims = (point_hidden, 1024, 2048, llama_hidden)
    for index, (in_features, out_features) in enumerate(zip(projector_dims, projector_dims[1:])):
        operations.append(Operation(
            f"projector_fc{index}",
            "projector",
            "linear",
            m=batch_size * point_tokens,
            n=out_features,
            k=in_features,
            weight_bits=policy["projector"],
            attributes={"mapping": "dense", "gelu_after": index < 2},
        ))

    prefill_m = batch_size * input_tokens
    operations.extend([
        Operation(
            "prefill_qkv",
            "llm_prefill",
            "linear",
            repeat=llama_layers,
            m=prefill_m,
            n=3 * llama_hidden,
            k=llama_hidden,
            weight_bits=policy["decoder_qkv"],
            attributes={"mapping": "dense", "layers": llama_layers},
        ),
        Operation(
            "prefill_attention",
            "llm_prefill",
            "attention",
            repeat=llama_layers,
            batch=batch_size,
            q_len=input_tokens,
            kv_len=input_tokens,
            heads=llama_heads,
            hidden=llama_hidden,
            weight_bits=policy["decoder_qkv"],
            attributes={"causal": True, "paged": False},
        ),
        Operation(
            "prefill_o",
            "llm_prefill",
            "linear",
            repeat=llama_layers,
            m=prefill_m,
            n=llama_hidden,
            k=llama_hidden,
            weight_bits=policy["decoder_o"],
            attributes={"mapping": "dense", "layers": llama_layers},
        ),
        Operation(
            "prefill_gate_up",
            "llm_prefill",
            "linear",
            repeat=2 * llama_layers,
            m=prefill_m,
            n=llama_intermediate,
            k=llama_hidden,
            weight_bits=policy["decoder_mlp"],
            attributes={"mapping": "dense", "layers": llama_layers},
        ),
        Operation(
            "prefill_down",
            "llm_prefill",
            "linear",
            repeat=llama_layers,
            m=prefill_m,
            n=llama_hidden,
            k=llama_intermediate,
            weight_bits=policy["decoder_mlp"],
            attributes={"mapping": "dense", "layers": llama_layers},
        ),
        Operation(
            "prefill_last_token_lm_head",
            "llm_prefill",
            "linear",
            m=batch_size,
            n=vocab,
            k=llama_hidden,
            weight_bits=policy["lm_head"],
            attributes={"mapping": "split_k", "last_token_only": True},
        ),
    ])

    decode_m = batch_size
    linear_repeat = llama_layers * output_tokens
    operations.extend([
        Operation(
            "decode_qkv",
            "llm_decode",
            "linear",
            repeat=linear_repeat,
            m=decode_m,
            n=3 * llama_hidden,
            k=llama_hidden,
            weight_bits=policy["decoder_qkv"],
            attributes={"mapping": "split_k", "layers": llama_layers, "tokens": output_tokens},
        ),
        Operation(
            "decode_o",
            "llm_decode",
            "linear",
            repeat=linear_repeat,
            m=decode_m,
            n=llama_hidden,
            k=llama_hidden,
            weight_bits=policy["decoder_o"],
            attributes={"mapping": "split_k", "layers": llama_layers, "tokens": output_tokens},
        ),
        Operation(
            "decode_gate_up",
            "llm_decode",
            "linear",
            repeat=2 * linear_repeat,
            m=decode_m,
            n=llama_intermediate,
            k=llama_hidden,
            weight_bits=policy["decoder_mlp"],
            attributes={"mapping": "split_k", "layers": llama_layers, "tokens": output_tokens},
        ),
        Operation(
            "decode_down",
            "llm_decode",
            "linear",
            repeat=linear_repeat,
            m=decode_m,
            n=llama_hidden,
            k=llama_intermediate,
            weight_bits=policy["decoder_mlp"],
            attributes={"mapping": "split_k", "layers": llama_layers, "tokens": output_tokens},
        ),
    ])
    for token_index in range(output_tokens):
        operations.append(Operation(
            f"decode_attention_t{token_index:03d}",
            "llm_decode",
            "attention",
            repeat=llama_layers,
            batch=batch_size,
            q_len=1,
            kv_len=input_tokens + token_index,
            heads=llama_heads,
            hidden=llama_hidden,
            weight_bits=policy["decoder_qkv"],
            attributes={"causal": True, "paged": True, "token_index": token_index},
        ))
    operations.append(Operation(
        "decode_lm_head",
        "llm_decode",
        "linear",
        repeat=output_tokens,
        m=decode_m,
        n=vocab,
        k=llama_hidden,
        weight_bits=policy["lm_head"],
        attributes={"mapping": "split_k", "tokens": output_tokens},
    ))

    workload = Workload(
        name=f"pointllm_7b_b{batch_size}_s{input_tokens}_o{output_tokens}",
        operations=tuple(operations),
        metadata={
            "model": "PointLLM-7B-v1.2",
            "batch_size": batch_size,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "source_checkpoint": "PointLLM_7B_v1.2_safetensors/config.json",
            "point_shape": "8192 -> 512x32 -> 513x384",
            "llama": {
                "hidden": llama_hidden,
                "intermediate": llama_intermediate,
                "layers": llama_layers,
                "heads": llama_heads,
                "vocab": vocab,
            },
            "precision_policy": policy,
            "evidence": [
                "docs/results/pointbert_ncu_roofline_2026-10-01",
                "docs/results/ncu_weight_bound_2026-09-30",
                "docs/results/compression_sensitivity_2026-10-01",
            ],
        },
    )
    workload.validate()
    return workload
