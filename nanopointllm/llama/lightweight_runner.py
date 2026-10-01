from __future__ import annotations

import os

import torch
import torch.nn as nn
import triton
import triton.language as tl

from nanopointllm.engine.forward_context import get_forward_context
from nanopointllm.profiling import nvtx_stage


def _env_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    return value if value else default


def _decode_rows_view(hidden_states: torch.Tensor) -> torch.Tensor:
    if hidden_states.dim() == 2:
        rows = hidden_states
    else:
        rows = hidden_states[0]
    return rows if rows.is_contiguous() else rows.contiguous()


def _decode_rows_layout(hidden_states: torch.Tensor, *, mode: str) -> torch.Tensor:
    rows = hidden_states if hidden_states.dim() == 2 else hidden_states[0]
    if mode == "auto":
        return rows if rows.is_contiguous() else rows.contiguous()
    if mode == "contiguous":
        return rows.contiguous()
    if mode == "clone":
        return rows.contiguous().clone()
    raise ValueError(f"unsupported decode rows layout mode: {mode}")


def _decode_linear_rows(rows: torch.Tensor, linear: nn.Module) -> torch.Tensor:
    return torch.nn.functional.linear(
        rows,
        linear.weight,
        getattr(linear, "bias", None),
    )


def _decode_linear_rows_matmul(
    rows: torch.Tensor,
    weight_t: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    out = torch.matmul(rows, weight_t)
    if bias is not None:
        out = out + bias
    return out


def _pack_gateup_tensors(
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    gate_bias: torch.Tensor | None,
    up_bias: torch.Tensor | None,
    *,
    layout: str,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if layout == "split":
        packed_weight = torch.cat([gate_weight, up_weight], dim=0).contiguous()
        if gate_bias is None and up_bias is None:
            packed_bias = None
        else:
            parts = []
            for bias, out_features, ref in (
                (gate_bias, gate_weight.shape[0], gate_weight),
                (up_bias, up_weight.shape[0], up_weight),
            ):
                if bias is None:
                    parts.append(ref.new_zeros(out_features))
                else:
                    parts.append(bias)
            packed_bias = torch.cat(parts, dim=0).contiguous()
        return packed_weight, packed_bias
    if layout == "interleave":
        packed_weight = torch.stack([gate_weight, up_weight], dim=1).reshape(
            gate_weight.shape[0] * 2,
            gate_weight.shape[1],
        ).contiguous()
        if gate_bias is None and up_bias is None:
            packed_bias = None
        else:
            gate_bias = gate_bias if gate_bias is not None else gate_weight.new_zeros(gate_weight.shape[0])
            up_bias = up_bias if up_bias is not None else up_weight.new_zeros(up_weight.shape[0])
            packed_bias = torch.stack([gate_bias, up_bias], dim=1).reshape(-1).contiguous()
        return packed_weight, packed_bias
    raise ValueError(f"unsupported gate_up weight layout: {layout}")


def _decode_staged_linear(hidden_states: torch.Tensor, linear: nn.Module) -> torch.Tensor:
    staged = _decode_rows_view(hidden_states)
    out = _decode_linear_rows(staged, linear)
    return out.unsqueeze(0)


@triton.jit
def _rmsnorm_kernel(
    x_ptr,
    weight_ptr,
    out_ptr,
    stride_row: tl.constexpr,
    hidden_size: tl.constexpr,
    block_h: tl.constexpr,
    eps: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, block_h)
    mask = offs < hidden_size
    x = tl.load(x_ptr + row * stride_row + offs, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / hidden_size
    y = x * tl.rsqrt(var + eps) * weight
    tl.store(out_ptr + row * stride_row + offs, y, mask=mask)


@triton.jit
def _silu_mul_kernel(
    gate_ptr,
    up_ptr,
    out_ptr,
    n_elements,
    block_n: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * block_n + tl.arange(0, block_n)
    mask = offs < n_elements
    gate = tl.load(gate_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(up_ptr + offs, mask=mask, other=0.0)
    silu = gate / (1.0 + tl.exp(-gate))
    tl.store(out_ptr + offs, silu * up, mask=mask)


@triton.jit
def _residual_rmsnorm_kernel(
    x_ptr,
    residual_ptr,
    weight_ptr,
    sum_ptr,
    norm_ptr,
    stride_row: tl.constexpr,
    hidden_size: tl.constexpr,
    block_h: tl.constexpr,
    eps: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, block_h)
    mask = offs < hidden_size
    x = tl.load(x_ptr + row * stride_row + offs, mask=mask, other=0.0).to(tl.float32)
    residual = tl.load(residual_ptr + row * stride_row + offs, mask=mask, other=0.0).to(tl.float32)
    summed = x + residual
    weight = tl.load(weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(summed * summed, axis=0) / hidden_size
    normed = summed * tl.rsqrt(var + eps) * weight
    tl.store(sum_ptr + row * stride_row + offs, summed, mask=mask)
    tl.store(norm_ptr + row * stride_row + offs, normed, mask=mask)


def _rmsnorm_triton(hidden_states: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    hidden_size = hidden_states.shape[-1]
    rows = hidden_states.numel() // hidden_size
    x = hidden_states.contiguous().view(rows, hidden_size)
    out = torch.empty_like(x)
    block_h = triton.next_power_of_2(hidden_size)
    _rmsnorm_kernel[(rows,)](
        x,
        weight,
        out,
        x.stride(0),
        hidden_size=hidden_size,
        block_h=block_h,
        eps=float(eps),
    )
    return out.view_as(hidden_states)


def _silu_mul_triton(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    gate = gate.contiguous()
    up = up.contiguous()
    out = torch.empty_like(gate)
    n_elements = gate.numel()
    block_n = 256
    _silu_mul_kernel[(triton.cdiv(n_elements, block_n),)](
        gate,
        up,
        out,
        n_elements,
        block_n=block_n,
    )
    return out


def _residual_rmsnorm_triton(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    hidden_size = hidden_states.shape[-1]
    rows = hidden_states.numel() // hidden_size
    x = hidden_states.contiguous().view(rows, hidden_size)
    r = residual.contiguous().view(rows, hidden_size)
    summed = torch.empty_like(x)
    normed = torch.empty_like(x)
    block_h = triton.next_power_of_2(hidden_size)
    _residual_rmsnorm_kernel[(rows,)](
        x,
        r,
        weight,
        summed,
        normed,
        x.stride(0),
        hidden_size=hidden_size,
        block_h=block_h,
        eps=float(eps),
    )
    return summed.view_as(hidden_states), normed.view_as(hidden_states)


class LightweightRMSNorm(nn.Module):
    def __init__(
        self,
        hf_norm: nn.Module,
        *,
        use_triton: bool = False,
        use_fused_residual_norm: bool = False,
    ) -> None:
        super().__init__()
        self.weight = hf_norm.weight
        self.variance_epsilon = getattr(hf_norm, "variance_epsilon", 1.0e-6)
        self.use_triton = use_triton
        self.use_fused_residual_norm = use_fused_residual_norm

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.use_triton and hidden_states.is_cuda:
            return _rmsnorm_triton(hidden_states, self.weight, self.variance_epsilon)
        input_dtype = hidden_states.dtype
        x = hidden_states.float()
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.variance_epsilon)
        return self.weight * x.to(input_dtype)

    def residual_norm(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.use_fused_residual_norm and hidden_states.is_cuda:
            return _residual_rmsnorm_triton(
                hidden_states,
                residual,
                self.weight,
                self.variance_epsilon,
            )
        summed = hidden_states + residual
        return summed, self.forward(summed)


class LightweightMLP(nn.Module):
    def __init__(
        self,
        hf_mlp: nn.Module,
        *,
        use_triton: bool = False,
        use_packed_gateup_decode: bool = False,
        gateup_input_layout: str = "auto",
        downproj_input_layout: str = "auto",
        gateup_split_impl: str = "split",
        gateup_weight_layout: str = "split",
        gateup_gemm_impl: str = "linear",
        downproj_gemm_impl: str = "linear",
    ) -> None:
        super().__init__()
        self.gate_proj = hf_mlp.gate_proj
        self.up_proj = hf_mlp.up_proj
        self.down_proj = hf_mlp.down_proj
        self.act_fn = hf_mlp.act_fn
        self.use_triton = use_triton
        self.use_packed_gateup_decode = use_packed_gateup_decode
        self.use_staged_down_proj_decode = use_packed_gateup_decode
        self.profile_enabled = os.environ.get("NANOPOINTLLM_PROFILE_LAYERS", "0") == "1"
        self.gateup_input_layout = gateup_input_layout
        self.downproj_input_layout = downproj_input_layout
        self.gateup_split_impl = gateup_split_impl
        self.gateup_weight_layout = gateup_weight_layout
        self.gateup_gemm_impl = gateup_gemm_impl
        self.downproj_gemm_impl = downproj_gemm_impl
        self.packed_gateup_weight = None
        self.packed_gateup_bias = None
        self.packed_gateup_weight_t = None
        self.down_proj_weight_t = None
        self.down_proj_bias = getattr(self.down_proj, "bias", None)
        if self.use_packed_gateup_decode:
            gate_bias = getattr(self.gate_proj, "bias", None)
            up_bias = getattr(self.up_proj, "bias", None)
            self.packed_gateup_weight, self.packed_gateup_bias = _pack_gateup_tensors(
                self.gate_proj.weight,
                self.up_proj.weight,
                gate_bias,
                up_bias,
                layout=self.gateup_weight_layout,
            )
            self.packed_gateup_weight_t = self.packed_gateup_weight.t().contiguous()
            self.down_proj_weight_t = self.down_proj.weight.t().contiguous()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        def _mark():
            if not self.profile_enabled or not hidden_states.is_cuda:
                return None
            e = torch.cuda.Event(enable_timing=True)
            e.record()
            return e

        if self.use_packed_gateup_decode and hidden_states.is_cuda:
            assert self.packed_gateup_weight is not None
            assert self.packed_gateup_weight_t is not None
            assert self.down_proj_weight_t is not None
            staged = _decode_rows_layout(
                hidden_states,
                mode=self.gateup_input_layout,
            )
            t0 = _mark()
            if self.gateup_gemm_impl == "linear":
                gate_up = torch.nn.functional.linear(
                    staged,
                    self.packed_gateup_weight,
                    self.packed_gateup_bias,
                )
            elif self.gateup_gemm_impl == "matmul":
                gate_up = _decode_linear_rows_matmul(
                    staged,
                    self.packed_gateup_weight_t,
                    self.packed_gateup_bias,
                )
            else:
                raise ValueError(f"unsupported gate_up gemm impl: {self.gateup_gemm_impl}")
            t1 = _mark()
            split = self.gate_proj.out_features
            if self.gateup_weight_layout == "interleave":
                gate_up = gate_up.view(*gate_up.shape[:-1], split, 2)
                gate = gate_up[..., 0]
                up = gate_up[..., 1]
            elif self.gateup_split_impl == "split":
                gate, up = torch.split(gate_up, (split, split), dim=-1)
            elif self.gateup_split_impl == "slice":
                gate = gate_up[..., :split]
                up = gate_up[..., split:]
            else:
                raise ValueError(
                    f"unsupported gate_up split impl: {self.gateup_split_impl}"
                )
            fused = self.act_fn(gate) * up
            t2 = _mark()
            if self.use_staged_down_proj_decode:
                down_input = _decode_rows_layout(
                    fused,
                    mode=self.downproj_input_layout,
                )
                t3 = _mark()
                if self.downproj_gemm_impl == "linear":
                    out = _decode_linear_rows(down_input, self.down_proj).unsqueeze(0)
                elif self.downproj_gemm_impl == "matmul":
                    out = _decode_linear_rows_matmul(
                        down_input,
                        self.down_proj_weight_t,
                        self.down_proj_bias,
                    ).unsqueeze(0)
                else:
                    raise ValueError(
                        f"unsupported down_proj gemm impl: {self.downproj_gemm_impl}"
                    )
                t4 = _mark()
                if self.profile_enabled and hidden_states.is_cuda:
                    self._last_profile_events = (t0, t1, t2, t3, t4)
                return out
            out = self.down_proj(fused)
            t3 = _mark()
            if self.profile_enabled and hidden_states.is_cuda:
                self._last_profile_events = (t0, t1, t2, None, t3)
            return out.unsqueeze(0)
        t0 = _mark()
        gate = self.gate_proj(hidden_states)
        up = self.up_proj(hidden_states)
        t1 = _mark()
        if self.use_triton and hidden_states.is_cuda:
            fused = _silu_mul_triton(gate, up)
        else:
            fused = self.act_fn(gate) * up
        t2 = _mark()
        out = self.down_proj(fused)
        t3 = _mark()
        if self.profile_enabled and hidden_states.is_cuda:
            self._last_profile_events = (t0, t1, t2, None, t3)
        return out


class LightweightLlamaLayer(nn.Module):
    """Llama decoder layer without calling the HF decoder-layer wrapper."""

    def __init__(
        self,
        hf_layer: nn.Module,
        *,
        use_triton: bool = False,
        use_fused_residual_norm: bool = False,
        use_packed_qkv_decode: bool = False,
        use_fused_rmsnorm_packed_qkv: bool = False,
        use_rmsnorm_staged_packed_qkv: bool = False,
        use_packed_gateup_decode: bool = False,
        gateup_input_layout: str = "auto",
        downproj_input_layout: str = "auto",
        gateup_split_impl: str = "split",
        gateup_weight_layout: str = "split",
        gateup_gemm_impl: str = "linear",
        downproj_gemm_impl: str = "linear",
    ) -> None:
        super().__init__()
        self.input_layernorm = LightweightRMSNorm(
            hf_layer.input_layernorm,
            use_triton=use_triton,
        )
        self.self_attn = hf_layer.self_attn
        self.post_attention_layernorm = LightweightRMSNorm(
            hf_layer.post_attention_layernorm,
            use_triton=use_triton,
            use_fused_residual_norm=use_fused_residual_norm,
        )
        self.mlp = LightweightMLP(
            hf_layer.mlp,
            use_triton=use_triton,
            use_packed_gateup_decode=use_packed_gateup_decode,
            gateup_input_layout=gateup_input_layout,
            downproj_input_layout=downproj_input_layout,
            gateup_split_impl=gateup_split_impl,
            gateup_weight_layout=gateup_weight_layout,
            gateup_gemm_impl=gateup_gemm_impl,
            downproj_gemm_impl=downproj_gemm_impl,
        )
        self.fuse_residual_norm = use_fused_residual_norm
        self.profile_enabled = os.environ.get("NANOPOINTLLM_PROFILE_LAYERS", "0") == "1"
        self.profile_name = getattr(hf_layer, "layer_idx", None)
        self.use_packed_qkv_decode = (
            use_packed_qkv_decode and hasattr(self.self_attn, "project_qkv_decode_packed")
        )
        self.use_fused_rmsnorm_packed_qkv = (
            use_fused_rmsnorm_packed_qkv
            and hasattr(self.self_attn, "project_qkv_decode_fused")
        )
        self.use_rmsnorm_staged_packed_qkv = (
            use_rmsnorm_staged_packed_qkv
            and hasattr(self.self_attn, "project_qkv_decode_rmsnorm_staged")
        )
        if hasattr(self.self_attn, "use_packed_qkv_decode"):
            self.self_attn.use_packed_qkv_decode = self.use_packed_qkv_decode

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        position_embeddings=None,
        attention_mask=None,
        position_ids=None,
    ) -> torch.Tensor:
        profile = []
        def _mark():
            if not self.profile_enabled or not hidden_states.is_cuda:
                return None
            e = torch.cuda.Event(enable_timing=True)
            e.record()
            return e

        t0 = _mark()
        residual = hidden_states
        ctx = get_forward_context()
        precomputed_qkv = None
        decode_only = (
            (
                self.use_fused_rmsnorm_packed_qkv
                or self.use_rmsnorm_staged_packed_qkv
                or self.use_packed_qkv_decode
            )
            and hidden_states.is_cuda
            and ctx is not None
            and ctx.seq_lens
            and all(q_len == 1 for q_len in ctx.seq_lens)
        )
        with nvtx_stage("pointllm_qkv", hidden_states):
            if decode_only and self.use_fused_rmsnorm_packed_qkv:
                precomputed_qkv = self.self_attn.project_qkv_decode_fused(
                    hidden_states,
                    self.input_layernorm.weight,
                    self.input_layernorm.variance_epsilon,
                )
            elif decode_only and self.use_rmsnorm_staged_packed_qkv:
                precomputed_qkv = self.self_attn.project_qkv_decode_rmsnorm_staged(
                    hidden_states,
                    self.input_layernorm.weight,
                    self.input_layernorm.variance_epsilon,
                )
            else:
                hidden_states = self.input_layernorm(hidden_states)
            if decode_only and self.use_packed_qkv_decode and precomputed_qkv is None:
                precomputed_qkv = self.self_attn.project_qkv_decode_packed(hidden_states)
        t1 = _mark()
        attn_out = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            use_cache=False,
            precomputed_qkv=precomputed_qkv,
        )
        hidden_states = attn_out[0] if isinstance(attn_out, tuple) else attn_out
        t2 = _mark()
        if self.fuse_residual_norm and hidden_states.is_cuda:
            residual, hidden_states = self.post_attention_layernorm.residual_norm(
                hidden_states,
                residual,
            )
        else:
            hidden_states = residual + hidden_states
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
        t3 = _mark()
        with nvtx_stage("pointllm_mlp", hidden_states):
            hidden_states = self.mlp(hidden_states)
        t4 = _mark()
        out = residual + hidden_states
        t5 = _mark()
        if self.profile_enabled and hidden_states.is_cuda:
            setattr(
                self,
                "_last_profile_events",
                (t0, t1, t2, t3, t4, t5),
            )
        return out


class LightweightLlamaRunner(nn.Module):
    """Minimal Llama execution stack independent of HF model.forward."""

    def __init__(
        self,
        hf_model: nn.Module,
        *,
        decode_graph_mode: bool = False,
        use_triton: bool = False,
        use_fused_residual_norm: bool = False,
        use_packed_qkv_decode: bool = False,
        use_fused_rmsnorm_packed_qkv: bool = False,
        use_rmsnorm_staged_packed_qkv: bool = False,
        use_packed_gateup_decode: bool = False,
        gateup_input_layout: str = "auto",
        downproj_input_layout: str = "auto",
        gateup_split_impl: str = "split",
        gateup_weight_layout: str = "split",
        gateup_gemm_impl: str = "linear",
        downproj_gemm_impl: str = "linear",
    ) -> None:
        super().__init__()
        self.hf_model = hf_model
        model = hf_model.model if hasattr(hf_model, "model") else hf_model
        self.embed_tokens = model.embed_tokens
        self.rotary_emb = getattr(model, "rotary_emb", None)
        self.layers = nn.ModuleList(
            [
                LightweightLlamaLayer(
                    layer,
                    use_triton=use_triton,
                    use_fused_residual_norm=use_fused_residual_norm,
                    use_packed_qkv_decode=use_packed_qkv_decode,
                    use_fused_rmsnorm_packed_qkv=use_fused_rmsnorm_packed_qkv,
                    use_rmsnorm_staged_packed_qkv=use_rmsnorm_staged_packed_qkv,
                    use_packed_gateup_decode=use_packed_gateup_decode,
                    gateup_input_layout=gateup_input_layout,
                    downproj_input_layout=downproj_input_layout,
                    gateup_split_impl=gateup_split_impl,
                    gateup_weight_layout=gateup_weight_layout,
                    gateup_gemm_impl=gateup_gemm_impl,
                    downproj_gemm_impl=downproj_gemm_impl,
                )
                for layer in model.layers
            ]
        )
        self.norm = LightweightRMSNorm(model.norm, use_triton=use_triton)
        self.lm_head = hf_model.lm_head
        self.use_staged_lm_head_decode = decode_graph_mode or (
            os.environ.get("NANOPOINTLLM_STAGED_LM_HEAD_DECODE", "0") == "1"
        )
        self.profile_enabled = os.environ.get("NANOPOINTLLM_PROFILE_LAYERS", "0") == "1"
        self.last_profile: list[dict[str, float]] = []
        self.last_global_profile: dict[str, float] = {}

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        *,
        inputs_embeds: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_embeddings=None,
    ) -> torch.Tensor:
        profile_tensor = inputs_embeds if inputs_embeds is not None else input_ids

        def _mark():
            if (
                not self.profile_enabled
                or profile_tensor is None
                or not profile_tensor.is_cuda
            ):
                return None
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            return event

        profile_start = _mark()
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("input_ids or inputs_embeds is required")
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = inputs_embeds

        if position_embeddings is None and self.rotary_emb is not None:
            if position_ids is None:
                position_ids = torch.arange(
                    hidden_states.shape[1],
                    dtype=torch.long,
                    device=hidden_states.device,
                ).unsqueeze(0)
            position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)

        embed_end = _mark()

        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
        layers_end = _mark()
        hidden_states = self.norm(hidden_states)
        norm_end = _mark()
        with nvtx_stage("pointllm_lm_head", hidden_states):
            if self.use_staged_lm_head_decode and hidden_states.is_cuda:
                logits = _decode_staged_linear(hidden_states, self.lm_head)
            else:
                logits = self.lm_head(hidden_states)
        head_end = _mark()

        if (
            self.profile_enabled
            and self.layers
            and hidden_states.is_cuda
            and not torch.cuda.is_current_stream_capturing()
        ):
            assert all(event is not None for event in (
                profile_start, embed_end, layers_end, norm_end, head_end,
            ))
            head_end.synchronize()
            prof = []
            for idx, layer in enumerate(self.layers):
                events = getattr(layer, "_last_profile_events", None)
                if events is None or any(ev is None for ev in events):
                    continue
                a, b, c, d, e, f = events
                mlp_events = getattr(layer.mlp, "_last_profile_events", None)
                mlp_gateup_ms = 0.0
                mlp_act_mul_ms = 0.0
                mlp_downproj_ms = 0.0
                mlp_downproj_staging_ms = 0.0
                if mlp_events is not None and len(mlp_events) == 5 and not any(
                    ev is None for ev in (mlp_events[0], mlp_events[1], mlp_events[2], mlp_events[4])
                ):
                    m0, m1, m2, m3, m4 = mlp_events
                    mlp_gateup_ms = m0.elapsed_time(m1)
                    mlp_act_mul_ms = m1.elapsed_time(m2)
                    if m3 is None:
                        mlp_downproj_ms = m2.elapsed_time(m4)
                    else:
                        mlp_downproj_staging_ms = m2.elapsed_time(m3)
                        mlp_downproj_ms = m3.elapsed_time(m4)
                prof.append({
                    "layer": idx,
                    "pre_attn_ms": a.elapsed_time(b),
                    "attn_ms": b.elapsed_time(c),
                    "post_attn_norm_ms": c.elapsed_time(d),
                    "mlp_ms": d.elapsed_time(e),
                    "mlp_gateup_ms": mlp_gateup_ms,
                    "mlp_act_mul_ms": mlp_act_mul_ms,
                    "mlp_downproj_staging_ms": mlp_downproj_staging_ms,
                    "mlp_downproj_ms": mlp_downproj_ms,
                    "residual_ms": e.elapsed_time(f),
                })
            self.last_profile = prof
            self.last_global_profile = {
                "embedding_rope_ms": profile_start.elapsed_time(embed_end),
                "layers_ms": embed_end.elapsed_time(layers_end),
                "final_norm_ms": layers_end.elapsed_time(norm_end),
                "lm_head_ms": norm_end.elapsed_time(head_end),
                "model_total_ms": profile_start.elapsed_time(head_end),
            }
        return logits


def build_lightweight_llama_runner(
    hf_model: nn.Module,
    *,
    decode_graph_mode: bool = False,
) -> LightweightLlamaRunner:
    use_triton = os.environ.get("NANOPOINTLLM_TRITON_LIGHTWEIGHT_OPS", "0") == "1"
    use_fused_residual_norm = decode_graph_mode or (
        os.environ.get("NANOPOINTLLM_FUSED_DECODE_RMSNORM", "0") == "1"
    )
    use_packed_qkv_decode = os.environ.get("NANOPOINTLLM_PACKED_QKV_DECODE", "0") == "1"
    use_fused_rmsnorm_packed_qkv = (
        os.environ.get("NANOPOINTLLM_FUSED_RMSNORM_PACKED_QKV", "0") == "1"
    )
    use_rmsnorm_staged_packed_qkv = decode_graph_mode or (
        os.environ.get("NANOPOINTLLM_RMSNORM_STAGED_PACKED_QKV", "0") == "1"
    )
    use_packed_gateup_decode = decode_graph_mode or (
        os.environ.get("NANOPOINTLLM_PACKED_GATEUP_DECODE", "0") == "1"
    )
    gateup_input_layout = _env_str(
        "NANOPOINTLLM_GATEUP_INPUT_LAYOUT",
        "contiguous" if decode_graph_mode else "auto",
    )
    downproj_input_layout = _env_str(
        "NANOPOINTLLM_DOWNPROJ_INPUT_LAYOUT",
        "contiguous" if decode_graph_mode else "auto",
    )
    gateup_split_impl = _env_str("NANOPOINTLLM_GATEUP_SPLIT_IMPL", "split")
    gateup_weight_layout = _env_str("NANOPOINTLLM_GATEUP_WEIGHT_LAYOUT", "split")
    gateup_gemm_impl = _env_str("NANOPOINTLLM_GATEUP_GEMM_IMPL", "linear")
    downproj_gemm_impl = _env_str("NANOPOINTLLM_DOWNPROJ_GEMM_IMPL", "linear")
    return LightweightLlamaRunner(
        hf_model,
        decode_graph_mode=decode_graph_mode,
        use_triton=use_triton,
        use_fused_residual_norm=use_fused_residual_norm,
        use_packed_qkv_decode=use_packed_qkv_decode,
        use_fused_rmsnorm_packed_qkv=use_fused_rmsnorm_packed_qkv,
        use_rmsnorm_staged_packed_qkv=use_rmsnorm_staged_packed_qkv,
        use_packed_gateup_decode=use_packed_gateup_decode,
        gateup_input_layout=gateup_input_layout,
        downproj_input_layout=downproj_input_layout,
        gateup_split_impl=gateup_split_impl,
        gateup_weight_layout=gateup_weight_layout,
        gateup_gemm_impl=gateup_gemm_impl,
        downproj_gemm_impl=downproj_gemm_impl,
    )
