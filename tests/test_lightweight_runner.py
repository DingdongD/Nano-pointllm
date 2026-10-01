import torch
import torch.nn as nn
import torch.nn.functional as F

from nanopointllm.llama.lightweight_runner import build_lightweight_llama_runner


class _RMSNorm(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden))
        self.variance_epsilon = 1.0e-6

    def forward(self, x):
        y = x.float()
        y = y * torch.rsqrt(y.pow(2).mean(dim=-1, keepdim=True) + self.variance_epsilon)
        return self.weight * y.to(x.dtype)


class _MLP(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden, hidden, bias=False)
        self.up_proj = nn.Linear(hidden, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, hidden, bias=False)
        self.act_fn = F.silu

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class _SelfAttn(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden, hidden, bias=False)

    def forward(self, hidden_states, **kwargs):
        return self.proj(hidden_states), None


class _Layer(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.input_layernorm = _RMSNorm(hidden)
        self.self_attn = _SelfAttn(hidden)
        self.post_attention_layernorm = _RMSNorm(hidden)
        self.mlp = _MLP(hidden)

    def forward(self, hidden_states, **kwargs):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, **kwargs)[0]
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return (residual + hidden_states,)


class _FakeHF(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        hidden = 4
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(8, hidden)
        self.model.layers = nn.ModuleList([_Layer(hidden), _Layer(hidden)])
        self.model.norm = _RMSNorm(hidden)
        self.lm_head = nn.Linear(hidden, 8, bias=False)

    def forward(self, input_ids):
        hidden = self.model.embed_tokens(input_ids)
        for layer in self.model.layers:
            hidden = layer(hidden)[0]
        return self.lm_head(self.model.norm(hidden))


def test_lightweight_runner_matches_manual_llama_components():
    torch.manual_seed(0)
    hf = _FakeHF()
    runner = build_lightweight_llama_runner(hf)
    input_ids = torch.tensor([[1, 2, 3]])
    torch.testing.assert_close(runner(input_ids), hf(input_ids))


def test_lightweight_runner_enables_fused_residual_norm_in_decode_graph_mode():
    hf = _FakeHF()
    runner = build_lightweight_llama_runner(hf, decode_graph_mode=True)

    assert all(layer.fuse_residual_norm for layer in runner.layers)
    assert all(
        layer.post_attention_layernorm.use_fused_residual_norm
        for layer in runner.layers
    )
    assert all(layer.mlp.gateup_input_layout == "contiguous" for layer in runner.layers)
    assert all(layer.mlp.downproj_input_layout == "contiguous" for layer in runner.layers)
    assert all(layer.mlp.gateup_split_impl == "split" for layer in runner.layers)


def test_lightweight_runner_enables_packed_qkv_in_decode_graph_mode():
    hf = _FakeHF()
    runner = build_lightweight_llama_runner(hf, decode_graph_mode=True)

    assert all(layer.use_packed_qkv_decode is False for layer in runner.layers)
    assert all(layer.use_fused_rmsnorm_packed_qkv is False for layer in runner.layers)
    assert all(layer.use_rmsnorm_staged_packed_qkv is False for layer in runner.layers)


def test_lightweight_runner_enables_rmsnorm_staged_packed_qkv_in_decode_graph_mode_with_paged_attn():
    hf = _FakeHF()
    class _FakePagedAttn(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.use_packed_qkv_decode = False

        def project_qkv_decode_rmsnorm_staged(self, *args, **kwargs):
            return None

        def project_qkv_decode_packed(self, *args, **kwargs):
            return None

        def forward(self, hidden_states, **kwargs):
            return hidden_states, None

    fake_attn = _FakePagedAttn()
    hf.model.layers[0].self_attn = fake_attn

    runner = build_lightweight_llama_runner(hf, decode_graph_mode=True)

    assert runner.layers[0].use_rmsnorm_staged_packed_qkv is True


def test_lightweight_mlp_packed_gateup_matches_reference():
    torch.manual_seed(0)
    mlp = _MLP(4)
    from nanopointllm.llama.lightweight_runner import LightweightMLP

    packed = LightweightMLP(mlp, use_packed_gateup_decode=True)
    x = torch.randn(1, 3, 4)
    torch.testing.assert_close(packed(x), mlp(x))


def test_lightweight_mlp_does_not_pack_unused_weights():
    from nanopointllm.llama.lightweight_runner import LightweightMLP

    lightweight = LightweightMLP(_MLP(4), use_packed_gateup_decode=False)

    assert lightweight.packed_gateup_weight is None
    assert lightweight.packed_gateup_weight_t is None
    assert lightweight.down_proj_weight_t is None


def test_lightweight_mlp_layout_modes_match_reference(monkeypatch):
    torch.manual_seed(0)
    mlp = _MLP(4)
    from nanopointllm.llama.lightweight_runner import LightweightMLP

    x = torch.randn(1, 3, 4)
    ref = mlp(x)
    for gateup_layout in ("auto", "contiguous", "clone"):
        for downproj_layout in ("auto", "contiguous", "clone"):
            for split_impl in ("split", "slice"):
                monkeypatch.setenv("NANOPOINTLLM_GATEUP_INPUT_LAYOUT", gateup_layout)
                monkeypatch.setenv("NANOPOINTLLM_DOWNPROJ_INPUT_LAYOUT", downproj_layout)
                monkeypatch.setenv("NANOPOINTLLM_GATEUP_SPLIT_IMPL", split_impl)
                packed = LightweightMLP(mlp, use_packed_gateup_decode=True)
                torch.testing.assert_close(packed(x), ref)


def test_lightweight_mlp_gemm_impls_match_reference():
    torch.manual_seed(0)
    mlp = _MLP(4)
    from nanopointllm.llama.lightweight_runner import LightweightMLP

    x = torch.randn(1, 3, 4)
    ref = mlp(x)
    for gateup_gemm_impl in ("linear", "matmul"):
        for downproj_gemm_impl in ("linear", "matmul"):
            packed = LightweightMLP(
                mlp,
                use_packed_gateup_decode=True,
                gateup_gemm_impl=gateup_gemm_impl,
                downproj_gemm_impl=downproj_gemm_impl,
            )
            torch.testing.assert_close(packed(x), ref)


def test_lightweight_mlp_weight_layouts_match_reference():
    torch.manual_seed(0)
    mlp = _MLP(4)
    from nanopointllm.llama.lightweight_runner import LightweightMLP

    x = torch.randn(1, 3, 4)
    ref = mlp(x)
    for gateup_weight_layout in ("split", "interleave"):
        packed = LightweightMLP(
            mlp,
            use_packed_gateup_decode=True,
            gateup_weight_layout=gateup_weight_layout,
        )
        torch.testing.assert_close(packed(x), ref)
