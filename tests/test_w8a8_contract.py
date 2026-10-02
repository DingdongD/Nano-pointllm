import pytest
import torch
import torch.nn as nn

from nanopointllm.compression.w8a8 import (
    CONTRACT, ActivationAbsmaxCollector, SequentialActivationQDQ,
    apply_smoothquant_transform, apply_weight_qdq, dynamic_per_token_activation_qdq,
    groupwise_weight_qdq, integer_linear_per_output_reference,
    per_output_weight_qdq, quantize_symmetric, static_activation_qdq,
    static_activation_scale,
)


def test_locked_symmetric_rounding_and_clipping_contract():
    scale = torch.tensor(1.0)
    values = torch.tensor([-200.0, -127.5, -2.5, -1.5, 1.5, 2.5, 127.5, 200.0])
    quantized = quantize_symmetric(values, scale)

    assert CONTRACT.rounding == "round_to_nearest_ties_to_even"
    assert quantized.tolist() == [-127, -127, -2, -2, 2, 2, 127, 127]


def test_per_output_and_group_weight_shapes_and_error():
    torch.manual_seed(4)
    weight = torch.randn(5, 259)
    per_output, q_per_output, per_output_scales = per_output_weight_qdq(weight)
    grouped, q_grouped, group_scales = groupwise_weight_qdq(weight, group_size=128)

    assert per_output.shape == grouped.shape == weight.shape
    assert q_per_output.dtype == q_grouped.dtype == torch.int8
    assert per_output_scales.shape == (5,)
    assert group_scales.shape == (5, 3)
    assert torch.mean((grouped - weight).square()) <= torch.mean((per_output - weight).square())

    convolution = torch.randn(4, 6, 1)
    conv_grouped, conv_quantized, conv_scales = groupwise_weight_qdq(
        convolution, group_size=4,
    )
    assert conv_grouped.shape == conv_quantized.shape == convolution.shape
    assert conv_scales.shape == (4, 2)


def test_static_and_dynamic_activation_qdq():
    values = torch.tensor([[[0.0, 1.0, -2.0], [0.1, 40.0, -0.2]]])
    scale = static_activation_scale(values.abs().max())
    static = static_activation_qdq(values, scale)
    dynamic, dynamic_scales = dynamic_per_token_activation_qdq(values)

    assert static.shape == dynamic.shape == values.shape
    assert dynamic_scales.shape == (1, 2, 1)
    assert torch.mean((dynamic - values).square()) < torch.mean((static - values).square())


def test_integer_linear_reference_uses_int32_accumulator():
    values = torch.tensor([[1.0, -2.0, 3.0, -4.0]])
    weight = torch.tensor([[2.0, 1.0, -1.0, 3.0], [-2.0, 4.0, 1.0, 0.0]])
    scale = static_activation_scale(values.abs().max())
    output, accumulator = integer_linear_per_output_reference(
        values, weight, activation_scale=scale, bias=torch.tensor([0.5, -0.5]),
    )

    assert accumulator.dtype == torch.int32
    assert output.shape == (1, 2)
    assert torch.isfinite(output).all()


def test_sequential_activation_qdq_propagates_to_later_layer():
    model = nn.Sequential(nn.Linear(4, 4, bias=False), nn.Linear(4, 2, bias=False))
    modules = {"first": model[0], "second": model[1]}
    values = torch.tensor([[0.03, -0.7, 1.1, 4.3]])
    with ActivationAbsmaxCollector(modules) as collector:
        baseline = model(values)
    scales = collector.static_scales()
    captured_second_input = []
    handle = model[1].register_forward_pre_hook(
        lambda _module, inputs: captured_second_input.append(inputs[0].detach().clone())
    )
    try:
        with SequentialActivationQDQ(modules, mode="static", static_scales=scales):
            candidate = model(values)
    finally:
        handle.remove()

    expected_first = model[0](static_activation_qdq(values, scales["first"]))
    torch.testing.assert_close(captured_second_input[0], expected_first)
    assert not torch.equal(candidate, baseline)


def test_apply_weight_qdq_changes_all_selected_modules():
    model = nn.Sequential(nn.Linear(7, 5), nn.Linear(5, 3))
    originals = [module.weight.detach().clone() for module in model]
    report = apply_weight_qdq(
        {"first": model[0], "second": model[1]}, granularity="group", group_size=4,
    )

    assert set(report) == {"first", "second"}
    assert all(not torch.equal(module.weight, original) for module, original in zip(model, originals))


def test_smoothquant_transform_is_exact_before_qdq_for_linear_and_conv():
    torch.manual_seed(8)
    linear = nn.Linear(5, 3)
    linear_input = torch.randn(2, 4, 5)
    linear_reference = linear(linear_input)
    with ActivationAbsmaxCollector({"linear": linear}) as collector:
        linear(linear_input)
    smoothing, activation_scales = apply_smoothquant_transform(
        {"linear": linear}, collector.channel_absmax, alpha=0.5,
    )
    transformed = linear(linear_input / smoothing["linear"])
    torch.testing.assert_close(transformed, linear_reference, rtol=2e-3, atol=2e-3)
    assert activation_scales["linear"].numel() == 1

    conv = nn.Conv1d(4, 6, 1)
    conv_input = torch.randn(2, 4, 7)
    conv_reference = conv(conv_input)
    with ActivationAbsmaxCollector({"conv": conv}) as collector:
        conv(conv_input)
    smoothing, _ = apply_smoothquant_transform(
        {"conv": conv}, collector.channel_absmax, alpha=0.5,
    )
    transformed = conv(conv_input / smoothing["conv"].reshape(1, 4, 1))
    torch.testing.assert_close(transformed, conv_reference, rtol=2e-3, atol=2e-3)


def test_invalid_contract_arguments_fail_closed():
    with pytest.raises(ValueError, match="positive"):
        quantize_symmetric(torch.ones(1), torch.tensor(0.0))
    with pytest.raises(ValueError, match="group_size"):
        groupwise_weight_qdq(torch.ones(2, 2), group_size=0)
    with pytest.raises(ValueError, match="alpha"):
        apply_smoothquant_transform({}, {}, alpha=1.1)
