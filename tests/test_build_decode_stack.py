"""无 CUDA：仅解析 decode 栈字符串。"""
from nanopointllm.engine.decode_stack import TorchCompileDecodeWrapper, build_decode_stack


def test_build_hf():
    b = build_decode_stack("hf")
    assert b is not None


def test_build_hf_inner():
    b = build_decode_stack("hf_inner")
    assert b is not None


def test_build_hf_inner_compile_wraps_inner():
    b = build_decode_stack("hf_inner|compile")
    assert isinstance(b, TorchCompileDecodeWrapper)


def test_build_torch_loop():
    b = build_decode_stack("torch_loop")
    assert b is not None


def test_build_torch_loop_compile():
    b = build_decode_stack("torch_loop|compile")
    assert isinstance(b, TorchCompileDecodeWrapper)
