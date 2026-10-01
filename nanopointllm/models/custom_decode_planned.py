"""
M3+ 占位：自定义 decode（CUDA / Triton / 块 KV）。

实现时请继承 `nanopointllm.engine.DecodeBackend`，使 `decode_step` 返回
与 HF 一致的 `(logits, past_key_values)`，再接入 `hybrid_greedy_decode(..., decode_backend=...)`。
"""


class CustomLlamaDecodeNotImplemented(NotImplementedError):
    pass


def decode_step_stub(*args, **kwargs) -> None:
    raise CustomLlamaDecodeNotImplemented(
        "实现 `DecodeBackend.decode_step`，与 HF KV layout 对齐后再接 bench；"
        "见 /home/PointLLM/docs/fork_nano_decode_plan.md"
    )
