"""Bit-exact edge model for the post-accumulator INT32-to-A8 unit."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import struct


@dataclass(frozen=True)
class RequantConfig:
    source_stall_mod: int = 4
    source_stall_phase: int = 1
    output_stall_mod: int = 5
    output_stall_phase: int = 2
    max_cycles: int = 10_000

    def validate(self) -> None:
        if self.max_cycles <= 0:
            raise ValueError("max_cycles must be positive")

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class RequantVector:
    tag: int
    accumulator: int
    bias: int
    multiplier: int
    right_shift: int


@dataclass(frozen=True)
class RequantEvent:
    cycle: int
    event: str
    tag: int
    value: int


@dataclass(frozen=True)
class RequantResult:
    cycles: int
    events: tuple[RequantEvent, ...]
    outputs: tuple[int, ...]
    counters: dict[str, int]


@dataclass(frozen=True)
class CompiledRequantScale:
    activation_scale_fp16: float
    weight_scale_fp16: float
    output_scale_fp16: float
    exact_ratio: float
    multiplier: int
    right_shift: int
    represented_ratio: float
    relative_error: float


def compile_fp16_requant_scale(
    activation_scale: float, weight_scale: float, output_scale: float,
) -> CompiledRequantScale:
    """Compile positive FP16-stored scales into multiplier/right-shift form."""
    activation = _fp16(activation_scale)
    weight = _fp16(weight_scale)
    output = _fp16(output_scale)
    if min(activation, weight, output) <= 0 or not all(
        math.isfinite(item) for item in (activation, weight, output)
    ):
        raise ValueError("requant scales must be positive finite FP16 values")
    ratio = activation * weight / output
    multiplier, right_shift = compile_requant_ratio(ratio)
    represented = multiplier / (1 << right_shift)
    return CompiledRequantScale(
        activation, weight, output, ratio, multiplier, right_shift,
        represented, abs(represented - ratio) / ratio,
    )


def compile_requant_ratio(ratio: float) -> tuple[int, int]:
    if ratio <= 0 or not math.isfinite(ratio):
        raise ValueError("requant ratio must be positive and finite")
    max_multiplier = (1 << 32) - 1
    right_shift = min(63, math.floor(math.log2(max_multiplier / ratio)))
    if right_shift < 0:
        raise OverflowError("requant ratio exceeds UINT32 multiplier range")
    multiplier = round(ratio * (1 << right_shift))
    if multiplier > max_multiplier:
        right_shift -= 1
        multiplier = round(ratio * (1 << right_shift))
    if multiplier <= 0 or right_shift < 0:
        raise OverflowError("requant ratio is outside multiplier/shift range")
    return multiplier, right_shift


def requantize_int32(vector: RequantVector) -> int:
    """Apply unsigned multiplier, RNE right shift, and symmetric A8 clipping."""
    if not -(1 << 31) <= vector.accumulator < (1 << 31):
        raise ValueError("accumulator exceeds INT32")
    if not -(1 << 31) <= vector.bias < (1 << 31):
        raise ValueError("bias exceeds INT32")
    if not 0 <= vector.multiplier < (1 << 32):
        raise ValueError("multiplier exceeds UINT32")
    if not 0 <= vector.right_shift <= 63:
        raise ValueError("right_shift must be in [0, 63]")
    product = (vector.accumulator + vector.bias) * vector.multiplier
    magnitude = abs(product)
    if vector.right_shift:
        truncated = magnitude >> vector.right_shift
        remainder = magnitude & ((1 << vector.right_shift) - 1)
        half = 1 << (vector.right_shift - 1)
        if remainder > half or (remainder == half and truncated & 1):
            truncated += 1
    else:
        truncated = magnitude
    rounded = -truncated if product < 0 else truncated
    return max(-127, min(127, rounded))


def locked_requant_vectors() -> tuple[RequantVector, ...]:
    return (
        RequantVector(0, 0, 0, 0x40000000, 31),
        RequantVector(1, 253, 0, 0x40000000, 31),
        RequantVector(2, -253, 0, 0x40000000, 31),
        RequantVector(3, 5, 0, 1, 1),       # +2.5 -> +2, ties to even
        RequantVector(4, 7, 0, 1, 1),       # +3.5 -> +4
        RequantVector(5, -5, 0, 1, 1),      # -2.5 -> -2
        RequantVector(6, -7, 0, 1, 1),      # -3.5 -> -4
        RequantVector(7, 1000, 20, 1, 0),   # positive saturation
        RequantVector(8, -1000, -20, 1, 0), # symmetric negative saturation
        RequantVector(9, 100, -100, 0xFFFFFFFF, 31),
        RequantVector(10, 2_000_000_000, 0, 0x7FFFFFFF, 55),
        RequantVector(11, -2_000_000_000, 0, 0x7FFFFFFF, 55),
    )


def run_requant_model(
    vectors: tuple[RequantVector, ...] | None = None,
    config: RequantConfig | None = None,
) -> RequantResult:
    config = config or RequantConfig()
    config.validate()
    vectors = vectors or locked_requant_vectors()
    cycle = 0
    source_index = 0
    output_register: tuple[int, int] | None = None
    events: list[RequantEvent] = []
    outputs: list[int] = []
    counters = {"input_accepts": 0, "output_accepts": 0,
                "source_backpressure_cycles": 0, "output_backpressure_cycles": 0}
    while source_index < len(vectors) or output_register is not None:
        if cycle >= config.max_cycles:
            raise RuntimeError("requant model exceeded max_cycles")
        source_valid = source_index < len(vectors) and _ready(
            cycle, config.source_stall_mod, config.source_stall_phase,
        )
        output_ready = _ready(cycle, config.output_stall_mod, config.output_stall_phase)
        input_ready = output_register is None or output_ready
        accept_input = source_valid and input_ready
        accept_output = output_register is not None and output_ready
        if source_valid and not input_ready:
            counters["source_backpressure_cycles"] += 1
        if output_register is not None and not output_ready:
            counters["output_backpressure_cycles"] += 1
        incoming = None
        if accept_input:
            vector = vectors[source_index]
            incoming = (vector.tag, requantize_int32(vector))
            events.append(RequantEvent(cycle, "INPUT_ACCEPT", vector.tag, 0))
            counters["input_accepts"] += 1
            source_index += 1
        if accept_output:
            assert output_register is not None
            tag, value = output_register
            events.append(RequantEvent(cycle, "OUTPUT_ACCEPT", tag, value))
            outputs.append(value)
            counters["output_accepts"] += 1
        if incoming is not None:
            output_register = incoming
        elif accept_output:
            output_register = None
        cycle += 1
    return RequantResult(cycle, tuple(events), tuple(outputs), counters)


def _ready(cycle: int, modulus: int, phase: int) -> bool:
    return modulus == 0 or cycle % modulus != phase


def _fp16(value: float) -> float:
    return struct.unpack("e", struct.pack("e", value))[0]
