"""Bit-exact INT32 x FP16 x FP16 to BF16 dequantization model."""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Bf16DequantConfig:
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
class Bf16DequantVector:
    tag: int
    accumulator: int
    activation_scale_fp16: int
    weight_scale_fp16: int


@dataclass(frozen=True)
class Bf16DequantEvent:
    cycle: int
    event: str
    tag: int
    value: int


@dataclass(frozen=True)
class Bf16DequantResult:
    cycles: int
    events: tuple[Bf16DequantEvent, ...]
    outputs: tuple[int, ...]
    counters: dict[str, int]


def dequantize_int32_to_bf16(vector: Bf16DequantVector) -> int:
    """Round the exact product to BF16 without host floating-point arithmetic."""
    if not -(1 << 31) <= vector.accumulator < (1 << 31):
        raise ValueError("accumulator exceeds INT32")
    mant_a, exponent_a, sign_a = _decode_fp16(vector.activation_scale_fp16)
    mant_w, exponent_w, sign_w = _decode_fp16(vector.weight_scale_fp16)
    if vector.accumulator == 0 or mant_a == 0 or mant_w == 0:
        return 0

    sign = int(vector.accumulator < 0) ^ sign_a ^ sign_w
    product = abs(vector.accumulator) * mant_a * mant_w
    exponent = exponent_a + exponent_w
    top = product.bit_length() - 1
    unbiased = top + exponent

    if unbiased < -126:
        fraction = _round_power_of_two(product, -(exponent + 133))
        if fraction == 0:
            return sign << 15
        if fraction >= 128:
            return (sign << 15) | (1 << 7)
        return (sign << 15) | fraction

    significand = _round_power_of_two(product, top - 7)
    if significand >= 256:
        significand >>= 1
        unbiased += 1
    if unbiased > 127:
        return (sign << 15) | 0x7F80
    exponent_bits = unbiased + 127
    return (sign << 15) | (exponent_bits << 7) | (significand - 128)


def locked_bf16_dequant_vectors() -> tuple[Bf16DequantVector, ...]:
    return (
        Bf16DequantVector(0, 0, 0x3C00, 0x3C00),
        Bf16DequantVector(1, 1, 0x3C00, 0x3C00),
        Bf16DequantVector(2, -1, 0x3C00, 0x3C00),
        Bf16DequantVector(3, 127, 0x3400, 0x3800),
        Bf16DequantVector(4, -253, 0x2E5E, 0x2389),
        Bf16DequantVector(5, (1 << 31) - 1, 0x0400, 0x0400),
        Bf16DequantVector(6, -(1 << 31), 0x3C00, 0x3C00),
        Bf16DequantVector(7, 65537, 0x3555, 0x3AAB),
        Bf16DequantVector(8, 3, 0x0001, 0x0001),
        Bf16DequantVector(9, 123456789, 0x8001, 0x3C00),
        Bf16DequantVector(10, -7654321, 0x3C00, 0xBC00),
        Bf16DequantVector(11, 0x00FFFFFF, 0x7BFF, 0x7BFF),
    )


def run_bf16_dequant_model(
    vectors: tuple[Bf16DequantVector, ...] | None = None,
    config: Bf16DequantConfig | None = None,
) -> Bf16DequantResult:
    config = config or Bf16DequantConfig()
    config.validate()
    vectors = vectors or locked_bf16_dequant_vectors()
    cycle = source_index = 0
    output_register: tuple[int, int] | None = None
    events: list[Bf16DequantEvent] = []
    outputs: list[int] = []
    counters = {"input_accepts": 0, "output_accepts": 0,
                "source_backpressure_cycles": 0, "output_backpressure_cycles": 0}
    while source_index < len(vectors) or output_register is not None:
        if cycle >= config.max_cycles:
            raise RuntimeError("BF16 dequant model exceeded max_cycles")
        source_valid = source_index < len(vectors) and _ready(
            cycle, config.source_stall_mod, config.source_stall_phase,
        )
        output_ready = _ready(cycle, config.output_stall_mod, config.output_stall_phase)
        input_ready = output_register is None or output_ready
        input_fire = source_valid and input_ready
        output_fire = output_register is not None and output_ready
        if source_valid and not input_ready:
            counters["source_backpressure_cycles"] += 1
        if output_register is not None and not output_ready:
            counters["output_backpressure_cycles"] += 1
        incoming = None
        if input_fire:
            vector = vectors[source_index]
            incoming = (vector.tag, dequantize_int32_to_bf16(vector))
            events.append(Bf16DequantEvent(cycle, "INPUT_ACCEPT", vector.tag, 0))
            counters["input_accepts"] += 1
            source_index += 1
        if output_fire:
            assert output_register is not None
            tag, value = output_register
            events.append(Bf16DequantEvent(cycle, "OUTPUT_ACCEPT", tag, value))
            outputs.append(value)
            counters["output_accepts"] += 1
        if incoming is not None:
            output_register = incoming
        elif output_fire:
            output_register = None
        cycle += 1
    return Bf16DequantResult(cycle, tuple(events), tuple(outputs), counters)


def _decode_fp16(bits: int) -> tuple[int, int, int]:
    if not 0 <= bits < (1 << 16):
        raise ValueError("FP16 scale bits exceed 16 bits")
    sign = bits >> 15
    exponent = (bits >> 10) & 0x1F
    fraction = bits & 0x3FF
    if exponent == 0x1F:
        raise ValueError("BF16 dequant scales must be finite FP16 values")
    if exponent == 0:
        return fraction, -24, sign
    return 1024 + fraction, exponent - 25, sign


def _round_power_of_two(value: int, right_shift: int) -> int:
    if right_shift <= 0:
        return value << -right_shift
    truncated = value >> right_shift
    remainder = value & ((1 << right_shift) - 1)
    half = 1 << (right_shift - 1)
    return truncated + int(remainder > half or (remainder == half and truncated & 1))


def _ready(cycle: int, modulus: int, phase: int) -> bool:
    return modulus == 0 or cycle % modulus != phase
