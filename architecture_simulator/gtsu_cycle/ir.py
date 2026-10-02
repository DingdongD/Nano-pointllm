"""Shared micro-op contract for RTL-locked GTSU slices."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

from .splitk_gemv import SplitKGemvConfig


class MicroOpcode(str, Enum):
    LD_W = "LD_W"
    UNPACK_W8 = "UNPACK_W8"
    GEMV_SPLITK = "GEMV_SPLITK"
    PSUM_REDUCE = "PSUM_REDUCE"
    STORE = "STORE"


@dataclass(frozen=True)
class MicroOp:
    sequence: int
    opcode: MicroOpcode
    operator: str
    n: int
    partition: int
    chunk: int
    bytes: int
    precision: str
    dependencies: tuple[int, ...]

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["opcode"] = self.opcode.value
        return payload


def lower_splitk_gemv(config: SplitKGemvConfig) -> tuple[MicroOp, ...]:
    """Lower the locked nested-loop schedule into dependency-ordered micro-ops."""
    config.validate()
    operations: list[MicroOp] = []

    def emit(
        opcode: MicroOpcode,
        n_index: int,
        partition: int,
        chunk: int,
        byte_count: int,
        dependencies: tuple[int, ...],
    ) -> int:
        sequence = len(operations)
        operations.append(MicroOp(
            sequence=sequence,
            opcode=opcode,
            operator="w8_splitk_gemv_vertical_slice",
            n=n_index,
            partition=partition,
            chunk=chunk,
            bytes=byte_count,
            precision="W8A8_ACC32",
            dependencies=dependencies,
        ))
        return sequence

    for n_index in range(config.n_outputs):
        previous_partial: int | None = None
        for partition in range(config.split_k):
            previous_compute: int | None = None
            for chunk in range(config.k_chunks):
                load = emit(
                    MicroOpcode.LD_W, n_index, partition, chunk,
                    config.lanes, (),
                )
                unpack = emit(
                    MicroOpcode.UNPACK_W8, n_index, partition, chunk,
                    config.lanes * 2, (load,),
                )
                previous_compute = emit(
                    MicroOpcode.GEMV_SPLITK, n_index, partition, chunk,
                    0,
                    (unpack,) if previous_compute is None else (unpack, previous_compute),
                )
            reduction_dependencies = (
                (previous_compute,)
                if previous_partial is None
                else (previous_compute, previous_partial)
            )
            previous_partial = emit(
                MicroOpcode.PSUM_REDUCE, n_index, partition,
                config.k_chunks - 1, 4, reduction_dependencies,
            )
        emit(
            MicroOpcode.STORE, n_index, config.split_k - 1,
            config.k_chunks - 1, 4, (previous_partial,),
        )
    return tuple(operations)
