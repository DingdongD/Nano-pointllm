"""Analytical engines for geometry, tensor, attention, vector, and memory work."""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

from .schema import HardwareConfig, Operation, OperationResult


def ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


@dataclass
class _Estimate:
    cycles: int
    compute_cycles: int
    memory_cycles: int
    sram_cycles: int
    overhead_cycles: int
    macs: int
    flops: int
    hbm_read_bytes: int
    hbm_write_bytes: int
    sram_bytes: int
    utilization: float
    bottleneck: str
    dataflow: str
    details: dict[str, Any] = field(default_factory=dict)


class EngineModels:
    def __init__(self, config: HardwareConfig):
        config.validate()
        self.config = config

    def estimate(self, operation: Operation) -> OperationResult:
        operation.validate()
        if operation.kind == "fps":
            estimate = self._fps(operation)
        elif operation.kind == "knn":
            estimate = self._knn(operation)
        elif operation.kind == "linear":
            estimate = self._linear(operation)
        elif operation.kind == "attention":
            estimate = self._attention(operation)
        elif operation.kind == "vector":
            estimate = self._vector(operation)
        else:  # pragma: no cover - schema validation owns this guard
            raise ValueError(operation.kind)

        scale = self._calibration_scale(operation)
        if scale != 1.0:
            unscaled = estimate.cycles
            estimate.cycles = max(1, math.ceil(estimate.cycles * scale))
            estimate.details["uncalibrated_cycles"] = unscaled
            estimate.details["calibration_scale"] = scale
        energy = self._energy(operation, estimate)
        return OperationResult(
            name=operation.name,
            phase=operation.phase,
            kind=operation.kind,
            cycles=estimate.cycles,
            compute_cycles=estimate.compute_cycles,
            memory_cycles=estimate.memory_cycles,
            sram_cycles=estimate.sram_cycles,
            overhead_cycles=estimate.overhead_cycles,
            macs=estimate.macs,
            flops=estimate.flops,
            hbm_read_bytes=estimate.hbm_read_bytes,
            hbm_write_bytes=estimate.hbm_write_bytes,
            sram_bytes=estimate.sram_bytes,
            utilization=estimate.utilization,
            bottleneck=estimate.bottleneck,
            dataflow=estimate.dataflow,
            energy_pj=energy,
            details=estimate.details,
        )

    def _dense_tensor_cycles(self, m: int, n: int, k: int, bits: int, repeat: int) -> tuple[int, float]:
        tensor = self.config.tensor
        packing = tensor.precision_macs_per_cycle[bits]
        m_tiles = ceil_div(m, tensor.rows)
        n_tiles = ceil_div(n, tensor.cols)
        last_m = m - (m_tiles - 1) * tensor.rows
        last_n = n - (n_tiles - 1) * tensor.cols
        sum_m = (m_tiles - 1) * tensor.rows + last_m
        sum_n = (n_tiles - 1) * tensor.cols + last_n
        k_cycles = ceil_div(k, packing)
        constant = k_cycles + tensor.pipeline_fill_cycles - 2
        cycles_once = (
            m_tiles * n_tiles * constant
            + n_tiles * sum_m
            + m_tiles * sum_n
        )
        cycles = max(1, cycles_once * repeat)
        macs = m * n * k * repeat
        capacity = cycles * tensor.rows * tensor.cols * packing
        return cycles, min(1.0, macs / capacity)

    def _split_k_cycles(self, m: int, n: int, k: int, bits: int, repeat: int) -> tuple[int, float, int]:
        tensor = self.config.tensor
        packing = tensor.precision_macs_per_cycle[bits]
        active_m = min(m, tensor.rows)
        partitions = min(tensor.split_k, max(1, tensor.rows // active_m), k)
        m_groups = ceil_div(m, active_m)
        n_tiles = ceil_div(n, tensor.cols)
        reduction = (
            math.ceil(math.log2(partitions)) * tensor.reduction_stage_cycles
            if partitions > 1 else 0
        )
        k_cycles = ceil_div(k, partitions * packing)
        cycles_once = 0
        for tile in range(n_tiles):
            active_n = min(tensor.cols, n - tile * tensor.cols)
            cycles_once += (
                k_cycles
                + reduction
                + active_n
                + tensor.pipeline_fill_cycles
            ) * m_groups
        cycles = max(1, cycles_once * repeat)
        macs = m * n * k * repeat
        capacity = cycles * tensor.rows * tensor.cols * packing
        return cycles, min(1.0, macs / capacity), partitions

    def _linear(self, operation: Operation) -> _Estimate:
        mapping = str(operation.attributes.get("mapping", "auto"))
        use_split_k = mapping == "split_k" or (
            mapping == "auto"
            and operation.phase == "llm_decode"
            and operation.m <= self.config.tensor.decode_split_k_threshold
        )
        if use_split_k:
            compute_cycles, utilization, partitions = self._split_k_cycles(
                operation.m,
                operation.n,
                operation.k,
                operation.weight_bits,
                operation.repeat,
            )
            dataflow = "split_k_output_channel"
        else:
            compute_cycles, utilization = self._dense_tensor_cycles(
                operation.m,
                operation.n,
                operation.k,
                operation.weight_bits,
                operation.repeat,
            )
            partitions = 1
            dataflow = "dense_output_stationary"

        macs = operation.m * operation.n * operation.k * operation.repeat
        weight_elements = operation.n * operation.k * operation.repeat
        weight_payload = ceil_div(weight_elements * operation.weight_bits, 8)
        scale_count = ceil_div(weight_elements, self.config.memory.weight_scale_group)
        scale_bytes = scale_count * self.config.memory.weight_scale_bytes
        weight_bytes = weight_payload + scale_bytes
        activation_bytes_per = operation.activation_bits // 8
        input_bytes = operation.m * operation.k * activation_bytes_per * operation.repeat
        output_bytes = operation.m * operation.n * activation_bytes_per * operation.repeat
        hbm_read_bytes = weight_bytes + input_bytes
        hbm_write_bytes = output_bytes
        memory_cycles = self._hbm_cycles(hbm_read_bytes + hbm_write_bytes)
        unpack_cycles = ceil_div(weight_bytes, self.config.memory.weight_unpack_bytes_per_cycle)
        sram_bytes = weight_bytes + input_bytes + output_bytes
        sram_cycles = self._sram_cycles(sram_bytes)
        weight_partition = self.config.memory.partition_bytes(operation.phase, "weight")
        buffer_capacity = min(self.config.memory.weight_buffer_bytes, weight_partition)
        double_buffered = buffer_capacity >= 2 * self.config.memory.weight_tile_bytes
        stream_cycles = max(memory_cycles, unpack_cycles)
        if double_buffered:
            cycles = max(compute_cycles, stream_cycles, sram_cycles)
        else:
            cycles = compute_cycles + stream_cycles + sram_cycles
        cycles += self.config.tensor.pipeline_fill_cycles
        bottleneck = self._bottleneck(
            compute=compute_cycles,
            hbm=memory_cycles,
            unpack=unpack_cycles,
            sram=sram_cycles,
        )
        return _Estimate(
            cycles=cycles,
            compute_cycles=compute_cycles,
            memory_cycles=memory_cycles,
            sram_cycles=sram_cycles,
            overhead_cycles=self.config.tensor.pipeline_fill_cycles,
            macs=macs,
            flops=2 * macs,
            hbm_read_bytes=hbm_read_bytes,
            hbm_write_bytes=hbm_write_bytes,
            sram_bytes=sram_bytes,
            utilization=utilization,
            bottleneck=bottleneck,
            dataflow=dataflow,
            details={
                "split_k_partitions": partitions,
                "weight_payload_bytes": weight_payload,
                "weight_scale_bytes": scale_bytes,
                "weight_buffer_capacity_bytes": buffer_capacity,
                "double_buffered_weight_stream": double_buffered,
                "weight_unpack_cycles": unpack_cycles,
                "precision_packing": self.config.tensor.precision_macs_per_cycle[
                    operation.weight_bits
                ],
            },
        )

    def _fps(self, operation: Operation) -> _Estimate:
        geometry = self.config.geometry
        points = int(operation.attributes["points"])
        samples = int(operation.attributes["samples"])
        coordinate_dims = int(operation.attributes.get("coordinate_dims", 3))
        coordinate_bytes = int(operation.attributes.get("coordinate_bytes", 2))
        distance_bytes = int(operation.attributes.get("distance_bytes", 2))
        point_bytes = points * coordinate_dims * coordinate_bytes
        min_distance_bytes = points * distance_bytes
        resident_bytes = point_bytes + min_distance_bytes
        point_capacity = self.config.memory.partition_bytes(operation.phase, "point")
        resident = resident_bytes <= point_capacity
        reduction_cycles = (
            math.ceil(math.log2(geometry.distance_lanes))
            * geometry.reduction_stage_cycles
        )
        per_iteration = (
            ceil_div(points, geometry.distance_lanes)
            + geometry.distance_pipeline_cycles
            + reduction_cycles
            + geometry.feedback_cycles
        )
        compute_cycles = operation.batch * samples * per_iteration
        if resident:
            hbm_read_bytes = operation.batch * resident_bytes
            hbm_write_bytes = operation.batch * samples * coordinate_dims * coordinate_bytes
        else:
            hbm_read_bytes = operation.batch * samples * resident_bytes
            hbm_write_bytes = operation.batch * samples * min_distance_bytes
        memory_cycles = self._hbm_cycles(hbm_read_bytes + hbm_write_bytes)
        sram_bytes = (
            operation.batch * samples * resident_bytes if resident else 0
        )
        sram_cycles = self._sram_cycles(sram_bytes)
        preload_cycles = self._hbm_cycles(operation.batch * resident_bytes) if resident else 0
        cycles = compute_cycles + preload_cycles if resident else max(compute_cycles, memory_cycles)
        distance_updates = operation.batch * samples * points
        lane_capacity = max(1, compute_cycles * geometry.distance_lanes)
        return _Estimate(
            cycles=max(1, cycles),
            compute_cycles=compute_cycles,
            memory_cycles=memory_cycles,
            sram_cycles=sram_cycles,
            overhead_cycles=preload_cycles,
            macs=3 * distance_updates,
            flops=8 * distance_updates,
            hbm_read_bytes=hbm_read_bytes,
            hbm_write_bytes=hbm_write_bytes,
            sram_bytes=sram_bytes,
            utilization=min(1.0, distance_updates / lane_capacity),
            bottleneck="serial_feedback" if resident else self._bottleneck(compute=compute_cycles, hbm=memory_cycles),
            dataflow="persistent_distance_min_argmax",
            details={
                "resident_point_state": resident,
                "resident_working_set_bytes": resident_bytes,
                "point_partition_bytes": point_capacity,
                "cycles_per_fps_iteration": per_iteration,
                "reduction_cycles_per_iteration": reduction_cycles,
            },
        )

    def _knn(self, operation: Operation) -> _Estimate:
        distance_cycles, tensor_util = self._dense_tensor_cycles(
            operation.batch * operation.m,
            operation.n,
            operation.k,
            operation.activation_bits,
            operation.repeat,
        )
        neighbors = int(operation.attributes.get("neighbors", 1))
        comparisons = (
            operation.batch
            * operation.m
            * operation.n
            * max(1, math.ceil(math.log2(neighbors + 1)))
            * operation.repeat
        )
        selection_cycles = ceil_div(
            comparisons * self.config.geometry.topk_compare_cycles,
            self.config.geometry.topk_lanes,
        )
        compute_cycles = distance_cycles + selection_cycles
        coordinate_bytes = int(operation.attributes.get("coordinate_bytes", 2))
        input_bytes = operation.batch * (operation.m + operation.n) * operation.k * coordinate_bytes
        distance_matrix_bytes = operation.batch * operation.m * operation.n * 2
        index_bytes = operation.batch * operation.m * neighbors * 4
        hbm_read_bytes = input_bytes
        hbm_write_bytes = index_bytes
        memory_cycles = self._hbm_cycles(input_bytes + distance_matrix_bytes + index_bytes)
        sram_bytes = input_bytes + distance_matrix_bytes + index_bytes
        sram_cycles = self._sram_cycles(sram_bytes)
        cycles = max(compute_cycles, memory_cycles, sram_cycles)
        macs = operation.batch * operation.m * operation.n * operation.k * operation.repeat
        return _Estimate(
            cycles=max(1, cycles),
            compute_cycles=compute_cycles,
            memory_cycles=memory_cycles,
            sram_cycles=sram_cycles,
            overhead_cycles=selection_cycles,
            macs=macs,
            flops=2 * macs + comparisons,
            hbm_read_bytes=hbm_read_bytes,
            hbm_write_bytes=hbm_write_bytes,
            sram_bytes=sram_bytes,
            utilization=tensor_util,
            bottleneck=self._bottleneck(
                distance=distance_cycles,
                selection=selection_cycles,
                hbm=memory_cycles,
                sram=sram_cycles,
            ),
            dataflow="tensor_distance_plus_topk",
            details={
                "distance_cycles": distance_cycles,
                "selection_cycles": selection_cycles,
                "comparisons": comparisons,
                "materialized_distance_bytes": distance_matrix_bytes,
            },
        )

    def _attention(self, operation: Operation) -> _Estimate:
        head_dim = operation.hidden // operation.heads
        repeats = operation.repeat * operation.batch * operation.heads
        qk_cycles, qk_util = self._dense_tensor_cycles(
            operation.q_len,
            operation.kv_len,
            head_dim,
            operation.activation_bits,
            repeats,
        )
        av_cycles, av_util = self._dense_tensor_cycles(
            operation.q_len,
            head_dim,
            operation.kv_len,
            operation.activation_bits,
            repeats,
        )
        score_elements = (
            operation.repeat
            * operation.batch
            * operation.heads
            * operation.q_len
            * operation.kv_len
        )
        softmax_cycles = ceil_div(
            math.ceil(score_elements * self.config.attention.softmax_passes),
            self.config.attention.softmax_lanes,
        )
        compute_cycles = qk_cycles + av_cycles + softmax_cycles
        activation_bytes = operation.activation_bits // 8
        q_bytes = operation.repeat * operation.batch * operation.q_len * operation.hidden * activation_bytes
        kv_bytes = 2 * operation.repeat * operation.batch * operation.kv_len * operation.hidden * activation_bytes
        output_bytes = q_bytes
        score_bytes = score_elements * activation_bytes
        if self.config.attention.fused_online_softmax:
            hbm_read_bytes = q_bytes + kv_bytes
            hbm_write_bytes = output_bytes
            materialized_score_bytes = 0
            dataflow = "fused_online_softmax"
        else:
            hbm_read_bytes = q_bytes + kv_bytes + score_bytes
            hbm_write_bytes = output_bytes + score_bytes
            materialized_score_bytes = 2 * score_bytes
            dataflow = "materialized_score_attention"
        memory_cycles = self._hbm_cycles(hbm_read_bytes + hbm_write_bytes)
        if operation.attributes.get("paged", False):
            memory_cycles = ceil_div(
                memory_cycles * 1000,
                max(1, int(self.config.attention.paged_kv_efficiency * 1000)),
            )
            dataflow += "_paged_kv"
        sram_bytes = q_bytes + kv_bytes + output_bytes + materialized_score_bytes
        sram_cycles = self._sram_cycles(sram_bytes)
        cycles = max(compute_cycles, memory_cycles, sram_cycles)
        macs = (
            2
            * operation.repeat
            * operation.batch
            * operation.q_len
            * operation.kv_len
            * operation.hidden
        )
        tensor_capacity = max(1, (qk_cycles + av_cycles) * self.config.tensor.rows * self.config.tensor.cols)
        return _Estimate(
            cycles=max(1, cycles),
            compute_cycles=compute_cycles,
            memory_cycles=memory_cycles,
            sram_cycles=sram_cycles,
            overhead_cycles=softmax_cycles,
            macs=macs,
            flops=2 * macs + 5 * score_elements,
            hbm_read_bytes=hbm_read_bytes,
            hbm_write_bytes=hbm_write_bytes,
            sram_bytes=sram_bytes,
            utilization=min(1.0, macs / tensor_capacity),
            bottleneck=self._bottleneck(compute=compute_cycles, hbm=memory_cycles, sram=sram_cycles),
            dataflow=dataflow,
            details={
                "qk_cycles": qk_cycles,
                "av_cycles": av_cycles,
                "softmax_cycles": softmax_cycles,
                "qk_utilization": qk_util,
                "av_utilization": av_util,
                "score_elements": score_elements,
                "materialized_score_bytes": materialized_score_bytes,
            },
        )

    def _vector(self, operation: Operation) -> _Estimate:
        operations_per_element = int(operation.attributes.get("operations_per_element", 1))
        scalar_ops = operation.elements * operations_per_element * operation.repeat
        throughput = self.config.vector.lanes * self.config.vector.operations_per_lane_cycle
        compute_cycles = ceil_div(scalar_ops, throughput)
        data_bytes = operation.elements * operation.activation_bits // 8 * operation.repeat
        memory_cycles = self._hbm_cycles(2 * data_bytes)
        sram_bytes = 2 * data_bytes
        sram_cycles = self._sram_cycles(sram_bytes)
        cycles = max(compute_cycles, memory_cycles, sram_cycles)
        return _Estimate(
            cycles=max(1, cycles),
            compute_cycles=compute_cycles,
            memory_cycles=memory_cycles,
            sram_cycles=sram_cycles,
            overhead_cycles=0,
            macs=0,
            flops=scalar_ops,
            hbm_read_bytes=data_bytes,
            hbm_write_bytes=data_bytes,
            sram_bytes=sram_bytes,
            utilization=min(1.0, scalar_ops / max(1, compute_cycles * throughput)),
            bottleneck=self._bottleneck(compute=compute_cycles, hbm=memory_cycles, sram=sram_cycles),
            dataflow="fused_vector_reduction" if operation.attributes.get("fusion_candidate") else "vector",
            details={"scalar_operations": scalar_ops},
        )

    def _hbm_cycles(self, byte_count: int) -> int:
        bytes_per_cycle = (
            self.config.memory.hbm_bandwidth_gbps
            * 1e9
            / self.config.clock_hz
            * self.config.memory.hbm_efficiency
        )
        return max(1, math.ceil(byte_count / bytes_per_cycle)) if byte_count else 0

    def _sram_cycles(self, byte_count: int) -> int:
        return (
            max(1, math.ceil(byte_count / self.config.memory.sram_bytes_per_cycle))
            if byte_count else 0
        )

    @staticmethod
    def _bottleneck(**cycles: int) -> str:
        return max(cycles, key=cycles.get) if cycles else "none"

    def _calibration_scale(self, operation: Operation) -> float:
        scales = self.config.calibration_scales
        for key in (operation.name, f"kind:{operation.kind}", f"phase:{operation.phase}"):
            if key in scales:
                return float(scales[key])
        return 1.0

    def _energy(self, operation: Operation, estimate: _Estimate) -> float | None:
        characterization = self.config.characterization
        if not characterization.energy_enabled:
            return None
        mac_energy = characterization.mac_energy_pj.get(operation.weight_bits)
        if mac_energy is None:
            return None
        return (
            estimate.macs * mac_energy
            + (estimate.hbm_read_bytes + estimate.hbm_write_bytes)
            * float(characterization.hbm_energy_pj_per_byte)
            + estimate.sram_bytes * float(characterization.sram_energy_pj_per_byte)
        )
