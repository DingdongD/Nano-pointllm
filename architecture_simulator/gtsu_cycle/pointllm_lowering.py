"""Checkpoint-backed PointLLM decoder lowering into memory-addressed W8 tiles.

This module validates source tensor shapes directly from safetensors headers,
then describes the proposed accelerator's quantized target layout.  It does not
pretend that the FP32 checkpoint is already physically stored as W8.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import struct
from typing import Any, Iterator

from .sram import SramConfig, decode_sram_address


PROJECTIONS = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj", "lm_head",
)


@dataclass(frozen=True)
class LoweringConfig:
    target_weight_bits: int = 8
    target_activation_bits: int = 8
    scale_group: int = 128
    scale_bytes: int = 2
    split_k: int = 16
    output_tile: int = 16
    dram_burst_bytes: int = 64
    dram_base: int = 0x1_0000_0000
    dram_alignment: int = 4096
    activation_sram_base: int = 0
    weight_sram_base: int = 1 * 1024 * 1024
    partial_sram_base: int = 3 * 1024 * 1024
    output_sram_base: int = 3 * 1024 * 1024 + 64 * 1024

    def validate(self) -> None:
        if self.target_weight_bits != 8 or self.target_activation_bits != 8:
            raise ValueError("the current RTL-locked compute contract is W8A8")
        for name in (
            "scale_group", "scale_bytes", "split_k", "output_tile",
            "dram_burst_bytes", "dram_alignment",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.dram_alignment % self.dram_burst_bytes:
            raise ValueError("DRAM alignment must be a multiple of burst bytes")


@dataclass(frozen=True)
class SourceTensor:
    name: str
    shard: str
    dtype: str
    shape: tuple[int, ...]
    data_offsets: tuple[int, int]

    @property
    def payload_bytes(self) -> int:
        return self.data_offsets[1] - self.data_offsets[0]

    @property
    def expected_payload_bytes(self) -> int:
        dtype_bytes = {"F32": 4, "F16": 2, "BF16": 2}.get(self.dtype)
        if dtype_bytes is None:
            raise ValueError(f"unsupported source dtype {self.dtype!r}")
        return math.prod(self.shape) * dtype_bytes


@dataclass(frozen=True)
class MemoryRegion:
    name: str
    space: str
    base: int
    payload_bytes: int
    allocated_bytes: int
    precision: str

    @property
    def end(self) -> int:
        return self.base + self.allocated_bytes


@dataclass(frozen=True)
class LinearTile:
    tile_id: int
    n_start: int
    n_count: int
    split_index: int
    k_start: int
    k_count: int
    weight_payload_bytes: int
    weight_bursts: int
    activation_sram_address: int
    weight_sram_address: int
    partial_sram_address: int


@dataclass(frozen=True)
class DramBurst:
    sequence: int
    address: int
    size: int
    stream: str
    tile_id: int
    output_row: int
    split_index: int
    chunk_index: int
    chunks_in_split: int


@dataclass(frozen=True)
class LoweredLinear:
    projection: str
    layer: int | None
    token: int
    m: int
    n: int
    k: int
    source: SourceTensor
    config: LoweringConfig
    sram_config: SramConfig
    regions: tuple[MemoryRegion, ...]
    tiles: tuple[LinearTile, ...]

    @property
    def target_weight_payload_bytes(self) -> int:
        return self.n * self.k * self.config.target_weight_bits // 8

    @property
    def target_scale_payload_bytes(self) -> int:
        groups = math.ceil(self.k / self.config.scale_group)
        return self.n * groups * self.config.scale_bytes

    @property
    def weight_burst_count(self) -> int:
        return sum(tile.weight_bursts for tile in self.tiles)

    @property
    def scale_burst_count(self) -> int:
        return math.ceil(self.target_scale_payload_bytes / self.config.dram_burst_bytes)

    def iter_weight_bursts(self) -> Iterator[DramBurst]:
        weight = _region(self.regions, "weight_w8")
        sequence = 0
        bytes_per_weight = self.config.target_weight_bits // 8
        for tile in self.tiles:
            for output_row in range(tile.n_start, tile.n_start + tile.n_count):
                row_address = weight.base + (
                    output_row * self.k + tile.k_start
                ) * bytes_per_weight
                remaining = tile.k_count * bytes_per_weight
                chunk_index = 0
                while remaining:
                    size = min(self.config.dram_burst_bytes, remaining)
                    if row_address % self.config.dram_burst_bytes:
                        raise AssertionError("lowered weight burst is not DRAM aligned")
                    if size != self.config.dram_burst_bytes:
                        raise AssertionError("partial weight bursts are not supported")
                    yield DramBurst(
                        sequence, row_address, size, "weight", tile.tile_id,
                        output_row, tile.split_index, chunk_index,
                        tile.k_count * bytes_per_weight // self.config.dram_burst_bytes,
                    )
                    sequence += 1
                    chunk_index += 1
                    row_address += size
                    remaining -= size

    def iter_scale_bursts(self) -> Iterator[DramBurst]:
        scale = _region(self.regions, "weight_scale")
        if scale.payload_bytes % self.config.dram_burst_bytes:
            raise AssertionError("scale payload must be padded before DRAM transfer")
        for index in range(self.scale_burst_count):
            yield DramBurst(
                self.weight_burst_count + index,
                scale.base + index * self.config.dram_burst_bytes,
                self.config.dram_burst_bytes,
                "scale",
                -1,
                -1,
                -1,
                -1,
                -1,
            )

    def sram_region_bank_histograms(self) -> dict[str, list[int]]:
        histograms: dict[str, list[int]] = {}
        for region in self.regions:
            if region.space != "sram":
                continue
            counts = [0] * self.sram_config.banks
            for offset in range(0, region.allocated_bytes, self.sram_config.word_bytes):
                bank, _ = decode_sram_address(self.sram_config, region.base + offset)
                counts[bank] += 1
            histograms[region.name] = counts
        return histograms

    def as_dict(self, *, sample_tiles: int = 4, sample_bursts: int = 16) -> dict[str, Any]:
        bursts = []
        for burst in self.iter_weight_bursts():
            if len(bursts) >= sample_bursts:
                break
            bursts.append(asdict(burst))
        region_rows = [asdict(region) | {"end": region.end} for region in self.regions]
        return {
            "schema_version": "0.1",
            "projection": self.projection,
            "layer": self.layer,
            "token": self.token,
            "shape": {"m": self.m, "n": self.n, "k": self.k},
            "source_tensor": asdict(self.source) | {
                "payload_bytes": self.source.payload_bytes,
                "expected_payload_bytes": self.source.expected_payload_bytes,
            },
            "target_layout": {
                "precision": "W8A8_ACC32",
                "quantization_required": True,
                "source_is_target_layout": False,
                "weight_payload_bytes": self.target_weight_payload_bytes,
                "scale_payload_bytes": self.target_scale_payload_bytes,
                "weight_burst_count": self.weight_burst_count,
                "scale_burst_count": self.scale_burst_count,
                "dram_request_count": self.weight_burst_count + self.scale_burst_count,
                "tile_count": len(self.tiles),
                "config": asdict(self.config),
            },
            "sram": {
                "config": asdict(self.sram_config),
                "address_mapping": "bank=(address/16)%16,row=(address/16)/16",
                "region_bank_word_histograms": self.sram_region_bank_histograms(),
            },
            "regions": region_rows,
            "sample_tiles": [asdict(tile) for tile in self.tiles[:sample_tiles]],
            "sample_weight_bursts": bursts,
            "validation": validate_lowered_linear(self),
            "fidelity": {
                "checkpoint_shape_exact": True,
                "address_and_byte_exact": True,
                "dram_timing_included": False,
                "compute_memory_cycle_correlated": False,
                "full_model_cycle_accurate": False,
            },
        }


def load_source_tensor(checkpoint: str | Path, tensor_name: str) -> SourceTensor:
    root = Path(checkpoint)
    index_path = root / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    shard_name = index["weight_map"].get(tensor_name)
    if shard_name is None:
        raise KeyError(f"tensor {tensor_name!r} is absent from {index_path}")
    shard = root / shard_name
    with shard.open("rb") as handle:
        header_size_raw = handle.read(8)
        if len(header_size_raw) != 8:
            raise ValueError(f"invalid safetensors header in {shard}")
        header_size = struct.unpack("<Q", header_size_raw)[0]
        header = json.loads(handle.read(header_size))
    metadata = header[tensor_name]
    return SourceTensor(
        name=tensor_name,
        shard=shard_name,
        dtype=str(metadata["dtype"]),
        shape=tuple(int(value) for value in metadata["shape"]),
        data_offsets=tuple(int(value) for value in metadata["data_offsets"]),
    )


def lower_decoder_linear(
    checkpoint: str | Path,
    projection: str,
    *,
    layer: int = 0,
    token: int = 0,
    batch_size: int = 1,
    config: LoweringConfig | None = None,
    sram_config: SramConfig | None = None,
) -> LoweredLinear:
    """Lower one PointLLM decode linear invocation using real tensor metadata."""
    if projection not in PROJECTIONS:
        raise ValueError(f"unsupported projection {projection!r}; expected one of {PROJECTIONS}")
    if batch_size <= 0 or token < 0 or layer < 0:
        raise ValueError("batch_size must be positive and layer/token non-negative")
    config = config or LoweringConfig()
    sram_config = sram_config or SramConfig()
    config.validate()
    sram_config.validate()

    model_config = json.loads((Path(checkpoint) / "config.json").read_text(encoding="utf-8"))
    layers = int(model_config["num_hidden_layers"])
    if projection != "lm_head" and layer >= layers:
        raise ValueError(f"layer {layer} is outside the {layers}-layer checkpoint")
    tensor_name = _tensor_name(projection, layer)
    source = load_source_tensor(checkpoint, tensor_name)
    if len(source.shape) != 2:
        raise ValueError(f"linear tensor {tensor_name} is not rank two: {source.shape}")
    n, k = source.shape
    expected = _expected_shape(model_config, projection)
    if (n, k) != expected:
        raise ValueError(f"checkpoint shape mismatch for {tensor_name}: {(n, k)} != {expected}")
    bytes_per_weight = config.target_weight_bits // 8
    if (k * bytes_per_weight) % config.dram_burst_bytes:
        raise ValueError("each target weight row must contain whole DRAM bursts")
    row_bursts = k * bytes_per_weight // config.dram_burst_bytes
    if config.split_k > row_bursts:
        raise ValueError("split_k cannot exceed the number of bursts in one weight row")
    base_bursts, extra_bursts = divmod(row_bursts, config.split_k)
    k_ranges: list[tuple[int, int]] = []
    k_cursor = 0
    for split_index in range(config.split_k):
        partition_bursts = base_bursts + int(split_index < extra_bursts)
        k_count = partition_bursts * config.dram_burst_bytes // bytes_per_weight
        k_ranges.append((k_cursor, k_count))
        k_cursor += k_count
    if k_cursor != k:
        raise AssertionError(f"balanced burst partition lost K elements: {k_cursor} != {k}")

    weight_payload = n * k * bytes_per_weight
    scales_per_row = math.ceil(k / config.scale_group)
    scale_payload = n * scales_per_row * config.scale_bytes
    weight_base = _target_tensor_base(model_config, projection, layer, config)
    scale_base = _align(weight_base + weight_payload, config.dram_alignment)

    activation_bytes = k * batch_size * config.target_activation_bits // 8
    weight_tile_bytes = config.output_tile * max(count for _, count in k_ranges) * bytes_per_weight
    partial_bytes = config.output_tile * config.split_k * 4
    output_bytes = config.output_tile * 4
    regions = (
        MemoryRegion("weight_w8", "dram", weight_base, weight_payload,
                     _align(weight_payload, config.dram_alignment), "W8"),
        MemoryRegion("weight_scale", "dram", scale_base, scale_payload,
                     _align(scale_payload, config.dram_alignment), "FP16"),
        MemoryRegion("activation_a8", "sram", config.activation_sram_base,
                     activation_bytes, _align(activation_bytes, sram_config.word_bytes), "A8"),
        MemoryRegion("weight_tile_w8", "sram", config.weight_sram_base,
                     weight_tile_bytes, _align(weight_tile_bytes, sram_config.word_bytes), "W8"),
        MemoryRegion("partial_acc32", "sram", config.partial_sram_base,
                     partial_bytes, _align(partial_bytes, sram_config.word_bytes), "ACC32"),
        MemoryRegion("output_acc32", "sram", config.output_sram_base,
                     output_bytes, _align(output_bytes, sram_config.word_bytes), "ACC32"),
    )

    tiles: list[LinearTile] = []
    tile_id = 0
    for n_start in range(0, n, config.output_tile):
        n_count = min(config.output_tile, n - n_start)
        for split_index, (k_start, k_count) in enumerate(k_ranges):
            tile_payload = n_count * k_count * bytes_per_weight
            tiles.append(LinearTile(
                tile_id=tile_id,
                n_start=n_start,
                n_count=n_count,
                split_index=split_index,
                k_start=k_start,
                k_count=k_count,
                weight_payload_bytes=tile_payload,
                weight_bursts=tile_payload // config.dram_burst_bytes,
                activation_sram_address=(
                    config.activation_sram_base + k_start
                    * config.target_activation_bits // 8
                ),
                weight_sram_address=config.weight_sram_base,
                partial_sram_address=(
                    config.partial_sram_base + split_index * config.output_tile * 4
                ),
            ))
            tile_id += 1

    lowered = LoweredLinear(
        projection=projection,
        layer=None if projection == "lm_head" else layer,
        token=token,
        m=batch_size,
        n=n,
        k=k,
        source=source,
        config=config,
        sram_config=sram_config,
        regions=regions,
        tiles=tuple(tiles),
    )
    validation = validate_lowered_linear(lowered)
    if not validation["passed"]:
        raise AssertionError(f"invalid PointLLM lowering: {validation}")
    return lowered


def lower_decoder_manifest(
    checkpoint: str | Path,
    *,
    layer: int = 0,
    token: int = 0,
    batch_size: int = 1,
    config: LoweringConfig | None = None,
) -> dict[str, Any]:
    """Lower every decoder projection for one layer/token into a compact manifest."""
    config = config or LoweringConfig()
    lowered = [
        lower_decoder_linear(
            checkpoint, projection, layer=layer, token=token,
            batch_size=batch_size, config=config,
        )
        for projection in PROJECTIONS
    ]
    all_dram_regions = sorted(
        (
            (region.base, region.end, f"{operator.projection}:{region.name}")
            for operator in lowered
            for region in operator.regions
            if region.space == "dram"
        ),
        key=lambda item: item[0],
    )
    no_overlap = all(
        left[1] <= right[0]
        for left, right in zip(all_dram_regions, all_dram_regions[1:])
    )
    if not no_overlap:
        raise AssertionError(f"global decoder DRAM regions overlap: {all_dram_regions}")
    return {
        "schema_version": "0.1",
        "checkpoint": str(Path(checkpoint).resolve()),
        "layer": layer,
        "token": token,
        "batch_size": batch_size,
        "operators": [operator.as_dict() for operator in lowered],
        "totals": {
            "target_weight_payload_bytes": sum(
                operator.target_weight_payload_bytes for operator in lowered
            ),
            "target_scale_payload_bytes": sum(
                operator.target_scale_payload_bytes for operator in lowered
            ),
            "weight_burst_count": sum(operator.weight_burst_count for operator in lowered),
            "scale_burst_count": sum(operator.scale_burst_count for operator in lowered),
            "dram_request_count": sum(
                operator.weight_burst_count + operator.scale_burst_count
                for operator in lowered
            ),
            "tile_count": sum(len(operator.tiles) for operator in lowered),
            "global_dram_span": {
                "base": all_dram_regions[0][0],
                "end": all_dram_regions[-1][1],
                "allocated_bytes_between": all_dram_regions[-1][1] - all_dram_regions[0][0],
                "regions_nonoverlap": no_overlap,
            },
        },
        "fidelity": {
            "checkpoint_shape_exact": True,
            "address_and_byte_exact": True,
            "target_quantization_executed": False,
            "cycle_accurate": False,
        },
    }


def validate_lowered_linear(lowered: LoweredLinear) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def check(name: str, condition: bool, detail: str) -> None:
        checks.append({"name": name, "passed": bool(condition), "detail": detail})

    check(
        "source_tensor_byte_conservation",
        lowered.source.payload_bytes == lowered.source.expected_payload_bytes,
        f"header={lowered.source.payload_bytes}, expected={lowered.source.expected_payload_bytes}",
    )
    check(
        "tile_weight_byte_conservation",
        sum(tile.weight_payload_bytes for tile in lowered.tiles)
        == lowered.target_weight_payload_bytes,
        f"tiles={sum(tile.weight_payload_bytes for tile in lowered.tiles)}, "
        f"tensor={lowered.target_weight_payload_bytes}",
    )
    check(
        "burst_byte_conservation",
        lowered.weight_burst_count * lowered.config.dram_burst_bytes
        == lowered.target_weight_payload_bytes,
        f"bursts={lowered.weight_burst_count}, bytes={lowered.target_weight_payload_bytes}",
    )
    check(
        "scale_burst_byte_conservation",
        lowered.scale_burst_count * lowered.config.dram_burst_bytes
        == lowered.target_scale_payload_bytes,
        f"bursts={lowered.scale_burst_count}, bytes={lowered.target_scale_payload_bytes}",
    )
    dram_regions = sorted(
        (region for region in lowered.regions if region.space == "dram"),
        key=lambda region: region.base,
    )
    sram_regions = sorted(
        (region for region in lowered.regions if region.space == "sram"),
        key=lambda region: region.base,
    )
    check(
        "dram_regions_nonoverlap",
        all(left.end <= right.base for left, right in zip(dram_regions, dram_regions[1:])),
        ", ".join(f"{region.name}=0x{region.base:x}:0x{region.end:x}" for region in dram_regions),
    )
    check(
        "sram_regions_nonoverlap",
        all(left.end <= right.base for left, right in zip(sram_regions, sram_regions[1:])),
        ", ".join(f"{region.name}=0x{region.base:x}:0x{region.end:x}" for region in sram_regions),
    )
    check(
        "sram_capacity",
        max(region.end for region in sram_regions) <= lowered.sram_config.capacity_bytes,
        f"max_end={max(region.end for region in sram_regions)}, "
        f"capacity={lowered.sram_config.capacity_bytes}",
    )
    address_samples = []
    for tile in lowered.tiles[: min(64, len(lowered.tiles))]:
        for address in (
            tile.activation_sram_address,
            tile.weight_sram_address,
            tile.partial_sram_address,
        ):
            aligned = address - address % lowered.sram_config.word_bytes
            address_samples.append(decode_sram_address(lowered.sram_config, aligned))
    check("sram_addresses_decodable", bool(address_samples), f"samples={len(address_samples)}")
    passed = all(item["passed"] for item in checks)
    return {"passed": passed, "checks": checks}


def _tensor_name(projection: str, layer: int) -> str:
    if projection == "lm_head":
        return "lm_head.weight"
    if projection in {"q_proj", "k_proj", "v_proj", "o_proj"}:
        return f"model.layers.{layer}.self_attn.{projection}.weight"
    return f"model.layers.{layer}.mlp.{projection}.weight"


def _expected_shape(model_config: dict[str, Any], projection: str) -> tuple[int, int]:
    hidden = int(model_config["hidden_size"])
    intermediate = int(model_config["intermediate_size"])
    if projection in {"q_proj", "k_proj", "v_proj", "o_proj"}:
        return hidden, hidden
    if projection in {"gate_proj", "up_proj"}:
        return intermediate, hidden
    if projection == "down_proj":
        return hidden, intermediate
    return int(model_config["vocab_size"]), hidden


def _target_tensor_base(
    model_config: dict[str, Any],
    projection: str,
    layer: int,
    config: LoweringConfig,
) -> int:
    """Assign every decoder tensor a deterministic global target address."""
    cursor = _align(config.dram_base, config.dram_alignment)
    layers = int(model_config["num_hidden_layers"])
    ordered: list[tuple[int | None, str]] = [
        (layer_index, name)
        for layer_index in range(layers)
        for name in PROJECTIONS
        if name != "lm_head"
    ]
    ordered.append((None, "lm_head"))
    target_key = (None, "lm_head") if projection == "lm_head" else (layer, projection)
    for layer_index, name in ordered:
        n, k = _expected_shape(model_config, name)
        weight_payload = n * k * config.target_weight_bits // 8
        scale_payload = n * math.ceil(k / config.scale_group) * config.scale_bytes
        weight_base = _align(cursor, config.dram_alignment)
        scale_base = _align(weight_base + weight_payload, config.dram_alignment)
        if (layer_index, name) == target_key:
            return weight_base
        cursor = scale_base + _align(scale_payload, config.dram_alignment)
    raise AssertionError(f"failed to allocate target tensor {target_key}")


def _region(regions: tuple[MemoryRegion, ...], name: str) -> MemoryRegion:
    return next(region for region in regions if region.name == name)


def _align(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment
