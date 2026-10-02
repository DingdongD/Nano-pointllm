"""DRAMsim3 completion to ordered 128-bit SRAM payload DMA model."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .dense_sram_pipeline import DensePipelineEvent, PayloadBeat
from .dramsim3_backend import DramTimingResult
from .pointllm_lowering import DramBurst


@dataclass(frozen=True)
class DramPayloadLine:
    tag: int
    address: int
    data: int


@dataclass(frozen=True)
class DramLineCompletion:
    available_cycle: int
    line: DramPayloadLine


def pack_payload_lines(
    beats: tuple[PayloadBeat, ...], *, base_address: int = 0x4000_0000,
) -> tuple[DramPayloadLine, ...]:
    if not beats or len(beats) % 4:
        raise ValueError("strict DMA packing requires a non-empty multiple of four words")
    if base_address < 0 or base_address % 64:
        raise ValueError("DRAM payload base address must be 64-byte aligned")
    lines = []
    for tag, start in enumerate(range(0, len(beats), 4)):
        data = 0
        for word, beat in enumerate(beats[start:start + 4]):
            data |= beat.data << (word * 128)
        lines.append(DramPayloadLine(tag, base_address + tag * 64, data))
    return tuple(lines)


def payload_lines_as_bursts(
    lines: tuple[DramPayloadLine, ...],
) -> tuple[DramBurst, ...]:
    return tuple(DramBurst(
        sequence=line.tag, address=line.address, size=64,
        stream="dense_payload", tile_id=0, output_row=-1,
        split_index=-1, chunk_index=line.tag, chunks_in_split=len(lines),
    ) for line in lines)


def completions_from_timing(
    timing: DramTimingResult, lines: tuple[DramPayloadLine, ...],
) -> tuple[DramLineCompletion, ...]:
    by_tag = {line.tag: line for line in lines}
    completions = []
    for event in timing.events:
        if event.event != "COMPLETE":
            continue
        if event.tag not in by_tag:
            raise ValueError(f"DRAMsim3 returned unknown payload tag {event.tag}")
        completions.append(DramLineCompletion(event.cycle, by_tag[event.tag]))
    if len(completions) != len(lines):
        raise ValueError("DRAMsim3 payload completion stream is incomplete")
    return tuple(completions)


class OrderedDmaPayloadSource:
    """Finite completion ROB followed by one 128-bit word/cycle unpack."""

    def __init__(
        self, completions: tuple[DramLineCompletion, ...], *,
        payload_count: int, block_chunks: int, rob_depth: int = 4,
    ) -> None:
        if rob_depth <= 0 or payload_count <= 0 or block_chunks <= 0:
            raise ValueError("DMA dimensions must be positive")
        if payload_count % 4:
            raise ValueError("strict DMA source requires full four-word DRAM lines")
        if len(completions) * 4 != payload_count:
            raise ValueError("completion count does not cover the payload stream")
        self.completions = completions
        self.payload_count = payload_count
        self.block_chunks = block_chunks
        self.rob_depth = rob_depth
        self.arrival_index = 0
        self.retire_line = 0
        self.word_offset = 0
        self.rob: dict[int, DramPayloadLine] = {}
        self.accepted_line_this_cycle: int | None = None
        self._events: list[DensePipelineEvent] = []
        self._counters = {
            "dma_completion_accepts": 0,
            "dma_words": 0,
            "dma_rob_backpressure_cycles": 0,
            "dma_rob_peak": 0,
        }

    def begin_cycle(self, cycle: int) -> None:
        self.accepted_line_this_cycle = None
        if self.arrival_index >= len(self.completions):
            return
        completion = self.completions[self.arrival_index]
        if completion.available_cycle > cycle:
            return
        tag = completion.line.tag
        in_window = self.retire_line <= tag < self.retire_line + self.rob_depth
        slot_collision = any(
            present % self.rob_depth == tag % self.rob_depth for present in self.rob
        )
        if len(self.rob) >= self.rob_depth or not in_window or slot_collision:
            self._counters["dma_rob_backpressure_cycles"] += 1
            return
        self.rob[tag] = completion.line
        self.accepted_line_this_cycle = tag
        self.arrival_index += 1
        self._counters["dma_completion_accepts"] += 1
        self._counters["dma_rob_peak"] = max(
            self._counters["dma_rob_peak"], len(self.rob),
        )
        self._events.append(DensePipelineEvent(
            cycle, "DRAM_COMPLETE_ACCEPT", tag, completion.available_cycle,
        ))

    def peek(self) -> PayloadBeat | None:
        if self.accepted_line_this_cycle == self.retire_line:
            return None
        line = self.rob.get(self.retire_line)
        if line is None:
            return None
        ordinal = self.retire_line * 4 + self.word_offset
        return PayloadBeat(
            sequence=ordinal // self.block_chunks,
            chunk=ordinal % self.block_chunks,
            data=(line.data >> (self.word_offset * 128)) & ((1 << 128) - 1),
        )

    def accept(self, cycle: int) -> None:
        beat = self.peek()
        if beat is None:
            raise AssertionError("DMA word accepted without an ordered line")
        ordinal = self.retire_line * 4 + self.word_offset
        self._events.append(DensePipelineEvent(
            cycle, "DMA_WORD_ACCEPT", ordinal, self.word_offset,
        ))
        self._counters["dma_words"] += 1
        if self.word_offset == 3:
            del self.rob[self.retire_line]
            self.retire_line += 1
            self.word_offset = 0
        else:
            self.word_offset += 1

    def take_events(self) -> tuple[DensePipelineEvent, ...]:
        events = tuple(self._events)
        self._events.clear()
        return events

    def counters(self) -> dict[str, int]:
        if self.arrival_index != len(self.completions) or self.rob:
            raise AssertionError("Dense pipeline completed before DMA source drained")
        return dict(self._counters)


def write_completion_trace(
    path: Path, completions: tuple[DramLineCompletion, ...],
) -> None:
    with path.open("w", encoding="ascii") as handle:
        for completion in completions:
            value = completion.available_cycle & 0xFFFFFFFF
            value |= completion.line.tag << 32
            value |= completion.line.data << 64
            handle.write(f"{value:0144x}\n")
