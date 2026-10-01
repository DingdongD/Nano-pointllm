import pytest
from nanopointllm.engine.sequence import PointLLMSequence, SequenceStatus
from nanopointllm.engine.block_manager import BlockManager
from nanopointllm.engine.scheduler import Scheduler
from nanopointllm.sampling_params import SamplingParams


def _seq(length: int, max_tokens: int = 10) -> PointLLMSequence:
    return PointLLMSequence(
        token_ids=list(range(length)),
        sampling_params=SamplingParams(max_tokens=max_tokens),
    )


def test_empty_scheduler_is_finished():
    s = Scheduler(max_num_seqs=4, max_num_batched_tokens=512, eos_token_id=2)
    assert s.is_finished()


def test_add_and_schedule_prefill():
    s = Scheduler(max_num_seqs=4, max_num_batched_tokens=512, eos_token_id=2)
    seq = _seq(10)
    s.add(seq)
    prefill_seqs, decode_seqs = s.schedule()
    assert seq in prefill_seqs
    assert decode_seqs == []
    assert seq.status == SequenceStatus.RUNNING


def test_schedule_decode_after_prefill():
    s = Scheduler(max_num_seqs=4, max_num_batched_tokens=512, eos_token_id=2)
    seq = _seq(10)
    s.add(seq)
    s.schedule()                                    # prefill moves seq to running
    prefill_seqs, decode_seqs = s.schedule()        # decode: no new requests
    assert prefill_seqs == []
    assert seq in decode_seqs


def test_max_num_seqs_respected():
    s = Scheduler(max_num_seqs=2, max_num_batched_tokens=512, eos_token_id=2)
    for _ in range(5):
        s.add(_seq(10))
    prefill_seqs, decode_seqs = s.schedule()
    assert len(prefill_seqs) <= 2
    assert decode_seqs == []


def test_max_num_batched_tokens_respected():
    s = Scheduler(max_num_seqs=8, max_num_batched_tokens=20, eos_token_id=2)
    s.add(_seq(15))
    s.add(_seq(15))     # total=30 > 20; only first fits
    seqs, _ = s.schedule()
    assert len(seqs) == 1


def test_postprocess_eos_finishes_sequence():
    s = Scheduler(max_num_seqs=4, max_num_batched_tokens=512, eos_token_id=2)
    seq = _seq(5, max_tokens=100)
    s.add(seq)
    seqs, _ = s.schedule()
    s.postprocess(seqs, [2])            # EOS token
    assert seq.is_finished
    assert s.is_finished()


def test_postprocess_max_tokens_finishes_sequence():
    s = Scheduler(max_num_seqs=4, max_num_batched_tokens=512, eos_token_id=2)
    seq = _seq(5, max_tokens=1)
    s.add(seq)
    seqs, _ = s.schedule()
    s.postprocess(seqs, [99])           # non-EOS but max_tokens reached
    assert seq.is_finished


def test_postprocess_appends_token():
    s = Scheduler(max_num_seqs=4, max_num_batched_tokens=512, eos_token_id=2)
    seq = _seq(5, max_tokens=10)
    s.add(seq)
    seqs, _ = s.schedule()
    s.postprocess(seqs, [77])
    assert seq.last_token == 77
    assert not seq.is_finished


def test_paged_scheduler_preempts_when_decode_append_needs_unavailable_block():
    bm = BlockManager(num_blocks=1, block_size=4)
    s = Scheduler(
        max_num_seqs=1,
        max_num_batched_tokens=512,
        eos_token_id=2,
        block_manager=bm,
    )
    seq = _seq(4, max_tokens=10)
    s.add(seq)

    prefill_seqs, decode_seqs = s.schedule()
    assert prefill_seqs == [seq]
    assert decode_seqs == []

    s.postprocess(prefill_seqs, [99])  # seq now has 5 tokens; needs a new block
    prefill_seqs, decode_seqs = s.schedule()

    assert prefill_seqs == []
    assert decode_seqs == []
    assert seq.status == SequenceStatus.WAITING
    assert list(s.waiting) == [seq]
    assert seq.block_table == []
    assert len(bm.free_block_ids) == 1


def test_chunked_prefill_requeues_without_deallocating_blocks():
    bm = BlockManager(num_blocks=8, block_size=4)
    s = Scheduler(
        max_num_seqs=1,
        max_num_batched_tokens=512,
        eos_token_id=2,
        block_manager=bm,
        max_prefill_chunk_tokens=3,
    )
    seq = _seq(8, max_tokens=10)
    s.add(seq)

    prefill_seqs, decode_seqs = s.schedule()
    assert prefill_seqs == [seq]
    assert decode_seqs == []
    assert seq.prefill_chunk_end == 3
    assert seq.block_table

    seq.num_cached_tokens = 3
    s.postprocess(prefill_seqs, [None])

    assert seq.status == SequenceStatus.WAITING
    assert seq.block_table
    assert list(s.waiting) == [seq]

    prefill_seqs, decode_seqs = s.schedule()
    assert prefill_seqs == [seq]
    assert decode_seqs == []
    assert seq.prefill_chunk_end == 6
