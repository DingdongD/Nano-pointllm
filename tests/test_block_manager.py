import pytest
from nanopointllm.engine.sequence import PointLLMSequence
from nanopointllm.engine.block_manager import BlockManager
from nanopointllm.sampling_params import SamplingParams


def _seq(length: int) -> PointLLMSequence:
    return PointLLMSequence(token_ids=list(range(length)))


def test_can_allocate_fits():
    bm = BlockManager(num_blocks=10, block_size=4)
    seq = _seq(8)   # needs 2 blocks
    assert bm.can_allocate(seq)


def test_can_allocate_too_large():
    bm = BlockManager(num_blocks=3, block_size=4)
    seq = _seq(16)  # needs 4 blocks, only 3 free
    assert not bm.can_allocate(seq)


def test_allocate_fills_block_table():
    bm = BlockManager(num_blocks=10, block_size=4)
    seq = _seq(8)   # 2 full blocks
    bm.allocate(seq)
    assert len(seq.block_table) == 2


def test_deallocate_frees_blocks():
    bm = BlockManager(num_blocks=4, block_size=4)
    seq = _seq(8)
    bm.allocate(seq)
    assert len(bm.free_block_ids) == 2
    bm.deallocate(seq)
    assert len(bm.free_block_ids) == 4


def test_can_append_same_block():
    bm = BlockManager(num_blocks=10, block_size=4)
    seq = _seq(5)   # 2 blocks: [4 tokens] + [1 token]
    bm.allocate(seq)
    # next token stays in current block → no new block needed
    assert bm.can_append(seq)


def test_can_append_needs_new_block():
    bm = BlockManager(num_blocks=2, block_size=4)
    seq = _seq(4)   # exactly 1 full block after prefill
    bm.allocate(seq)
    seq.append_token(99)  # next decode writes token 99 into block 1, offset 0
    assert bm.can_append(seq)  # 1 free block is enough


def test_can_append_no_free_blocks():
    bm = BlockManager(num_blocks=1, block_size=4)
    seq = _seq(4)   # uses the only block
    bm.allocate(seq)
    seq.append_token(99)
    # next decode would write the appended token into a new block, but none is free
    assert not bm.can_append(seq)


def test_may_append_allocates_new_block():
    bm = BlockManager(num_blocks=4, block_size=4)
    seq = _seq(4)   # full block
    bm.allocate(seq)
    assert len(seq.block_table) == 1
    seq.append_token(99)    # simulate postprocess appending token into next block
    bm.may_append(seq)
    assert len(seq.block_table) == 2


def test_may_append_hashes_newly_completed_block():
    bm = BlockManager(num_blocks=4, block_size=4)
    seq = _seq(4)
    bm.allocate(seq)
    seq.append_token(99)
    bm.may_append(seq)

    seq.append_token(100)
    seq.append_token(101)
    seq.append_token(102)  # second block now has 4 tokens
    bm.may_append(seq)

    second_block = bm.blocks[seq.block_table[1]]
    assert second_block.hash != -1
    assert second_block.token_ids == [99, 100, 101, 102]


def test_prefix_cache_no_same_step_sharing():
    """
    Two text-only seqs with the same token_ids allocated in the SAME scheduling
    step must NOT share blocks.  Their KV data hasn't been written yet, so reusing
    in-flight blocks would read stale/zero values.
    """
    bm = BlockManager(num_blocks=10, block_size=4)
    tokens = list(range(8))
    seq1 = PointLLMSequence(token_ids=list(tokens))
    seq2 = PointLLMSequence(token_ids=list(tokens))

    bm.allocate(seq1)
    bm.allocate(seq2)

    assert seq1.block_table != seq2.block_table, (
        "Same-step seqs must not share blocks (KV not written yet)"
    )
    assert seq2.num_cached_tokens == 0, (
        "Seq allocated in same step as its prefix source must not be marked cached"
    )


def test_prefix_cache_cross_request_reuse():
    """
    A text-only seq whose blocks have been deallocated (KV data written + seq
    finished) should be reused as prefix cache for a new seq with the same tokens.
    """
    bm = BlockManager(num_blocks=10, block_size=4)
    tokens = list(range(8))
    seq1 = PointLLMSequence(token_ids=list(tokens))
    seq2 = PointLLMSequence(token_ids=list(tokens))

    bm.allocate(seq1)
    seq1_blocks = list(seq1.block_table)  # save before dealloc clears it
    # Simulate seq1 finishing: blocks go back to free list with hashes intact
    bm.deallocate(seq1)

    # seq1's blocks are now free — seq2 should find them via prefix cache
    bm.allocate(seq2)

    assert seq2.num_cached_tokens == 8, (
        f"Expected 8 cached tokens from cross-request prefix cache, "
        f"got {seq2.num_cached_tokens}"
    )
    assert seq2.block_table == seq1_blocks, (
        "Cross-request prefix cache should reuse the same physical blocks"
    )
    assert bm.prefix_cache.stats.hits >= 2
    assert bm.prefix_cache.stats.inserts >= 2
