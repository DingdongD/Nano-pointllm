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
    seq = _seq(4)   # exactly 1 full block
    bm.allocate(seq)
    assert bm.can_append(seq)  # 1 free block is enough


def test_can_append_no_free_blocks():
    bm = BlockManager(num_blocks=1, block_size=4)
    seq = _seq(4)   # uses the only block
    bm.allocate(seq)
    # next append would cross boundary but no free blocks
    assert not bm.can_append(seq)


def test_may_append_allocates_new_block():
    bm = BlockManager(num_blocks=4, block_size=4)
    seq = _seq(4)   # full block
    bm.allocate(seq)
    assert len(seq.block_table) == 1
    seq.token_ids.append(99)    # simulate new token pushing to next block
    bm.may_append(seq)
    assert len(seq.block_table) == 2


def test_prefix_cache_reuse():
    """相同 prompt prefix 的两个请求应复用 cache block。"""
    bm = BlockManager(num_blocks=10, block_size=4)
    tokens = list(range(8))
    seq1 = PointLLMSequence(token_ids=list(tokens))
    seq2 = PointLLMSequence(token_ids=list(tokens))

    bm.allocate(seq1)
    bm.allocate(seq2)

    # 两个序列的完整 block 都相同，至少一个 block 应被复用（ref_count>=2）
    shared = False
    for bid in seq1.block_table:
        block = bm.blocks[bid]
        if block.hash != -1 and block.ref_count >= 2:
            shared = True
            break
    assert shared, "No prefix cache block shared between identical sequences"
