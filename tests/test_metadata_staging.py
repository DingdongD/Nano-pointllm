import torch

from nanopointllm.engine.metadata_staging import DecodeMetadataStagingBuffer, next_bucket_size
from nanopointllm.engine.sequence import PointLLMSequence


def test_next_bucket_size():
    assert next_bucket_size(1) == 1
    assert next_bucket_size(3) == 4
    assert next_bucket_size(17) == 32
    assert next_bucket_size(80) == 80


def test_decode_metadata_staging_prepare_decode():
    seq0 = PointLLMSequence([1, 2, 3, 4, 5])
    seq0.block_table = [7, 9]
    seq1 = PointLLMSequence([6, 8])
    seq1.block_table = [4]

    staging = DecodeMetadataStagingBuffer(
        device=torch.device("cpu"),
        max_batch_size=4,
        max_blocks_per_seq=3,
    )
    view = staging.prepare_decode([seq0, seq1], block_size=4)

    assert view.batch_size == 2
    assert view.bucket_size == 4
    assert view.input_ids.tolist() == [[5, 8]]
    assert view.position_ids.tolist() == [[4, 1]]
    assert view.context.slot_mapping.tolist() == [9 * 4, 4 * 4 + 1]
    assert view.context.context_lens.tolist() == [5, 2]
    assert view.context.block_tables.tolist() == [[7, 9], [4, -1]]
    assert view.context.seq_lens == [1, 1]
