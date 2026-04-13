import pytest
from nanopointllm.sampling_params import SamplingParams
from nanopointllm.engine.sequence import PointLLMSequence, SequenceStatus


def test_sampling_params_defaults():
    p = SamplingParams()
    assert p.temperature == 1.0
    assert p.max_tokens == 128
    assert p.ignore_eos is False


def test_sequence_init():
    seq = PointLLMSequence(token_ids=[1, 2, 3])
    assert seq.num_tokens == 3
    assert seq.num_prompt_tokens == 3
    assert seq.num_completion_tokens == 0
    assert seq.status == SequenceStatus.WAITING
    assert seq.last_token == 3


def test_sequence_append_token():
    seq = PointLLMSequence(token_ids=[1, 2, 3])
    seq.append_token(99)
    assert seq.num_tokens == 4
    assert seq.last_token == 99
    assert seq.num_completion_tokens == 1


def test_sequence_status_transition():
    seq = PointLLMSequence(token_ids=[1, 2, 3])
    assert not seq.is_finished
    seq.status = SequenceStatus.FINISHED
    assert seq.is_finished


def test_sequence_with_point_clouds():
    import torch
    pc = torch.randn(1024, 3)
    seq = PointLLMSequence(token_ids=[1, 2, 3], point_clouds=pc)
    assert seq.point_clouds is not None
    assert seq.inputs_embeds_cached is None  # not encoded yet
