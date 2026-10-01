import torch

from nanopointllm.engine.sampler import GreedySampler, Sampler
from nanopointllm.engine.sequence import PointLLMSequence
from nanopointllm.sampling_params import SamplingParams


def test_greedy_sampler_2d_logits():
    sampler = GreedySampler()
    logits = torch.tensor([[0.1, 2.0, 1.0], [3.0, 0.0, 1.0]])
    assert sampler(logits).tolist() == [1, 0]


def test_greedy_sampler_3d_logits_uses_last_position():
    sampler = GreedySampler()
    logits = torch.tensor([
        [[10.0, 0.0, 0.0], [0.1, 0.2, 0.9]],
        [[0.0, 9.0, 0.0], [1.0, 2.0, 0.5]],
    ])
    assert sampler(logits).tolist() == [2, 1]


def test_sampler_defaults_to_greedy_for_compatibility():
    sampler = Sampler()
    seqs = [PointLLMSequence([1, 2]), PointLLMSequence([3, 4])]
    logits = torch.tensor([[0.1, 2.0, 1.0], [3.0, 0.0, 1.0]])
    assert sampler(logits, seqs).tolist() == [1, 0]


def test_sampler_top_k_seeded_sampling_is_deterministic():
    sampler = Sampler()
    seq = PointLLMSequence(
        [1, 2],
        sampling_params=SamplingParams(do_sample=True, top_k=1, seed=123),
    )
    logits = torch.tensor([[0.1, 2.0, 4.0, 3.0]])
    assert sampler(logits, [seq]).tolist() == [2]


def test_sampler_repetition_penalty_can_change_greedy_choice():
    sampler = Sampler()
    seq = PointLLMSequence(
        [2, 2],
        sampling_params=SamplingParams(
            do_sample=True,
            top_k=1,
            repetition_penalty=2.0,
            frequency_penalty=1.0,
        ),
    )
    logits = torch.tensor([[0.0, 4.0, 5.0]])
    assert sampler(logits, [seq]).tolist() == [1]
