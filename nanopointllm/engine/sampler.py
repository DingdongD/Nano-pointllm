from __future__ import annotations

import torch
import torch.nn as nn

from nanopointllm.engine.sequence import PointLLMSequence
from nanopointllm.sampling_params import SamplingParams


def _last_logits(logits: torch.Tensor) -> torch.Tensor:
    if logits.dim() == 3:
        return logits[:, -1, :]
    return logits


def _greedy_impl(logits: torch.Tensor) -> torch.Tensor:
    return logits.float().argmax(dim=-1).to(torch.long)


def _batch_sample_impl(
    logits: torch.Tensor,
    token_counts: torch.Tensor,
    do_sample: torch.Tensor,
    temperature: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    repetition_penalty: torch.Tensor,
    frequency_penalty: torch.Tensor,
    presence_penalty: torch.Tensor,
) -> torch.Tensor:
    logits = logits.float()
    greedy = logits.argmax(dim=-1)

    repeated = token_counts > 0
    adjusted = torch.where(
        repeated & (logits < 0),
        logits * repetition_penalty[:, None],
        logits,
    )
    adjusted = torch.where(
        repeated & (logits >= 0),
        adjusted / repetition_penalty[:, None],
        adjusted,
    )
    adjusted = adjusted - token_counts * frequency_penalty[:, None]
    adjusted = adjusted - repeated.to(adjusted.dtype) * presence_penalty[:, None]
    adjusted = adjusted / torch.clamp(temperature[:, None], min=1.0e-6)

    vocab = adjusted.shape[-1]
    ranks = torch.arange(vocab, device=adjusted.device)[None, :]
    sorted_logits, sorted_indices = torch.sort(adjusted, descending=True, dim=-1)
    valid_k = (top_k[:, None] <= 0) | (ranks < top_k[:, None])
    sorted_logits = torch.where(valid_k, sorted_logits, -torch.inf)

    sorted_probs = torch.softmax(sorted_logits, dim=-1)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    remove_p = cumulative > top_p[:, None]
    shifted = torch.cat([torch.zeros_like(remove_p[:, :1]), remove_p[:, :-1]], dim=-1)
    sorted_logits = torch.where(shifted, -torch.inf, sorted_logits)

    filtered = torch.full_like(adjusted, -torch.inf)
    filtered.scatter_(1, sorted_indices, sorted_logits)
    probs = torch.softmax(filtered, dim=-1)
    sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)
    return torch.where(do_sample, sampled, greedy).to(torch.long)


class BatchCompiledSampler(nn.Module):
    """Compiled batch sampler.

    Sampling parameters are materialized into compact tensors once per request
    batch; token penalties are represented as a `[B, vocab]` count matrix.
    Actual sampling is a single tensor program, optionally `torch.compile`d on
    CUDA, with no per-row Python sampling loop.
    """

    def __init__(self, *, compile_cuda: bool = False) -> None:
        super().__init__()
        self.compile_cuda = compile_cuda
        self._compiled = None
        self._compiled_greedy = None

    def _impl(self, *args) -> torch.Tensor:
        if self.compile_cuda and args[0].is_cuda and hasattr(torch, "compile"):
            if self._compiled is None:
                self._compiled = torch.compile(_batch_sample_impl, mode="reduce-overhead")
            return self._compiled(*args)
        return _batch_sample_impl(*args)

    def _greedy(self, logits: torch.Tensor) -> torch.Tensor:
        if self.compile_cuda and logits.is_cuda and hasattr(torch, "compile"):
            if self._compiled_greedy is None:
                self._compiled_greedy = torch.compile(_greedy_impl, mode="reduce-overhead")
            return self._compiled_greedy(logits)
        return _greedy_impl(logits)

    @staticmethod
    def _is_default_greedy(params: SamplingParams) -> bool:
        return (
            not params.do_sample
            and params.temperature == 1.0
            and params.top_k == -1
            and params.top_p == 1.0
            and params.repetition_penalty == 1.0
            and params.frequency_penalty == 0.0
            and params.presence_penalty == 0.0
            and params.seed is None
        )

    def forward(
        self,
        logits: torch.Tensor,
        seqs: list[PointLLMSequence] | None = None,
    ) -> torch.Tensor:
        logits = _last_logits(logits)
        if seqs is None:
            return self._greedy(logits)
        if len(seqs) != logits.shape[0]:
            raise ValueError(f"sampler got {logits.shape[0]} rows for {len(seqs)} sequences")
        if all(self._is_default_greedy(seq.sampling_params) for seq in seqs):
            return self._greedy(logits)

        device = logits.device
        B, vocab = logits.shape
        token_counts = torch.zeros(B, vocab, dtype=torch.float32, device=device)
        do_sample = torch.empty(B, dtype=torch.bool, device=device)
        temperature = torch.empty(B, dtype=torch.float32, device=device)
        top_k = torch.empty(B, dtype=torch.long, device=device)
        top_p = torch.empty(B, dtype=torch.float32, device=device)
        repetition_penalty = torch.empty(B, dtype=torch.float32, device=device)
        frequency_penalty = torch.empty(B, dtype=torch.float32, device=device)
        presence_penalty = torch.empty(B, dtype=torch.float32, device=device)

        explicit_seeds = [seq.sampling_params.seed for seq in seqs if seq.sampling_params.seed is not None]
        if explicit_seeds:
            if len(set(explicit_seeds)) != 1:
                raise NotImplementedError("compiled batch sampler requires one shared explicit seed per batch")
            torch.manual_seed(explicit_seeds[0] + max(seq.num_completion_tokens for seq in seqs))

        for row, seq in enumerate(seqs):
            params = seq.sampling_params
            ids = torch.tensor(seq.token_ids, dtype=torch.long, device=device)
            if ids.numel() > 0:
                valid = ids[(ids >= 0) & (ids < vocab)]
                token_counts[row].scatter_add_(
                    0,
                    valid,
                    torch.ones_like(valid, dtype=token_counts.dtype),
                )
            do_sample[row] = params.do_sample
            temperature[row] = params.temperature
            top_k[row] = params.top_k
            top_p[row] = params.top_p
            repetition_penalty[row] = params.repetition_penalty
            frequency_penalty[row] = params.frequency_penalty
            presence_penalty[row] = params.presence_penalty

        return self._impl(
            logits,
            token_counts,
            do_sample,
            temperature,
            top_k,
            top_p,
            repetition_penalty,
            frequency_penalty,
            presence_penalty,
        )


GreedySampler = BatchCompiledSampler
Sampler = BatchCompiledSampler


def build_greedy_sampler(*, compile_sampler: bool = False) -> nn.Module:
    return BatchCompiledSampler(compile_cuda=compile_sampler)


def build_sampler(*, compile_greedy_cuda: bool = False) -> nn.Module:
    return BatchCompiledSampler(compile_cuda=compile_greedy_cuda)
