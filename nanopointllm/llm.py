from __future__ import annotations

from typing import Any, Optional

import torch.nn as nn

from nanopointllm.engine.llm_engine import PointLLMLLMEngine
from nanopointllm.sampling_params import SamplingParams


class LLM(PointLLMLLMEngine):
    """Nano-vLLM-style offline inference wrapper for already-loaded PointLLM models.

    PointLLM loading remains caller-owned because local forks/checkpoints often need
    project-specific construction. This class aligns the public batch API with
    ``nanovllm.LLM.generate`` once an HF/PointLLM-compatible model is available.
    """

    def __init__(
        self,
        hf_model: nn.Module,
        tokenizer: Optional[Any] = None,
        eos_token_id: Optional[int] = None,
        **engine_kwargs: Any,
    ) -> None:
        self.tokenizer = tokenizer
        if eos_token_id is None and tokenizer is not None:
            eos_token_id = getattr(tokenizer, "eos_token_id", None)
        if eos_token_id is None:
            eos_token_id = getattr(getattr(hf_model, "config", None), "eos_token_id", None)
        if eos_token_id is None:
            raise ValueError("eos_token_id must be provided when tokenizer/model config lacks one")
        super().__init__(hf_model=hf_model, eos_token_id=eos_token_id, **engine_kwargs)

    def _prompt_to_request(self, prompt: str | list[int] | dict[str, Any]) -> tuple[list[int], dict[str, Any]]:
        point_clouds = None
        extra: dict[str, Any] = {}
        if isinstance(prompt, dict):
            point_clouds = prompt.get("point_clouds")
            if prompt.get("point_cloud_cache_key") is not None:
                extra["point_cloud_cache_key"] = prompt["point_cloud_cache_key"]
            if prompt.get("point_features_cached") is not None:
                extra["point_features_cached"] = prompt["point_features_cached"]
            if "token_ids" in prompt:
                prompt = prompt["token_ids"]
            elif "prompt_token_ids" in prompt:
                prompt = prompt["prompt_token_ids"]
            else:
                prompt = prompt["prompt"]
        if isinstance(prompt, str):
            if self.tokenizer is None:
                raise ValueError("A tokenizer is required when prompts are strings")
            prompt = self.tokenizer.encode(prompt)
        extra["point_clouds"] = point_clouds
        return list(prompt), extra

    def generate(
        self,
        prompts: list[str] | list[list[int]] | list[dict[str, Any]],
        sampling_params: SamplingParams | list[SamplingParams] | None = None,
        use_tqdm: bool = False,
    ) -> list[dict[str, Any]]:
        if sampling_params is None:
            sampling_params = SamplingParams()
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        if len(sampling_params) != len(prompts):
            raise ValueError("sampling_params length must match prompts length")

        iterator = zip(prompts, sampling_params)
        if use_tqdm:
            from tqdm.auto import tqdm

            iterator = tqdm(list(iterator), total=len(prompts), desc="Generating", dynamic_ncols=True)

        seqs = []
        for prompt, sp in iterator:
            token_ids, req = self._prompt_to_request(prompt)
            if isinstance(prompt, dict) and prompt.get("sampling_params") is not None:
                sp = prompt["sampling_params"]
            seqs.append(super().add_request(token_ids=token_ids, sampling_params=sp, **req))

        while not self.is_finished():
            self.step()

        outputs = []
        for seq in seqs:
            completion_ids = seq.token_ids[seq.num_prompt_tokens:]
            text = self.tokenizer.decode(completion_ids) if self.tokenizer is not None else None
            outputs.append({"text": text, "token_ids": completion_ids, "sequence": seq})
        return outputs
