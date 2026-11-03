"""A history-dependent single-binding copy task, with no external dataset.

Prompt: [key, v1,v2,v3,v4, noise..., QUERY, key]; response: [v1,v2,v3,v4, EOS].
Two prompts can have identical recent suffixes yet different correct responses.
This is a synthetic copy task, not a natural-language or multi-entity benchmark.
"""

from __future__ import annotations

import torch

__all__ = ["recall_batch"]


def recall_batch(batch: int, prompt_length: int, generator: torch.Generator,
                 device="cpu") -> tuple[torch.Tensor, torch.Tensor]:
    if batch < 1 or prompt_length < 7:
        raise ValueError("recall requires batch >= 1 and prompt length >= 7")
    key = torch.randint(1, 9, (batch, 1), generator=generator)
    values = torch.randint(9, 17, (batch, 4), generator=generator)
    noise = torch.randint(17, 31, (batch, prompt_length - 7), generator=generator)
    query = torch.full((batch, 1), 31, dtype=torch.long)
    eos = torch.zeros(batch, 1, dtype=torch.long)
    prompt = torch.cat([key, values, noise, query, key], 1)
    response = torch.cat([values, eos], 1)
    return prompt.to(device), response.to(device)
