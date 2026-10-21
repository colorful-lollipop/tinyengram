"""Inference: KV cache, sampling, and a naive reference decoder.

The systems story only matters if decoding is autoregressive: at step ``t`` we
must not recompute the whole prefix.  With a KV cache, the only new Engram work
is the lookup for the *new* token — and that lookup could have been prefetched
before the step even started.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from .transformer import TinyLM

__all__ = ["sample_next", "generate", "naive_generate"]


def sample_next(
    logits: torch.Tensor,
    *,
    temperature: float = 0.0,
    top_k: Optional[int] = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Pick one token per batch row from ``logits [B, V]``."""
    if temperature < 0:
        raise ValueError("temperature 必须 >= 0")
    if temperature == 0:
        return logits.argmax(dim=-1, keepdim=True)
    logits = logits / temperature
    if top_k is not None:
        if top_k < 1:
            raise ValueError("top_k 必须 >= 1")
        k = min(top_k, logits.size(-1))
        values, _ = torch.topk(logits, k, dim=-1)
        cutoff = values[..., -1, None]
        logits = logits.masked_fill(logits < cutoff, float("-inf"))
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator)


@torch.no_grad()
def generate(
    model: TinyLM,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    *,
    temperature: float = 0.0,
    top_k: Optional[int] = None,
    eos_id: Optional[int] = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Autoregressive decoding with a KV cache. Returns the full sequence."""
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens 必须 >= 0")
    model.eval()
    device = next(model.parameters()).device
    ids = input_ids.to(device)
    if ids.dim() != 2:
        raise ValueError("input_ids 应为 [B, T]")

    logits, _, caches = model(ids)
    out = ids
    pos = ids.shape[1]
    finished = torch.zeros(ids.shape[0], dtype=torch.bool, device=device)

    for _ in range(max_new_tokens):
        next_id = sample_next(
            logits[:, -1, :], temperature=temperature, top_k=top_k, generator=generator
        )
        out = torch.cat([out, next_id], dim=1)
        if eos_id is not None:
            finished |= next_id.squeeze(1) == eos_id
            if bool(finished.all()):
                break
        logits, _, caches = model(
            next_id, caches=caches, position_offset=pos, context_ids=out
        )
        pos += 1
    return out


@torch.no_grad()
def naive_generate(
    model: TinyLM,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    *,
    temperature: float = 0.0,
    top_k: Optional[int] = None,
    eos_id: Optional[int] = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Reference decoder: recompute the full prefix at every step (slow)."""
    model.eval()
    out = input_ids
    for _ in range(max_new_tokens):
        logits, _, _ = model(out)
        next_id = sample_next(
            logits[:, -1, :], temperature=temperature, top_k=top_k, generator=generator
        )
        out = torch.cat([out, next_id], dim=1)
        if eos_id is not None and bool((out[:, -1] == eos_id).all()):
            break
    return out
