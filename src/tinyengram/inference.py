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
    raise NotImplementedError("TODO(L7): 请实现本关卡的核心逻辑；先读 docs/ 对应讲义")


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
    raise NotImplementedError("TODO(L7): 请实现本关卡的核心逻辑；先读 docs/ 对应讲义")


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
    raise NotImplementedError("TODO(L7): 请实现本关卡的核心逻辑；先读 docs/ 对应讲义")
