"""Context-aware gating (paper §2.3, Eq. 3-4).

Retrieved n-gram embeddings are *static*: "the Great" always maps to the same
vector, whether the sentence is about Alexander or about a failure.  Engram
therefore computes a scalar gate from the current hidden state:

    k_t = W_K e_t
    alpha_t = sigmoid( signed_sqrt( <RMSNorm(h_t), RMSNorm(k_t)> / sqrt(d) ) )
    v_t = W_V e_t
    u_t = alpha_t * v_t

Two details are worth internalizing:

1. **Why normalize both sides?** Without RMSNorm the logit scale drifts with
   training and the gate saturates.  Normalizing makes the logit a cosine
   similarity in ``[-sqrt(d), sqrt(d)]``; dividing by ``sqrt(d)`` bounds it to
   ``[-1, 1]``.

2. **Why signed sqrt?** The official implementation applies
   ``sign(x) * sqrt(|x|)`` before the sigmoid.  This compresses large logits,
   keeping gradients alive for confidently-retrieved memories, while
   preserving the sign (suppression stays suppression).  Plain sigmoid on
   cosine similarity already works, but the sqrt variant trains faster at
   27B scale.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["rms_norm", "RMSNorm", "signed_sqrt", "context_gate"]


def rms_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Root-mean-square normalization over the last dimension (no mean removal)."""
    raise NotImplementedError("TODO(L3): 请实现本关卡的核心逻辑；先读 docs/ 对应讲义")


def signed_sqrt(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """``sign(x) * sqrt(|x|)`` with a floor on the magnitude.

    The floor avoids ``sqrt(0)`` having an infinite derivative at exactly zero,
    which happens often right after initialization.
    """
    raise NotImplementedError("TODO(L3): 请实现本关卡的核心逻辑；先读 docs/ 对应讲义")


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("TODO(L3): 请实现本关卡的核心逻辑；先读 docs/ 对应讲义")


def context_gate(
    query: torch.Tensor,
    key: torch.Tensor,
    *,
    eps: float = 1e-6,
    use_signed_sqrt: bool = True,
    normalize: bool = True,
) -> torch.Tensor:
    """Scalar gate ``[..., 1]`` in ``(0, 1)``.

    Args:
        query: ``[..., D]`` current hidden state (one per branch).
        key:   ``[..., D]`` projected memory.
        normalize: if False, caller has already applied (learnable) RMSNorm.
    """
    raise NotImplementedError("TODO(L3): 请实现本关卡的核心逻辑；先读 docs/ 对应讲义")


def gated_value(gate: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    """Broadcast ``alpha_t`` over the value width: ``[..., 1] * [..., D]``."""
    return gate * value
