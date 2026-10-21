"""Mechanistic analysis tools from paper §6.

* **LogitLens / KL divergence** (Fig. 4a): project every intermediate hidden
  state through the final norm + LM head, then measure KL to the final
  distribution.  Engram's curve drops faster -> predictions become
  "ready" earlier.
* **CKA** (Fig. 4b-c): centered kernel alignment between layer activations of
  two models.  The soft alignment index (Eq. 9) summarizes "Engram layer j
  behaves like MoE layer a_j"; ``a_j > j`` means extra effective depth.
* **Gate statistics** (Fig. 7): Engram should fire on complete static patterns.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F

__all__ = ["linear_cka", "logit_lens_kl", "soft_alignment", "gate_stats", "binary_entropy"]


def linear_cka(x: torch.Tensor, y: torch.Tensor) -> float:
    """Linear-kernel CKA between two representation matrices.

    Args:
        x: ``[N, D1]``, y: ``[N, D2]`` (same N samples).
    Returns:
        scalar in ``[0, 1]``; 1 means identical up to an orthogonal transform.

    Feature-space form (Kornblith et al., 2019):

        CKA = ||Yc^T Xc||_F^2 / (||Xc^T Xc||_F * ||Yc^T Yc||_F)
    """
    raise NotImplementedError("TODO(L8): 请实现本关卡的核心逻辑；先读 docs/ 对应讲义")


def logit_lens_kl(
    hidden_states: Sequence[torch.Tensor],
    norm,
    lm_head,
    final_logits: torch.Tensor,
) -> list[float]:
    """KL(final || lens) for each intermediate hidden state.

    Args:
        hidden_states: list of ``[..., D]`` tensors (one per layer).
        norm: final RMSNorm module (or callable).
        lm_head: ``Linear(D, V)``.
        final_logits: ``[..., V]`` logits of the last layer.
    """
    raise NotImplementedError("TODO(L8): 请实现本关卡的核心逻辑；先读 docs/ 对应讲义")


def soft_alignment(similarity: torch.Tensor, k: int = 5) -> torch.Tensor:
    """Eq. (9): weighted centroid of the top-k most similar rows per column.

    Args:
        similarity: ``[L_moe, L_engram]``; rows are MoE layers, columns are
            Engram layers.
    Returns:
        ``[L_engram]`` with ``a_j`` in layer-index units.
    """
    raise NotImplementedError("TODO(L8): 请实现本关卡的核心逻辑；先读 docs/ 对应讲义")


def binary_entropy(p: torch.Tensor) -> torch.Tensor:
    """Entropy (nats) of Bernoulli(p), elementwise; ``0*log 0 := 0``."""
    raise NotImplementedError("TODO(L8): 请实现本关卡的核心逻辑；先读 docs/ 对应讲义")


def gate_stats(gates: torch.Tensor) -> dict[str, float]:
    """Summary of gate activations ``[..., M]`` for a batch of tokens."""
    raise NotImplementedError("TODO(L8): 请实现本关卡的核心逻辑；先读 docs/ 对应讲义")
