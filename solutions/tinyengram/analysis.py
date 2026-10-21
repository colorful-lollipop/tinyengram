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
    if x.dim() != 2 or y.dim() != 2:
        raise ValueError("x/y 必须是 [N, D]")
    if x.shape[0] != y.shape[0]:
        raise ValueError("x/y 的样本数必须一致")
    if x.shape[0] < 2:
        raise ValueError("至少需要 2 个样本")
    xc = x - x.mean(dim=0, keepdim=True)
    yc = y - y.mean(dim=0, keepdim=True)
    xty = torch.linalg.matrix_norm(yc.T @ xc, ord="fro") ** 2
    xtx = torch.linalg.matrix_norm(xc.T @ xc, ord="fro")
    yty = torch.linalg.matrix_norm(yc.T @ yc, ord="fro")
    denom = xtx * yty
    if float(denom) == 0.0:
        return 0.0
    return float((xty / denom).clamp(0.0, 1.0))


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
    if len(hidden_states) == 0:
        raise ValueError("hidden_states 不能为空")
    target = F.log_softmax(final_logits.float(), dim=-1).detach()
    out = []
    for h in hidden_states:
        logits = lm_head(norm(h)).float().detach()
        logp = F.log_softmax(logits, dim=-1)
        kl = (target.exp() * (target - logp)).sum(dim=-1).mean()
        out.append(float(kl.detach()))
    return out


def soft_alignment(similarity: torch.Tensor, k: int = 5) -> torch.Tensor:
    """Eq. (9): weighted centroid of the top-k most similar rows per column.

    Args:
        similarity: ``[L_moe, L_engram]``; rows are MoE layers, columns are
            Engram layers.
    Returns:
        ``[L_engram]`` with ``a_j`` in layer-index units.
    """
    if similarity.dim() != 2:
        raise ValueError("similarity 必须是二维矩阵")
    rows, cols = similarity.shape
    if not 1 <= k <= rows:
        raise ValueError(f"k 必须在 [1, {rows}] 内")
    idx = similarity.topk(k, dim=0).indices  # [k, cols]
    weights = torch.gather(similarity, 0, idx)
    positions = idx.float()
    denom = weights.sum(dim=0).clamp_min(1e-12)
    return (weights * positions).sum(dim=0) / denom


def binary_entropy(p: torch.Tensor) -> torch.Tensor:
    """Entropy (nats) of Bernoulli(p), elementwise; ``0*log 0 := 0``."""
    p = p.clamp(1e-6, 1.0 - 1e-6)
    return -(p * p.log() + (1 - p) * (1 - p).log())


def gate_stats(gates: torch.Tensor) -> dict[str, float]:
    """Summary of gate activations ``[..., M]`` for a batch of tokens."""
    g = gates.detach().float()
    return {
        "mean": float(g.mean()),
        "std": float(g.std()) if g.numel() > 1 else 0.0,
        "frac_above_0.5": float((g > 0.5).float().mean()),
        "frac_below_0.1": float((g < 0.1).float().mean()),
        "entropy": float(binary_entropy(g).mean()),
    }
