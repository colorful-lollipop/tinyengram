"""Multi-head embedding tables with a single fused gather.

The paper stores one table per (n-gram order, hash head), written
``E_{n,k} in R^{M_{n,k} x d_head}``.  Allocating them as 16 separate
``nn.Embedding`` modules would launch 16 tiny kernels per forward.  Instead we
concatenate them into **one flat table** and add precomputed offsets to the
hashed ids, so a token needs exactly one ``index_select``:

    flat_id[h] = z[h] + sum_{j<h} M_j

This also keeps the parameter layout friendly to optimizer sharding: row ``i``
of the flat table always belongs to the same head.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from .hashing import NgramHashMapping

__all__ = ["MultiHeadEmbedding", "EngramMemory"]


class MultiHeadEmbedding(nn.Module):
    """One flat embedding table logically split into ``len(list_of_N)`` heads."""

    def __init__(self, list_of_N: Sequence[int], embedding_dim: int) -> None:
        raise NotImplementedError("TODO(L2): 请实现本关卡的核心逻辑；先读 docs/ 对应讲义")

    @property
    def total_slots(self) -> int:
        return sum(self.head_sizes)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Args: ``[..., H]`` ids. Returns: ``[..., H, D]`` embeddings."""
        raise NotImplementedError("TODO(L2): 请实现本关卡的核心逻辑；先读 docs/ 对应讲义")


class EngramMemory(nn.Module):
    """Hash -> gather -> concatenate, i.e. Eq. (1)-(2) without the gate."""

    def __init__(self, mapping: NgramHashMapping, layer_id: int, head_dim: int) -> None:
        raise NotImplementedError("TODO(L2): 请实现本关卡的核心逻辑；先读 docs/ 对应讲义")

    @property
    def dmem(self) -> int:
        return self.table.num_heads * self.head_dim

    @property
    def total_slots(self) -> int:
        return self.table.total_slots

    def hash_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Deterministic slot ids — usable *before* the forward pass."""
        raise NotImplementedError("TODO(L2): 请实现本关卡的核心逻辑；先读 docs/ 对应讲义")

    def retrieve(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Returns ``[B, T, d_mem]`` (heads concatenated in hash order)."""
        raise NotImplementedError("TODO(L2): 请实现本关卡的核心逻辑；先读 docs/ 对应讲义")

    def extra_repr(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"layer_id={self.layer_id}, heads={self.table.num_heads}, "
            f"head_dim={self.head_dim}, slots={self.total_slots}, dmem={self.dmem}"
        )
