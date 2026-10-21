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
        super().__init__()
        if len(list_of_N) == 0:
            raise ValueError("至少需要一个 head")
        if any(int(n) < 1 for n in list_of_N):
            raise ValueError("每个 head 的表大小必须 >= 1")
        self.head_sizes = tuple(int(n) for n in list_of_N)
        self.embedding_dim = int(embedding_dim)
        self.num_heads = len(self.head_sizes)

        offsets = [0]
        for n in self.head_sizes[:-1]:
            offsets.append(offsets[-1] + n)
        self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.long))
        self.embedding = nn.Embedding(sum(self.head_sizes), self.embedding_dim)

    @property
    def total_slots(self) -> int:
        return sum(self.head_sizes)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Args: ``[..., H]`` ids. Returns: ``[..., H, D]`` embeddings."""
        if input_ids.shape[-1] != self.num_heads:
            raise ValueError(
                f"最后一维应为 head 数 {self.num_heads}，收到 {input_ids.shape[-1]}"
            )
        flat = input_ids + self.offsets.to(input_ids.device)
        return self.embedding(flat)


class EngramMemory(nn.Module):
    """Hash -> gather -> concatenate, i.e. Eq. (1)-(2) without the gate."""

    def __init__(self, mapping: NgramHashMapping, layer_id: int, head_dim: int) -> None:
        super().__init__()
        self.mapping = mapping
        self.layer_id = int(layer_id)
        self.head_dim = int(head_dim)
        self.table = MultiHeadEmbedding(mapping.flat_table_sizes(layer_id), head_dim)

    @property
    def dmem(self) -> int:
        return self.table.num_heads * self.head_dim

    @property
    def total_slots(self) -> int:
        return self.table.total_slots

    def hash_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Deterministic slot ids — usable *before* the forward pass."""
        with torch.no_grad():
            return self.mapping.hash(input_ids, self.layer_id)

    def retrieve(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Returns ``[B, T, d_mem]`` (heads concatenated in hash order)."""
        ids = self.hash_ids(input_ids)
        return self.table(ids).flatten(start_dim=-2)

    def extra_repr(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"layer_id={self.layer_id}, heads={self.table.num_heads}, "
            f"head_dim={self.head_dim}, slots={self.total_slots}, dmem={self.dmem}"
        )
