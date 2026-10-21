"""Systems layer: sharding, caching, prefetch, and host offload.

Why Engram is systems-friendly (paper §2.5, §6.4):

* Retrieval indices depend **only on the token ids**, not on hidden states.
  Before layer 2 runs, we already know every row it will touch.
* Therefore the host can prefetch rows over PCIe while layer 1 computes.
* The transfer volume per step is ``active_slots * d_head * bytes`` — it does
  **not** grow with table size.  A 100B-parameter table costs the same PCIe
  traffic as a 1B table for the same batch.
* N-gram frequencies are Zipfian, so a small cache in HBM absorbs most hits.

The functions here are pure simulators with explicit units, so the behaviour is
testable on CPU and the assumptions are visible in the lecture notes.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import torch

__all__ = [
    "zipf_probs",
    "LRUCache",
    "simulate_cache",
    "prefetch_plan",
    "OffloadResult",
    "simulate_offload",
    "shard_gather",
    "all_to_all_counts",
    "estimate_transfer_ms",
]


def zipf_probs(n: int, s: float = 1.0) -> torch.Tensor:
    """Normalized ``1 / rank^s`` distribution over ``n`` items."""
    raise NotImplementedError("TODO(L7): 请实现本关卡的核心逻辑；先读 docs/ 对应讲义")


class LRUCache:
    """Tiny ordered-dict LRU, used to model the HBM tier of the hierarchy."""

    def __init__(self, capacity: int) -> None:
        if capacity < 0:
            raise ValueError("capacity 必须 >= 0")
        self.capacity = int(capacity)
        self._data: OrderedDict[int, int] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def __len__(self) -> int:
        return len(self._data)

    def get(self, key: int) -> int | None:
        raise NotImplementedError("TODO(L7): 请实现本关卡的核心逻辑；先读 docs/ 对应讲义")

    def put(self, key: int, value: int = 0) -> None:
        raise NotImplementedError("TODO(L7): 请实现本关卡的核心逻辑；先读 docs/ 对应讲义")

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


def simulate_cache(
    accesses: Sequence[int] | torch.Tensor,
    capacity: int,
    *,
    n_items: int | None = None,
) -> dict[str, float]:
    """Run LRU over a trace; returns hits/misses/hit_rate/unique."""
    raise NotImplementedError("TODO(L7): 请实现本关卡的核心逻辑；先读 docs/ 对应讲义")


def prefetch_plan(
    input_ids: torch.Tensor,
    mapping,
    layer_ids: Iterable[int] | None = None,
) -> dict[int, torch.Tensor]:
    """All slot ids for all Engram layers, computed before any forward pass."""
    raise NotImplementedError("TODO(L7): 请实现本关卡的核心逻辑；先读 docs/ 对应讲义")


@dataclass
class OffloadResult:
    stalls_ms: float
    transfer_ms: float
    bytes: int
    hit_rate: float
    per_layer_stalls_ms: list[float]

    @property
    def stall_ratio(self) -> float:
        total = self.transfer_ms
        return self.stalls_ms / total if total else 0.0


def estimate_transfer_ms(n_bytes: int, bandwidth_gbps: float) -> float:
    """PCIe/NVLink transfer time. ``bandwidth_gbps`` is decimal Gbit/s."""
    raise NotImplementedError("TODO(L7): 请实现本关卡的核心逻辑；先读 docs/ 对应讲义")


def simulate_offload(
    plan: Mapping[int, torch.Tensor],
    *,
    bytes_per_slot: int,
    compute_window_ms: float = 2.0,
    bandwidth_gbps: float = 64.0,
    cache_capacity: int = 0,
) -> OffloadResult:
    """Prefetch each layer's rows from host DRAM, overlapping with compute.

    Model: for every layer in order, the unique missing rows are transferred
    over PCIe.  The preceding layer's compute provides ``compute_window_ms``
    of cover.  A stall occurs only when transfer time exceeds that window.

    ``cache_capacity`` is the number of *distinct slots* resident in HBM; the
    simulator treats the cache as perfectly hot (best case), which is the right
    way to upper-bound the benefit of the hierarchy.
    """
    raise NotImplementedError("TODO(L7): 请实现本关卡的核心逻辑；先读 docs/ 对应讲义")


def shard_gather(
    shards: Sequence[torch.Tensor],
    ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Simulate tensor-parallel embedding lookup with contiguous shards.

    Returns ``(gathered, owner, local_ids)``.  ``owner`` is what an All-to-All
    forward pass would use to route index requests; ``local_ids`` indexes into
    the destination shard.  The backward pass is the mirror image (dispatch
    gradients to owners).
    """
    raise NotImplementedError("TODO(L7): 请实现本关卡的核心逻辑；先读 docs/ 对应讲义")


def all_to_all_counts(owner: torch.Tensor, world_size: int) -> torch.Tensor:
    """Histogram of index requests per destination rank (the All-to-All load)."""
    return torch.bincount(owner.reshape(-1), minlength=world_size)
