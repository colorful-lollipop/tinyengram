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
    if n < 1:
        raise ValueError("n 必须 >= 1")
    if s <= 0:
        raise ValueError("s 必须 > 0")
    ranks = torch.arange(1, n + 1, dtype=torch.float64)
    p = ranks.pow(-s)
    return (p / p.sum()).float()


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
        if self.capacity == 0:
            self.misses += 1
            return None
        if key in self._data:
            self._data.move_to_end(key)
            self.hits += 1
            return self._data[key]
        self.misses += 1
        return None

    def put(self, key: int, value: int = 0) -> None:
        if self.capacity == 0:
            return
        if key in self._data:
            self._data.move_to_end(key)
        self._data[key] = value
        while len(self._data) > self.capacity:
            self._data.popitem(last=False)

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
    cache = LRUCache(capacity)
    keys = torch.as_tensor(accesses).reshape(-1).tolist()
    for k in keys:
        if cache.get(int(k)) is None:
            cache.put(int(k), int(k))
    total = len(keys)
    return {
        "hits": float(cache.hits),
        "misses": float(cache.misses),
        "hit_rate": cache.hit_rate,
        "unique": float(len(set(keys))),
        "total": float(total),
    }


def prefetch_plan(
    input_ids: torch.Tensor,
    mapping,
    layer_ids: Iterable[int] | None = None,
) -> dict[int, torch.Tensor]:
    """All slot ids for all Engram layers, computed before any forward pass."""
    layers = tuple(layer_ids) if layer_ids is not None else tuple(mapping.layer_ids)
    return {layer: mapping.hash(input_ids, layer) for layer in layers}


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
    if n_bytes < 0:
        raise ValueError("n_bytes 必须 >= 0")
    if bandwidth_gbps <= 0:
        raise ValueError("bandwidth_gbps 必须 > 0")
    return n_bytes * 8.0 / (bandwidth_gbps * 1e9) * 1000.0


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
    if bytes_per_slot < 1:
        raise ValueError("bytes_per_slot 必须 >= 1")
    if compute_window_ms < 0:
        raise ValueError("compute_window_ms 必须 >= 0")
    if cache_capacity < 0:
        raise ValueError("cache_capacity 必须 >= 0")

    stalls = []
    total_bytes = 0
    total_transfer = 0.0
    cached: set[int] = set()
    touches = hits = 0

    for layer, ids in plan.items():
        unique = torch.unique(ids.detach().cpu()).tolist()
        touches += len(unique)
        if cache_capacity:
            # Keep the hottest `capacity` slots (best-case LRU).
            for slot in unique:
                if len(cached) < cache_capacity:
                    cached.add(int(slot))
            missing = [s for s in unique if int(s) not in cached]
            hits += len(unique) - len(missing)
        else:
            missing = unique
        n_bytes = len(missing) * bytes_per_slot
        transfer = estimate_transfer_ms(n_bytes, bandwidth_gbps)
        total_bytes += n_bytes
        total_transfer += transfer
        stalls.append(max(0.0, transfer - compute_window_ms))

    return OffloadResult(
        stalls_ms=float(sum(stalls)),
        transfer_ms=float(total_transfer),
        bytes=int(total_bytes),
        hit_rate=(hits / touches) if touches else 0.0,
        per_layer_stalls_ms=[float(s) for s in stalls],
    )


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
    if len(shards) == 0:
        raise ValueError("至少需要一个 shard")
    sizes = torch.tensor([s.shape[0] for s in shards], dtype=torch.long)
    offsets = torch.cat([torch.zeros(1, dtype=torch.long), sizes.cumsum(0)[:-1]])
    total = int(sizes.sum())
    flat = ids.reshape(-1)
    if bool((flat < 0).any()) or bool((flat >= total).any()):
        raise ValueError("ids 越界")
    owner = torch.bucketize(flat, offsets[1:], right=True)
    local = flat - offsets[owner]
    gathered = torch.stack(
        [shards[int(o)][int(i)] for o, i in zip(owner.tolist(), local.tolist())]
    )
    return gathered.reshape(*ids.shape, shards[0].shape[-1]), owner, local


def all_to_all_counts(owner: torch.Tensor, world_size: int) -> torch.Tensor:
    """Histogram of index requests per destination rank (the All-to-All load)."""
    return torch.bincount(owner.reshape(-1), minlength=world_size)
