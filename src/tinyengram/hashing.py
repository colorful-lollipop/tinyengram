"""Deterministic multi-head hashing of suffix n-grams.

Paper Eq. (1)-(2):

    z_{t,n,k} = phi_{n,k}(g_{t,n})          g_{t,n} = (x'_{t-n+1}, ..., x'_t)
    e_{t,n,k} = E_{n,k}[z_{t,n,k}]
    e_t       = concat_{n=2..N} concat_{k=1..K} e_{t,n,k}

Design choices that matter (and are easy to get wrong):

* **Suffix** n-grams, not prefix: position ``t`` may only see ``<= t``.
* **XOR mixing** with odd multipliers, not addition.  XOR has no carries, so
  the result stays inside 63 bits when ``token * multiplier < 2**62``.
* **Prime table sizes**, distinct *globally* (across layers, orders, heads).
  Co-prime moduli make collision events as independent as possible, and a
  prime modulus avoids structure from power-of-two masking.
* **Per-layer multipliers** so layer 2 and layer 15 do not learn the same
  accidental collisions.
* The whole function is **pure and differentiable-free**: ids in, ids out.
  This is what enables prefetching before the forward pass.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import torch
import torch.nn.functional as F

__all__ = [
    "is_prime",
    "next_primes",
    "make_odd_multipliers",
    "NgramHashMapping",
]

INT64_MAX = (1 << 63) - 1


def is_prime(n: int) -> bool:
    """Deterministic trial division (adequate for table sizes up to ~1e8)."""
    raise NotImplementedError("TODO(L1): 请实现本关卡的核心逻辑；先读 docs/ 对应讲义")


def next_primes(start: int, count: int, seen: Iterable[int] | None = None) -> list[int]:
    """Return ``count`` primes strictly greater than ``start``, skipping ``seen``.

    ``seen`` is updated in place when it is a mutable set; this is how the
    constructor guarantees globally distinct moduli.
    """
    raise NotImplementedError("TODO(L1): 请实现本关卡的核心逻辑；先读 docs/ 对应讲义")


def make_odd_multipliers(
    max_ngram_size: int,
    vocab_size: int,
    seed: int,
) -> torch.Tensor:
    """Random *odd* multipliers that cannot overflow int64 during mixing.

    ``m < INT64_MAX / vocab_size`` guarantees ``token * m < 2**63`` for every
    token id.  Forcing odd values makes the map ``x -> x*m mod 2**64`` a
    bijection on the low bits, which spreads ids more uniformly than even
    multipliers (whose low bits are always zero).
    """
    raise NotImplementedError("TODO(L1): 请实现本关卡的核心逻辑；先读 docs/ 对应讲义")


class NgramHashMapping:
    """Maps token id sequences to per-head slot ids, deterministically.

    Parameters
    ----------
    vocab_size:
        Size of the *compressed* vocabulary ``V'``.
    slots_per_head:
        Requested rows per hash head; one int or one per n-gram order.
    layer_ids:
        Layers that own an Engram module.  Each gets independent multipliers
        and a disjoint block of prime table sizes.
    """

    def __init__(
        self,
        vocab_size: int,
        slots_per_head: int | Sequence[int],
        max_ngram_size: int = 3,
        n_head_per_ngram: int = 8,
        layer_ids: Sequence[int] = (1,),
        pad_id: int = 0,
        seed: int = 0,
    ) -> None:
        raise NotImplementedError("TODO(L1): 请实现本关卡的核心逻辑；先读 docs/ 对应讲义")

    # ------------------------------------------------------------------ shape
    @property
    def n_orders(self) -> int:
        return self.max_ngram_size - 1

    @property
    def n_heads_total(self) -> int:
        return self.n_orders * self.n_head_per_ngram

    def flat_table_sizes(self, layer_id: int) -> list[int]:
        """Head table sizes in exactly the order produced by :meth:`hash`."""
        return [size for order in self.table_sizes[layer_id] for size in order]

    def slot_count(self, layer_id: int) -> int:
        return sum(self.flat_table_sizes(layer_id))

    def footprint(self, layer_id: int, head_dim: int, bytes_per_param: int = 4) -> int:
        """Bytes of one layer's embedding table (fp32 by default)."""
        return self.slot_count(layer_id) * head_dim * bytes_per_param

    # ------------------------------------------------------------------- hash
    def hash(self, input_ids: torch.Tensor, layer_id: int) -> torch.Tensor:
        """Hash a batch of already-compressed ids.

        Args:
            input_ids: ``LongTensor[B, T]`` in ``[0, vocab_size)``.
            layer_id: must be in ``self.layer_ids``.
        Returns:
            ``LongTensor[B, T, n_orders * n_head_per_ngram]`` with each entry
            in ``[0, table_size)`` for the corresponding head.
        """
        raise NotImplementedError("TODO(L1): 请实现本关卡的核心逻辑；先读 docs/ 对应讲义")

    def hash_all(self, input_ids: torch.Tensor) -> dict[int, torch.Tensor]:
        return {layer_id: self.hash(input_ids, layer_id) for layer_id in self.layer_ids}


def _check_hash_bounds(hashes: torch.Tensor, sizes: Sequence[int]) -> None:
    """Debug helper used by tests: every head stays inside its table."""
    for h, size in enumerate(sizes):
        col = hashes[..., h]
        if bool((col < 0).any()) or bool((col >= size).any()):
            raise AssertionError(f"head {h} 越界: size={size}")
