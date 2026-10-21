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
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    if n % 3 == 0:
        return n == 3
    d = 5
    while d * d <= n:
        if n % d == 0 or n % (d + 2) == 0:
            return False
        d += 6
    return True


def next_primes(start: int, count: int, seen: Iterable[int] | None = None) -> list[int]:
    """Return ``count`` primes strictly greater than ``start``, skipping ``seen``.

    ``seen`` is updated in place when it is a mutable set; this is how the
    constructor guarantees globally distinct moduli.
    """
    if count < 0:
        raise ValueError("count 必须 >= 0")
    used = seen if seen is not None else set()
    out: list[int] = []
    candidate = int(start) + 1
    while len(out) < count:
        if is_prime(candidate) and candidate not in used:
            out.append(candidate)
            if isinstance(used, set):
                used.add(candidate)
            else:  # pragma: no cover - defensive
                used = set(used) | {candidate}
        candidate += 1
    return out


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
    if vocab_size < 1:
        raise ValueError("vocab_size 必须 >= 1")
    bound = max(1, INT64_MAX // vocab_size)
    half = max(1, bound // 2)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    r = torch.randint(0, half, (max_ngram_size,), generator=gen, dtype=torch.int64)
    return r * 2 + 1


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
        if max_ngram_size < 2:
            raise ValueError("max_ngram_size 至少为 2")
        if n_head_per_ngram < 1:
            raise ValueError("n_head_per_ngram 必须 >= 1")
        if vocab_size < 1:
            raise ValueError("vocab_size 必须 >= 1")
        if not 0 <= pad_id < vocab_size:
            raise ValueError("pad_id 必须落在 [0, vocab_size) 内")

        self.vocab_size = int(vocab_size)
        self.max_ngram_size = int(max_ngram_size)
        self.n_head_per_ngram = int(n_head_per_ngram)
        self.layer_ids = tuple(int(i) for i in layer_ids)
        self.pad_id = int(pad_id)
        self.seed = int(seed)

        if len(set(self.layer_ids)) != len(self.layer_ids):
            raise ValueError("layer_ids 不能重复")

        if isinstance(slots_per_head, int):
            slots_per_head = (slots_per_head,) * self.n_orders
        slots_per_head = tuple(int(s) for s in slots_per_head)
        if len(slots_per_head) != self.n_orders:
            raise ValueError(f"slots_per_head 需要 {self.n_orders} 个值")
        if any(s < 1 for s in slots_per_head):
            raise ValueError("slots_per_head 必须 >= 1")
        self.requested_slots = slots_per_head

        self.multipliers: dict[int, torch.Tensor] = {}
        self.table_sizes: dict[int, tuple[tuple[int, ...], ...]] = {}

        seen_primes: set[int] = set()
        for layer_id in self.layer_ids:
            # Different layers must not share multipliers; PRIME_1 decorrelates
            # layer ids that are close together (2, 3, ...).
            layer_seed = self.seed + 10007 * layer_id
            self.multipliers[layer_id] = make_odd_multipliers(
                self.max_ngram_size, self.vocab_size, layer_seed
            )
            per_order: list[tuple[int, ...]] = []
            for order_slots in slots_per_head:
                per_order.append(
                    tuple(
                        next_primes(order_slots - 1, self.n_head_per_ngram, seen_primes)
                    )
                )
            self.table_sizes[layer_id] = tuple(per_order)

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
        if layer_id not in self.table_sizes:
            raise KeyError(f"layer {layer_id} 不在 layer_ids={self.layer_ids} 中")
        if input_ids.dim() != 2:
            raise ValueError(f"input_ids 应为 [B, T]，收到 {tuple(input_ids.shape)}")
        if input_ids.dtype != torch.long:
            input_ids = input_ids.long()
        if bool((input_ids < 0).any()) or bool((input_ids >= self.vocab_size).any()):
            raise ValueError("input_ids 超出压缩词表范围")

        b, t = input_ids.shape
        multipliers = self.multipliers[layer_id].to(input_ids.device)

        shifted: list[torch.Tensor] = []
        for k in range(self.max_ngram_size):
            if k == 0:
                shifted.append(input_ids)
            else:
                padded = F.pad(input_ids, (k, 0), value=self.pad_id)
                shifted.append(padded[:, :t])

        hashes: list[torch.Tensor] = []
        for order in range(2, self.max_ngram_size + 1):
            mix = shifted[0] * multipliers[0]
            for k in range(1, order):
                # Python ints stay in int64 because of the multiplier bound.
                mix = torch.bitwise_xor(mix, shifted[k] * multipliers[k])
            sizes = self.table_sizes[layer_id][order - 2]
            for size in sizes:
                hashes.append(torch.remainder(mix, int(size)))

        return torch.stack(hashes, dim=2).to(torch.long)

    def hash_all(self, input_ids: torch.Tensor) -> dict[int, torch.Tensor]:
        return {layer_id: self.hash(input_ids, layer_id) for layer_id in self.layer_ids}


def _check_hash_bounds(hashes: torch.Tensor, sizes: Sequence[int]) -> None:
    """Debug helper used by tests: every head stays inside its table."""
    for h, size in enumerate(sizes):
        col = hashes[..., h]
        if bool((col < 0).any()) or bool((col >= size).any()):
            raise AssertionError(f"head {h} 越界: size={size}")
