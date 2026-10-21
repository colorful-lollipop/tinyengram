"""Configuration objects shared by the whole project.

The paper uses two nested config spaces:

* ``EngramConfig``  – everything that determines the memory module (vocabulary,
  n-gram orders, number of hash heads, table capacity, insertion layers);
* ``TinyLMConfig``   – the backbone around it.

Keeping them separate is deliberate: in the paper the *backbone* is held fixed
while Engram capacity is swept (Figure 3, right), and vice versa (Table 5).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

__all__ = ["EngramConfig", "TinyLMConfig"]


def _as_int_tuple(values: Sequence[int], name: str) -> tuple[int, ...]:
    out = tuple(int(v) for v in values)
    if len(out) == 0:
        raise ValueError(f"{name} 不能为空")
    return out


@dataclass
class EngramConfig:
    """Hyper-parameters of one Engram memory module family.

    Notes
    -----
    ``n_embed_per_ngram`` is the width contributed by **one n-gram order**
    (paper: 640, with two orders -> ``d_mem = 1280``).  It is split evenly
    across ``n_head_per_ngram`` hash heads, so every retrieved slot stores
    ``head_dim = n_embed_per_ngram // n_head_per_ngram`` floats.

    ``slots_per_head`` is the *requested* number of rows in each head's table.
    The actual table size is the next prime >= that value, and primes are
    globally unique across layers/orders/heads (see :mod:`tinyengram.hashing`).
    A sequence is interpreted as one value per n-gram order.

    ``vocab_size`` / ``pad_id`` are in the id space the module sees.  If an
    external ``CompressedTokenizer`` is supplied to :class:`Engram`, they are
    interpreted in the *raw* tokenizer space instead, and the module remaps
    both internally.
    """

    vocab_size: int = 4096
    hidden_size: int = 128
    hc_mult: int = 4
    max_ngram_size: int = 3
    n_head_per_ngram: int = 8
    n_embed_per_ngram: int = 32
    slots_per_head: int | Sequence[int] = 1024
    layer_ids: Sequence[int] = (1,)
    pad_id: int = 0
    seed: int = 0
    conv_kernel_size: int = 4
    conv_zero_init: bool = True

    # filled in __post_init__
    layer_ids_: tuple[int, ...] = field(init=False, repr=False)
    slots_per_head_: tuple[int, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.layer_ids_ = _as_int_tuple(self.layer_ids, "layer_ids")
        if len(set(self.layer_ids_)) != len(self.layer_ids_):
            raise ValueError("layer_ids 不能重复")
        if any(i < 0 for i in self.layer_ids_):
            raise ValueError("layer_ids 必须是非负整数")

        if self.max_ngram_size < 2:
            raise ValueError("max_ngram_size 至少为 2（2-gram 是最小的上下文记忆）")
        if self.n_head_per_ngram < 1:
            raise ValueError("n_head_per_ngram 必须 >= 1")
        if self.n_embed_per_ngram % self.n_head_per_ngram != 0:
            raise ValueError("n_embed_per_ngram 必须能被 n_head_per_ngram 整除")
        if self.vocab_size < 1:
            raise ValueError("vocab_size 必须 >= 1")
        if not 0 <= self.pad_id < self.vocab_size:
            raise ValueError("pad_id 必须落在 [0, vocab_size) 内")
        if self.conv_kernel_size < 1:
            raise ValueError("conv_kernel_size 必须 >= 1")

        slots = self.slots_per_head
        if isinstance(slots, int):
            slots = (slots,) * self.n_orders
        slots = _as_int_tuple(slots, "slots_per_head")
        if len(slots) != self.n_orders:
            raise ValueError(
                f"slots_per_head 需要 {self.n_orders} 个值（每个 n-gram order 一个），"
                f"收到 {len(slots)} 个"
            )
        if any(s < 1 for s in slots):
            raise ValueError("slots_per_head 必须 >= 1")
        self.slots_per_head_ = slots

    @property
    def n_orders(self) -> int:
        """Number of n-gram orders: N=3 -> 2 orders (2-gram, 3-gram)."""
        return self.max_ngram_size - 1

    @property
    def head_dim(self) -> int:
        return self.n_embed_per_ngram // self.n_head_per_ngram

    @property
    def dmem(self) -> int:
        """Width of the concatenated memory vector e_t."""
        return self.n_orders * self.n_embed_per_ngram

    @property
    def n_heads_total(self) -> int:
        """Total hash heads per token = orders * heads_per_order."""
        return self.n_orders * self.n_head_per_ngram

    def slots_for_order(self, order_index: int) -> int:
        return self.slots_per_head_[order_index]


@dataclass
class TinyLMConfig:
    """A deliberately small GPT-style backbone.

    4 layers, 128 hidden, 4 heads is enough to see Engram's effect on a
    synthetic associative-recall corpus on CPU in a few minutes.
    """

    vocab_size: int = 4096
    hidden_size: int = 128
    n_layers: int = 4
    n_heads: int = 4
    max_seq: int = 256
    ffn_mult: int = 4
    hc_mult: int = 1
    engram_layers: Sequence[int] = ()
    engram: EngramConfig | None = None
    dropout: float = 0.0
    tie_weights: bool = True
    rope_theta: float = 10000.0

    engram_layers_: tuple[int, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.hidden_size % self.n_heads != 0:
            raise ValueError("hidden_size 必须能被 n_heads 整除")
        if self.hc_mult < 1:
            raise ValueError("hc_mult 必须 >= 1")
        self.engram_layers_ = _as_int_tuple(self.engram_layers, "engram_layers") if self.engram_layers else ()
        if any(i < 0 or i >= self.n_layers for i in self.engram_layers_):
            raise ValueError("engram_layers 必须落在 [0, n_layers) 内")
        if self.engram is not None:
            if self.engram.vocab_size != self.vocab_size:
                raise ValueError("EngramConfig.vocab_size 必须与 TinyLMConfig.vocab_size 一致")
            if self.engram.hidden_size != self.hidden_size:
                raise ValueError("EngramConfig.hidden_size 必须与 TinyLMConfig.hidden_size 一致")
            if self.engram.hc_mult != self.hc_mult:
                raise ValueError("EngramConfig.hc_mult 必须与 TinyLMConfig.hc_mult 一致")
            if tuple(self.engram.layer_ids_) != self.engram_layers_:
                raise ValueError("EngramConfig.layer_ids 必须与 TinyLMConfig.engram_layers 一致")
        elif self.engram_layers_:
            raise ValueError("给了 engram_layers 但没有给 engram 配置")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.n_heads
