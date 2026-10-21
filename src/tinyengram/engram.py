"""The Engram module: retrieval + context-aware gating + short conv.

Data flow (paper Figure 1):

    input_ids ──► compressed ids ──► hash ids ──► flat gather ──► e_t
                                                                  │
    hidden h_t^(m) ──────────────► gate alpha_t^(m) ◄── W_K^(m) ──┤
                                                                  │
                              u_t^(m) = alpha_t^(m) * (W_V e_t) ◄──┘
                              Y_t^(m) = u_t^(m) + ShortConv(u^(m))
                              H^(l)  <- H^(l) + Y

Parameter sharing in the multi-branch (mHC) setting:

* **shared** across branches: the huge embedding table, ``W_V``;
* **branch-specific**: ``W_K^(m)`` and the gate, so each residual stream can
  decide independently whether the memory is relevant;
* one depthwise conv per branch (channel groups = ``M``).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

from .config import EngramConfig
from .conv import ShortConv
from .embedding import EngramMemory
from .gating import RMSNorm, context_gate
from .hashing import NgramHashMapping
from .tokenizer import CompressedTokenizer

__all__ = ["Engram", "build_engram_mapping", "build_engram_modules"]


def build_engram_mapping(
    cfg: EngramConfig,
    vocab_size: Optional[int] = None,
    pad_id: Optional[int] = None,
) -> NgramHashMapping:
    """One mapping shared by all layers (global prime uniqueness)."""
    raise NotImplementedError("TODO(L4): 请实现本关卡的核心逻辑；先读 docs/ 对应讲义")


class Engram(nn.Module):
    def __init__(
        self,
        cfg: EngramConfig,
        layer_id: int,
        *,
        mapping: Optional[NgramHashMapping] = None,
        compressed_tokenizer: Optional[CompressedTokenizer] = None,
    ) -> None:
        """If ``compressed_tokenizer`` is given, ``cfg.vocab_size`` and
        ``cfg.pad_id`` are interpreted in the **raw** tokenizer space; hashing
        uses the compressed ids and the remapped pad id."""
        raise NotImplementedError("TODO(L4): 请实现本关卡的核心逻辑；先读 docs/ 对应讲义")

    # ---------------------------------------------------------------- helpers
    @property
    def dmem(self) -> int:
        return self.memory.dmem

    def compress(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self.compressed_tokenizer is None:
            return input_ids
        return self.compressed_tokenizer.compress(input_ids)

    def _ids(self, input_ids: torch.Tensor, context_ids: Optional[torch.Tensor]) -> torch.Tensor:
        """Compress the *full* context; hashing needs the preceding tokens."""
        raise NotImplementedError("TODO(L4): 请实现本关卡的核心逻辑；先读 docs/ 对应讲义")

    def retrieve(
        self, input_ids: torch.Tensor, context_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Memory vector ``e_t`` of shape ``[B, T, d_mem]``."""
        raise NotImplementedError("TODO(L4): 请实现本关卡的核心逻辑；先读 docs/ 对应讲义")

    def gates(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        context_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Per-branch gate values ``[B, T, M]`` (for analysis / visualization)."""
        raise NotImplementedError("TODO(L4): 请实现本关卡的核心逻辑；先读 docs/ 对应讲义")

    # ---------------------------------------------------------------- forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        context_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: ``[B, T, M, D]`` multi-branch residual stream.
            input_ids:     ``[B, T]`` raw (or already compressed) ids.
        Returns:
            ``[B, T, M, D]`` contribution to be *added* to ``hidden_states``.
        """
        raise NotImplementedError("TODO(L4): 请实现本关卡的核心逻辑；先读 docs/ 对应讲义")

    def extra_repr(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"layer_id={self.layer_id}, dmem={self.dmem}, "
            f"slots={self.memory.total_slots}, branches={self.cfg.hc_mult}"
        )


def build_engram_modules(cfg: EngramConfig) -> nn.ModuleDict:
    """Build all layers with one shared mapping (the paper's configuration)."""
    mapping = build_engram_mapping(cfg)
    return nn.ModuleDict(
        {str(layer_id): Engram(cfg, layer_id, mapping=mapping) for layer_id in cfg.layer_ids_}
    )
