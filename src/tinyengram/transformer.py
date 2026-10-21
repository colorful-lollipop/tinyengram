"""A small GPT backbone with optional Engram layers.

This is intentionally *not* a full DeepSeek-V3 clone: no MLA, no MoE, no mHC
manifold projection.  Those are orthogonal to understanding conditional
memory, and the sister project ``mHc/`` covers the last one in depth.  What we
keep is the part Engram interacts with:

* a multi-branch residual stream (``hc_mult`` parallel streams);
* pre-norm blocks with causal attention and a SwiGLU FFN;
* Engram inserted *before* attention at selected layers, exactly as in
  Figure 1;
* RoPE, because n-gram memory is position-local but attention is not.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import TinyLMConfig
from .engram import Engram, build_engram_mapping
from .gating import RMSNorm

__all__ = [
    "RotaryEmbedding",
    "apply_rope",
    "CausalSelfAttention",
    "SwiGLU",
    "Block",
    "TinyLM",
    "count_parameters",
]

KVCache = list[tuple[torch.Tensor, torch.Tensor]]


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_seq: int, theta: float = 10000.0) -> None:
        raise NotImplementedError("TODO(L5): 请实现本关卡的核心逻辑；先读 docs/ 对应讲义")

    def forward(
        self, seq_len: int, *, offset: int = 0, device=None, dtype=None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("TODO(L5): 请实现本关卡的核心逻辑；先读 docs/ 对应讲义")


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate pairs ``(x0, x1), (x2, x3), ...`` of the last dimension."""
    raise NotImplementedError("TODO(L5): 请实现本关卡的核心逻辑；先读 docs/ 对应讲义")


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: TinyLMConfig) -> None:
        raise NotImplementedError("TODO(L5): 请实现本关卡的核心逻辑；先读 docs/ 对应讲义")

    def forward(
        self,
        x: torch.Tensor,
        cache: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        position_offset: int = 0,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Args: ``[B, T, M, D]``. Returns ``([B,T,M,D], (k, v))``."""
        raise NotImplementedError("TODO(L5): 请实现本关卡的核心逻辑；先读 docs/ 对应讲义")


class SwiGLU(nn.Module):
    def __init__(self, dim: int, mult: int = 4) -> None:
        raise NotImplementedError("TODO(L5): 请实现本关卡的核心逻辑；先读 docs/ 对应讲义")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("TODO(L5): 请实现本关卡的核心逻辑；先读 docs/ 对应讲义")


class Block(nn.Module):
    def __init__(self, cfg: TinyLMConfig, layer_id: int, mapping=None) -> None:
        raise NotImplementedError("TODO(L5): 请实现本关卡的核心逻辑；先读 docs/ 对应讲义")

    def forward(
        self,
        hidden: torch.Tensor,
        input_ids: torch.Tensor,
        cache: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        position_offset: int = 0,
        context_ids: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        raise NotImplementedError("TODO(L5): 请实现本关卡的核心逻辑；先读 docs/ 对应讲义")


class TinyLM(nn.Module):
    def __init__(self, cfg: TinyLMConfig) -> None:
        raise NotImplementedError("TODO(L5): 请实现本关卡的核心逻辑；先读 docs/ 对应讲义")

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        caches: Optional[KVCache] = None,
        position_offset: int = 0,
        context_ids: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], KVCache]:
        """Returns ``(logits, loss, new_caches)``."""
        raise NotImplementedError("TODO(L5): 请实现本关卡的核心逻辑；先读 docs/ 对应讲义")

    def engram_modules(self) -> list[Engram]:
        return [b.engram for b in self.blocks if b.engram is not None]

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def num_parameters_excluding_engram(self) -> int:
        engram_ids = {id(p) for module in self.engram_modules() for p in module.parameters()}
        return sum(p.numel() for p in self.parameters() if id(p) not in engram_ids)


def count_parameters(module: nn.Module, *, trainable_only: bool = True) -> int:
    ps = module.parameters()
    if trainable_only:
        return sum(p.numel() for p in ps if p.requires_grad)
    return sum(p.numel() for p in ps)
