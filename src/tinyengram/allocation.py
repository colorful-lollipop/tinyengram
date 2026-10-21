"""Sparsity allocation and scaling-law arithmetic (paper §3).

Definitions (paper §3.1):

* ``P_tot``    total trainable parameters, excluding vocab embedding / LM head
* ``P_act``    parameters touched per token (determines FLOPs)
* ``P_sparse`` = ``P_tot - P_act``: "free" capacity, either unrouted experts
  or unretrieved embedding rows.
* ``rho`` = fraction of ``P_sparse`` assigned to MoE experts:
  ``P_moe = rho * P_sparse``, ``P_engram = (1 - rho) * P_sparse``.

Two empirical laws from the paper:

1. validation loss vs ``rho`` is **U-shaped**, with the optimum at
   ``rho ~= 0.75-0.80`` across two compute budgets;
2. validation loss vs ``log`` number of memory slots is **linear** (power law),
   with no sign of saturation up to 1e7 slots.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

__all__ = [
    "Allocation",
    "engram_parameter_count",
    "allocate",
    "u_shaped_loss",
    "quadratic_fit",
    "argmin_quadratic",
    "power_law_fit",
    "power_law_predict",
    "iso_flops",
]


@dataclass(frozen=True)
class Allocation:
    total_params: int
    active_params: int
    rho: float
    moe_params: int
    engram_params: int
    n_routed_experts: int
    slots_total: int
    slots_per_head: int

    @property
    def sparse_params(self) -> int:
        return self.total_params - self.active_params

    @property
    def sparsity(self) -> float:
        return self.total_params / self.active_params


def engram_parameter_count(
    *,
    layers: int,
    slots_per_head: int,
    n_heads_total: int,
    head_dim: int,
    include_projections: bool = False,
    hidden_size: int = 0,
    branches: int = 1,
    conv_kernel_size: int = 4,
) -> int:
    """Embedding rows dominate: ``layers * slots_per_head * heads * head_dim``.

    Table 5 reports ``Engram Vocab Size`` per hash head (2,262,400); there are
    16 heads of 80 dims, so one layer is
    ``2,262,400 * 16 * 80 = 2.90B`` parameters.
    """
    raise NotImplementedError("TODO(L6): 请实现本关卡的核心逻辑；先读 docs/ 对应讲义")


def allocate(
    *,
    total_params: int,
    active_params: int,
    rho: float,
    params_per_expert: int,
    engram_dmem: int,
    engram_layers: int,
    n_heads_total: int,
    n_shared_experts: int = 2,
) -> Allocation:
    """Turn a budget split ``rho`` into expert count and embedding slots."""
    raise NotImplementedError("TODO(L6): 请实现本关卡的核心逻辑；先读 docs/ 对应讲义")


def u_shaped_loss(
    rho: float | torch.Tensor,
    *,
    floor: float = 1.710,
    optimum: float = 0.78,
    curvature: float = 0.12,
    asymmetry: float = 0.0,
) -> float | torch.Tensor:
    """Analytic stand-in for Figure 3 (left).

    ``L(rho) = floor + curvature * (rho - optimum)^2
              + asymmetry * (optimum - rho)^3 * 1[rho < optimum]``

    The cubic term makes the left wall steeper (memory cannot replace
    computation), while keeping the minimum at ``optimum``.
    """
    raise NotImplementedError("TODO(L6): 请实现本关卡的核心逻辑；先读 docs/ 对应讲义")


def quadratic_fit(x: Sequence[float], y: Sequence[float]) -> tuple[float, float, float]:
    """Least-squares fit ``y = a x^2 + b x + c``. Returns ``(a, b, c)``."""
    raise NotImplementedError("TODO(L6): 请实现本关卡的核心逻辑；先读 docs/ 对应讲义")


def argmin_quadratic(a: float, b: float, c: float, *, lo: float = 0.0, hi: float = 1.0) -> float:
    raise NotImplementedError("TODO(L6): 请实现本关卡的核心逻辑；先读 docs/ 对应讲义")


def power_law_fit(
    slots: Sequence[float], losses: Sequence[float]
) -> tuple[float, float]:
    """Fit ``loss = intercept + slope * log10(slots)``.

    Returns ``(intercept, slope)``; ``slope`` should be negative.
    """
    raise NotImplementedError("TODO(L6): 请实现本关卡的核心逻辑；先读 docs/ 对应讲义")


def power_law_predict(intercept: float, slope: float, slots: float) -> float:
    raise NotImplementedError("TODO(L6): 请实现本关卡的核心逻辑；先读 docs/ 对应讲义")


def iso_flops(active_params: int, tokens: int, flops_per_param_token: float = 6.0) -> float:
    """Standard estimate: forward+backward ~= 6 * P_act * tokens."""
    raise NotImplementedError("TODO(L6): 请实现本关卡的核心逻辑；先读 docs/ 对应讲义")
