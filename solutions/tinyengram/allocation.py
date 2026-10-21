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
    table = layers * slots_per_head * n_heads_total * head_dim
    if not include_projections:
        return table
    dmem = n_heads_total * head_dim
    proj = layers * (
        dmem * hidden_size  # W_V
        + branches * dmem * hidden_size  # W_K per branch
        + branches * hidden_size * conv_kernel_size  # depthwise conv
    )
    return table + proj


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
    if not 0.0 <= rho <= 1.0:
        raise ValueError("rho 必须在 [0, 1] 内")
    if total_params <= active_params:
        raise ValueError("total_params 必须大于 active_params")
    if params_per_expert < 1 or engram_dmem < 1 or engram_layers < 1:
        raise ValueError("params_per_expert / engram_dmem / engram_layers 必须 >= 1")

    sparse = total_params - active_params
    moe_params = int(round(rho * sparse))
    engram_params = sparse - moe_params

    # Shared experts are part of the activated budget in the paper; here we
    # only convert the *routed* share into a count.
    n_routed = int(round(moe_params / params_per_expert))
    # params = slots_per_head * dmem * layers  (dmem already sums all heads)
    slots_per_head = engram_params // (engram_dmem * engram_layers)
    slots_total = slots_per_head * n_heads_total

    return Allocation(
        total_params=total_params,
        active_params=active_params,
        rho=rho,
        moe_params=moe_params,
        engram_params=engram_params,
        n_routed_experts=max(0, n_routed),
        slots_total=slots_total,
        slots_per_head=max(0, slots_per_head),
    )


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
    if not 0.0 <= optimum <= 1.0:
        raise ValueError("optimum 必须在 [0, 1] 内")
    if curvature < 0:
        raise ValueError("curvature 必须 >= 0")
    delta = rho - optimum
    out = floor + curvature * delta**2
    if asymmetry:
        if isinstance(delta, torch.Tensor):
            out = out + asymmetry * torch.clamp(-delta, min=0.0) ** 3
        else:
            out = out + asymmetry * max(-delta, 0.0) ** 3
    return out


def quadratic_fit(x: Sequence[float], y: Sequence[float]) -> tuple[float, float, float]:
    """Least-squares fit ``y = a x^2 + b x + c``. Returns ``(a, b, c)``."""
    if len(x) != len(y) or len(x) < 3:
        raise ValueError("至少需要 3 个点，且 x/y 等长")
    x_t = torch.tensor(list(x), dtype=torch.float64)
    y_t = torch.tensor(list(y), dtype=torch.float64)
    design = torch.stack([x_t**2, x_t, torch.ones_like(x_t)], dim=1)
    sol = torch.linalg.lstsq(design, y_t.unsqueeze(1)).solution.squeeze(1)
    return float(sol[0]), float(sol[1]), float(sol[2])


def argmin_quadratic(a: float, b: float, c: float, *, lo: float = 0.0, hi: float = 1.0) -> float:
    if a <= 0:
        raise ValueError("不是开口向上的抛物线（a 必须 > 0）")
    vertex = -b / (2 * a)
    return float(min(hi, max(lo, vertex)))


def power_law_fit(
    slots: Sequence[float], losses: Sequence[float]
) -> tuple[float, float]:
    """Fit ``loss = intercept + slope * log10(slots)``.

    Returns ``(intercept, slope)``; ``slope`` should be negative.
    """
    if len(slots) != len(losses) or len(slots) < 2:
        raise ValueError("至少需要 2 个点，且 slots/losses 等长")
    x = torch.log10(torch.tensor(list(slots), dtype=torch.float64))
    y = torch.tensor(list(losses), dtype=torch.float64)
    x_mean, y_mean = x.mean(), y.mean()
    denom = ((x - x_mean) ** 2).sum()
    if float(denom) == 0.0:
        raise ValueError("slots 不能全部相同")
    slope = ((x - x_mean) * (y - y_mean)).sum() / denom
    intercept = y_mean - slope * x_mean
    return float(intercept), float(slope)


def power_law_predict(intercept: float, slope: float, slots: float) -> float:
    if slots <= 0:
        raise ValueError("slots 必须 > 0")
    import math

    return intercept + slope * math.log10(slots)


def iso_flops(active_params: int, tokens: int, flops_per_param_token: float = 6.0) -> float:
    """Standard estimate: forward+backward ~= 6 * P_act * tokens."""
    if active_params < 0 or tokens < 0:
        raise ValueError("参数与 token 数必须非负")
    return flops_per_param_token * active_params * tokens
