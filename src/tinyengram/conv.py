"""Short depthwise causal convolution (paper Eq. 5).

    Y = SiLU( Conv1D( RMSNorm(V~) ) ) + V~

with kernel size ``w = 4`` and dilation ``delta = max N``.  A dilated causal
convolution lets one position mix memory retrieved at ``t, t-1, t-2, ...``
without touching the future, broadening the effective n-gram order at almost
zero FLOPs (``w * d`` per channel).

Implementation notes:

* ``groups = T? no, channels`` — every channel has its own filter (depthwise).
* Padding is applied on the **left** and then truncated, which is exactly
  causal: output ``t`` sees inputs ``t - j*delta`` for ``j < w``.
* The convolution is initialized to **zero**, so at step 0
  ``Y = 0 + V~ = V~`` and the Engram module is a pure gated lookup.  This
  identity-preserving initialization is what the paper reports in Table 5
  ("Engram Conv Zero Init: True").
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .gating import RMSNorm

__all__ = ["ShortConv"]


class ShortConv(nn.Module):
    """Causal depthwise conv over ``[B, T, G, C]`` with per-branch RMSNorm."""

    def __init__(
        self,
        hidden_size: int,
        *,
        groups: int = 1,
        kernel_size: int = 4,
        dilation: int = 3,
        norm_eps: float = 1e-6,
        activation: bool = True,
        zero_init: bool = True,
    ) -> None:
        raise NotImplementedError("TODO(L4): 请实现本关卡的核心逻辑；先读 docs/ 对应讲义")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: ``[B, T, G, C]``. Returns: same shape, causal, residual-free."""
        raise NotImplementedError("TODO(L4): 请实现本关卡的核心逻辑；先读 docs/ 对应讲义")
