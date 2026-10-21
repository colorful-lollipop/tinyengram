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
        super().__init__()
        if groups < 1:
            raise ValueError("groups 必须 >= 1")
        if kernel_size < 1:
            raise ValueError("kernel_size 必须 >= 1")
        if dilation < 1:
            raise ValueError("dilation 必须 >= 1")

        self.hidden_size = int(hidden_size)
        self.groups = int(groups)
        self.kernel_size = int(kernel_size)
        self.dilation = int(dilation)
        self.activation = bool(activation)

        channels = self.groups * self.hidden_size
        self.norms = nn.ModuleList(
            [RMSNorm(self.hidden_size, eps=norm_eps) for _ in range(self.groups)]
        )
        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=self.kernel_size,
            groups=channels,
            bias=False,
            dilation=self.dilation,
            padding=(self.kernel_size - 1) * self.dilation,
        )
        if zero_init:
            nn.init.zeros_(self.conv.weight)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: ``[B, T, G, C]``. Returns: same shape, causal, residual-free."""
        if x.dim() != 4:
            raise ValueError(f"期望 [B, T, G, C]，收到 {tuple(x.shape)}")
        b, t, g, c = x.shape
        if g != self.groups or c != self.hidden_size:
            raise ValueError(
                f"期望 G={self.groups}, C={self.hidden_size}，收到 G={g}, C={c}"
            )

        normed = torch.cat([self.norms[i](x[:, :, i, :]) for i in range(g)], dim=-1)
        y = self.conv(normed.transpose(1, 2))
        y = y[..., :t]  # remove right padding introduced for causality
        if self.activation:
            y = self.act(y)
        return y.transpose(1, 2).contiguous().view(b, t, g, c)
