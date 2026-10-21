#!/usr/bin/env python3
"""实验 1：U 形稀疏分配曲线（论文 Figure 3 左）。

    python experiments/exp1_allocation.py

纯解析实验：用论文的稀疏预算公式扫 rho，再用二次多项式拟合最优分配点。
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import torch

from tinyengram import (
    allocate,
    argmin_quadratic,
    quadratic_fit,
    u_shaped_loss,
)


def main() -> None:
    rhos = torch.linspace(0.30, 1.00, 29)
    losses = u_shaped_loss(rhos, floor=1.7109, optimum=0.78, curvature=0.12, asymmetry=0.25)
    a, b, c = quadratic_fit(rhos.tolist(), losses.tolist())
    best = argmin_quadratic(a, b, c)

    print("\n rho     val loss     Engram slots (26.7B budget)")
    for rho, loss in zip(rhos.tolist(), losses.tolist()):
        alloc = allocate(
            total_params=26_700_000_000,
            active_params=3_800_000_000,
            rho=rho,
            params_per_expert=309_000_000,
            engram_dmem=1280,
            engram_layers=2,
            n_heads_total=16,
        )
        print(f"{rho:4.2f}     {loss:.4f}       {alloc.slots_total / 1e6:6.2f}M")

    print(f"\n二次拟合: L(rho) = {a:.4f} rho^2 + {b:.4f} rho + {c:.4f}")
    print(f"拟合最优 rho* = {best:.3f}  (论文: 0.75-0.80)")
    print(f"纯 MoE (rho=1.0) loss = {float(u_shaped_loss(1.0, floor=1.7109, optimum=0.78, curvature=0.12, asymmetry=0.25)):.4f}")

    _bootstrap.write_markdown(
        "allocation.md",
        "# 稀疏分配实验\n\n"
        f"- 拟合最优 rho* = **{best:.3f}**（论文 0.75–0.80）\n"
        "- 机制：rho→1 时缺少静态记忆，模型用深度重建 n-gram；\n"
        "  rho→0 时缺少条件计算，动态推理能力受损。\n",
    )


if __name__ == "__main__":
    main()
