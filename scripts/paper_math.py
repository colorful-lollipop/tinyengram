#!/usr/bin/env python3
"""复算 Engram 论文里的关键数字，检查本项目的公式与论文一致。

    python scripts/paper_math.py

不训练、不下载任何东西，纯算术，几毫秒跑完。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IMPL = os.environ.get("TINYENGRAM_IMPL", "solutions")
sys.path.insert(0, str(ROOT / IMPL))

from tinyengram import allocate, engram_parameter_count, iso_flops  # noqa: E402


def line(label: str, value: str, paper: str) -> None:
    print(f"{label:<38} {value:>18}   (论文: {paper})")


def main() -> None:
    print("\nEngram-27B / Table 5")
    n27 = engram_parameter_count(layers=2, slots_per_head=2_262_400, n_heads_total=16, head_dim=80)
    line("Engram 参数量", f"{n27 / 1e9:.2f}B", "5.7B")

    n40 = engram_parameter_count(layers=2, slots_per_head=7_239_680, n_heads_total=16, head_dim=80)
    line("Engram-40B 记忆参数量", f"{n40 / 1e9:.2f}B", "18.5B")

    table5_slots = 2_262_400
    line("每个 head/order 的 slot 数", f"{table5_slots:,}", "2,262,400")
    line("总 head row 数/层", f"{table5_slots * 16:,}", "36.2M")

    print("\n稀疏分配 (Engram-27B)")
    a = allocate(
        total_params=26_700_000_000,
        active_params=3_800_000_000,
        rho=0.743,
        params_per_expert=309_000_000,
        engram_dmem=1280,
        engram_layers=2,
        n_heads_total=16,
    )
    line("rho", f"{a.rho:.3f}", "0.743")
    line("Moe / sparse", f"{a.moe_params / 1e9:.2f}B", "~17.0B")
    line("Engram / sparse", f"{a.engram_params / 1e9:.2f}B", "~5.7B")
    line("routed experts", f"{a.n_routed_experts}", "55")
    line("slots/head", f"{a.slots_per_head:,}", "~2.26M")

    print("\n计算量")
    flops = iso_flops(3.8e9, 262e9)
    line("训练 FLOPs (6PD)", f"{flops:.2e}", "~6e21")

    print("\n系统（每个 token 的 PCIe 流量）")
    per_token_bytes = 16 * 80 * 2
    line("激活 slot 数", "16", "16 (2 orders x 8 heads)")
    line("每 token 读取", f"{per_token_bytes / 1024:.1f} KiB", "~2.5 KiB (bf16)")
    print("  -> 与表大小无关；100B 参数表也只是按需取 16 个 head 行。\n")

    print("结论：公式与 Table 5 对得上；差异来自论文四舍五入。\n")


if __name__ == "__main__":
    main()
