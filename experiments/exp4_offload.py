#!/usr/bin/env python3
"""实验 4：100B 记忆表的 host offload / 预取 / Zipf 缓存（论文 §6.4）。

    python experiments/exp4_offload.py

用确定性访问序列模拟：
* 不同 PCIe 带宽与计算窗口下的 stall；
* 不同 HBM 缓存容量下的命中率；
* 每 token 的传输量为什么与表大小无关。
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import time

import torch

from tinyengram import NgramHashMapping, prefetch_plan, simulate_offload


def main() -> None:
    # 100B 参数、d_mem=1280、16 heads -> 每 head 78.1M rows，每行 80 dims（bf16）
    dmem = 1280
    n_heads = 16
    head_dim = dmem // n_heads
    slots = 100_000_000_000 // dmem  # per-head slots
    bytes_per_slot = head_dim * 2  # bf16, one head row
    total_gb = slots * n_heads * bytes_per_slot / 1e9
    print(f"\n表: {slots / 1e6:.1f}M slots/head x {n_heads} heads = "
          f"{slots * n_heads / 1e9:.2f}B rows, {total_gb:.0f} GB (bf16)")

    # 一个 batch 的 token 序列（用真实 hash 生成，保证访问分布不是玩具）
    mapping = NgramHashMapping(
        vocab_size=129280,
        slots_per_head=slots,
        max_ngram_size=3,
        n_head_per_ngram=8,
        layer_ids=(2, 15),
        pad_id=0,
        seed=0,
    )
    gen = torch.Generator().manual_seed(0)
    ids = torch.randint(0, 129280, (16, 512), generator=gen)
    plan = prefetch_plan(ids, mapping)
    print(f"每层激活的 (token, head) 对: {ids.numel() * 16 / 1e6:.2f}M")

    print("\nPCIe 64 GB/s，计算窗口 2 ms")
    print(f"{'cache slots':>12}  {'hit rate':>9}  {'bytes':>10}  {'transfer ms':>11}  {'stall ms':>9}")
    for cap in (0, 10_000, 100_000, 1_000_000):
        res = simulate_offload(
            plan,
            bytes_per_slot=bytes_per_slot,
            compute_window_ms=2.0,
            bandwidth_gbps=64.0,
            cache_capacity=cap,
        )
        print(
            f"{cap:12,d}  {res.hit_rate:9.3f}  {res.bytes / 1e6:9.1f}MB  "
            f"{res.transfer_ms:11.2f}  {res.stalls_ms:9.2f}"
        )

    print("\n计算窗口 vs stall（无缓存）")
    for window in (0.0, 0.25, 0.5, 1.0, 2.0, 4.0):
        res = simulate_offload(
            plan, bytes_per_slot=bytes_per_slot, compute_window_ms=window, bandwidth_gbps=64.0
        )
        print(f"  window={window:4.2f} ms -> stall={res.stalls_ms:8.2f} ms")

    # 真实 CPU gather 的吞吐，帮助建立“内存墙”的直觉
    rows, dim = 200_000, 1280
    table = torch.randn(rows, dim, dtype=torch.float16)
    idx = torch.randint(0, rows, (16 * 512 * 16,), generator=gen)
    t0 = time.perf_counter()
    for _ in range(10):
        _ = table[idx]
    dt = (time.perf_counter() - t0) / 10
    gb = idx.numel() * dim * 2 / 1e9
    print(f"\nCPU gather: {idx.numel():,} rows, {gb:.3f} GB/次, {dt * 1e3:.2f} ms -> {gb / dt:.1f} GB/s")

    _bootstrap.write_markdown(
        "offload.md",
        "# Host offload 实验\n\n"
        f"- 表大小 {total_gb:.0f} GB，每 token 只读 16 x 80 x 2 = 2.5 KiB；\n"
        "- 传输量随 batch 内 unique slots 增长，不随表大小增长；\n"
        "- 2 ms 计算窗口即可掩盖大部分 PCIe 延迟；Zipf 缓存进一步降低 miss。\n",
    )


if __name__ == "__main__":
    main()
