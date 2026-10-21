#!/usr/bin/env python3
"""Engram 检索 kernel / 模块基准。

    python perf/bench.py --breakdown

有 CUDA + Triton 时比较 eager / torch.compile / Triton；否则只跑 CPU gather
基准并打印提示。重点看**有效带宽**，因为 Engram 检索是 memory-bound。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "solutions"))
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from tinyengram import EngramConfig, Engram  # noqa: E402
from perf.kernels import (  # noqa: E402
    HAS_TRITON,
    eager_gather_concat,
    triton_gather_concat,
)


def time_cuda(fn, warmup: int = 10, iters: int = 50) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def time_cpu(fn, warmup: int = 3, iters: int = 20) -> float:
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) * 1000 / iters


def bench_gather(breakdown: bool) -> None:
    torch.manual_seed(0)
    n, heads, dim = 16384, 16, 80
    slots = heads * 80_000
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    table = torch.randn(slots, dim, dtype=dtype)
    hashes = torch.randint(0, 80_000, (n, heads), dtype=torch.long)
    offsets = torch.arange(0, slots, 80_000, dtype=torch.long)
    gb = n * heads * dim * table.element_size() / 1e9

    print(f"\nGather: N={n:,}  H={heads}  D={dim}  table={slots * dim * 2 / 1e9:.1f} GB  dtype={dtype}")
    if not torch.cuda.is_available():
        ms = time_cpu(lambda: eager_gather_concat(table, hashes, offsets))
        print(f"  CPU eager: {ms:7.2f} ms  {gb / (ms / 1e3):7.1f} GB/s")
        print("  （没有 CUDA，Triton/compile 对比跳过）")
        return

    table = table.cuda()
    hashes = hashes.cuda()
    offsets = offsets.cuda()
    ref = eager_gather_concat(table, hashes, offsets)

    eager_ms = time_cuda(lambda: eager_gather_concat(table, hashes, offsets))
    print(f"  eager:        {eager_ms:7.3f} ms  {gb / (eager_ms / 1e3):7.1f} GB/s")

    compiled = torch.compile(lambda t, h, o: eager_gather_concat(t, h, o))
    compiled(table, hashes, offsets)  # trigger compile
    comp_ms = time_cuda(lambda: compiled(table, hashes, offsets))
    print(f"  torch.compile:{comp_ms:7.3f} ms  {gb / (comp_ms / 1e3):7.1f} GB/s")

    if HAS_TRITON:
        out = triton_gather_concat(table, hashes, offsets)
        ok = torch.allclose(out, ref)
        tri_ms = time_cuda(lambda: triton_gather_concat(table, hashes, offsets))
        print(f"  Triton fused: {tri_ms:7.3f} ms  {gb / (tri_ms / 1e3):7.1f} GB/s  (allclose={ok})")
        print(f"  加速: {eager_ms / tri_ms:.2f}x vs eager")
    else:
        print("  Triton 未安装，跳过")


def bench_module() -> None:
    torch.manual_seed(0)
    cfg = EngramConfig(
        vocab_size=32000,
        hidden_size=512,
        hc_mult=4,
        max_ngram_size=3,
        n_head_per_ngram=8,
        n_embed_per_ngram=320,
        slots_per_head=25_000,
        layer_ids=(1,),
        pad_id=0,
    )
    module = Engram(cfg, 1)
    if torch.cuda.is_available():
        module = module.cuda()
    module.eval()
    ids = torch.randint(0, 32000, (4, 128))
    hidden = torch.randn(4, 128, cfg.hc_mult, cfg.hidden_size)
    if torch.cuda.is_available():
        ids, hidden = ids.cuda(), hidden.cuda()
    device = "CUDA" if torch.cuda.is_available() else "CPU"
    timer = time_cuda if torch.cuda.is_available() else time_cpu
    with torch.no_grad():
        ms = timer(lambda: module(hidden, ids))
    params = sum(p.numel() for p in module.parameters())
    print(f"\nEngram module ({device}): {params / 1e6:.1f}M params, "
          f"B=4 T=128 -> {ms:.2f} ms/forward")
    print(f"  每 token 读取 {cfg.n_heads_total * cfg.head_dim * 2 / 1024:.1f} KiB (bf16)")

    if torch.cuda.is_available():
        compiled = torch.compile(module)
        with torch.no_grad():
            compiled(hidden, ids)
            cms = timer(lambda: compiled(hidden, ids))
        print(f"  torch.compile: {cms:.2f} ms/forward  ({ms / cms:.2f}x)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--breakdown", action="store_true", help="额外跑模块级 forward")
    args = ap.parse_args()
    bench_gather(breakdown=args.breakdown)
    if args.breakdown:
        bench_module()


if __name__ == "__main__":
    main()
