#!/usr/bin/env python3
"""分阶段评分：显示每一关的通过情况。

    python scripts/grade.py              # 你的实现（src/tinyengram）
    python scripts/grade.py --solution   # 参考实现
    python scripts/grade.py --stage 3    # 只跑 L3
    python scripts/grade.py --all        # 包含 slow 的端到端复现
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

STAGES = [
    (0, "Tokenizer 压缩", "tests/test_l0_tokenizer.py", False),
    (1, "多头部哈希", "tests/test_l1_hashing.py", False),
    (2, "检索与内存布局", "tests/test_l2_embedding.py", False),
    (3, "上下文门控", "tests/test_l3_gating.py", False),
    (4, "ShortConv 与融合", "tests/test_l4_engram.py", False),
    (5, "Transformer 与训练", "tests/test_l5_transformer.py", False),
    (6, "稀疏分配与缩放律", "tests/test_l6_allocation.py", False),
    (7, "推理系统与 offload", "tests/test_l7_systems.py", False),
    (8, "机制分析", "tests/test_l8_analysis.py", False),
    (9, "端到端复现", "tests/test_l9_reproduction.py", True),
]

# 可选的 GPU kernel 测试并到 L7，不单独占一关。
EXTRA = [(7, "tests/test_l7_perf.py")]

ANSI = {
    "green": "\033[32m",
    "red": "\033[31m",
    "yellow": "\033[33m",
    "dim": "\033[2m",
    "reset": "\033[0m",
}


def color(text: str, name: str, enabled: bool) -> str:
    return f"{ANSI[name]}{text}{ANSI['reset']}" if enabled else text


def parse_counts(output: str) -> tuple[int, int, int]:
    def find(pattern: str) -> int:
        m = re.search(pattern, output)
        return int(m.group(1)) if m else 0

    return find(r"(\d+) passed"), find(r"(\d+) failed"), find(r"(\d+) skipped")


def run_files(files: list[str], impl: str, include_slow: bool):
    cmd = [sys.executable, "-m", "pytest", *files, "-q", "--tb=line", "-p", "no:cacheprovider"]
    if not include_slow:
        cmd += ["-m", "not slow"]
    cmd += ["-m", "not gpu"]
    env = os.environ.copy()
    env["TINYENGRAM_IMPL"] = impl
    proc = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    output = proc.stdout + proc.stderr
    passed, failed, skipped = parse_counts(output)
    if proc.returncode == 0 and passed == 0 and skipped > 0:
        status = "SKIP"
    elif proc.returncode == 0:
        status = "PASS"
    elif passed > 0 and failed > 0:
        status = "PART"
    else:
        status = "FAIL"
    return status, output, passed, failed, skipped


def first_error(output: str) -> str:
    for line in output.splitlines():
        if line.startswith("FAILED") or "NotImplementedError" in line or line.startswith("E "):
            return line.strip()[:120]
    return ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--solution", action="store_true", help="用 solutions/ 里的参考实现评分")
    ap.add_argument("--stage", type=int, help="只跑某一关")
    ap.add_argument("--all", action="store_true", help="包含 slow 测试")
    ap.add_argument("--verbose", action="store_true", help="失败时打印完整输出")
    args = ap.parse_args()

    try:
        import pytest  # noqa: F401
    except ImportError:
        raise SystemExit("需要 pytest：pip install pytest")

    impl = "solutions" if args.solution else "src"
    ansi = sys.stdout.isatty()
    stages = [s for s in STAGES if args.stage is None or s[0] == args.stage]
    if not stages:
        raise SystemExit(f"没有 L{args.stage}，可选 0-9")

    title = "TinyEngram 参考实现评分" if args.solution else "TinyEngram 学习进度"
    print(f"\n{title}  (impl = {impl}/tinyengram)\n")
    print(f"{'关卡':<4}{'主题':<30}{'状态':<8}{'通过/失败/跳过':<16}")
    print("-" * 70)

    total_pass = total_fail = total_skip = 0
    failed_outputs = []
    for stage, name, path, slow in stages:
        if slow and not (args.all or args.stage == stage):
            print(f"L{stage:<3}{name:<30}{'SKIP':<8}{'需要 --all':<16}")
            continue
        files = [path] + [f for s, f in EXTRA if s == stage]
        status, output, passed, failed, skipped = run_files(files, impl, args.all)
        total_pass += passed
        total_fail += failed
        total_skip += skipped

        status_color = {"PASS": "green", "SKIP": "yellow", "PART": "yellow", "FAIL": "red"}[status]
        print(
            f"L{stage:<3}{name:<30}"
            f"{color(status, status_color, ansi):<8}"
            f"{passed:>3} / {failed:>3} / {skipped:>3}"
        )
        if status in ("FAIL", "PART"):
            failed_outputs.append((stage, output))

    print("-" * 70)
    total = total_pass + total_fail
    pct = 100 * total_pass / total if total else 0.0
    print(f"合计: {total_pass} passed, {total_fail} failed, {total_skip} skipped  ({pct:.0f}%)\n")

    for stage, output in failed_outputs:
        err = first_error(output)
        if err:
            print(f"L{stage} 第一个错误: {err}")
    if failed_outputs and not args.verbose:
        print("\n加 --verbose 或直接运行 pytest <file> -x 查看完整报错。")
    if args.verbose:
        for stage, output in failed_outputs:
            print(f"\n===== L{stage} =====\n{output[-3000:]}")


if __name__ == "__main__":
    main()
