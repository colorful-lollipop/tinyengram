#!/usr/bin/env python3
"""CPU-only first success: nonzero Engram conv + KV + speculative rollback.

Run from the repository root: python scripts/hello_engram.py
Uses the completed reference implementation; this does not fill student TODOs.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "solutions"))
sys.path.insert(0, str(ROOT))

try:
    import torch
except ImportError:
    raise SystemExit("缺少 PyTorch。请先按 docs/start-here.md 安装依赖。") from None

from experiments.speculative import greedy_speculative_generate  # noqa: E402
from tinyengram import EngramConfig, TinyLM, TinyLMConfig, generate, naive_generate  # noqa: E402


def build(seed):
    torch.manual_seed(seed)
    memory = EngramConfig(
        vocab_size=16, hidden_size=16, hc_mult=1, layer_ids=(0,),
        n_head_per_ngram=2, n_embed_per_ngram=8, slots_per_head=17,
        conv_kernel_size=4, conv_zero_init=False,
    )
    return TinyLM(TinyLMConfig(
        vocab_size=16, hidden_size=16, n_heads=2, n_layers=1, max_seq=32,
        tie_weights=False, engram_layers=(0,), engram=memory,
    )).eval()


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    target, draft = build(0), build(1)
    prompt = torch.tensor([[1, 2, 3]])
    print("使用 solutions 参考实现 / CPU / 未训练的两个 TinyLM")
    print("token ID 只是数字；本例不生成自然语言，也不衡量模型质量。")
    def audit(ids, tc, dc, stats):
        for model, actual in ((target, tc), (draft, dc)):
            _, _, expected = model(ids)
            for (ak, av), (bk, bv) in zip(actual, expected):
                torch.testing.assert_close(ak, bk, atol=2e-6, rtol=2e-5)
                torch.testing.assert_close(av, bv, atol=2e-6, rtol=2e-5)
            for a, b in zip(actual.conv_states, expected.conv_states):
                torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)
        print(f"  草拟 {stats.proposed}，接受 {stats.accepted}，输出 {stats.emitted}；"
              f"缓存长度 {ids.shape[1]}，KV/卷积历史审计 PASS")
    result = greedy_speculative_generate(target, draft, prompt, 8, draft_length=4,
                                          on_commit=audit)
    ordinary = generate(target, prompt, 8)
    naive = naive_generate(target, prompt, 8)
    assert torch.equal(result.ids, ordinary) and torch.equal(ordinary, naive)
    print("PASS：普通解码 = 完整重算 = 投机解码")
    print("输出 token IDs:", result.ids[0].tolist())
    print("下一步：docs/start-here.md；做题从 python scripts/grade.py --stage 0 开始。")


if __name__ == "__main__":
    main()
