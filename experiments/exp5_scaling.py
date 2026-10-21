#!/usr/bin/env python3
"""实验 5：记忆容量 scaling（论文 Figure 3 右）。

    python experiments/exp5_scaling.py

在合成任务上扫 slots，画 log-slots vs loss；同时给出论文 10^7 量级的解析外推。
真实 27B 的幂律斜率不能靠这个玩具复现，但"更多 slot 单调更好 + log 线性"的
形状应该出现。
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import math

import torch

from tinyengram import (
    EngramConfig,
    EntityCorpus,
    TinyLM,
    TinyLMConfig,
    TrainConfig,
    Trainer,
    power_law_fit,
    power_law_predict,
)


def run(vocab: int, slots: int, corpus: EntityCorpus, seed: int = 0) -> float:
    torch.manual_seed(seed)
    engram = EngramConfig(
        vocab_size=vocab,
        hidden_size=48,
        hc_mult=2,
        max_ngram_size=3,
        n_head_per_ngram=3,
        n_embed_per_ngram=24,
        slots_per_head=slots,
        layer_ids=(1,),
        pad_id=0,
        seed=seed,
    )
    cfg = TinyLMConfig(
        vocab_size=vocab,
        hidden_size=48,
        n_layers=2,
        n_heads=3,
        max_seq=40,
        ffn_mult=2,
        hc_mult=2,
        engram_layers=(1,),
        engram=engram,
    )
    model = TinyLM(cfg)
    trainer = Trainer(model, TrainConfig(lr=3e-3, warmup_steps=20, total_steps=250))
    trainer.fit(corpus, steps=250, batch_size=24, seed=1, log_every=0)
    metrics = trainer.evaluate(corpus, batch_size=24, batches=8, split="train")
    print(f"  slots/head={slots:6d}  params={model.num_parameters / 1e3:7.1f}K  train_loss={metrics['loss']:.4f}")
    return metrics["loss"]


def main() -> None:
    corpus = EntityCorpus(n_entities=32, n_train_entities=24, seed=0)
    print("\n小规模 slot sweep（train loss；heldout 实体没进过记忆表，不适合看容量）")
    slots = [128, 512, 2048, 8192]
    losses = [run(corpus.vocab_size, s, corpus) for s in slots]
    intercept, slope = power_law_fit(slots, losses)
    print(f"\n拟合: loss = {intercept:.4f} + {slope:.4f} * log10(slots)")
    print(f"外推到 1e7 slots: {power_law_predict(intercept, slope, 1e7):.4f}")

    print("\n论文 Figure 3 的量级（解析，非本实验拟合）")
    for s in (2.58e5, 1e6, 4e6, 1e7):
        print(f"  slots={s:9.0f}  loss~{1.80 - 0.015 * math.log10(s):.4f}")

    _bootstrap.write_markdown(
        "scaling.md",
        "# 记忆容量 scaling\n\n"
        f"- 玩具 sweep: slots={slots}, train loss={['%.4f' % x for x in losses]}\n"
        f"- 拟合斜率 {slope:.4f} / decade（论文报告严格 log-linear）\n",
    )


if __name__ == "__main__":
    main()
