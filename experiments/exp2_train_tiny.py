#!/usr/bin/env python3
"""实验 2：在小规模合成语料上训练 MoE-less baseline vs Engram（论文 §4 的玩具版）。

    python experiments/exp2_train_tiny.py

任务：多 token 实体名 -> 固定属性 token。3-gram 恰好能覆盖完整实体名，
所以 Engram 可以把"记住实体"这件事从计算里拿出来。
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import torch

from tinyengram import (
    EngramConfig,
    EntityCorpus,
    TinyLM,
    TinyLMConfig,
    TrainConfig,
    Trainer,
)


def build(vocab: int, with_engram: bool, seed: int) -> TinyLM:
    torch.manual_seed(seed)
    engram = None
    if with_engram:
        engram = EngramConfig(
            vocab_size=vocab,
            hidden_size=48,
            hc_mult=2,
            max_ngram_size=3,
            n_head_per_ngram=3,
            n_embed_per_ngram=24,
            slots_per_head=512,
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
        engram_layers=(1,) if with_engram else (),
        engram=engram,
    )
    return TinyLM(cfg)


def run(name: str, with_engram: bool, corpus: EntityCorpus) -> dict[str, float]:
    model = build(corpus.vocab_size, with_engram, seed=0)
    trainer = Trainer(
        model,
        TrainConfig(lr=5e-3, warmup_steps=20, total_steps=240, grad_clip=1.0),
    )
    trainer.fit(corpus, steps=240, batch_size=24, seed=1, log_every=80)
    train = trainer.evaluate(corpus, batch_size=24, batches=6, split="train")
    held = trainer.evaluate(corpus, batch_size=24, batches=6, split="heldout")
    print(
        f"{name:<12} params={model.num_parameters / 1e3:8.1f}K  "
        f"train_loss={train['loss']:.4f}  train_acc={train['token_acc']:.3f}  "
        f"heldout_loss={held['loss']:.4f}  heldout_acc={held['token_acc']:.3f}"
    )
    return {
        "params": float(model.num_parameters),
        "train_loss": train["loss"],
        "train_acc": train["token_acc"],
        "heldout_loss": held["loss"],
        "heldout_acc": held["token_acc"],
    }


def main() -> None:
    corpus = EntityCorpus(n_entities=48, n_train_entities=36, seed=0)
    print(f"vocab={corpus.vocab_size}  train_entities=36  heldout=12  templates=4\n")
    baseline = run("baseline", False, corpus)
    engram = run("engram", True, corpus)

    delta = baseline["heldout_loss"] - engram["heldout_loss"]
    print(f"\nheldout loss 改善: {delta:+.4f}（正数表示 Engram 更好）")
    print("注意：这是机制 demo，不是 27B 结论；规模太小，随机性很大。")

    _bootstrap.write_markdown(
        "train_tiny.md",
        "# 小规模训练\n\n"
        f"| 模型 | 参数量 | train loss | train acc | heldout loss | heldout acc |\n"
        f"|---|---|---|---|---|---|\n"
        f"| baseline | {baseline['params'] / 1e3:.1f}K | {baseline['train_loss']:.4f} | "
        f"{baseline['train_acc']:.3f} | {baseline['heldout_loss']:.4f} | {baseline['heldout_acc']:.3f} |\n"
        f"| Engram | {engram['params'] / 1e3:.1f}K | {engram['train_loss']:.4f} | "
        f"{engram['train_acc']:.3f} | {engram['heldout_loss']:.4f} | {engram['heldout_acc']:.3f} |\n\n"
        f"heldout loss 改善 **{delta:+.4f}**。\n",
    )


if __name__ == "__main__":
    main()
