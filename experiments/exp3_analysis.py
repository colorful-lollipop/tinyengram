#!/usr/bin/env python3
"""实验 3：LogitLens / CKA / 门控可视化（论文 §6.1、§6.5）。

    python experiments/exp3_analysis.py

训练一对小模型（baseline / Engram），比较：
* 每层 hidden state 经 LM head 后的 KL（LogitLens）；
* 两个模型层间表示的 CKA 与软对齐；
* Engram 门控在哪些 token 上激活。
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
    gate_stats,
    linear_cka,
    logit_lens_kl,
    soft_alignment,
)


def build(vocab: int, with_engram: bool, seed: int = 0) -> TinyLM:
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


def layer_hiddens(model: TinyLM, ids: torch.Tensor) -> list[torch.Tensor]:
    model.eval()
    with torch.no_grad():
        h = model.tok_emb(ids).unsqueeze(2).expand(-1, -1, model.cfg.hc_mult, -1)
        out = [h.mean(dim=2)]
        for block in model.blocks:
            h, _ = block(h, ids)
            out.append(h.mean(dim=2))
        return out


def main() -> None:
    corpus = EntityCorpus(n_entities=24, n_train_entities=20, seed=0)
    models = {}
    for name, flag in (("baseline", False), ("engram", True)):
        model = build(corpus.vocab_size, flag)
        trainer = Trainer(model, TrainConfig(lr=5e-3, warmup_steps=10, total_steps=120))
        trainer.fit(corpus, steps=120, batch_size=16, seed=2, log_every=0)
        models[name] = model

    ids, _ = corpus.sample(8, split="train", generator=torch.Generator().manual_seed(0))
    hs = {k: layer_hiddens(v, ids) for k, v in models.items()}

    print("\nLogitLens KL(final || layer)  —— 越小说明越早“准备好”预测")
    for name, model in models.items():
        logits, _, _ = model(ids)
        kls = logit_lens_kl(hs[name], model.norm_f, model.lm_head, logits)
        print(f"  {name:<9} " + "  ".join(f"L{i}={v:.3f}" for i, v in enumerate(kls)))

    print("\nCKA（Engram vs baseline，逐层）")
    sim = torch.zeros(len(hs["engram"]), len(hs["baseline"]))
    for i, a in enumerate(hs["engram"]):
        for j, b in enumerate(hs["baseline"]):
            sim[i, j] = linear_cka(a.reshape(-1, a.shape[-1]), b.reshape(-1, b.shape[-1]))
    for i in range(sim.shape[0]):
        print(f"  Engram L{i}: " + "  ".join(f"{v:.3f}" for v in sim[i].tolist()))
    align = soft_alignment(sim, k=min(3, sim.shape[0]))
    print("  软对齐 a_j: " + "  ".join(f"{v:.2f}" for v in align.tolist()))

    module = models["engram"].engram_modules()[0]
    # 用 layer 1 的真实输入（经过 block 0）算门控
    h = models["engram"].blocks[0](
        models["engram"].tok_emb(ids).unsqueeze(2).expand(-1, -1, 2, -1), ids
    )[0]
    gates = module.gates(h, ids)
    mask = ids != 0  # 忽略 padding
    print("\n门控统计:", {k: round(v, 3) for k, v in gate_stats(gates[mask]).items()})

    print("\n门控最高的 token（样本 0，layer 1，忽略 pad）:")
    g = gates[0, :, 0].masked_fill(ids[0] == 0, -1.0)
    values, idx = g.topk(min(8, int(mask[0].sum())))
    for v, i in zip(values.tolist(), idx.tolist()):
        print(f"  {v:.3f}  {corpus.tokenizer.decode(ids[0, i].item())!r}")

    _bootstrap.write_markdown(
        "analysis.md",
        "# 机制分析\n\n"
        "LogitLens（KL，越小越接近最终预测）：\n\n"
        + "\n".join(
            f"- {name}: " + ", ".join(f"L{i}={v:.3f}" for i, v in enumerate(logit_lens_kl(hs[name], models[name].norm_f, models[name].lm_head, models[name](ids)[0])))
            for name in models
        )
        + "\n\n软对齐 a_j: " + ", ".join(f"{v:.2f}" for v in align.tolist()) + "\n",
    )


if __name__ == "__main__":
    main()
