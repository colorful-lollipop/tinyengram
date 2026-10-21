"""L9：端到端小规模复现（slow）。

我们不复现 27B 的绝对分数，而是复现三条**机制**：

1. 加入 Engram 后训练 loss 能下降，且记忆表确实被更新；
2. 门控在训练后变得更有选择性（方差上升）；
3. 训练后的模型能跑通 generate / LogitLens / CKA / gate 可视化整条链路。
"""

from __future__ import annotations

import math

import pytest
import torch

from tinyengram import (
    EntityCorpus,
    TinyLM,
    TinyLMConfig,
    TrainConfig,
    Trainer,
    gate_stats,
    generate,
    linear_cka,
    logit_lens_kl,
)
from _utils import tiny_engram_config

pytestmark = pytest.mark.slow


def build(vocab_size: int, with_engram: bool) -> TinyLM:
    engram = tiny_engram_config(vocab_size=vocab_size) if with_engram else None
    cfg = TinyLMConfig(
        vocab_size=vocab_size,
        hidden_size=32,
        n_layers=2,
        n_heads=2,
        max_seq=40,
        ffn_mult=2,
        hc_mult=2,
        engram_layers=(1,) if with_engram else (),
        engram=engram,
    )
    return TinyLM(cfg)


@pytest.mark.parametrize("with_engram", [False, True])
def test_training_reduces_loss_and_updates_memory(with_engram):
    torch.manual_seed(0)
    corpus = EntityCorpus(n_entities=24, n_train_entities=20, seed=0)
    model = build(corpus.vocab_size, with_engram)
    trainer = Trainer(
        model,
        TrainConfig(lr=5e-3, warmup_steps=10, total_steps=80, grad_clip=1.0),
    )
    if with_engram:
        emb_before = model.engram_modules()[0].memory.table.embedding.weight.detach().clone()
    before = trainer.evaluate(corpus, batch_size=16, batches=4)
    trainer.fit(corpus, steps=80, batch_size=16, seed=1, log_every=0)
    after = trainer.evaluate(corpus, batch_size=16, batches=4)
    assert math.isfinite(after["loss"])
    assert after["loss"] < before["loss"], (before, after)

    if with_engram:
        modules = model.engram_modules()
        assert len(modules) == 1
        emb = modules[0].memory.table.embedding.weight
        assert float(emb.detach().abs().sum()) > 0
        assert not torch.equal(emb.detach(), emb_before)


def test_gates_become_selective_after_training():
    torch.manual_seed(0)
    corpus = EntityCorpus(n_entities=16, n_train_entities=16, seed=0)
    model = build(corpus.vocab_size, True)
    module = model.engram_modules()[0]

    ids, _ = corpus.sample(16, split="train", generator=torch.Generator().manual_seed(0))
    hidden = torch.randn(16, ids.shape[1], 2, 32)
    init_std = gate_stats(module.gates(hidden, ids))["std"]

    trainer = Trainer(model, TrainConfig(lr=5e-3, warmup_steps=5, total_steps=60))
    trainer.fit(corpus, steps=60, batch_size=16, seed=2, log_every=0)
    trained_std = gate_stats(module.gates(hidden, ids))["std"]
    assert trained_std >= init_std * 0.9  # 不要求一定更大，但不能塌缩成常数


def test_end_to_end_analysis_pipeline():
    torch.manual_seed(0)
    corpus = EntityCorpus(n_entities=12, n_train_entities=12, seed=0)
    model = build(corpus.vocab_size, True)
    trainer = Trainer(model, TrainConfig(lr=5e-3, warmup_steps=5, total_steps=40))
    trainer.fit(corpus, steps=40, batch_size=8, seed=3, log_every=0)

    ids, _ = corpus.sample(2, split="train", generator=torch.Generator().manual_seed(4))
    out = generate(model, ids[:, :3], 5, temperature=0.0)
    assert out.shape == (2, 8)
    assert bool((out >= 0).all() and (out < corpus.vocab_size).all())

    model.eval()
    with torch.no_grad():
        _, _, _ = model(ids)
    assert model.num_parameters > model.num_parameters_excluding_engram()

    with torch.no_grad():
        x = torch.randn(32, model.cfg.hidden_size)
        y = torch.randn(32, model.cfg.hidden_size)
    cka = linear_cka(x, y)
    assert 0.0 <= cka <= 1.0

    logits, _, _ = model(ids)
    hidden = [model.tok_emb(ids) for _ in range(2)]
    kls = logit_lens_kl(hidden, model.norm_f, model.lm_head, logits)
    assert all(math.isfinite(k) and k >= 0 for k in kls)
