"""L5：Transformer 集成与训练 —— 因果性、KV cache、参数组。"""

from __future__ import annotations

import math

import torch

from tinyengram import (
    RotaryEmbedding,
    TinyLM,
    TrainConfig,
    Trainer,
    apply_rope,
    count_parameters,
    generate,
    make_param_groups,
    naive_generate,
)
from _utils import make_ids, tiny_lm_config


def build_model(seed: int = 0):
    torch.manual_seed(seed)
    return TinyLM(tiny_lm_config(with_engram=True))


def test_forward_shapes_and_loss():
    model = build_model()
    ids = make_ids(2, 12)
    logits, loss, caches = model(ids, targets=ids)
    assert logits.shape == (2, 12, 64)
    assert loss is not None and math.isfinite(float(loss.detach()))
    assert len(caches) == 2
    assert caches[0][0].shape[0] == 2 * 2  # B * hc_mult


def test_lm_is_causal():
    model = build_model()
    model.eval()
    ids = make_ids(1, 10, seed=3)
    changed = ids.clone()
    changed[:, 6:] = (changed[:, 6:] + 5) % 64
    with torch.no_grad():
        a, _, _ = model(ids)
        b, _, _ = model(changed)
    assert torch.allclose(a[:, :6], b[:, :6], atol=1e-5)
    assert not torch.allclose(a[:, 6:], b[:, 6:], atol=1e-5)


def test_kv_cache_matches_full_forward():
    model = build_model()
    model.eval()
    ids = make_ids(2, 12, seed=4)
    with torch.no_grad():
        full, _, _ = model(ids)
        prefix = 3
        part, _, caches = model(ids[:, :prefix])
        assert torch.allclose(part, full[:, :prefix], atol=1e-5)
        # Engram needs at least max_ngram_size-1 previous ids, so a cached
        # step must be given the full context (generate() does this for you).
        context = ids[:, :prefix].clone()
        for t in range(prefix, ids.shape[1]):
            context = torch.cat([context, ids[:, t : t + 1]], dim=1)
            step, _, caches = model(
                ids[:, t : t + 1],
                caches=caches,
                position_offset=t,
                context_ids=context,
            )
            assert torch.allclose(step[:, 0], full[:, t], atol=1e-5), f"t={t}"


def test_kv_cache_chunked_prefill_matches_full_forward():
    model = build_model()
    model.eval()
    ids = make_ids(2, 10, seed=6)
    with torch.no_grad():
        full, _, _ = model(ids)
        prefix = 3
        part, _, caches = model(ids[:, :prefix])
        assert torch.allclose(part, full[:, :prefix], atol=1e-5)

        context = ids[:, :prefix].clone()
        t = prefix
        for chunk in (2, 2, 3):
            context = torch.cat([context, ids[:, t : t + chunk]], dim=1)
            out, _, caches = model(
                ids[:, t : t + chunk],
                caches=caches,
                position_offset=t,
                context_ids=context,
            )
            assert torch.allclose(out, full[:, t : t + chunk], atol=1e-5), f"t={t}"
            t += chunk


def test_cached_generation_matches_naive_greedy():
    model = build_model()
    model.eval()
    ids = make_ids(1, 5, seed=5)
    with torch.no_grad():
        fast = generate(model, ids, 6, temperature=0.0)
        slow = naive_generate(model, ids, 6, temperature=0.0)
    assert torch.equal(fast, slow)


def test_weight_tying_and_parameter_counts():
    model = build_model()
    assert model.lm_head.weight is model.tok_emb.weight
    total = model.num_parameters
    without = model.num_parameters_excluding_engram()
    assert 0 < without < total
    assert count_parameters(model) == total


def test_engram_is_inserted_at_requested_layer_only():
    model = build_model()
    modules = model.engram_modules()
    assert len(modules) == 1
    assert modules[0].layer_id == 1
    assert model.blocks[0].engram is None


def test_rope_dot_product_depends_on_relative_position():
    torch.manual_seed(0)
    rope = RotaryEmbedding(head_dim=8, max_seq=32)
    q = torch.randn(1, 1, 1, 8)
    k = torch.randn(1, 1, 1, 8)
    c0, s0 = rope(1, offset=0)
    c5, s5 = rope(1, offset=5)
    c3, s3 = rope(1, offset=3)
    c8, s8 = rope(1, offset=8)
    dot_a = (apply_rope(q, c0, s0) * apply_rope(k, c5, s5)).sum()
    dot_b = (apply_rope(q, c3, s3) * apply_rope(k, c8, s8)).sum()
    assert torch.allclose(dot_a, dot_b, atol=1e-5)


def test_param_groups_put_engram_embedding_at_5x_without_weight_decay():
    model = build_model()
    groups = make_param_groups(model, lr=1e-3, weight_decay=0.1, engram_lr_mult=5.0)
    names = {g["name"]: g for g in groups}
    assert "engram_embedding" in names
    emb_group = names["engram_embedding"]
    assert abs(emb_group["lr"] - 5e-3) < 1e-12
    assert emb_group["weight_decay"] == 0.0

    seen = [p for g in groups for p in g["params"]]
    assert len(seen) == sum(1 for p in model.parameters() if p.requires_grad)
    assert len({id(p) for p in seen}) == len(seen), "参数不能重复出现在多个组里"


def test_engram_embedding_receives_gradient():
    model = build_model()
    ids = make_ids(2, 8)
    _, loss, _ = model(ids, targets=ids)
    loss.backward()
    emb = model.engram_modules()[0].memory.table.embedding.weight
    assert emb.grad is not None
    assert float(emb.grad.abs().sum()) > 0
    assert int((emb.grad.abs().sum(-1) > 0).sum()) > 0


def test_trainer_step_updates_parameters_and_history():
    model = build_model()
    trainer = Trainer(model, TrainConfig(lr=1e-2, warmup_steps=1, total_steps=10))
    before = [p.detach().clone() for p in model.parameters()]
    ids = make_ids(2, 8)
    loss = trainer.step(ids, ids)
    assert math.isfinite(loss)
    assert len(trainer.history["loss"]) == 1
    changed = any(not torch.equal(a, b) for a, b in zip(before, model.parameters()))
    assert changed


def test_trainer_evaluate_returns_finite_metrics():
    model = build_model()
    trainer = Trainer(model)
    from tinyengram import EntityCorpus

    corpus = EntityCorpus(n_entities=8, n_train_entities=8)
    metrics = trainer.evaluate(corpus, batch_size=4, batches=2)
    assert 0 <= metrics["token_acc"] <= 1
    assert math.isfinite(metrics["loss"])
