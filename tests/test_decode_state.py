"""Nonzero temporal Engram: state equivalence, rollback, and bounded lookup work."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from experiments.speculative import greedy_speculative_generate
from tinyengram import EngramConfig, TinyLM, TinyLMConfig, generate, naive_generate


def model(seed=0):
    torch.manual_seed(seed)
    engram = EngramConfig(
        vocab_size=32, hidden_size=16, hc_mult=2, layer_ids=(0, 1),
        n_head_per_ngram=2, n_embed_per_ngram=8, slots_per_head=17,
        conv_kernel_size=4, conv_zero_init=False,
    )
    return TinyLM(TinyLMConfig(
        vocab_size=32, hidden_size=16, hc_mult=2, n_heads=2, n_layers=2,
        max_seq=100, ffn_mult=2, tie_weights=False,
        engram_layers=(0, 1), engram=engram,
    )).eval()


def check_cache(actual, expected):
    for (ak, av), (bk, bv) in zip(actual, expected):
        torch.testing.assert_close(ak, bk, atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(av, bv, atol=2e-6, rtol=2e-5)
    for a, b in zip(actual.conv_states, expected.conv_states):
        if a is not None:
            torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)
            assert a.shape[1] <= 9  # (kernel_size-1) * dilation


@pytest.mark.parametrize("chunks", [[1] * 13, [2, 4, 7], [7, 1, 5]])
def test_nonzero_temporal_conv_matches_full_forward(chunks):
    m = model()
    ids = torch.randint(0, 32, (2, 13), generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        full, _, expected_cache = m(ids)
        cache, pos = None, 0
        outputs = []
        for size in chunks:
            logits, _, cache = m(
                ids[:, pos:pos+size], caches=cache, position_offset=pos,
                context_ids=ids[:, :pos+size],
            )
            pos += size
            outputs.append(logits)
        torch.testing.assert_close(torch.cat(outputs, 1), full, atol=2e-6, rtol=2e-5)
        check_cache(cache, expected_cache)


@pytest.mark.parametrize("keep", [0, 1, 3, 6])
def test_crop_restores_temporal_state_and_subsequent_outputs(keep):
    m = model()
    ids = torch.randint(0, 32, (1, 20), generator=torch.Generator().manual_seed(2))
    with torch.no_grad():
        _, _, base = m(ids[:, :10])
        _, _, speculative = m(ids[:, 10:16], caches=base, position_offset=10,
                              context_ids=ids[:, :16])
        cropped = speculative.crop(10 + keep)
        _, _, expected = m(ids[:, :10+keep])
        check_cache(cropped, expected)
        # A new, different continuation must not inherit rejected activations.
        continuation = torch.tensor([[21, 22]])
        context = torch.cat([ids[:, :10+keep], continuation], 1)
        cached, _, _ = m(continuation, caches=cropped, position_offset=10+keep,
                          context_ids=context)
        full, _, _ = m(context)
        torch.testing.assert_close(cached, full[:, -2:], atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("k", [1, 4, 12])
def test_temporal_speculation_rolls_back_both_models_even_beyond_ring_size(k):
    target, draft = model(1), model(2)
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    def audit(committed, tc, dc, stats):
        with torch.no_grad():
            for m, cache in ((target, tc), (draft, dc)):
                _, _, full = m(committed)
                check_cache(cache, full)
    out = greedy_speculative_generate(target, draft, ids, 18, draft_length=k, on_commit=audit)
    assert torch.equal(out.ids, generate(target, ids, 18))
    assert torch.equal(out.ids, naive_generate(target, ids, 18))
    assert any(r.rejected for r in out.rounds)


def test_temporal_identical_models_all_accept():
    target = model()
    ids = torch.tensor([[1, 2, 3]])
    out = greedy_speculative_generate(target, deepcopy(target), ids, 17, draft_length=6)
    assert all(r.accepted == r.proposed for r in out.rounds)
    assert torch.equal(out.ids, naive_generate(target, ids, 17))


@pytest.mark.parametrize("prefix,chunk", [(3, 1), (32, 1), (64, 4)])
def test_hash_history_and_embedding_rows_are_bounded(prefix, chunk):
    m = model()
    ids = torch.randint(0, 32, (1, prefix+chunk))
    engram = m.engram_modules()[0]
    hash_lengths, lookup_lengths = [], []
    original = engram.memory.hash_ids
    def record_hash(context):
        hash_lengths.append(context.shape[1])
        return original(context)
    with torch.no_grad():
        _, _, cache = m(ids[:, :prefix])
        engram.memory.hash_ids = record_hash
        hook = engram.memory.table.embedding.register_forward_pre_hook(
            lambda module, args: lookup_lengths.append(args[0].shape[1]),
        )
        m(ids[:, prefix:], caches=cache, position_offset=prefix, context_ids=ids)
        hook.remove()
    assert hash_lengths == [chunk + 2], "3-gram only needs two preceding token IDs"
    assert lookup_lengths == [chunk], "do not gather embeddings for old tokens"


def test_cached_context_and_offset_errors_are_clear():
    m = model()
    ids = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        _, _, cache = m(ids[:, :3])
        with pytest.raises(ValueError, match="position_offset"):
            m(ids[:, 3:], caches=cache, position_offset=2, context_ids=ids)
        with pytest.raises(ValueError, match="context_ids"):
            m(ids[:, 3:], caches=cache, position_offset=3)
        with pytest.raises(ValueError, match="context_ids"):
            m(ids[:, 3:], caches=cache, position_offset=3, context_ids=ids[:, -1:])
        with pytest.raises(ValueError, match="conv|卷积"):
            m(ids[:, 3:], caches=list(cache), position_offset=3, context_ids=ids)
        with pytest.raises(ValueError):
            cache.crop(-1)
        with pytest.raises(ValueError):
            cache.crop(4)


def test_crop_before_last_chunk_requires_a_checkpoint():
    m = model()
    with torch.no_grad():
        _, _, cache = m(torch.tensor([[1, 2, 3]]))
        _, _, cache = m(torch.tensor([[4]]), caches=cache, position_offset=3,
                        context_ids=torch.tensor([[1, 2, 3, 4]]))
    with pytest.raises(ValueError, match="checkpoint"):
        cache.crop(2)


def test_local_lookup_matches_full_history_reference_and_gradients():
    m = model()
    engram = m.engram_modules()[0]
    ids = torch.randint(0, 32, (2, 17))
    actual = engram.retrieve(ids[:, -4:], ids)
    reference = engram.memory.retrieve(engram.compress(ids))[:, -4:]
    torch.testing.assert_close(actual, reference)
    weight = engram.memory.table.embedding.weight
    ga, = torch.autograd.grad(actual.square().sum(), weight)
    gb, = torch.autograd.grad(reference.square().sum(), weight)
    torch.testing.assert_close(ga, gb)


def test_temporal_forward_still_backpropagates_to_conv():
    m = model().train()
    ids = torch.randint(0, 32, (2, 12))
    _, loss, _ = m(ids, targets=ids)
    loss.backward()
    for module in m.engram_modules():
        assert module.short_conv.conv.weight.grad.abs().sum() > 0


def test_plain_model_legacy_list_cache_is_compatible():
    m = TinyLM(replace(model().cfg, engram=None, engram_layers=())).eval()
    with torch.no_grad():
        _, _, cache = m(torch.tensor([[1, 2]]))
        cached, _, _ = m(torch.tensor([[3]]), caches=list(cache), position_offset=2)
        full, _, _ = m(torch.tensor([[1, 2, 3]]))
    torch.testing.assert_close(cached, full[:, -1:])


@pytest.mark.parametrize("fault", [4, 6, 15, None])
def test_forced_partial_acceptance_with_temporal_state(fault):
    from test_speculative import ScriptedTinyLM

    base = model()
    target, draft = ScriptedTinyLM(base.cfg).eval(), ScriptedTinyLM(base.cfg).eval()
    target.load_state_dict(base.state_dict())
    draft.load_state_dict(base.state_dict())
    draft.fault_position = fault
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    def audit(committed, tc, dc, stats):
        with torch.no_grad():
            _, _, a = target(committed)
            _, _, b = draft(committed)
            check_cache(tc, a)
            check_cache(dc, b)
    result = greedy_speculative_generate(target, draft, ids, 18, draft_length=12,
                                          on_commit=audit)
    assert result.rounds[0].accepted == (12 if fault is None else fault-4)
    assert torch.equal(result.ids, naive_generate(target, ids, 18))


def test_interleaved_requests_do_not_share_conv_state():
    m = model()
    with torch.no_grad():
        a, b = torch.tensor([[1, 2, 3]]), torch.tensor([[6, 7]])
        _, _, ca = m(a)
        _, _, cb = m(b)
        for prompt, cache in [(a, ca), (b, cb), (a, ca)]:
            context = torch.cat([prompt, torch.tensor([[9]])], 1)
            step, _, updated = m(context[:, -1:], caches=cache,
                                  position_offset=prompt.shape[1], context_ids=context)
            full, _, expected = m(context)
            torch.testing.assert_close(step, full[:, -1:])
            check_cache(updated, expected)
            assert cache.seq_len == prompt.shape[1], "forward must not mutate its input cache"
