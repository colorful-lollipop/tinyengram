"""Experimental greedy decoder: output equivalence and transactional state."""

from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from experiments.speculative import greedy_speculative_generate
from tinyengram import EngramConfig, TinyLM, TinyLMConfig, generate, naive_generate


def build(seed=0, engram=False, max_seq=80):
    torch.manual_seed(seed)
    ecfg = EngramConfig(
        vocab_size=16, hidden_size=16, hc_mult=2, layer_ids=(0,),
        n_head_per_ngram=2, n_embed_per_ngram=8, slots_per_head=17,
        conv_kernel_size=1, conv_zero_init=False,
    ) if engram else None
    return TinyLM(TinyLMConfig(
        vocab_size=16, hidden_size=16, n_heads=2, n_layers=2, hc_mult=2,
        ffn_mult=2, max_seq=max_seq, tie_weights=False,
        engram=ecfg, engram_layers=(0,) if engram else (),
    )).eval()


def audit(target, draft, seen):
    def check(ids, tc, dc, stats):
        # Independent full-context recomputation catches poisoned accepted KV,
        # stale draft state, wrong RoPE offsets, and retained rejected IDs.
        with torch.no_grad():
            for model, cache in ((target, tc), (draft, dc)):
                _, _, expected = model(ids)
                for actual_layer, expected_layer in zip(cache, expected):
                    for actual, ref in zip(actual_layer, expected_layer):
                        assert actual.shape[-2] == ids.shape[1]
                        torch.testing.assert_close(actual, ref, atol=2e-6, rtol=2e-5)
        assert stats.emitted >= 1
        seen.append(stats)
    return check


@pytest.mark.parametrize("engram", [False, True])
@pytest.mark.parametrize("k", [1, 2, 4, 8])
@pytest.mark.parametrize("prompt_length", [1, 7, 37])
def test_real_models_match_both_references_at_every_commit(engram, k, prompt_length):
    target, draft = build(1, engram), build(2, engram)
    ids = torch.randint(0, 16, (1, prompt_length), generator=torch.Generator().manual_seed(3))
    seen = []
    result = greedy_speculative_generate(
        target, draft, ids, 11, draft_length=k, on_commit=audit(target, draft, seen),
    )
    assert torch.equal(result.ids, generate(target, ids, 11))
    assert torch.equal(result.ids, naive_generate(target, ids, 11))
    assert seen and sum(r.emitted for r in seen) == 11
    assert result.target_calls == 1 + len(seen) + sum(r.emitted > r.accepted for r in seen)


class ScriptedTinyLM(TinyLM):
    """Real attention/KV/Engram, but position-controlled logits force branches."""
    fault_position: int | None = None

    def forward(self, input_ids, *args, **kwargs):
        logits, loss, caches = super().forward(input_ids, *args, **kwargs)
        offset = kwargs.get("position_offset", 0)
        positions = torch.arange(offset, offset + input_ids.shape[1], device=logits.device)
        tokens = (positions + 1) % self.cfg.vocab_size
        if self.fault_position is not None:
            tokens = (tokens + (positions == self.fault_position)) % self.cfg.vocab_size
        logits = torch.full_like(logits, -10)
        logits.scatter_(2, tokens[None, :, None].expand(input_ids.shape[0], -1, 1), 10)
        return logits, loss, caches


def scripted(seed=0):
    base = build(seed, engram=True)
    model = ScriptedTinyLM(base.cfg)
    model.load_state_dict(base.state_dict())
    return model.eval()


@pytest.mark.parametrize("rejection_index", [0, 1, 3, None])
def test_forced_first_middle_last_rejection_and_full_accept(rejection_index):
    target, draft = scripted(), scripted(1)
    ids = torch.tensor([[1, 2, 3]])
    if rejection_index is not None:
        draft.fault_position = ids.shape[1] - 1 + rejection_index
    seen = []
    calls = []
    hook = target.register_forward_pre_hook(
        lambda module, args: calls.append(args[0].shape[1]),
    )
    result = greedy_speculative_generate(target, draft, ids, 12, draft_length=4)
    hook.remove()
    assert calls[1] == 4, "all proposals must be verified in one target forward"
    first = result.rounds[0]
    assert first.accepted == (4 if rejection_index is None else rejection_index)
    assert first.emitted == first.accepted + 1
    assert first.rejected == (rejection_index is not None)
    result = greedy_speculative_generate(
        target, draft, ids, 12, draft_length=4, on_commit=audit(target, draft, seen),
    )
    assert torch.equal(result.ids, naive_generate(target, ids, 12))
    assert result.summary()["rejection_rounds"] == (0 if rejection_index is None else 1)


@pytest.mark.parametrize("k", [1, 4, 8])
def test_identical_models_accept_every_proposal_and_budget_clips(k):
    target = build(8)
    ids = torch.tensor([[4, 2]])
    result = greedy_speculative_generate(target, deepcopy(target), ids, 9, draft_length=k)
    assert all(r.accepted == r.proposed for r in result.rounds)
    assert result.ids.shape[1] == 11
    assert torch.equal(result.ids, generate(target, ids, 9))


@pytest.mark.parametrize("eos", [3, 4, 6, 7])
@pytest.mark.parametrize("fault", [None, 2, 3, 5])
def test_eos_in_accepted_proposal_correction_bonus_or_prompt(eos, fault):
    target, draft = scripted(), scripted(4)
    draft.fault_position = fault
    ids = torch.tensor([[1, 2, 3]])  # prompt EOS does not itself stop generation
    result = greedy_speculative_generate(
        target, draft, ids, 10, draft_length=4, eos_id=eos,
        on_commit=audit(target, draft, []),
    )
    assert torch.equal(result.ids, generate(target, ids, 10, eos_id=eos))
    assert torch.equal(result.ids, naive_generate(target, ids, 10, eos_id=eos))


def test_zero_budget_no_forwards_and_no_input_mutation():
    model = build()
    ids = torch.tensor([[1]])
    before = ids.clone()
    result = greedy_speculative_generate(model, model, ids, 0)
    assert torch.equal(result.ids, ids) and torch.equal(ids, before)
    assert result.target_calls == result.draft_calls == 0
    assert result.summary()["mean_accepted_draft_tokens"] == 0


@pytest.mark.parametrize("ids,n,k", [
    (torch.empty(1, 0, dtype=torch.long), 1, 4),
    (torch.ones(2, 2, dtype=torch.long), 1, 4),
    (torch.ones(1, 2), 1, 4),
    (torch.tensor([[16]]), 1, 4),
    (torch.tensor([[1]]), -1, 4),
    (torch.tensor([[1]]), 1, 0),
    (torch.tensor([[1]]), 80, 4),
])
def test_invalid_requests_fail_early(ids, n, k):
    model = build()
    with pytest.raises(ValueError):
        greedy_speculative_generate(model, model, ids, n, draft_length=k)


@pytest.mark.parametrize("bad_role", ["target", "draft"])
def test_temporal_shortconv_is_explicitly_unsupported(bad_role):
    from tinyengram.conv import ShortConv

    target, draft = build(0, True), build(1, True)
    bad = target if bad_role == "target" else draft
    bad.engram_modules()[0].short_conv = ShortConv(
        16, groups=2, kernel_size=4, dilation=3, zero_init=False,
    )
    with pytest.raises(ValueError, match="Temporal ShortConv"):
        greedy_speculative_generate(target, draft, torch.tensor([[1, 2]]), 5)


def test_zero_temporal_taps_remain_supported():
    from tinyengram.conv import ShortConv

    model = build(0, True)
    model.engram_modules()[0].short_conv = ShortConv(16, groups=2, zero_init=True)
    ids = torch.tensor([[2, 3, 4]])
    result = greedy_speculative_generate(model, deepcopy(model), ids, 5)
    assert torch.equal(result.ids, naive_generate(model, ids, 5))


@pytest.mark.parametrize("eos", [-1, 16])
def test_invalid_eos(eos):
    model = build()
    with pytest.raises(ValueError, match="eos_id"):
        greedy_speculative_generate(model, model, torch.tensor([[1]]), 5, eos_id=eos)


def test_different_vocabulary_sizes_are_rejected():
    target = build()
    draft = TinyLM(TinyLMConfig(vocab_size=8, hidden_size=16, n_heads=2, n_layers=1))
    with pytest.raises(ValueError, match="vocabulary"):
        greedy_speculative_generate(target, draft, torch.tensor([[1]]), 5)
