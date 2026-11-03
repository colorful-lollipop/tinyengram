"""Single-request greedy speculative decoding using TinyLM's existing KV API.

Kept outside the graded curriculum: this helper runs with either completed
student models or the reference implementation. No weights or training required.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch

from tinyengram.transformer import KVCache, TinyLM

__all__ = ["RoundStats", "SpeculativeResult", "greedy_speculative_generate"]


@dataclass(frozen=True)
class RoundStats:
    prefix_length: int
    proposed: int
    accepted: int  # committed draft tokens; excludes correction/bonus
    emitted: int  # accepted + correction/bonus, clipped at EOS/budget
    rejected: bool


@dataclass
class SpeculativeResult:
    ids: torch.Tensor
    target_cache: KVCache | None = None
    draft_cache: KVCache | None = None
    rounds: list[RoundStats] = field(default_factory=list)
    target_calls: int = 0
    draft_calls: int = 0

    def summary(self) -> dict[str, float | int]:
        n = len(self.rounds)
        proposed = sum(r.proposed for r in self.rounds)
        accepted = sum(r.accepted for r in self.rounds)
        emitted = sum(r.emitted for r in self.rounds)
        return {
            "rounds": n,
            "proposed_tokens": proposed,
            "accepted_draft_tokens": accepted,
            "acceptance_rate": accepted / proposed if proposed else 0.0,
            "mean_accepted_draft_tokens": accepted / n if n else 0.0,
            "mean_emitted_tokens": emitted / n if n else 0.0,
            "rejection_rounds": sum(r.rejected for r in self.rounds),
            "discarded_draft_tokens": proposed - accepted,
            "target_calls": self.target_calls,
            "draft_calls": self.draft_calls,
        }


def _crop(caches: KVCache, length: int) -> KVCache:
    # TinyLM stores [B * hc_mult, heads, time, head_dim]. Never mutate views.
    return [(k[..., :length, :], v[..., :length, :]) for k, v in caches]


def _check_model(model: TinyLM) -> None:
    for engram in model.engram_modules():
        # context_ids restores n-gram history, NOT hidden-dependent conv state.
        # Earlier taps must be zero; the last tap sees only the current token.
        weight = engram.short_conv.conv.weight
        if bool(torch.count_nonzero(weight[..., :-1])):
            raise ValueError(
                "Temporal ShortConv needs activation-state caching/rollback; "
                "use conv_kernel_size=1 (or zero past taps) for this experiment"
            )


@torch.no_grad()
def greedy_speculative_generate(
    target: TinyLM,
    draft: TinyLM,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    *,
    draft_length: int = 4,
    eos_id: int | None = None,
    on_commit: Callable[[torch.Tensor, KVCache, KVCache, RoundStats], None] | None = None,
) -> SpeculativeResult:
    """Return target-greedy ids [1, prompt + generated] and auditable statistics.

    One target forward validates the entire proposal chunk with a causal mask.
    After each round, BOTH caches contain exactly ``ids``; all context_ids are
    rebuilt from that committed prefix. A correction/bonus is processed by both
    models in one extra forward each. This conservative flush simplifies the
    invariant; it is slower than a production decoder carrying a pending token.

    Only batch=1, shared token IDs/vocabulary, same device, and greedy argmax
    (lowest token ID on ties) are supported. Near-tied floating-point logits can
    differ between chunked and single-token kernels; this is not a bitwise
    numerical guarantee for arbitrary hardware/dtypes. No sampling claim.
    on_commit is a read-only audit hook, excluded from benchmark timing.
    """
    if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] == 0:
        raise ValueError("input_ids must be a nonempty [1, T] tensor")
    if input_ids.dtype != torch.long:
        raise ValueError("input_ids must have dtype torch.long")
    if max_new_tokens < 0 or draft_length < 1:
        raise ValueError("max_new_tokens >= 0 and draft_length >= 1 required")
    if target.cfg.vocab_size != draft.cfg.vocab_size:
        raise ValueError("target and draft must share vocabulary/token semantics")
    device = next(target.parameters()).device
    if next(draft.parameters()).device != device:
        raise ValueError("target and draft must be on the same device")
    if eos_id is not None and not 0 <= eos_id < target.cfg.vocab_size:
        raise ValueError("eos_id must be in the vocabulary")
    if bool(((input_ids < 0) | (input_ids >= target.cfg.vocab_size)).any()):
        raise ValueError("input_ids outside vocabulary")
    end = input_ids.shape[1] + max_new_tokens
    if end > min(target.cfg.max_seq, draft.cfg.max_seq):
        raise ValueError("prompt + max_new_tokens exceeds model max_seq")
    _check_model(target)
    _check_model(draft)
    target.eval()
    draft.eval()
    out = input_ids.to(device).clone()
    result = SpeculativeResult(out)
    if max_new_tokens == 0:
        return result

    t_logits, _, t_cache = target(out)
    d_logits, _, d_cache = draft(out)
    result.target_calls = result.draft_calls = 1
    while out.shape[1] < end:
        length = out.shape[1]
        k = min(draft_length, end - length)
        proposal_context = out
        proposals = []
        for _ in range(k):
            token = d_logits[:, -1].argmax(-1, keepdim=True)
            proposals.append(token)
            proposal_context = torch.cat([proposal_context, token], dim=1)
            d_logits, _, d_cache = draft(
                token, caches=d_cache, position_offset=proposal_context.shape[1] - 1,
                context_ids=proposal_context,
            )
            result.draft_calls += 1
        proposed = torch.cat(proposals, dim=1)
        verified, _, t_cache = target(
            proposed, caches=t_cache, position_offset=length, context_ids=proposal_context,
        )
        result.target_calls += 1
        # Prefix logits predict proposal 0; output at proposal j predicts j+1.
        predictions = torch.cat([t_logits[:, -1:], verified], dim=1).argmax(-1)
        matches = (proposed == predictions[:, :k]).squeeze(0).tolist()
        accepted = next((i for i, match in enumerate(matches) if not match), k)
        rejected = accepted < k
        kept = proposed[:, :accepted]
        if eos_id is not None:
            eos_positions = (kept[0] == eos_id).nonzero()
            if eos_positions.numel():
                accepted = int(eos_positions[0, 0]) + 1
                kept = kept[:, :accepted]
                rejected = False  # stopped before a possible later mismatch

        stopped = eos_id is not None and kept.numel() > 0 and int(kept[0, -1]) == eos_id
        out = torch.cat([out, kept], dim=1)
        # Roll back both speculative KV suffixes, including unused EOS suffixes.
        t_cache = _crop(t_cache, length + accepted)
        d_cache = _crop(d_cache, length + accepted)
        if not stopped and out.shape[1] < end:
            # First mismatch -> target correction; all accepted -> target bonus.
            token = predictions[:, accepted : accepted + 1]
            out = torch.cat([out, token], dim=1)
            # Crucially, no rejected IDs remain in this context.
            t_logits, _, t_cache = target(
                token, caches=t_cache, position_offset=out.shape[1] - 1, context_ids=out,
            )
            d_logits, _, d_cache = draft(
                token, caches=d_cache, position_offset=out.shape[1] - 1, context_ids=out,
            )
            result.target_calls += 1
            result.draft_calls += 1
            stopped = eos_id is not None and int(token[0, 0]) == eos_id

        stats = RoundStats(length, k, accepted, out.shape[1] - length, rejected)
        result.rounds.append(stats)
        if on_commit is not None:
            on_commit(out, t_cache, d_cache, stats)
        if stopped:
            break

    result.ids, result.target_cache, result.draft_cache = out, t_cache, d_cache
    return result
