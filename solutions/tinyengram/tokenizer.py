"""Tokenizer compression: collapse textually equivalent subword tokens.

Paper §2.2 / Appendix C.  A 128k BPE vocabulary contains many tokens that are
semantically identical after Unicode normalization: ``Apple``, ``apple``,
`` apple``, ``APPLE`` ...  Engram hashes *token ids*, so every spelling variant
would create a different n-gram and waste table capacity.  The fix is a
surjective projection ``P : V -> V'`` precomputed once:

    t' = P(t),   g_{t,n} = (t'_{t-n+1}, ..., t'_t)

The paper reports a 23.43% reduction for a 128k tokenizer.  The projection is
surjective (every new id is hit) but not injective (many old ids collapse).

This module is dependency-free (standard library + torch) so the rest of the
course works without ``transformers`` installed.  ``from_pretrained`` is a thin
optional adapter.
"""

from __future__ import annotations

import unicodedata
from typing import Any, Iterable, Sequence

import torch

__all__ = ["normalize_token_text", "CompressedTokenizer"]


def normalize_token_text(text: str) -> str:
    """Canonical form used to decide whether two tokens are "the same".

    Order matters and mirrors the paper's normalizer:

    1. NFKC (full-width -> ASCII, ligatures, compatibility forms)
    2. NFD + drop combining marks (strip accents: café -> cafe)
    3. lowercase
    4. collapse any whitespace run to a single space
    5. strip; a token that was *only* whitespace becomes ``" "``

    Keeping whitespace-only tokens as a single canonical ``" "`` token is
    important: ``"\\n"`` and ``"  "`` are extremely frequent in BPE vocabularies
    and should share one Engram slot instead of fragmenting into dozens.
    """
    if not isinstance(text, str):
        text = str(text)
    if text == "":
        return ""

    was_space = text.strip() == ""
    text = unicodedata.normalize("NFKC", text)
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    text = " ".join(text.split())
    if was_space:
        return " "
    return text


def _decode_one(tokenizer: Any, token_id: int) -> str:
    """Call ``tokenizer.decode`` with the two common signatures."""
    try:
        return tokenizer.decode([token_id], skip_special_tokens=False)
    except TypeError:
        return tokenizer.decode([token_id])


class CompressedTokenizer:
    """Precomputed many-to-one map from a base tokenizer's ids to canonical ids.

    Parameters
    ----------
    tokenizer:
        Any object exposing ``__len__`` and ``decode``.  HuggingFace
        ``PreTrainedTokenizer`` and the toy tokenizers in ``tinyengram.data``
        both qualify.  ``convert_ids_to_tokens`` is used as a fallback for
        byte-level vocabularies whose ``decode`` emits U+FFFD.
    """

    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer
        self.lookup_table, self.new_tokens = self._build_lookup_table()
        self.num_new_token = len(self.new_tokens)

    def __len__(self) -> int:
        return self.num_new_token

    @classmethod
    def from_pretrained(cls, name_or_path: str, **kwargs: Any) -> "CompressedTokenizer":
        """Optional HuggingFace entry point (requires ``transformers``)."""
        try:
            from transformers import AutoTokenizer  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "需要 transformers：pip install -e '.[hf]'，或直接传入自己的 tokenizer"
            ) from exc
        return cls(AutoTokenizer.from_pretrained(name_or_path, **kwargs))

    # ------------------------------------------------------------------ build
    def _build_lookup_table(self) -> tuple[torch.Tensor, list[str]]:
        vocab_size = len(self.tokenizer)
        key_to_new: dict[str, int] = {}
        new_tokens: list[str] = []
        old_to_new = [0] * vocab_size

        for tid in range(vocab_size):
            text = _decode_one(self.tokenizer, tid)
            # Byte-level BPE: decode() may return U+FFFD for partial bytes.
            if "\ufffd" in text:
                converter = getattr(self.tokenizer, "convert_ids_to_tokens", None)
                key = str(converter(tid)) if converter is not None else text
            else:
                key = normalize_token_text(text)
                if key == "":
                    # A token decoding to "" (special tokens) keeps a unique key
                    # so we do not collapse the whole special-token block into one id.
                    key = f"\x00special:{tid}"

            nid = key_to_new.get(key)
            if nid is None:
                nid = len(new_tokens)
                key_to_new[key] = nid
                new_tokens.append(key)
            old_to_new[tid] = nid

        return torch.tensor(old_to_new, dtype=torch.long), new_tokens

    # ------------------------------------------------------------------ apply
    def compress(self, input_ids: torch.Tensor | Sequence[int] | Iterable[int]) -> torch.Tensor:
        """Map raw ids to canonical ids, preserving shape and negative padding."""
        ids = torch.as_tensor(input_ids, dtype=torch.long)
        flat = ids.reshape(-1)
        out = flat.clone()
        valid = flat >= 0
        if bool(valid.any()):
            idx = flat[valid]
            if int(idx.max()) >= self.lookup_table.numel():
                raise IndexError(
                    f"token id {int(idx.max())} 超出 tokenizer 词表 "
                    f"{self.lookup_table.numel()}"
                )
            out[valid] = self.lookup_table.to(flat.device)[idx]
        return out.reshape(ids.shape)

    __call__ = compress

    # ------------------------------------------------------------------ stats
    @property
    def compression_ratio(self) -> float:
        """Fraction of vocabulary removed: ``1 - |V'| / |V|``."""
        return 1.0 - self.num_new_token / len(self.tokenizer)

    def merge_groups(self) -> dict[int, list[int]]:
        """Inverse map new id -> sorted old ids (useful for case studies)."""
        groups: dict[int, list[int]] = {}
        for old, new in enumerate(self.lookup_table.tolist()):
            groups.setdefault(int(new), []).append(old)
        return groups

    def top_merges(self, k: int = 10) -> list[tuple[int, str, int]]:
        """Largest merge groups: ``(new_id, canonical_text, group_size)``."""
        groups = self.merge_groups()
        rows = [
            (nid, self.new_tokens[nid], len(ids))
            for nid, ids in groups.items()
            if len(ids) > 1
        ]
        rows.sort(key=lambda r: (-r[2], r[0]))
        return rows[:k]
