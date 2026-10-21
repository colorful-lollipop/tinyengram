"""A synthetic corpus where conditional memory is *provably* useful.

Real pre-training data is too expensive for a CPU tutorial, so we construct a
task with the same structure the paper cares about:

    multi-token entity name  ->  a fixed attribute token

The mapping is deterministic and local (the attribute always follows the name),
so a 3-gram lookup can store it; a Transformer must spend layers reconstructing
it.  Four surface templates prevent the model from solving the task with a
single positional shortcut.

The corpus also has a **held-out entity split** so you can ask whether Engram
merely memorizes names or actually frees depth for the rest of the network.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import torch

__all__ = ["ToyTokenizer", "EntityCorpus", "pad_batch"]

PAD, BOS, EOS, ARROW, QMARK = 0, 1, 2, 3, 4
SPECIALS = ["<pad>", "<bos>", "<eos>", "->", "?"]
FILLERS = [
    "the",
    "a",
    "of",
    "is",
    "was",
    "who",
    "what",
    "answer",
    "for",
    "known",
    "as",
    "in",
    "and",
    "story",
    "about",
    "tell",
    "me",
    ".",
    ",",
    "!",
]


class ToyTokenizer:
    """Word-level tokenizer with the small interface ``CompressedTokenizer`` needs."""

    def __init__(self, words: Sequence[str]) -> None:
        self.itos = list(words)
        self.stoi = {w: i for i, w in enumerate(self.itos)}
        if len(self.stoi) != len(self.itos):
            raise ValueError("词表里有重复 token")

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, words: Sequence[str]) -> list[int]:
        return [self.stoi[w] for w in words]

    def decode(self, ids: Sequence[int] | int, skip_special_tokens: bool = False) -> str:
        if isinstance(ids, int):
            ids = [ids]
        out = []
        for i in ids:
            word = self.itos[int(i)]
            if skip_special_tokens and word in SPECIALS:
                continue
            out.append(word)
        return " ".join(out)


@dataclass
class EntityCorpus:
    """Generate ``name -> attribute`` sentences.

    Names are 3 tokens long.  With ``max_ngram_size=3`` the final token of a
    name sees the complete name in its suffix trigram — this is the exact
    condition under which Engram should shine.
    """

    n_entities: int = 64
    name_len: int = 3
    n_train_entities: int | None = None
    seed: int = 0
    templates: tuple[str, ...] = ("arrow", "qa", "story", "reverse")
    _words: list[str] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.name_len < 2:
            raise ValueError("name_len 至少为 2，否则 3-gram 看不到完整实体")
        if self.n_entities < 2:
            raise ValueError("n_entities 至少为 2")
        if self.n_train_entities is None:
            self.n_train_entities = self.n_entities
        if not 0 < self.n_train_entities <= self.n_entities:
            raise ValueError("n_train_entities 必须在 (0, n_entities] 内")

        words = list(SPECIALS) + list(FILLERS)
        self.name_tokens: list[list[int]] = []
        self.attr_tokens: list[int] = []
        for i in range(self.n_entities):
            name = []
            for j in range(self.name_len):
                name.append(len(words))
                words.append(f"name{i}_{j}")
            self.name_tokens.append(name)
            self.attr_tokens.append(len(words))
            words.append(f"attr{i}")
        self.tokenizer = ToyTokenizer(words)
        self._words = words

    @property
    def vocab_size(self) -> int:
        return len(self.tokenizer)

    @property
    def train_entities(self) -> range:
        return range(self.n_train_entities)

    @property
    def heldout_entities(self) -> range:
        return range(self.n_train_entities, self.n_entities)

    def _sentence(self, entity: int, template: str) -> list[int]:
        name = self.name_tokens[entity]
        attr = self.attr_tokens[entity]
        t = self.tokenizer
        if template == "arrow":
            words = ["<bos>", *[t.itos[i] for i in name], "->", t.itos[attr], "<eos>"]
        elif template == "qa":
            words = [
                "<bos>",
                "who",
                "is",
                *[t.itos[i] for i in name],
                "?",
                "the",
                "answer",
                "is",
                t.itos[attr],
                "<eos>",
            ]
        elif template == "story":
            words = [
                "<bos>",
                "the",
                "story",
                "of",
                *[t.itos[i] for i in name],
                "is",
                "about",
                t.itos[attr],
                ".",
                "<eos>",
            ]
        elif template == "reverse":
            words = [
                "<bos>",
                t.itos[attr],
                "is",
                "known",
                "as",
                *[t.itos[i] for i in name],
                "<eos>",
            ]
        else:  # pragma: no cover - guarded by dataclass callers
            raise ValueError(f"未知模板 {template}")
        return t.encode(words)

    def sample(
        self,
        batch_size: int,
        *,
        generator: torch.Generator | None = None,
        split: str = "train",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(input_ids, targets)`` padded to the batch max length."""
        if split == "train":
            entities = list(self.train_entities)
        elif split == "heldout":
            entities = list(self.heldout_entities)
        else:
            raise ValueError("split 必须是 'train' 或 'heldout'")
        if not entities:
            raise ValueError("该 split 没有实体")

        seqs = []
        for _ in range(batch_size):
            e = entities[int(torch.randint(len(entities), (1,), generator=generator))]
            t = self.templates[
                int(torch.randint(len(self.templates), (1,), generator=generator))
            ]
            seqs.append(self._sentence(e, t))
        return pad_batch(seqs, PAD)

    def all_sequences(self, split: str = "train") -> list[list[int]]:
        entities = self.train_entities if split == "train" else self.heldout_entities
        return [
            self._sentence(e, t)
            for e in entities
            for t in self.templates
        ]


def pad_batch(
    sequences: Sequence[Sequence[int]], pad_id: int = PAD
) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-pad and build next-token targets (``-100`` on padding)."""
    if len(sequences) == 0:
        raise ValueError("sequences 不能为空")
    max_len = max(len(s) for s in sequences)
    if max_len < 2:
        raise ValueError("序列至少需要 2 个 token 才能构造 next-token 目标")
    ids = torch.full((len(sequences), max_len), pad_id, dtype=torch.long)
    for i, seq in enumerate(sequences):
        ids[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
    inputs = ids[:, :-1]
    targets = ids[:, 1:].clone()
    targets[targets == pad_id] = -100
    return inputs, targets
