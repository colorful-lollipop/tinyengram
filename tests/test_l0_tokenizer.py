"""L0：Tokenizer 压缩 —— 把语义等价的子词折叠成一个规范 id。"""

from __future__ import annotations

import torch

from tinyengram import CompressedTokenizer, normalize_token_text


class FakeTokenizer:
    """Minimal duck-typed tokenizer: one token per vocabulary entry."""

    def __init__(self, tokens: list[str]) -> None:
        self.tokens = tokens

    def __len__(self) -> int:
        return len(self.tokens)

    def decode(self, ids, skip_special_tokens: bool = False) -> str:
        return self.tokens[int(ids[0])]


TOKENS = [
    "Apple",
    "apple",
    " apple",
    "APPLE",
    "café",
    "cafe",
    "\n",
    "  ",
    "hello",
    " world",
]


def test_normalize_unicode_case_and_accents():
    assert normalize_token_text("Ａpple") == "apple"  # 全角 -> ASCII
    assert normalize_token_text("APPLE") == "apple"
    assert normalize_token_text("Café") == "cafe"
    assert normalize_token_text("  hello   world ") == "hello world"


def test_whitespace_only_tokens_share_one_canonical_form():
    assert normalize_token_text("\n") == " "
    assert normalize_token_text("\t\t") == " "
    assert normalize_token_text("   ") == " "
    assert normalize_token_text("") == ""


def test_projection_is_surjective_and_compresses():
    ct = CompressedTokenizer(FakeTokenizer(TOKENS))
    # 10 个原始 token -> 5 个规范 token
    assert ct.num_new_token == 5
    assert abs(ct.compression_ratio - 0.5) < 1e-9

    used = set(ct.lookup_table.tolist())
    assert used == set(range(ct.num_new_token)), "每个新 id 都必须被至少一个旧 id 命中"


def test_equivalent_tokens_share_id():
    ct = CompressedTokenizer(FakeTokenizer(TOKENS))
    apple_ids = {int(ct.lookup_table[i]) for i in [0, 1, 2, 3]}
    assert len(apple_ids) == 1
    assert int(ct.lookup_table[4]) == int(ct.lookup_table[5])
    assert int(ct.lookup_table[6]) == int(ct.lookup_table[7])


def test_compress_preserves_shape_and_negative_padding():
    ct = CompressedTokenizer(FakeTokenizer(TOKENS))
    ids = torch.tensor([[0, 1, 2], [4, -1, 9]])
    out = ct.compress(ids)
    assert out.shape == ids.shape
    assert int(out[1, 1]) == -1
    assert int(out[0, 0]) == int(out[0, 1]) == int(out[0, 2])
    # 不同 token 不能误合并
    assert int(out[1, 0]) != int(out[1, 2])


def test_top_merges_reports_largest_group():
    ct = CompressedTokenizer(FakeTokenizer(TOKENS))
    top = ct.top_merges(3)
    assert top[0][2] == 4
    assert top[0][1] == "apple"
    assert [row[2] for row in top] == sorted([row[2] for row in top], reverse=True)
