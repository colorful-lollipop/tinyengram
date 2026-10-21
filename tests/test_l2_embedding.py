"""L2：检索与内存布局 —— 一张扁平表 + 偏移量完成多头 gather。"""

from __future__ import annotations

import torch

from tinyengram import EngramMemory, MultiHeadEmbedding, NgramHashMapping


def test_offsets_route_each_head_to_its_own_rows():
    torch.manual_seed(0)
    mhe = MultiHeadEmbedding([5, 7, 11], embedding_dim=3)
    ids = torch.tensor([[0, 0, 0], [4, 6, 10]])
    out = mhe(ids)
    assert out.shape == (2, 3, 3)
    w = mhe.embedding.weight
    for h, offset in enumerate([0, 5, 12]):
        expected = w[ids[:, h] + offset]
        assert torch.equal(out[:, h, :], expected)


def test_forward_shape_generic_leading_dims():
    mhe = MultiHeadEmbedding([3, 4], embedding_dim=2)
    ids = torch.zeros(2, 5, 2, dtype=torch.long)
    assert mhe(ids).shape == (2, 5, 2, 2)


def test_wrong_head_count_raises():
    mhe = MultiHeadEmbedding([3, 4], embedding_dim=2)
    try:
        mhe(torch.zeros(1, 3, dtype=torch.long))
    except ValueError:
        pass
    else:
        raise AssertionError("head 数不匹配应该报错")


def test_only_touched_rows_receive_gradient():
    torch.manual_seed(0)
    mhe = MultiHeadEmbedding([4, 4], embedding_dim=2)
    ids = torch.tensor([[1, 3]])
    mhe(ids).sum().backward()
    grad = mhe.embedding.weight.grad
    assert grad is not None
    nonzero = (grad.abs().sum(-1) > 0).nonzero().flatten().tolist()
    assert set(nonzero) == {1, 4 + 3}


def test_memory_retrieval_matches_manual_pipeline():
    torch.manual_seed(0)
    mapping = NgramHashMapping(
        vocab_size=32,
        slots_per_head=16,
        max_ngram_size=3,
        n_head_per_ngram=2,
        layer_ids=(1,),
        pad_id=0,
        seed=0,
    )
    mem = EngramMemory(mapping, layer_id=1, head_dim=4)
    ids = torch.randint(1, 32, (2, 6), generator=torch.Generator().manual_seed(0))
    out = mem.retrieve(ids)
    assert out.shape == (2, 6, mem.dmem)
    assert mem.dmem == mapping.n_heads_total * 4

    hashes = mapping.hash(ids, 1)
    expected = mem.table(hashes).flatten(start_dim=-2)
    assert torch.equal(out, expected)


def test_memory_shapes_match_hash_metadata():
    mapping = NgramHashMapping(
        vocab_size=32,
        slots_per_head=16,
        max_ngram_size=3,
        n_head_per_ngram=2,
        layer_ids=(1,),
        pad_id=0,
    )
    mem = EngramMemory(mapping, 1, head_dim=4)
    assert mem.total_slots == sum(mapping.flat_table_sizes(1))
    assert mem.table.num_heads == mapping.n_heads_total
    assert mem.table.total_slots * 4 == mem.total_slots * 4


def test_hash_ids_is_no_grad_and_deterministic():
    mapping = NgramHashMapping(
        vocab_size=32,
        slots_per_head=16,
        max_ngram_size=3,
        n_head_per_ngram=2,
        layer_ids=(1,),
        pad_id=0,
    )
    mem = EngramMemory(mapping, 1, head_dim=4)
    ids = torch.randint(1, 32, (2, 6), generator=torch.Generator().manual_seed(2))
    h = mem.hash_ids(ids)
    assert not h.requires_grad
    assert torch.equal(h, mem.hash_ids(ids))
