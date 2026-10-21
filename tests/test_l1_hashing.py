"""L1：多头部哈希 —— 确定性、无溢出、后缀因果、素数表。"""

from __future__ import annotations

import torch

from tinyengram import NgramHashMapping, is_prime, make_odd_multipliers, next_primes


def make_mapping(**kwargs):
    defaults = dict(
        vocab_size=64,
        slots_per_head=64,
        max_ngram_size=3,
        n_head_per_ngram=2,
        layer_ids=(1, 2),
        pad_id=0,
        seed=0,
    )
    defaults.update(kwargs)
    return NgramHashMapping(**defaults)


def test_is_prime_known_values():
    primes = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31}
    for n in range(0, 32):
        assert is_prime(n) == (n in primes), n
    assert is_prime(104729)
    assert not is_prime(104730)


def test_next_primes_are_distinct_and_global():
    seen: set[int] = set()
    first = next_primes(63, 3, seen)
    second = next_primes(63, 3, seen)
    assert first == [67, 71, 73]
    assert second == [79, 83, 89]
    assert len(set(first + second)) == 6


def test_table_sizes_are_prime_and_large_enough():
    m = make_mapping()
    all_sizes = []
    for layer in m.layer_ids:
        for size in m.flat_table_sizes(layer):
            assert is_prime(size)
            assert size >= 64
            all_sizes.append(size)
    assert len(all_sizes) == len(set(all_sizes)), "不同层/order/head 的素数必须互不相同"
    assert len(all_sizes) == len(m.layer_ids) * m.n_orders * m.n_head_per_ngram


def test_multipliers_are_odd_and_bounded():
    m = make_mapping()
    for layer in m.layer_ids:
        mult = m.multipliers[layer]
        assert mult.shape == (3,)
        assert bool((mult % 2 == 1).all())
        assert int(mult.max()) < (2**63 - 1) // m.vocab_size


def test_layers_get_different_multipliers_but_same_seed_is_reproducible():
    a = make_mapping(seed=7)
    b = make_mapping(seed=7)
    for layer in a.layer_ids:
        assert torch.equal(a.multipliers[layer], b.multipliers[layer])
    assert not torch.equal(a.multipliers[1], a.multipliers[2])


def test_hash_shape_and_bounds():
    m = make_mapping()
    ids = torch.randint(1, 64, (2, 8), generator=torch.Generator().manual_seed(0))
    hashes = m.hash(ids, layer_id=1)
    assert hashes.shape == (2, 8, m.n_heads_total)
    sizes = m.flat_table_sizes(1)
    for h, size in enumerate(sizes):
        assert int(hashes[..., h].min()) >= 0
        assert int(hashes[..., h].max()) < size


def test_hash_is_deterministic():
    m = make_mapping()
    ids = torch.randint(1, 64, (2, 8), generator=torch.Generator().manual_seed(1))
    assert torch.equal(m.hash(ids, 1), m.hash(ids, 1))


def test_hash_is_strictly_causal():
    m = make_mapping()
    ids = torch.tensor([[5, 9, 11, 13, 17]])
    changed = ids.clone()
    changed[0, 3:] = torch.tensor([21, 22])
    h1 = m.hash(ids, 1)
    h2 = m.hash(changed, 1)
    assert torch.equal(h1[:, :3], h2[:, :3]), "未来 token 不能影响过去的 hash"
    assert not torch.equal(h1[:, 3:], h2[:, 3:])


def test_first_positions_use_pad_token():
    m = make_mapping()
    ids = torch.tensor([[5, 9, 11]])
    hashes = m.hash(ids, 1)
    mult = m.multipliers[1]
    sizes = m.table_sizes[1]

    for order_index, order in enumerate((2, 3)):
        # t=0, 2-gram = (pad, 5)
        mix0 = int(ids[0, 0] * mult[0]) ^ int(torch.tensor(m.pad_id) * mult[1])
        if order == 2:
            expected = mix0 % sizes[order_index][0]
            assert int(hashes[0, 0, order_index * m.n_head_per_ngram]) == expected
        else:
            mix0_3 = (
                int(ids[0, 0] * mult[0])
                ^ int(torch.tensor(m.pad_id) * mult[1])
                ^ int(torch.tensor(m.pad_id) * mult[2])
            )
            expected = mix0_3 % sizes[order_index][0]
            assert int(hashes[0, 0, order_index * m.n_head_per_ngram]) == expected


def test_invalid_layer_and_ids_raise():
    m = make_mapping()
    ids = torch.ones(1, 4, dtype=torch.long)
    for bad in (0, 99):
        try:
            m.hash(ids, bad)
        except KeyError:
            pass
        else:
            raise AssertionError("未知 layer 应该 KeyError")

    try:
        m.hash(torch.full((1, 4), 64, dtype=torch.long), 1)
    except ValueError:
        pass
    else:
        raise AssertionError("越界 token 应该 ValueError")


def test_large_vocab_multipliers_do_not_overflow():
    m = make_mapping(vocab_size=129280, slots_per_head=128, layer_ids=(1,))
    ids = torch.full((1, 4), 129279, dtype=torch.long)
    hashes = m.hash(ids, 1)
    assert bool((hashes >= 0).all())
    for h, size in enumerate(m.flat_table_sizes(1)):
        assert int(hashes[..., h].max()) < size


def test_multiplier_function_shape():
    mult = make_odd_multipliers(4, 1000, seed=3)
    assert mult.shape == (4,)
    assert bool((mult > 0).all())
