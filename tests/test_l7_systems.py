"""L7：推理系统 —— 确定性预取、Zipf 缓存、分片、offload。"""

from __future__ import annotations

import pytest
import torch

from tinyengram import (
    LRUCache,
    NgramHashMapping,
    estimate_transfer_ms,
    prefetch_plan,
    shard_gather,
    simulate_cache,
    simulate_offload,
    zipf_probs,
)


def make_mapping(layers=(1, 2)):
    return NgramHashMapping(
        vocab_size=64,
        slots_per_head=32,
        max_ngram_size=3,
        n_head_per_ngram=2,
        layer_ids=layers,
        pad_id=0,
        seed=0,
    )


def test_zipf_probs_normalized_and_sorted():
    p = zipf_probs(1000)
    assert float(p.sum()) == pytest.approx(1.0, abs=1e-6)
    assert bool((p[:-1] >= p[1:]).all())
    assert float(zipf_probs(1000, s=1.5)[0]) > float(zipf_probs(1000, s=1.0)[0])


def test_lru_evicts_least_recently_used():
    cache = LRUCache(2)
    for key in (1, 2):
        assert cache.get(key) is None
        cache.put(key, key)
    assert cache.get(1) == 1  # 1 becomes most recent
    cache.put(3, 3)  # evicts 2
    assert cache.get(2) is None
    assert cache.get(1) == 1
    assert cache.get(3) == 3
    assert cache.hit_rate == pytest.approx(3 / 6)


def test_zero_capacity_cache_always_misses():
    cache = LRUCache(0)
    cache.put(1, 1)
    assert cache.get(1) is None
    assert cache.hit_rate == 0.0


def test_simulate_cache_upper_and_lower_bounds():
    trace = torch.tensor([1, 1, 2, 2, 3, 3, 1, 2, 3])
    cold = simulate_cache(trace, capacity=0)
    assert cold["hit_rate"] == 0.0
    hot = simulate_cache(trace, capacity=3)
    assert hot["hit_rate"] == pytest.approx(2 / 3)
    mid = simulate_cache(trace, capacity=1)
    assert 0.0 < mid["hit_rate"] < 1.0
    assert mid["unique"] == 3


def test_prefetch_plan_is_computed_before_forward():
    mapping = make_mapping()
    ids = torch.randint(1, 64, (2, 8), generator=torch.Generator().manual_seed(0))
    plan = prefetch_plan(ids, mapping)
    assert set(plan) == {1, 2}
    for layer, hashes in plan.items():
        assert torch.equal(hashes, mapping.hash(ids, layer))
        # 计划不依赖任何模型参数/隐藏状态
        assert torch.equal(hashes, mapping.hash(ids.clone(), layer))


def test_offload_traffic_scales_with_unique_active_slots():
    plan = {1: torch.tensor([[5, 5, 6, 7]])}
    res = simulate_offload(
        plan, bytes_per_slot=1000, compute_window_ms=0.0, bandwidth_gbps=64.0
    )
    assert res.bytes == 3 * 1000
    res_dup = simulate_offload(
        {1: torch.tensor([[5, 5, 5, 5]])}, bytes_per_slot=1000, compute_window_ms=0.0
    )
    assert res_dup.bytes == 1000


def test_large_compute_window_hides_all_stalls():
    plan = {1: torch.arange(100).reshape(1, -1)}
    hidden = simulate_offload(plan, bytes_per_slot=1280 * 4, compute_window_ms=1e9)
    assert hidden.stalls_ms == 0.0
    exposed = simulate_offload(plan, bytes_per_slot=1280 * 4, compute_window_ms=0.0)
    assert exposed.stalls_ms == pytest.approx(exposed.transfer_ms)
    assert exposed.transfer_ms > 0


def test_cache_reduces_traffic():
    plan = {1: torch.arange(64).reshape(1, -1), 2: torch.arange(64).reshape(1, -1)}
    no_cache = simulate_offload(
        plan, bytes_per_slot=512, compute_window_ms=0.0, cache_capacity=0
    )
    with_cache = simulate_offload(
        plan, bytes_per_slot=512, compute_window_ms=0.0, cache_capacity=64
    )
    assert with_cache.bytes < no_cache.bytes
    assert with_cache.hit_rate > 0.0


def test_shard_gather_matches_full_table():
    torch.manual_seed(0)
    full = torch.randn(10, 4)
    shards = [full[:3], full[3:7], full[7:]]
    ids = torch.randint(0, 10, (2, 5), generator=torch.Generator().manual_seed(0))
    gathered, owner, local = shard_gather(shards, ids)
    assert torch.equal(gathered, full[ids])
    assert gathered.shape == (2, 5, 4)
    assert bool((owner >= 0).all() and (owner < 3).all())
    for o, loc in zip(owner.reshape(-1).tolist(), local.reshape(-1).tolist()):
        assert 0 <= o < len(shards)
        assert 0 <= loc < shards[o].shape[0]


def test_transfer_time_formula():
    assert estimate_transfer_ms(1_000_000_000, 64.0) == pytest.approx(125.0)


def test_system_input_validation():
    with pytest.raises(ValueError):
        simulate_offload({1: torch.zeros(1, 1, dtype=torch.long)}, bytes_per_slot=0)
    with pytest.raises(ValueError):
        estimate_transfer_ms(10, 0.0)
    with pytest.raises(ValueError):
        shard_gather([torch.zeros(2, 2)], torch.tensor([[5]]))
