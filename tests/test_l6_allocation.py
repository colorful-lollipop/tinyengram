"""L6：稀疏分配与缩放律 —— 复算论文的数字。"""

from __future__ import annotations

import math

import pytest
import torch

from tinyengram import (
    allocate,
    argmin_quadratic,
    engram_parameter_count,
    iso_flops,
    power_law_fit,
    power_law_predict,
    quadratic_fit,
    u_shaped_loss,
)


def test_paper_engram_27b_parameter_count():
    # Table 5: 2 layers x 2,262,400 slots x d_mem=1280
    n = engram_parameter_count(layers=2, slots_per_head=2_262_400, n_heads_total=16, head_dim=80)
    assert n / 1e9 == pytest.approx(5.79, abs=0.02)


def test_paper_engram_40b_parameter_count():
    n = engram_parameter_count(layers=2, slots_per_head=7_239_680, n_heads_total=16, head_dim=80)
    assert n / 1e9 == pytest.approx(18.53, abs=0.02)


def test_allocation_conserves_sparse_budget():
    a = allocate(
        total_params=26_700_000_000,
        active_params=3_800_000_000,
        rho=0.743,
        params_per_expert=309_000_000,
        engram_dmem=1280,
        engram_layers=2,
        n_heads_total=16,
    )
    sparse = 26_700_000_000 - 3_800_000_000
    assert a.moe_params + a.engram_params == sparse
    assert a.rho == pytest.approx(0.743)
    assert 50 <= a.n_routed_experts <= 60
    assert a.slots_per_head / 1e6 == pytest.approx(2.26, abs=0.1)
    assert a.slots_total / 1e6 == pytest.approx(36.2, abs=1.0)
    assert a.sparsity == pytest.approx(26.7 / 3.8, abs=0.01)


def test_rho_extremes():
    kwargs = dict(
        total_params=10_000,
        active_params=1_000,
        params_per_expert=100,
        engram_dmem=16,
        engram_layers=1,
        n_heads_total=4,
    )
    pure_moe = allocate(rho=1.0, **kwargs)
    assert pure_moe.engram_params == 0 and pure_moe.slots_total == 0
    assert pure_moe.n_routed_experts == 90

    pure_memory = allocate(rho=0.0, **kwargs)
    assert pure_memory.moe_params == 0
    assert pure_memory.n_routed_experts == 0
    assert pure_memory.slots_per_head == 9000 // 16
    assert pure_memory.slots_total == (9000 // 16) * 4


def test_more_rho_means_more_experts_fewer_slots():
    kwargs = dict(
        total_params=1_000_000,
        active_params=100_000,
        params_per_expert=10_000,
        engram_dmem=64,
        engram_layers=1,
        n_heads_total=4,
    )
    prev_experts, prev_slots = -1, 10**12
    for rho in (0.2, 0.5, 0.8, 1.0):
        a = allocate(rho=rho, **kwargs)
        assert a.n_routed_experts >= prev_experts
        assert a.slots_total <= prev_slots
        prev_experts, prev_slots = a.n_routed_experts, a.slots_total


def test_u_shape_has_minimum_near_optimum():
    rhos = torch.linspace(0, 1, 101)
    losses = u_shaped_loss(rhos, optimum=0.78, curvature=0.2)
    idx = int(losses.argmin())
    assert abs(float(rhos[idx]) - 0.78) <= 0.01
    assert float(losses[0]) > float(losses[idx])
    assert float(losses[-1]) > float(losses[idx])


def test_asymmetry_keeps_minimum_but_steepens_left_wall():
    rhos = torch.linspace(0, 1, 201)
    sym = u_shaped_loss(rhos, optimum=0.78, asymmetry=0.0)
    asym = u_shaped_loss(rhos, optimum=0.78, asymmetry=0.5)
    assert int(asym.argmin()) == int(sym.argmin())
    assert float(asym[0]) > float(sym[0])


def test_quadratic_fit_recovers_coefficients():
    xs = torch.linspace(0, 1, 21).tolist()
    ys = [2.0 * x * x - 3.0 * x + 1.5 for x in xs]
    a, b, c = quadratic_fit(xs, ys)
    assert a == pytest.approx(2.0, abs=1e-8)
    assert b == pytest.approx(-3.0, abs=1e-8)
    assert c == pytest.approx(1.5, abs=1e-8)
    assert argmin_quadratic(a, b, c) == pytest.approx(0.75, abs=1e-8)


def test_quadratic_argmin_is_clipped_to_unit_interval():
    assert argmin_quadratic(1.0, -10.0, 0.0) == 1.0
    assert argmin_quadratic(1.0, 10.0, 0.0) == 0.0


def test_power_law_fit_and_predict():
    slots = [2.58e5, 1e6, 4e6, 1e7]
    losses = [1.80 - 0.02 * math.log10(s) for s in slots]
    intercept, slope = power_law_fit(slots, losses)
    assert slope == pytest.approx(-0.02, abs=1e-8)
    pred = power_law_predict(intercept, slope, 5e6)
    assert pred == pytest.approx(1.80 - 0.02 * math.log10(5e6), abs=1e-8)


def test_iso_flops_matches_paper_scale():
    flops = iso_flops(3.8e9, 262e9)
    assert flops / 1e21 == pytest.approx(5.97, abs=0.05)


def test_invalid_allocation_inputs():
    with pytest.raises(ValueError):
        allocate(
            total_params=100,
            active_params=10,
            rho=1.5,
            params_per_expert=1,
            engram_dmem=1,
            engram_layers=1,
            n_heads_total=1,
        )
    with pytest.raises(ValueError):
        u_shaped_loss(0.5, curvature=-1.0)
