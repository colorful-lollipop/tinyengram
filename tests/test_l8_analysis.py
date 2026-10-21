"""L8：机制分析 —— LogitLens、CKA、软对齐、门控统计。"""

from __future__ import annotations

import math

import pytest
import torch

from tinyengram import (
    RMSNorm,
    binary_entropy,
    gate_stats,
    linear_cka,
    logit_lens_kl,
    soft_alignment,
)


def test_cka_identity_is_one():
    x = torch.randn(64, 16, generator=torch.Generator().manual_seed(0))
    assert linear_cka(x, x) == pytest.approx(1.0, abs=1e-5)


def test_cka_is_invariant_to_scaling_and_rotation():
    x = torch.randn(64, 16, generator=torch.Generator().manual_seed(0))
    assert linear_cka(x, x * 7.5) == pytest.approx(1.0, abs=1e-5)

    q, _ = torch.linalg.qr(torch.randn(16, 16, generator=torch.Generator().manual_seed(1)))
    assert linear_cka(x, x @ q) == pytest.approx(1.0, abs=1e-4)


def test_cka_of_independent_noise_is_low():
    x = torch.randn(128, 32, generator=torch.Generator().manual_seed(0))
    y = torch.randn(128, 32, generator=torch.Generator().manual_seed(1))
    assert linear_cka(x, y) < 0.5


def test_cka_validation():
    with pytest.raises(ValueError):
        linear_cka(torch.randn(3), torch.randn(3))
    with pytest.raises(ValueError):
        linear_cka(torch.randn(4, 2), torch.randn(5, 2))


def test_soft_alignment_identity_is_diagonal():
    sim = torch.eye(10)
    align = soft_alignment(sim, k=1)
    assert torch.allclose(align, torch.arange(10).float())


def test_soft_alignment_recovers_shift():
    rows, cols = 12, 8
    sim = torch.zeros(rows, cols)
    shift = 2
    for j in range(cols):
        sim[j + shift, j] = 1.0
        sim[j + shift, j] = 1.0
    align = soft_alignment(sim, k=1)
    assert torch.allclose(align, torch.arange(cols).float() + shift)


def test_logit_lens_final_kl_is_zero_and_earlier_is_positive():
    torch.manual_seed(0)
    norm = RMSNorm(8)
    head = torch.nn.Linear(8, 5, bias=False)
    hidden = [torch.randn(2, 4, 8) for _ in range(3)]
    final_logits = head(norm(hidden[-1]))
    kls = logit_lens_kl(hidden, norm, head, final_logits)
    assert len(kls) == 3
    assert kls[-1] == pytest.approx(0.0, abs=1e-6)
    assert kls[0] > 0.0


def test_binary_entropy_peaks_at_half():
    p = torch.tensor([0.0, 0.1, 0.5, 0.9, 1.0])
    h = binary_entropy(p)
    assert float(h[2]) == pytest.approx(math.log(2), abs=1e-5)
    assert float(h[2]) > float(h[1]) > float(h[0])
    assert bool((h >= 0).all())


def test_gate_stats_ranges():
    torch.manual_seed(0)
    gates = torch.sigmoid(torch.randn(100, 4))
    stats = gate_stats(gates)
    assert 0.0 <= stats["mean"] <= 1.0
    assert 0.0 <= stats["frac_above_0.5"] <= 1.0
    assert 0.0 <= stats["frac_below_0.1"] <= 1.0
    assert stats["entropy"] <= math.log(2) + 1e-6
