"""L3：上下文门控 —— RMSNorm + 余弦 logit + signed sqrt。"""

from __future__ import annotations

import math

import torch

from tinyengram import RMSNorm, context_gate, rms_norm, signed_sqrt


def test_rms_norm_has_unit_rms():
    x = torch.randn(4, 16, generator=torch.Generator().manual_seed(0))
    y = rms_norm(x)
    rms = y.pow(2).mean(dim=-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-5)


def test_rms_norm_module_applies_weight():
    x = torch.randn(2, 8, generator=torch.Generator().manual_seed(0))
    norm = RMSNorm(8)
    with torch.no_grad():
        norm.weight.fill_(3.0)
    assert torch.allclose(norm(x), rms_norm(x) * 3.0)


def test_signed_sqrt_preserves_sign_and_magnitude():
    x = torch.tensor([-4.0, -0.25, 0.0, 0.25, 4.0])
    y = signed_sqrt(x)
    assert torch.allclose(y.sign(), x.sign())
    assert torch.allclose(y.abs()[x != 0], x.abs()[x != 0].sqrt(), atol=1e-6)
    assert float(y[2]) == 0.0


def test_gate_zero_logit_is_half():
    q = torch.zeros(2, 3, 8)
    k = torch.zeros(2, 3, 8)
    g = context_gate(q, k)
    assert g.shape == (2, 3, 1)
    assert torch.allclose(g, torch.full_like(g, 0.5))


def test_gate_is_monotone_in_similarity():
    d = 16
    q = torch.zeros(1, 1, d)
    q[..., 0] = 1.0
    gates = []
    for c in (-0.8, -0.2, 0.0, 0.2, 0.8):
        k = torch.zeros(1, 1, d)
        k[..., 0] = c
        k[..., 1] = math.sqrt(max(0.0, 1 - c * c))
        gates.append(float(context_gate(q, k)))
    assert gates == sorted(gates)
    assert gates[0] < 0.5 < gates[-1]
    assert abs(gates[2] - 0.5) < 1e-6


def test_gate_is_scale_invariant():
    torch.manual_seed(0)
    q = torch.randn(1, 4, 8)
    k = torch.randn(1, 4, 8)
    g1 = context_gate(q, k)
    g2 = context_gate(q * 7.0, k * 0.2)
    assert torch.allclose(g1, g2, atol=1e-5)


def test_gate_gradients_reach_both_inputs():
    q = torch.randn(1, 4, 8, requires_grad=True)
    k = torch.randn(1, 4, 8, requires_grad=True)
    context_gate(q, k).sum().backward()
    assert q.grad is not None and q.grad.abs().sum() > 0
    assert k.grad is not None and k.grad.abs().sum() > 0


def test_gate_shape_mismatch_raises():
    try:
        context_gate(torch.zeros(1, 2, 4), torch.zeros(1, 2, 5))
    except ValueError:
        pass
    else:
        raise AssertionError("形状不一致应该报错")
