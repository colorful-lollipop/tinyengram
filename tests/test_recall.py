from __future__ import annotations

import pytest
import torch

from experiments.recall import recall_batch


def test_response_requires_earlier_payload_not_just_recent_tokens():
    prompt, response = recall_batch(8, 32, torch.Generator().manual_seed(0))
    assert torch.equal(response[:, :4], prompt[:, 1:5])
    assert bool((response[:, -1] == 0).all())
    changed = prompt.clone()
    changed[:, 1:5] = (changed[:, 1:5] - 9 + 1) % 8 + 9
    assert torch.equal(changed[:, -20:], prompt[:, -20:])
    assert not torch.equal(changed[:, 1:5], response[:, :4])


def test_recall_rng_is_reproducible_and_payload_varies():
    a, ra = recall_batch(16, 8, torch.Generator().manual_seed(10))
    b, rb = recall_batch(16, 8, torch.Generator().manual_seed(10))
    assert torch.equal(a, b) and torch.equal(ra, rb)
    assert ra[:, :4].unique(dim=0).shape[0] > 1
    with pytest.raises(ValueError):
        recall_batch(1, 6, torch.Generator())
