"""L7（可选）：Triton gather 与 eager 数值一致。需要 CUDA。"""

from __future__ import annotations

import pytest
import torch

from perf.kernels import HAS_TRITON, eager_gather_concat, triton_gather_concat

pytestmark = pytest.mark.gpu


@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 CUDA")
@pytest.mark.skipif(not HAS_TRITON, reason="需要 Triton")
def test_triton_gather_matches_eager():
    torch.manual_seed(0)
    heads, dim, rows = 8, 64, 1000
    table = torch.randn(heads * rows, dim, device="cuda", dtype=torch.float16)
    hashes = torch.randint(0, rows, (128, heads), device="cuda")
    offsets = torch.arange(0, heads * rows, rows, device="cuda")
    expected = eager_gather_concat(table, hashes, offsets)
    actual = triton_gather_concat(table, hashes, offsets)
    assert torch.equal(actual, expected)
