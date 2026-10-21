"""Optional Triton kernel: fused multi-head gather + concat.

The eager path does ``H`` separate ``index_select`` calls (one per hash head),
each producing ``[N, D]``, then concatenates them into ``[N, H*D]``.  That is
``H`` kernel launches plus an extra pass over the output.  The fused kernel
launches one program per ``(position, head)`` and writes directly into the
final layout.

This is a *bandwidth* optimization: Engram retrieval is memory-bound, so the
goal is to touch each output byte once.  It does not change numerics.
"""

from __future__ import annotations

import torch

try:  # pragma: no cover - import guard
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:  # pragma: no cover
    triton = None
    tl = None
    HAS_TRITON = False

__all__ = ["HAS_TRITON", "triton_gather_concat", "eager_gather_concat"]


def eager_gather_concat(
    table, hashes: torch.Tensor, offsets: torch.Tensor
) -> torch.Tensor:
    """Reference: ``table[hashes + offsets].flatten(-2)``.

    ``table`` may be a Tensor or an ``nn.Embedding`` module.
    """
    ids = hashes + offsets
    if isinstance(table, torch.Tensor):
        return table[ids].flatten(-2)
    return table(ids).flatten(-2)


if HAS_TRITON:

    @triton.jit
    def _gather_concat_kernel(
        table_ptr,
        hash_ptr,
        offset_ptr,
        out_ptr,
        H: tl.constexpr,
        D: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        # One program per token: load H rows (one per hash head) and write
        # them contiguously.  This keeps the random-access work inside one
        # program instead of launching N*H tiny programs.
        n = tl.program_id(0)
        d = tl.arange(0, BLOCK_D)
        mask = d < D
        for h in tl.static_range(H):
            row = tl.load(hash_ptr + n * H + h).to(tl.int64) + tl.load(offset_ptr + h)
            value = tl.load(table_ptr + row * D + d, mask=mask)
            tl.store(out_ptr + n * (H * D) + h * D + d, value, mask=mask)

    def triton_gather_concat(
        table: torch.Tensor,
        hashes: torch.Tensor,
        offsets: torch.Tensor,
        *,
        block_d: int = 128,
    ) -> torch.Tensor:
        """Fused gather. Requires CUDA and contiguous inputs."""
        if not table.is_cuda:
            raise RuntimeError("triton_gather_concat 需要 CUDA tensor")
        if hashes.dim() != 2:
            raise ValueError("hashes 应为 [N, H]")
        n, h = hashes.shape
        if offsets.numel() != h:
            raise ValueError("offsets 长度必须等于 head 数")
        d = table.shape[1]
        out = torch.empty((n, h * d), device=table.device, dtype=table.dtype)
        grid = (n,)
        _gather_concat_kernel[grid](
            table,
            hashes,
            offsets,
            out,
            H=h,
            D=d,
            BLOCK_D=block_d,
        )
        return out

else:

    def triton_gather_concat(*args, **kwargs):  # type: ignore[misc]
        """Import-safe stub used on CPU-only installs."""
        raise RuntimeError(
            "triton_gather_concat 需要 CUDA + Triton；请安装 `.[perf]` 并在 GPU 上运行"
        )


def eager_from_module(module, hashes: torch.Tensor) -> torch.Tensor:
    """Convenience: use a MultiHeadEmbedding module's table/offsets."""
    return eager_gather_concat(module.embedding.weight, hashes, module.offsets)
