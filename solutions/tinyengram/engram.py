"""The Engram module: retrieval + context-aware gating + short conv.

Data flow (paper Figure 1):

    input_ids ──► compressed ids ──► hash ids ──► flat gather ──► e_t
                                                                  │
    hidden h_t^(m) ──────────────► gate alpha_t^(m) ◄── W_K^(m) ──┤
                                                                  │
                              u_t^(m) = alpha_t^(m) * (W_V e_t) ◄──┘
                              Y_t^(m) = u_t^(m) + ShortConv(u^(m))
                              H^(l)  <- H^(l) + Y

Parameter sharing in the multi-branch (mHC) setting:

* **shared** across branches: the huge embedding table, ``W_V``;
* **branch-specific**: ``W_K^(m)`` and the gate, so each residual stream can
  decide independently whether the memory is relevant;
* one depthwise conv per branch (channel groups = ``M``).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .config import EngramConfig
from .conv import ShortConv
from .embedding import EngramMemory
from .gating import RMSNorm, context_gate
from .hashing import NgramHashMapping
from .tokenizer import CompressedTokenizer

__all__ = ["Engram", "build_engram_mapping", "build_engram_modules"]


def build_engram_mapping(
    cfg: EngramConfig,
    vocab_size: Optional[int] = None,
    pad_id: Optional[int] = None,
) -> NgramHashMapping:
    """One mapping shared by all layers (global prime uniqueness)."""
    return NgramHashMapping(
        vocab_size=cfg.vocab_size if vocab_size is None else vocab_size,
        slots_per_head=cfg.slots_per_head_,
        max_ngram_size=cfg.max_ngram_size,
        n_head_per_ngram=cfg.n_head_per_ngram,
        layer_ids=cfg.layer_ids_,
        pad_id=cfg.pad_id if pad_id is None else pad_id,
        seed=cfg.seed,
    )


class Engram(nn.Module):
    def __init__(
        self,
        cfg: EngramConfig,
        layer_id: int,
        *,
        mapping: Optional[NgramHashMapping] = None,
        compressed_tokenizer: Optional[CompressedTokenizer] = None,
    ) -> None:
        """If ``compressed_tokenizer`` is given, ``cfg.vocab_size`` and
        ``cfg.pad_id`` are interpreted in the **raw** tokenizer space; hashing
        uses the compressed ids and the remapped pad id."""
        super().__init__()
        if layer_id not in cfg.layer_ids_:
            raise ValueError(f"layer_id={layer_id} 不在 cfg.layer_ids={cfg.layer_ids_} 中")
        self.cfg = cfg
        self.layer_id = int(layer_id)
        self.compressed_tokenizer = compressed_tokenizer

        vocab_size = len(compressed_tokenizer) if compressed_tokenizer is not None else cfg.vocab_size
        pad_id = cfg.pad_id
        if compressed_tokenizer is not None:
            # The pad token is also remapped by the projection; using the raw
            # id would pad with an unrelated canonical token.
            pad_id = int(compressed_tokenizer.lookup_table[cfg.pad_id])
        self.mapping = mapping or build_engram_mapping(
            cfg, vocab_size=vocab_size, pad_id=pad_id
        )
        if self.layer_id not in self.mapping.table_sizes:
            raise ValueError("传入的 mapping 不包含当前 layer_id")

        self.memory = EngramMemory(self.mapping, self.layer_id, cfg.head_dim)
        dmem = self.memory.dmem
        hidden = cfg.hidden_size
        branches = cfg.hc_mult

        self.value_proj = nn.Linear(dmem, hidden, bias=False)
        self.key_projs = nn.ModuleList(
            [nn.Linear(dmem, hidden, bias=False) for _ in range(branches)]
        )
        self.query_norms = nn.ModuleList([RMSNorm(hidden) for _ in range(branches)])
        self.key_norms = nn.ModuleList([RMSNorm(hidden) for _ in range(branches)])
        self.short_conv = ShortConv(
            hidden,
            groups=branches,
            kernel_size=cfg.conv_kernel_size,
            dilation=cfg.max_ngram_size,
            zero_init=cfg.conv_zero_init,
        )

    # ---------------------------------------------------------------- helpers
    @property
    def dmem(self) -> int:
        return self.memory.dmem

    @property
    def history_size(self) -> int:
        """Number of earlier gated activations required by ShortConv."""
        return (self.short_conv.kernel_size - 1) * self.short_conv.dilation

    def compress(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self.compressed_tokenizer is None:
            return input_ids
        return self.compressed_tokenizer.compress(input_ids)

    def _ids(self, input_ids: torch.Tensor, context_ids: Optional[torch.Tensor]) -> torch.Tensor:
        """Compress only the new input plus N-1 preceding raw token IDs."""
        ids = self.compress(input_ids)
        if context_ids is None:
            return ids
        if context_ids.shape[0] != input_ids.shape[0] or context_ids.shape[1] < input_ids.shape[1]:
            raise ValueError("context_ids 必须同 batch 且不短于 input_ids")
        width = input_ids.shape[1] + self.cfg.max_ngram_size - 1
        context = self.compress(context_ids[:, -width:])
        if not torch.equal(context[:, -ids.shape[1] :], ids):
            raise ValueError("input_ids 必须是 context_ids 的末尾切片")
        return context

    def retrieve(
        self, input_ids: torch.Tensor, context_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Memory vector ``e_t`` of shape ``[B, T, d_mem]``."""
        hashes = self.memory.hash_ids(self._ids(input_ids, context_ids))
        # Previous IDs affect the hash, but their embedding rows are not needed.
        return self.memory.table(hashes[:, -input_ids.shape[1]:]).flatten(-2)

    def gates(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        context_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Per-branch gate values ``[B, T, M]`` (for analysis / visualization)."""
        emb = self.retrieve(input_ids, context_ids)
        out = []
        for m in range(self.cfg.hc_mult):
            key = self.key_projs[m](emb)
            q = self.query_norms[m](hidden_states[:, :, m, :])
            k = self.key_norms[m](key)
            out.append(context_gate(q, k, normalize=False))
        return torch.stack(out, dim=2).squeeze(-1)

    # ---------------------------------------------------------------- forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        context_ids: Optional[torch.Tensor] = None,
        *,
        conv_state: Optional[torch.Tensor] = None,
        return_state: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states: ``[B, T, M, D]`` multi-branch residual stream.
            input_ids:     ``[B, T]`` raw (or already compressed) ids.
        Returns:
            Contribution ``[B,T,M,D]``. With return_state, also return the raw
            gated activation window (at most H previous + T new activations).
        """
        if hidden_states.dim() != 4:
            raise ValueError(f"hidden_states 应为 [B,T,M,D]，收到 {tuple(hidden_states.shape)}")
        b, t, m, d = hidden_states.shape
        if m != self.cfg.hc_mult or d != self.cfg.hidden_size:
            raise ValueError(
                f"期望 M={self.cfg.hc_mult}, D={self.cfg.hidden_size}，收到 M={m}, D={d}"
            )
        if input_ids.shape != (b, t):
            raise ValueError(f"input_ids 应为 [B,T]={(b, t)}，收到 {tuple(input_ids.shape)}")

        emb = self.retrieve(input_ids, context_ids)  # [B,T,dmem]

        value = self.value_proj(emb)  # shared W_V
        gated = []
        for branch in range(m):
            key = self.key_projs[branch](emb)
            q = self.query_norms[branch](hidden_states[:, :, branch, :])
            k = self.key_norms[branch](key)
            gate = context_gate(q, k, normalize=False)  # [B,T,1]
            gated.append(gate * value)
        u = torch.stack(gated, dim=2)  # [B,T,M,D]

        if conv_state is not None:
            if (conv_state.ndim != 4 or conv_state.shape[0] != b
                    or conv_state.shape[2:] != (m, d)
                    or conv_state.shape[1] > self.history_size):
                raise ValueError("invalid bounded conv_state shape")
            window = torch.cat([conv_state, u], dim=1)
        else:
            window = u
        out = u + self.short_conv(window)[:, -t:]
        return (out, window) if return_state else out

    def extra_repr(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"layer_id={self.layer_id}, dmem={self.dmem}, "
            f"slots={self.memory.total_slots}, branches={self.cfg.hc_mult}"
        )


def build_engram_modules(cfg: EngramConfig) -> nn.ModuleDict:
    """Build all layers with one shared mapping (the paper's configuration)."""
    mapping = build_engram_mapping(cfg)
    return nn.ModuleDict(
        {str(layer_id): Engram(cfg, layer_id, mapping=mapping) for layer_id in cfg.layer_ids_}
    )
