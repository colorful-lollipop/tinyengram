"""A small GPT backbone with optional Engram layers.

This is intentionally *not* a full DeepSeek-V3 clone: no MLA, no MoE, no mHC
manifold projection.  Those are orthogonal to understanding conditional
memory, and the sister project ``mHc/`` covers the last one in depth.  What we
keep is the part Engram interacts with:

* a multi-branch residual stream (``hc_mult`` parallel streams);
* pre-norm blocks with causal attention and a SwiGLU FFN;
* Engram inserted *before* attention at selected layers, exactly as in
  Figure 1;
* RoPE, because n-gram memory is position-local but attention is not.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import TinyLMConfig
from .engram import Engram, build_engram_mapping
from .gating import RMSNorm

__all__ = [
    "RotaryEmbedding",
    "apply_rope",
    "CausalSelfAttention",
    "SwiGLU",
    "Block",
    "TinyLM",
    "count_parameters",
]

KVCache = list[tuple[torch.Tensor, torch.Tensor]]


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_seq: int, theta: float = 10000.0) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("RoPE 需要偶数 head_dim")
        self.head_dim = head_dim
        self.max_seq = max_seq
        self.theta = theta
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self, seq_len: int, *, offset: int = 0, device=None, dtype=None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if offset + seq_len > self.max_seq:
            raise ValueError(
                f"位置 {offset + seq_len} 超过 max_seq={self.max_seq}；"
                "推理时请增大 max_seq 或缩短上下文"
            )
        pos = torch.arange(offset, offset + seq_len, device=device or self.inv_freq.device)
        freqs = torch.outer(pos.float(), self.inv_freq.float())
        emb = freqs.repeat_interleave(2, dim=-1)
        cos = emb.cos().to(dtype or torch.float32)[None, None, :, :]
        sin = emb.sin().to(dtype or torch.float32)[None, None, :, :]
        return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate pairs ``(x0, x1), (x2, x3), ...`` of the last dimension."""
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    rotated = torch.stack((-x_odd, x_even), dim=-1).flatten(-2)
    return x * cos + rotated * sin


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: TinyLMConfig) -> None:
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.dropout = cfg.dropout
        self.qkv = nn.Linear(cfg.hidden_size, 3 * cfg.hidden_size, bias=False)
        self.out = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
        self.rope = RotaryEmbedding(cfg.head_dim, cfg.max_seq, cfg.rope_theta)

    def forward(
        self,
        x: torch.Tensor,
        cache: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        position_offset: int = 0,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Args: ``[B, T, M, D]``. Returns ``([B,T,M,D], (k, v))``."""
        b, t, m, d = x.shape
        h = self.n_heads
        hd = self.head_dim
        xb = x.permute(0, 2, 1, 3).reshape(b * m, t, d)
        q, k, v = self.qkv(xb).split(d, dim=-1)
        q = q.view(b * m, t, h, hd).transpose(1, 2)
        k = k.view(b * m, t, h, hd).transpose(1, 2)
        v = v.view(b * m, t, h, hd).transpose(1, 2)

        cos, sin = self.rope(t, offset=position_offset, device=x.device, dtype=q.dtype)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        past_len = 0
        if cache is not None:
            past_k, past_v = cache
            past_len = past_k.shape[-2]
            k = torch.cat([past_k, k], dim=-2)
            v = torch.cat([past_v, v], dim=-2)

        # Without a cache and T>1, use the fused causal mask.  With a cache
        # and T==1, every cached position is visible.  For chunked prefill
        # (cache + T>1) we need an explicit offset mask, otherwise the new
        # queries could see later tokens inside the same chunk.
        if past_len > 0 and t > 1:
            q_pos = torch.arange(past_len, past_len + t, device=x.device)
            k_pos = torch.arange(past_len + t, device=x.device)
            attn_mask = k_pos.unsqueeze(0) <= q_pos.unsqueeze(1)
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=self.dropout if self.training else 0.0,
            )
        else:
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=past_len == 0 and t > 1,
            )
        y = y.transpose(1, 2).reshape(b * m, t, d)
        y = self.out(y).reshape(b, m, t, d).permute(0, 2, 1, 3)
        return y, (k, v)


class SwiGLU(nn.Module):
    def __init__(self, dim: int, mult: int = 4) -> None:
        super().__init__()
        hidden = int(dim * mult)
        # SwiGLU has 3 matrices; 2/3 of the width keeps FLOPs comparable to a
        # 4x MLP, following LLaMA.
        hidden = int(hidden * 2 / 3)
        self.w12 = nn.Linear(dim, 2 * hidden, bias=False)
        self.w3 = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.w12(x).chunk(2, dim=-1)
        return self.w3(F.silu(a) * b)


class Block(nn.Module):
    def __init__(self, cfg: TinyLMConfig, layer_id: int, mapping=None) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.hc_mult = cfg.hc_mult
        self.norm1 = nn.ModuleList([RMSNorm(cfg.hidden_size) for _ in range(cfg.hc_mult)])
        self.norm2 = nn.ModuleList([RMSNorm(cfg.hidden_size) for _ in range(cfg.hc_mult)])
        self.attn = CausalSelfAttention(cfg)
        self.ffn = SwiGLU(cfg.hidden_size, cfg.ffn_mult)
        self.engram: Optional[Engram] = None
        if layer_id in cfg.engram_layers_:
            assert cfg.engram is not None
            self.engram = Engram(cfg.engram, layer_id, mapping=mapping)

    def forward(
        self,
        hidden: torch.Tensor,
        input_ids: torch.Tensor,
        cache: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        position_offset: int = 0,
        context_ids: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if self.engram is not None:
            hidden = hidden + self.engram(hidden, input_ids, context_ids=context_ids)

        normed = torch.stack(
            [self.norm1[i](hidden[:, :, i, :]) for i in range(self.hc_mult)], dim=2
        )
        attn_out, new_cache = self.attn(normed, cache=cache, position_offset=position_offset)
        hidden = hidden + attn_out

        b, t, m, d = hidden.shape
        normed = torch.stack(
            [self.norm2[i](hidden[:, :, i, :]) for i in range(self.hc_mult)], dim=2
        )
        ffn_out = (
            self.ffn(normed.permute(0, 2, 1, 3).reshape(b * m, t, d))
            .reshape(b, m, t, d)
            .permute(0, 2, 1, 3)
        )
        hidden = hidden + ffn_out
        return hidden, new_cache


class TinyLM(nn.Module):
    def __init__(self, cfg: TinyLMConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        mapping = build_engram_mapping(cfg.engram) if cfg.engram is not None else None
        self.blocks = nn.ModuleList(
            [Block(cfg, i, mapping=mapping) for i in range(cfg.n_layers)]
        )
        self.norm_f = RMSNorm(cfg.hidden_size)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        if cfg.tie_weights:
            self.lm_head.weight = self.tok_emb.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        caches: Optional[KVCache] = None,
        position_offset: int = 0,
        context_ids: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], KVCache]:
        """Returns ``(logits, loss, new_caches)``."""
        b, t = input_ids.shape
        hidden = self.tok_emb(input_ids).unsqueeze(2).expand(-1, -1, self.cfg.hc_mult, -1)
        new_caches: KVCache = []
        for i, block in enumerate(self.blocks):
            cache = caches[i] if caches is not None else None
            hidden, kv = block(
                hidden,
                input_ids,
                cache=cache,
                position_offset=position_offset,
                context_ids=context_ids,
            )
            new_caches.append(kv)
        hidden = self.norm_f(hidden).mean(dim=2)
        logits = self.lm_head(hidden)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
            )
        return logits, loss, new_caches

    def engram_modules(self) -> list[Engram]:
        return [b.engram for b in self.blocks if b.engram is not None]

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def num_parameters_excluding_engram(self) -> int:
        engram_ids = {id(p) for module in self.engram_modules() for p in module.parameters()}
        return sum(p.numel() for p in self.parameters() if id(p) not in engram_ids)


def count_parameters(module: nn.Module, *, trainable_only: bool = True) -> int:
    ps = module.parameters()
    if trainable_only:
        return sum(p.numel() for p in ps if p.requires_grad)
    return sum(p.numel() for p in ps)
