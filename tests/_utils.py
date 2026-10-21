"""Shared test helpers."""

from __future__ import annotations

import torch

from tinyengram import EngramConfig, TinyLMConfig


def assert_close(name: str, actual: torch.Tensor, expected: torch.Tensor, atol: float = 1e-5) -> None:
    if actual.shape != expected.shape:
        raise AssertionError(f"{name}: 形状 {actual.shape} != {expected.shape}")
    diff = (actual.float() - expected.float()).abs().max().item()
    if diff > atol:
        raise AssertionError(f"{name}: 最大误差 {diff:.3e} > {atol:.1e}")


def tiny_engram_config(
    *,
    vocab_size: int = 64,
    hidden_size: int = 32,
    hc_mult: int = 2,
    layer_ids: tuple[int, ...] = (1,),
    slots_per_head: int = 64,
    seed: int = 0,
) -> EngramConfig:
    return EngramConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        hc_mult=hc_mult,
        max_ngram_size=3,
        n_head_per_ngram=2,
        n_embed_per_ngram=16,
        slots_per_head=slots_per_head,
        layer_ids=layer_ids,
        pad_id=0,
        seed=seed,
        conv_kernel_size=4,
    )


def tiny_lm_config(*, with_engram: bool = True, vocab_size: int = 64, max_seq: int = 32) -> TinyLMConfig:
    engram = tiny_engram_config(vocab_size=vocab_size) if with_engram else None
    return TinyLMConfig(
        vocab_size=vocab_size,
        hidden_size=32,
        n_layers=2,
        n_heads=2,
        max_seq=max_seq,
        ffn_mult=2,
        hc_mult=2,
        engram_layers=(1,) if with_engram else (),
        engram=engram,
    )


def make_ids(batch: int, seq: int, vocab: int = 64, seed: int = 0) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randint(1, vocab, (batch, seq), generator=gen)
