"""Training utilities.

Two details from the paper are implemented here because they are easy to miss
and matter for stability:

1. **Separate optimizer / LR for the memory table.**  The 5.7B embedding rows
   are updated with Adam at ``5x`` the backbone LR and **zero weight decay**
   (Table 5).  Sparse rows receive very few updates; decaying them towards zero
   between occurrences destroys exactly the long-tail knowledge Engram exists
   to store.

2. **Gradient clipping and warmup.**  A 5x LR on a freshly-initialized table
   produces large early gradients; warmup is not optional.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from .data import EntityCorpus

__all__ = ["TrainConfig", "make_param_groups", "Trainer", "train_step", "evaluate"]


@dataclass
class TrainConfig:
    lr: float = 3e-3
    weight_decay: float = 0.1
    engram_lr_mult: float = 5.0
    grad_clip: float = 1.0
    betas: tuple[float, float] = (0.9, 0.95)
    warmup_steps: int = 20
    total_steps: int = 500
    min_lr_ratio: float = 0.1


def _is_engram_embedding(name: str) -> bool:
    return name.endswith("memory.table.embedding.weight")


def make_param_groups(
    model: nn.Module,
    lr: float,
    weight_decay: float = 0.1,
    engram_lr_mult: float = 5.0,
) -> list[dict]:
    """Split parameters into embedding / decay / no-decay groups."""
    raise NotImplementedError("TODO(L5): 请实现本关卡的核心逻辑；先读 docs/ 对应讲义")


def _lr_lambda(cfg: TrainConfig):
    def fn(step: int) -> float:
        if step < cfg.warmup_steps:
            return (step + 1) / max(1, cfg.warmup_steps)
        progress = (step - cfg.warmup_steps) / max(1, cfg.total_steps - cfg.warmup_steps)
        progress = min(1.0, max(0.0, progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return cfg.min_lr_ratio + (1 - cfg.min_lr_ratio) * cosine

    return fn


class Trainer:
    def __init__(self, model: nn.Module, cfg: Optional[TrainConfig] = None, device: str = "cpu"):
        self.model = model.to(device)
        self.cfg = cfg or TrainConfig()
        self.device = device
        groups = make_param_groups(
            model,
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
            engram_lr_mult=self.cfg.engram_lr_mult,
        )
        self.optimizer = AdamW(groups, betas=self.cfg.betas)
        self.scheduler = LambdaLR(self.optimizer, _lr_lambda(self.cfg))
        self.step_count = 0
        self.history: dict[str, list[float]] = {"loss": [], "lr": []}

    def step(self, input_ids: torch.Tensor, targets: torch.Tensor) -> float:
        raise NotImplementedError("TODO(L5): 请实现本关卡的核心逻辑；先读 docs/ 对应讲义")

    def fit(
        self,
        corpus: EntityCorpus,
        *,
        steps: int | None = None,
        batch_size: int = 32,
        split: str = "train",
        seed: int = 0,
        log_every: int = 50,
    ) -> dict[str, list[float]]:
        raise NotImplementedError("TODO(L5): 请实现本关卡的核心逻辑；先读 docs/ 对应讲义")

    @torch.no_grad()
    def evaluate(
        self,
        corpus: EntityCorpus,
        *,
        batch_size: int = 32,
        batches: int = 8,
        split: str = "train",
        seed: int = 1234,
    ) -> dict[str, float]:
        raise NotImplementedError("TODO(L5): 请实现本关卡的核心逻辑；先读 docs/ 对应讲义")


def train_step(
    model: nn.Module,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    grad_clip: float = 1.0,
) -> float:
    raise NotImplementedError("TODO(L5): 请实现本关卡的核心逻辑；先读 docs/ 对应讲义")


@torch.no_grad()
def evaluate(
    model: nn.Module,
    corpus: EntityCorpus,
    *,
    batch_size: int = 32,
    batches: int = 4,
    split: str = "train",
    seed: int = 0,
) -> dict[str, float]:
    raise NotImplementedError("TODO(L5): 请实现本关卡的核心逻辑；先读 docs/ 对应讲义")
