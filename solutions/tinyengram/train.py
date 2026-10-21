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
from dataclasses import dataclass
from typing import Optional

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
    embedding, decay, no_decay = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if _is_engram_embedding(name):
            embedding.append(p)
        elif p.dim() >= 2:
            decay.append(p)
        else:
            no_decay.append(p)
    groups = [
        {"params": embedding, "lr": lr * engram_lr_mult, "weight_decay": 0.0, "name": "engram_embedding"},
        {"params": decay, "lr": lr, "weight_decay": weight_decay, "name": "decay"},
        {"params": no_decay, "lr": lr, "weight_decay": 0.0, "name": "no_decay"},
    ]
    return [g for g in groups if g["params"]]


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
        self.model.train()
        input_ids = input_ids.to(self.device)
        targets = targets.to(self.device)
        _, loss, _ = self.model(input_ids, targets=targets)
        assert loss is not None
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
        self.optimizer.step()
        self.scheduler.step()
        self.step_count += 1
        value = float(loss.detach())
        self.history["loss"].append(value)
        self.history["lr"].append(self.optimizer.param_groups[0]["lr"])
        return value

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
        steps = self.cfg.total_steps if steps is None else steps
        gen = torch.Generator().manual_seed(seed)
        for i in range(steps):
            ids, targets = corpus.sample(batch_size, generator=gen, split=split)
            loss = self.step(ids, targets)
            if log_every and (i + 1) % log_every == 0:
                print(f"step {i + 1:4d}  loss {loss:.4f}  lr {self.history['lr'][-1]:.2e}")
        return self.history

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
        self.model.eval()
        gen = torch.Generator().manual_seed(seed)
        losses, correct, total = [], 0, 0
        for _ in range(batches):
            ids, targets = corpus.sample(batch_size, generator=gen, split=split)
            logits, loss, _ = self.model(
                ids.to(self.device), targets=targets.to(self.device)
            )
            losses.append(float(loss))
            mask = targets.reshape(-1) != -100
            pred = logits.reshape(-1, logits.size(-1)).argmax(-1)
            correct += int((pred[mask] == targets.reshape(-1)[mask]).sum())
            total += int(mask.sum())
        return {
            "loss": sum(losses) / len(losses),
            "token_acc": correct / max(1, total),
        }


def train_step(
    model: nn.Module,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    grad_clip: float = 1.0,
) -> float:
    model.train()
    _, loss, _ = model(input_ids, targets=targets)
    assert loss is not None
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    if grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    return float(loss.detach())


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
    model.eval()
    gen = torch.Generator().manual_seed(seed)
    losses = []
    for _ in range(batches):
        ids, targets = corpus.sample(batch_size, generator=gen, split=split)
        _, loss, _ = model(ids, targets=targets)
        losses.append(float(loss))
    return {"loss": sum(losses) / len(losses)}
