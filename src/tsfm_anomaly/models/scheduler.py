"""Linear warm-up followed by cosine decay, stepped once per optimizer step."""

from __future__ import annotations

import math
from typing import Mapping

import torch


class LinearWarmupCosineScheduler:
    """
    Step-wise linear warm-up followed by cosine decay.

    The official research configuration uses this mechanism. For small
    datasets, warm-up steps are capped by a fraction of total steps so that
    the complete run is not trapped in warm-up.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        total_steps: int,
        warmup_steps: int,
        initial_lr: float,
        warmup_lr: float,
        min_lr: float,
    ):
        if total_steps < 1:
            raise ValueError("total_steps must be positive.")
        self.optimizer = optimizer
        self.total_steps = total_steps
        self.warmup_steps = max(0, min(warmup_steps, total_steps - 1))
        self.initial_lr = initial_lr
        self.warmup_lr = warmup_lr
        self.min_lr = min_lr
        self.step_number = 0
        self._set_lr(self.lr_at_step(0))

    def lr_at_step(self, step: int) -> float:
        if self.warmup_steps > 0 and step < self.warmup_steps:
            progress = step / max(self.warmup_steps, 1)
            return self.warmup_lr + progress * (self.initial_lr - self.warmup_lr)

        cosine_steps = max(self.total_steps - self.warmup_steps, 1)
        cosine_step = max(step - self.warmup_steps, 0)
        progress = min(cosine_step / cosine_steps, 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr + cosine * (self.initial_lr - self.min_lr)

    def _set_lr(self, value: float) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = value

    def step(self) -> None:
        self.step_number += 1
        self._set_lr(self.lr_at_step(self.step_number))

    def state_dict(self) -> dict[str, int]:
        return {"step_number": self.step_number}

    def load_state_dict(self, state: Mapping[str, int]) -> None:
        self.step_number = int(state["step_number"])
        self._set_lr(self.lr_at_step(self.step_number))
