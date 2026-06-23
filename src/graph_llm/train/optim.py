"""Optimiser and LR-schedule construction.

All hyperparameters come from :class:`~graph_llm.config.TrainConfig`.
No hardcoded values.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

if TYPE_CHECKING:
    from graph_llm.config import TrainConfig


def build_optimizer(model: nn.Module, cfg: TrainConfig) -> AdamW:
    """Build an AdamW optimiser with weight decay applied to non-bias/norm params.

    Parameters that are 1-D (biases, LayerNorm/RMSNorm weights) are placed in
    a zero-weight-decay group to match GPT-style training practice.

    Args:
        model: The model whose parameters to optimise.
        cfg: :class:`~graph_llm.config.TrainConfig` with ``lr`` and
            ``weight_decay``.

    Returns:
        Configured :class:`~torch.optim.AdamW` instance.
    """
    decay_params = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    no_decay_params = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    param_groups = [
        {"params": decay_params, "weight_decay": cfg.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    return AdamW(param_groups, lr=cfg.lr, betas=(0.9, 0.95), eps=1e-8)


def build_scheduler(optimizer: AdamW, cfg: TrainConfig) -> LambdaLR:
    """Build a cosine-with-warmup or constant LR schedule.

    Args:
        optimizer: The optimiser to attach the schedule to.
        cfg: :class:`~graph_llm.config.TrainConfig` with ``lr_schedule``,
            ``warmup_steps``, and ``max_steps``.

    Returns:
        A :class:`~torch.optim.lr_scheduler.LambdaLR` that tracks steps
        (call ``scheduler.step()`` once per *optimiser step*, not per batch).
    """
    warmup = cfg.warmup_steps
    total = cfg.max_steps
    schedule = cfg.lr_schedule.lower()

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return float(step) / max(1, warmup)
        if schedule == "constant":
            return 1.0
        # cosine decay from 1 → 0
        progress = float(step - warmup) / max(1, total - warmup)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)
