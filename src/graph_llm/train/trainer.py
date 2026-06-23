"""Model-agnostic Trainer with the 12 GB VRAM toolkit.

The Trainer interacts with the model only through the contract::

    loss, logits = model(x, targets)

No model-specific branches exist here.  Adding a new registered model
requires zero trainer changes.

12 GB toolkit (all config-toggleable):
* Mixed precision: bf16 / fp16 autocast + GradScaler
* Gradient accumulation
* Activation / gradient checkpointing hook (set on the model before training)
* Gradient clipping
* Cosine LR + warmup
* Checkpoint save / resume
* Deterministic seeding
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader

from graph_llm.train.optim import build_optimizer, build_scheduler
from graph_llm.utils.logging import get_logger, log_metrics, setup_logging
from graph_llm.utils.seed import seed_everything

if TYPE_CHECKING:
    from graph_llm.config import Config

_log = get_logger("trainer")


@dataclass
class TrainState:
    """Mutable training state (serialised to checkpoint)."""

    step: int = 0
    best_val_loss: float = float("inf")


class Trainer:
    """Model-agnostic training loop.

    Args:
        cfg:          Full :class:`~graph_llm.config.Config`.
        model:        Any ``nn.Module`` whose ``forward`` returns ``(loss, logits)``.
        train_loader: Training :class:`~torch.utils.data.DataLoader`.
        val_loader:   Validation :class:`~torch.utils.data.DataLoader`.
    """

    def __init__(
        self,
        cfg: Config,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> None:
        self.cfg = cfg
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

        setup_logging()
        seed_everything(cfg.train.seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        self.optimizer = build_optimizer(self.model, cfg.train)
        self.scheduler = build_scheduler(self.optimizer, cfg.train)
        self.state = TrainState()

        # Mixed precision
        mp = cfg.train.mixed_precision.lower()
        self._use_amp = mp in ("fp16", "bf16") and self.device.type == "cuda"
        self._amp_dtype = torch.bfloat16 if mp == "bf16" else torch.float16
        self._scaler: GradScaler | None = (
            GradScaler("cuda") if (self._use_amp and mp == "fp16") else None
        )

        # Gradient accumulation
        self._accum_steps = max(1, cfg.train.grad_accumulation_steps)

        # Resume from checkpoint if specified
        if cfg.train.resume_from:
            self._load_checkpoint(cfg.train.resume_from)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self) -> list[float]:
        """Run the training loop for ``cfg.train.max_steps`` steps.

        Returns:
            A list of per-step losses (one entry per *optimiser* step, i.e.
            after gradient accumulation is applied).
        """
        cfg_t = self.cfg.train
        loss_history: list[float] = []

        self.model.train()
        self.optimizer.zero_grad()

        data_iter: Iterator = iter(self.train_loader)
        micro_step = 0

        while self.state.step < cfg_t.max_steps:
            # Fetch next batch (cycle the loader)
            try:
                x, targets = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_loader)
                x, targets = next(data_iter)

            x = x.to(self.device)
            targets = targets.to(self.device)

            # Forward pass (micro-step)
            loss_val = self._forward_step(x, targets)
            micro_step += 1

            if micro_step < self._accum_steps:
                continue

            # Optimiser step
            micro_step = 0
            self._clip_and_step()
            self.scheduler.step()
            self.optimizer.zero_grad()
            self.state.step += 1

            loss_history.append(loss_val)

            if self.state.step % cfg_t.log_every == 0:
                lr = self.scheduler.get_last_lr()[0]
                log_metrics(self.state.step, loss=round(loss_val, 4), lr=round(lr, 6))

        return loss_history

    def evaluate(self) -> float:
        """Run one pass over the validation set and return average loss."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        with torch.no_grad():
            for x, targets in self.val_loader:
                x = x.to(self.device)
                targets = targets.to(self.device)
                with self._autocast():
                    loss, _ = self.model(x, targets)
                total_loss += loss.item()
                n_batches += 1
        self.model.train()
        return total_loss / max(n_batches, 1)

    def save_checkpoint(self, tag: str = "") -> Path:
        """Save model, optimiser, scheduler, and training state to disk.

        Args:
            tag: Optional suffix for the checkpoint filename.

        Returns:
            Path to the saved checkpoint file.
        """
        ckpt_dir = Path(self.cfg.train.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        filename = f"ckpt_step{self.state.step:06d}{('_' + tag) if tag else ''}.pt"
        path = ckpt_dir / filename
        torch.save(
            {
                "step": self.state.step,
                "best_val_loss": self.state.best_val_loss,
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "scheduler_state": self.scheduler.state_dict(),
                "scaler_state": self._scaler.state_dict() if self._scaler else None,
                "config": self.cfg,
            },
            path,
        )
        _log.info("Saved checkpoint: %s", path)
        return path

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _autocast(self):
        if self._use_amp:
            return torch.autocast(device_type="cuda", dtype=self._amp_dtype)
        return torch.autocast(device_type="cpu", enabled=False)

    def _forward_step(self, x: torch.Tensor, targets: torch.Tensor) -> float:
        """Run a single forward+backward micro-step.

        Scales loss by ``1/accum_steps`` for correct gradient accumulation.
        """
        scale = 1.0 / self._accum_steps
        with self._autocast():
            loss, _ = self.model(x, targets)
            scaled_loss = loss * scale

        if self._scaler is not None:
            self._scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        return loss.item()

    def _clip_and_step(self) -> None:
        """Unscale, clip gradients, and call optimizer.step()."""
        clip = self.cfg.train.grad_clip
        if self._scaler is not None:
            self._scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), clip)
            self._scaler.step(self.optimizer)
            self._scaler.update()
        else:
            nn.utils.clip_grad_norm_(self.model.parameters(), clip)
            self.optimizer.step()

    def _load_checkpoint(self, path: str) -> None:
        """Resume training state from a checkpoint file."""
        # weights_only=False is explicit: the checkpoint includes the Config
        # dataclass object, which pickle-deserialises outside the tensor-only
        # safe path.  Only load checkpoints you wrote yourself.
        # TODO (later card): migrate to a split checkpoint format — tensors in
        # a .safetensors file + config in a separate .yaml — so model weights
        # can be loaded with weights_only=True.
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.scheduler.load_state_dict(ckpt["scheduler_state"])
        if self._scaler is not None and ckpt.get("scaler_state"):
            self._scaler.load_state_dict(ckpt["scaler_state"])
        self.state.step = ckpt["step"]
        self.state.best_val_loss = ckpt["best_val_loss"]
        _log.info("Resumed from checkpoint: %s (step=%d)", path, self.state.step)
