"""Training CLI.

Usage::

    python scripts/train.py --config configs/smoke.yaml
    python scripts/train.py --config configs/smoke.yaml --steps 500
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as ``python scripts/train.py`` from the repo root when the
# package is installed in editable mode.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import graph_llm.models.baselines  # noqa: F401 — registers "transformer" and "mamba"
from graph_llm.config import load_config
from graph_llm.data import build_dataloaders
from graph_llm.models import build_model  # importing the package also registers "bilinear_lm"
from graph_llm.tokenizer.phonological_init import apply_embedding_init
from graph_llm.train import Trainer
from graph_llm.utils.logging import get_logger, setup_logging

_log = get_logger("scripts.train")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a graph_llm model")
    p.add_argument("--config", required=True, help="Path to YAML config file")
    p.add_argument("--steps", type=int, default=None, help="Override max_steps")
    p.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging()

    overrides: dict = {}
    if args.steps is not None:
        overrides = {"train": {"max_steps": args.steps}}
    if args.resume is not None:
        overrides.setdefault("train", {})["resume_from"] = args.resume

    cfg = load_config(args.config, overrides or None)

    _log.info("Config: model=%s, steps=%d, device=auto", cfg.model.name, cfg.train.max_steps)

    train_loader, val_loader = build_dataloaders(
        cfg.data,
        vocab_size=cfg.model.vocab_size,
        seed=cfg.train.seed,
    )
    model = build_model(cfg)
    apply_embedding_init(model, cfg)  # no-op unless cfg.model.embedding_init="phonological"
    _log.info(
        "Model: %s, params=%s",
        cfg.model.name,
        f"{model.num_parameters():,}" if hasattr(model, "num_parameters") else "?",
    )

    trainer = Trainer(cfg, model, train_loader, val_loader)
    loss_history = trainer.train()

    if loss_history:
        _log.info(
            "Training done. loss_start=%.4f  loss_end=%.4f",
            loss_history[0],
            loss_history[-1],
        )
        if loss_history[-1] >= loss_history[0]:
            _log.warning(
                "Loss did not decrease over this run (start=%.4f, end=%.4f) — "
                "expected for very short or noisy runs; not treated as a failure.",
                loss_history[0],
                loss_history[-1],
            )

    # Save final checkpoint
    ckpt_path = trainer.save_checkpoint(tag="final")
    _log.info("Checkpoint saved: %s", ckpt_path)

    val_loss = trainer.evaluate()
    _log.info("Val loss: %.4f", val_loss)


if __name__ == "__main__":
    main()
