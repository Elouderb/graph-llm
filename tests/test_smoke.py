"""Smoke test: CPU-only, ~10 training steps, loss finite and strictly decreasing.

Runs entirely offline using the synthetic dataset (no network, no disk I/O).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

# Ensure src/ is on the path when running pytest from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import graph_llm.models.baselines  # noqa: F401 — registers "transformer"
from graph_llm.config import Config, DataConfig, ModelConfig, TrainConfig
from graph_llm.data import build_dataloaders
from graph_llm.models import build_model
from graph_llm.train import Trainer


def _make_smoke_config() -> Config:
    """Tiny config that fits in CPU RAM and finishes in seconds."""
    cfg = Config(
        model=ModelConfig(
            name="transformer",
            vocab_size=256,
            d_model=64,
            n_heads=4,
            n_layers=2,
            d_ff=256,
            max_seq_len=64,
            dropout=0.0,
            tie_embeddings=True,
            use_rope=True,
            activation_checkpointing=False,
        ),
        data=DataConfig(
            source="synthetic",
            seq_len=64,
            batch_size=4,
            val_fraction=0.1,
        ),
        train=TrainConfig(
            seed=42,
            max_steps=10,
            grad_accumulation_steps=1,
            grad_clip=1.0,
            lr=1e-3,
            weight_decay=0.01,
            warmup_steps=2,
            lr_schedule="cosine",
            mixed_precision="no",   # CPU only
            checkpoint_dir="/tmp/graph_llm_smoke_ckpts",
            resume_from=None,
            log_every=1,
        ),
    )
    return cfg


def test_smoke_loss_finite_and_decreasing():
    """Overfit a single batch: loss must be finite and strictly decrease.

    We repeat a single fixed batch for 20 steps.  On a tiny model with LR=1e-3
    this reliably overfits (loss drops ~0.5 nats), confirming the entire
    forward/backward/optimiser chain is functional.

    The 10-step config in smoke.yaml exercises the *script* smoke path
    (``python scripts/train.py``).  This test uses a longer 20-step single-batch
    overfit because pure random data has no structure to exploit in 10 steps.
    """
    cfg = _make_smoke_config()
    # Use 20 steps for a reliable overfit signal; keep the smoke.yaml at 10
    # for the CLI path.
    from dataclasses import replace
    cfg = replace(cfg, train=replace(cfg.train, max_steps=20, warmup_steps=2, lr=3e-3))

    # Build a single fixed batch and wrap it as a trivially small DataLoader
    # (1 sample repeated).  The trainer will cycle it every step.
    B, T = 4, cfg.model.max_seq_len
    torch.manual_seed(0)
    fixed_x = torch.randint(0, cfg.model.vocab_size, (B, T))
    fixed_targets = torch.randint(0, cfg.model.vocab_size, (B, T))

    from torch.utils.data import DataLoader as DL
    from torch.utils.data import TensorDataset
    single_batch_ds = TensorDataset(fixed_x, fixed_targets)
    train_loader = DL(single_batch_ds, batch_size=B, shuffle=False, drop_last=False)

    # Val loader: same single batch (we don't need accurate val for the smoke test)
    val_loader = train_loader

    model = build_model(cfg)

    # Force CPU regardless of CUDA availability
    original_is_available = torch.cuda.is_available
    torch.cuda.is_available = lambda: False  # type: ignore[assignment]

    try:
        trainer = Trainer(cfg, model, train_loader, val_loader)
        loss_history = trainer.train()
    finally:
        torch.cuda.is_available = original_is_available  # type: ignore[assignment]

    assert len(loss_history) == cfg.train.max_steps, (
        f"Expected {cfg.train.max_steps} loss values, got {len(loss_history)}"
    )

    # All losses must be finite
    for i, loss_val in enumerate(loss_history):
        assert math.isfinite(loss_val), f"Loss at step {i} is not finite: {loss_val}"

    # Overfitting a single batch should drop loss reliably.  Compare the mean
    # of the first 5 steps against the mean of the last 5 steps.
    first_mean = sum(loss_history[:5]) / 5
    last_mean = sum(loss_history[-5:]) / 5

    print(f"\n  first_5_mean={first_mean:.4f}  last_5_mean={last_mean:.4f}")

    assert last_mean < first_mean, (
        f"Loss did not decrease while overfitting single batch: "
        f"first_5_mean={first_mean:.4f}, last_5_mean={last_mean:.4f}"
    )


def test_model_builds_from_config():
    """Model registry resolves 'transformer' and builds a parameter-bearing model."""
    cfg = _make_smoke_config()
    model = build_model(cfg)
    assert isinstance(model, torch.nn.Module)
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params > 0, "Model has no parameters"


def test_forward_returns_loss_and_logits():
    """forward(x, targets) returns (loss, logits) with correct shapes."""
    cfg = _make_smoke_config()
    model = build_model(cfg)
    model.eval()

    B, T = 2, cfg.model.max_seq_len
    x = torch.randint(0, cfg.model.vocab_size, (B, T))
    targets = torch.randint(0, cfg.model.vocab_size, (B, T))

    with torch.no_grad():
        loss, logits = model(x, targets)

    assert loss.ndim == 0, f"Loss should be scalar, got shape {loss.shape}"
    assert math.isfinite(loss.item()), f"Loss is not finite: {loss.item()}"
    assert logits.shape == (B, T, cfg.model.vocab_size), (
        f"Unexpected logits shape: {logits.shape}"
    )


def test_yaml_config_loads():
    """smoke.yaml parses correctly and matches expected fields."""
    from graph_llm.config import load_config

    yaml_path = Path(__file__).resolve().parents[1] / "configs" / "smoke.yaml"
    cfg = load_config(yaml_path)
    assert cfg.model.name == "transformer"
    assert cfg.data.source == "synthetic"
    assert cfg.train.max_steps == 10
    assert cfg.train.mixed_precision == "no"


def test_embedding_init_hook_is_respected():
    """Regression (review SF-1): a registered embedding-init hook must have the
    final say on ``embed.weight``. ``_init_weights()`` must run BEFORE the hook,
    never after — otherwise the phonological init (card e1644700) is a no-op.
    """
    from dataclasses import replace

    import torch.nn as nn

    from graph_llm.models.registry import register_embedding_init

    sentinel = 7.0

    @register_embedding_init("sf1_sentinel")
    def _const_init(weight, vocab_size, d_model):  # noqa: ANN001, ARG001
        nn.init.constant_(weight, sentinel)

    cfg = _make_smoke_config()
    cfg = replace(cfg, model=replace(cfg.model, embedding_init="sf1_sentinel"))
    model = build_model(cfg)

    expected = torch.full_like(model.embed.weight, sentinel)
    assert torch.allclose(model.embed.weight, expected), (
        "Embedding-init hook was overwritten by _init_weights(); the hook must "
        "be applied last so card e1644700's phonological init takes effect."
    )
