"""SegmentedTrainer periodic unified-eval-report hook (card 69776c3e, piece 3).

Confirms the "callable from the training loop periodically" half of the unified
eval harness: ``cfg.train.eval_every`` writes a JSON report to ``eval_run_dir``
every N steps, defaults to off, and does not otherwise perturb training (the model
ends the run in ``.train()`` mode; loss still decreases as it does without the
hook -- see test_segmented_training.py for that invariant on its own).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import graph_llm.models  # noqa: F401 — registers "delta_memory_lm"
from graph_llm.config import Config, DataConfig, ModelConfig, TrainConfig
from graph_llm.models import build_model
from graph_llm.train.segmented import SegmentedTrainer


def _mem_cfg(**overrides: Any) -> ModelConfig:
    base: dict[str, Any] = {
        "name": "delta_memory_lm",
        "vocab_size": 64,
        "d_model": 16,
        "delta_layers": 1,
        "delta_n_heads": 1,
        "delta_head_k_dim": 8,
        "delta_head_v_dim": 8,
        "delta_dropout": 0.0,
        "dropout": 0.0,
        "max_seq_len": 32,
        "front_end": "none",
        "segment_len": 8,
        "bptt_window": 2,
    }
    base.update(overrides)
    return ModelConfig(**base)


def _stream(period: int = 16, vocab: int = 64, repeats: int = 200) -> np.ndarray:
    return np.tile(np.arange(period, dtype=np.int64) % vocab, repeats)


def test_eval_hook_off_by_default_writes_nothing(tmp_path) -> None:
    m = _mem_cfg()
    cfg = Config(
        model=m,
        data=DataConfig(source="synthetic", seq_len=m.segment_len, batch_size=4),
        train=TrainConfig(max_steps=4, log_every=1000),  # eval_every defaults to 0
    )
    assert cfg.train.eval_every == 0
    model = build_model(cfg)
    trainer = SegmentedTrainer(cfg, model, _stream(), device=None)
    trainer.train()
    assert not list(tmp_path.glob("*.json"))


def test_eval_hook_writes_periodic_reports(tmp_path) -> None:
    m = _mem_cfg()
    run_dir = tmp_path / "eval_reports"
    cfg = Config(
        model=m,
        data=DataConfig(source="synthetic", seq_len=m.segment_len, batch_size=4),
        train=TrainConfig(
            max_steps=4, log_every=1000, eval_every=2, eval_run_dir=str(run_dir)
        ),
    )
    model = build_model(cfg)
    tokens = _stream()
    val_tokens = tokens[: len(tokens) // 4]
    trainer = SegmentedTrainer(cfg, model, tokens, device=None, val_tokens=val_tokens)
    trainer.train()

    written = sorted(run_dir.glob("eval_step*.json"))
    assert [p.name for p in written] == ["eval_step000002.json", "eval_step000004.json"]
    report = json.loads(written[0].read_text())
    assert report["step"] == 2
    assert report["tandem_enabled"] is False
    assert isinstance(report["val_bpb"], float)
    assert "cross_segment_retrieval" in report
    assert report["reasoning_depth_accuracy"] is None
    # Training resumes/ends in .train() mode -- the hook must not leave it in eval().
    assert model.training is True
