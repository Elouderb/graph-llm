"""Unified eval-report harness tests (card 69776c3e, piece 3).

Covers acceptance criterion 4: report structure (all metrics present), JSON
round-trip, and NO eval-time perturbation of model mode / RNG state -- on a tiny
CPU model, both with the tandem pathway off (reasoning/routing fields are ``None``)
and on (populated).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import graph_llm.models  # noqa: F401 — registers "delta_memory_lm"
from graph_llm.config import Config, DataConfig, ModelConfig, TrainConfig
from graph_llm.eval.report import build_eval_report, write_eval_report
from graph_llm.models import build_model


def _base_model_kwargs(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "name": "delta_memory_lm",
        "vocab_size": 64,
        "d_model": 16,
        "delta_layers": 1,
        "delta_n_heads": 1,
        "delta_head_k_dim": 8,
        "delta_head_v_dim": 8,
        "delta_conv_width": 4,
        "delta_chunk_size": 8,
        "delta_feature_map": "silu_l2",
        "delta_use_forget_gate": False,
        "delta_dropout": 0.0,
        "dropout": 0.0,
        "max_seq_len": 64,
        "front_end": "none",
        "segment_len": 16,
        "synthetic_key_digits": 1,
    }
    base.update(overrides)
    return base


def _plain_cfg() -> Config:
    """tandem_enabled=False (default) -- the reasoning/routing fields must be None."""
    m = ModelConfig(**_base_model_kwargs())
    return Config(
        model=m,
        data=DataConfig(source="synthetic", seq_len=m.segment_len, batch_size=4),
        train=TrainConfig(max_steps=1, log_every=1000),
    )


def _tandem_cfg() -> Config:
    """tandem_enabled=True -- the reasoning/routing fields must be populated.

    vocab_size=256 (byte-level): the synthetic reasoning examples
    (``make_reasoning_example``) encode arbitrary ASCII text byte-for-byte, which
    needs the full byte vocab -- matching ``build_tandem_model``'s hard-coded 256.
    """
    m = ModelConfig(
        **_base_model_kwargs(
            vocab_size=256,
            segment_len=256,
            max_seq_len=256,
            tandem_enabled=True,
            reasoning_segment_len=256,
            causal_reasoner_steps=4,
            causal_reasoner_gamma_floor=2.0,
            causal_reasoner_key_dim=8,
            causal_reasoner_conv_kernel=3,
            causal_reasoner_gru_layers=1,
            causal_reasoner_query_window=4,
            causal_reasoner_hard_seed=True,
        )
    )
    return Config(
        model=m,
        data=DataConfig(source="synthetic", seq_len=m.segment_len, batch_size=4),
        train=TrainConfig(max_steps=1, log_every=1000),
    )


def _val_loader(cfg: Config, n_batches: int = 3) -> DataLoader:
    seq_len = cfg.data.seq_len
    vocab = cfg.model.vocab_size
    n = n_batches * cfg.data.batch_size
    x = torch.randint(0, vocab, (n, seq_len))
    y = torch.randint(0, vocab, (n, seq_len))
    return DataLoader(TensorDataset(x, y), batch_size=cfg.data.batch_size)


def test_report_structure_tandem_off() -> None:
    cfg = _plain_cfg()
    model = build_model(cfg)
    device = torch.device("cpu")
    report = build_eval_report(
        model, cfg, _val_loader(cfg), device,
        step=42, retrieval_n_segments=(2, 3), retrieval_repeats=2,
    )
    assert report["step"] == 42
    assert report["tandem_enabled"] is False
    assert isinstance(report["val_bpb"], float)
    assert set(report["cross_segment_retrieval"]) == {"2", "3"}
    for v in report["cross_segment_retrieval"].values():
        assert "nll_carry" in v and "nll_reset" in v
    assert report["reasoning_depth_accuracy"] is None
    assert report["routing_health"] is None


def test_report_structure_tandem_on_all_metrics_present() -> None:
    cfg = _tandem_cfg()
    model = build_model(cfg)
    device = torch.device("cpu")
    report = build_eval_report(
        model, cfg, _val_loader(cfg), device,
        step=7,
        reasoning_depths=(4, 6),
        retrieval_n_segments=(2,),
        retrieval_repeats=2,
        reasoning_eval_batch=4,
        reasoning_eval_batches=1,
    )
    assert report["tandem_enabled"] is True
    assert isinstance(report["val_bpb"], float)
    assert set(report["cross_segment_retrieval"]) == {"2"}
    assert report["reasoning_depth_accuracy"] is not None
    assert set(report["reasoning_depth_accuracy"]) == {"M", "D4", "D6"}
    for v in report["reasoning_depth_accuracy"].values():
        assert 0.0 <= v <= 1.0
    assert report["routing_health"] is not None
    assert set(report["routing_health"]) == {"M", "D4", "D6"}


def test_report_val_bpb_none_without_val_loader() -> None:
    cfg = _plain_cfg()
    model = build_model(cfg)
    report = build_eval_report(model, cfg, None, torch.device("cpu"))
    assert report["val_bpb"] is None


def test_write_eval_report_json_round_trip(tmp_path) -> None:
    cfg = _plain_cfg()
    model = build_model(cfg)
    report = build_eval_report(
        model, cfg, _val_loader(cfg), torch.device("cpu"),
        step=5, retrieval_n_segments=(2,), retrieval_repeats=1,
    )
    path = write_eval_report(report, tmp_path, step=5)
    assert path.name == "eval_step000005.json"
    loaded = json.loads(path.read_text())
    assert loaded == report

    # Standalone (step=None) uses the fixed filename.
    path2 = write_eval_report(report, tmp_path)
    assert path2.name == "eval_report.json"


def test_build_eval_report_does_not_perturb_model_or_rng_state() -> None:
    cfg = _plain_cfg()
    model = build_model(cfg)
    device = torch.device("cpu")
    model.train()  # simulate mid-training call
    # Build the val loader (consumes RNG for its random fixture data) BEFORE taking
    # any "pre" snapshot, so the snapshot reflects exactly what build_eval_report
    # itself will see.
    loader = _val_loader(cfg)

    torch.manual_seed(123)
    pre_rng = torch.get_rng_state()
    pre_training = model.training
    # A tensor drawn AFTER the report call should be identical to one drawn from
    # the SAME seeded state before it, proving the report did not advance the
    # global RNG stream (it must save+restore around its own work).
    torch.manual_seed(123)
    expected_next_draw = torch.rand(4)

    torch.manual_seed(123)
    build_eval_report(model, cfg, loader, device, retrieval_repeats=1)

    assert model.training == pre_training
    assert torch.equal(torch.get_rng_state(), pre_rng)
    actual_next_draw = torch.rand(4)
    assert torch.equal(actual_next_draw, expected_next_draw)


def test_build_eval_report_restores_eval_mode_when_called_in_eval() -> None:
    cfg = _plain_cfg()
    model = build_model(cfg)
    model.eval()
    build_eval_report(model, cfg, _val_loader(cfg), torch.device("cpu"), retrieval_repeats=1)
    assert model.training is False
