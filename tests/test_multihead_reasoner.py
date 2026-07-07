"""Tests for the multi-head causal reasoner (card 22acac98).

``MultiHeadCausalReasoner`` runs ``n_heads`` parallel weight-independent locate-then-walk
heads over ONE shared encoder and combines them on their final reads.  The load-bearing
correctness checks:

* **Default-off / interface-stable**: the shipped ``CausalReasoner`` is untouched, and a
  ``MultiHeadCausalReasoner(n_heads=1, combine="mean")`` with weights copied from a
  ``CausalReasoner`` is BYTE-FOR-BYTE identical (forward, query_forward, aux, incl.
  tf_seed + walk_hard) — the multi-head is a faithful generalisation of the single walk.
* **Wiring**: per-head aux, the three combine modes, heterogeneous specs (different K /
  window / width per head), per-head step overrides, and the single-query fast path all
  produce the right shapes and behaviours.
* **Diversity pressure**: per-head ``read_dropout`` is train-only (stochastic in train,
  deterministic in eval).
* Invalid config raises.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from graph_llm.config import ModelConfig
from graph_llm.models.components.causal_reasoner import (
    CausalReasoner,
    MultiHeadCausalReasoner,
    WalkHeadSpec,
)


def _cfg(**overrides: Any) -> ModelConfig:
    """A tiny CPU-friendly reasoner config (faithful field names)."""
    base: dict[str, Any] = {
        "name": "delta_memory_lm",
        "vocab_size": 256,
        "d_model": 24,
        "tandem_enabled": True,
        "reasoning_segment_len": 32,
        "causal_reasoner_steps": 4,
        "causal_reasoner_gamma_floor": 2.0,
        "causal_reasoner_key_dim": 12,
        "causal_reasoner_conv_kernel": 5,
        "causal_reasoner_gru_layers": 1,
        "causal_reasoner_query_window": 6,
        "causal_reasoner_hard_seed": True,
    }
    base.update(overrides)
    return ModelConfig(**base)


def _copy_single_into_head(cr: CausalReasoner, mh: MultiHeadCausalReasoner) -> None:
    """Copy every ``CausalReasoner`` walk parameter into ``mh``'s single head + encoder so
    the two walks are numerically identical (``head_proj <- proj``)."""
    mh.encoder.load_state_dict(cr.encoder.state_dict())
    head = mh.heads[0]
    for name in ("name_key", "ptr_key", "val", "seed_name_key", "query_pool",
                 "gru", "to_query", "to_beta", "to_gate", "to_gamma"):
        getattr(head, name).load_state_dict(getattr(cr, name).state_dict())
    head.head_proj.load_state_dict(cr.proj.state_dict())
    with torch.no_grad():
        head.move_gain.copy_(cr.move_gain)
        head.key_scale.copy_(cr.key_scale)


# ---------------------------------------------------------------------------
# n_heads=1 is a faithful generalisation of the single CausalReasoner walk
# ---------------------------------------------------------------------------


def test_single_head_matches_causal_reasoner_full_forward() -> None:
    torch.manual_seed(0)
    cfg = _cfg()
    cr = CausalReasoner(cfg).eval()
    mh = MultiHeadCausalReasoner(cfg, n_heads=1, combine="mean").eval()
    _copy_single_into_head(cr, mh)
    x = torch.randint(1, 256, (3, 28))
    with torch.no_grad():
        a = cr(x)
        b = mh(x)
    assert isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)
    assert torch.allclose(a, b, atol=1e-6), "n_heads=1 diverges from the single walk"


def test_single_head_matches_causal_reasoner_query_and_aux() -> None:
    torch.manual_seed(1)
    cfg = _cfg(causal_reasoner_steps=5)
    cr = CausalReasoner(cfg).eval()
    mh = MultiHeadCausalReasoner(cfg, n_heads=1, combine="mean").eval()
    _copy_single_into_head(cr, mh)
    x = torch.randint(1, 256, (3, 28))
    qpos = torch.tensor([27, 15, 9])
    tf = torch.tensor([3, 5, -1])
    with torch.no_grad():
        cr_res = cr.query_forward(x, qpos, tf_seed=tf, return_aux=True, walk_hard=True)
        mh_res = mh.query_forward(x, qpos, tf_seed=tf, return_aux=True, walk_hard=True)
    assert isinstance(cr_res, tuple) and isinstance(mh_res, tuple)
    cr_out, cr_aux = cr_res
    mh_out, mh_aux = mh_res
    assert torch.allclose(cr_out, mh_out, atol=1e-6)
    assert torch.allclose(cr_aux["seed_logits"], mh_aux["seed_logits"], atol=1e-6)
    assert torch.allclose(cr_aux["walk_w"], mh_aux["walk_w"], atol=1e-6)
    # And the mirror aux equals the per-head[0] aux.
    assert torch.equal(mh_aux["walk_w"], mh_aux["per_head"][0]["walk_w"])


# ---------------------------------------------------------------------------
# Multi-head wiring: shapes, combine modes, aux
# ---------------------------------------------------------------------------


def test_multihead_forward_and_query_shapes() -> None:
    torch.manual_seed(0)
    cfg = _cfg()
    mh = MultiHeadCausalReasoner(cfg, n_heads=2, combine="mean").eval()
    x = torch.randint(1, 256, (4, 30))
    with torch.no_grad():
        full = mh(x)
        qpos = torch.tensor([29, 20, 10, 5])
        res = mh.query_forward(x, qpos, return_aux=True)
    assert isinstance(res, tuple)
    out, aux = res
    assert isinstance(full, torch.Tensor) and full.shape == (4, 30, cfg.d_model)
    assert out.shape == (4, cfg.d_model)
    assert len(aux["per_head"]) == 2
    for ph in aux["per_head"]:
        assert ph["walk_w"].shape == (4, cfg.causal_reasoner_steps, 30)
        assert ph["final_pos"].shape == (4,)
        assert ph["conf"].shape == (4,)


@pytest.mark.parametrize("combine", ["mean", "confidence", "concat"])
def test_multihead_combine_modes_run(combine: str) -> None:
    torch.manual_seed(2)
    cfg = _cfg()
    mh = MultiHeadCausalReasoner(cfg, n_heads=3, combine=combine).eval()
    x = torch.randint(1, 256, (2, 30))
    qpos = torch.tensor([29, 12])
    with torch.no_grad():
        res = mh.query_forward(x, qpos, return_aux=True)
    assert isinstance(res, tuple)
    out, aux = res
    assert out.shape == (2, cfg.d_model)
    assert torch.isfinite(out).all()
    if combine == "confidence":
        cw = aux["combine_weights"]
        assert cw.shape == (2, 3)
        assert torch.allclose(cw.sum(dim=-1), torch.ones(2), atol=1e-5)
    else:
        assert aux["combine_weights"] is None
    if combine == "concat":
        assert mh.final_proj is not None


def test_multihead_query_forward_matches_full_at_query_pos() -> None:
    torch.manual_seed(3)
    cfg = _cfg()
    mh = MultiHeadCausalReasoner(cfg, n_heads=2, combine="confidence").eval()
    x = torch.randint(1, 256, (3, 28))
    qpos = torch.tensor([27, 15, 9])
    with torch.no_grad():
        full_res = mh(x, aux_query_pos=qpos, return_aux=True)
        fast = mh.query_forward(x, qpos)
    assert isinstance(full_res, tuple) and isinstance(fast, torch.Tensor)
    full = full_res[0]
    rows = torch.arange(3)
    assert torch.allclose(full[rows, qpos], fast, atol=1e-6)


# ---------------------------------------------------------------------------
# Heterogeneous heads (different K / window / width) + per-head step overrides
# ---------------------------------------------------------------------------


def test_multihead_heterogeneous_specs_run() -> None:
    torch.manual_seed(4)
    cfg = _cfg()
    specs = [
        WalkHeadSpec(steps=3, query_window=4, d_key=8, d_ctrl=16),
        WalkHeadSpec(steps=6, query_window=10, d_key=12, d_ctrl=24, addr_range=20),
    ]
    mh = MultiHeadCausalReasoner(cfg, n_heads=2, head_specs=specs, combine="concat").eval()
    x = torch.randint(1, 256, (2, 30))
    qpos = torch.tensor([29, 14])
    with torch.no_grad():
        res = mh.query_forward(x, qpos, return_aux=True)
    assert isinstance(res, tuple)
    out, aux = res
    assert out.shape == (2, cfg.d_model)
    # Each head walks its OWN native depth when steps=None.
    assert aux["per_head"][0]["walk_w"].shape == (2, 3, 30)
    assert aux["per_head"][1]["walk_w"].shape == (2, 6, 30)


def test_multihead_per_head_steps_override() -> None:
    torch.manual_seed(5)
    cfg = _cfg()
    mh = MultiHeadCausalReasoner(cfg, n_heads=2, combine="mean").eval()
    x = torch.randint(1, 256, (2, 30))
    qpos = torch.tensor([29, 14])
    with torch.no_grad():
        res = mh.query_forward(x, qpos, steps=[7, 2], return_aux=True)
    assert isinstance(res, tuple)
    aux = res[1]
    assert aux["per_head"][0]["walk_w"].shape[1] == 7
    assert aux["per_head"][1]["walk_w"].shape[1] == 2


# ---------------------------------------------------------------------------
# Diversity pressure: per-head read_dropout is train-only
# ---------------------------------------------------------------------------


def test_read_dropout_is_train_only() -> None:
    torch.manual_seed(6)
    cfg = _cfg()
    specs = [WalkHeadSpec.from_cfg(cfg), WalkHeadSpec(steps=4, query_window=6, d_key=12,
                                                      d_ctrl=24, read_dropout=0.3)]
    mh = MultiHeadCausalReasoner(cfg, n_heads=2, head_specs=specs, combine="mean")
    x = torch.randint(1, 256, (2, 28))
    qpos = torch.tensor([27, 14])
    mh.train()
    torch.manual_seed(100)
    a = mh.query_forward(x, qpos)
    torch.manual_seed(101)
    b = mh.query_forward(x, qpos)
    assert isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)
    assert (a - b).abs().max().item() > 1e-6, "read_dropout should perturb the train walk"
    mh.eval()
    with torch.no_grad():
        c = mh.query_forward(x, qpos)
        d = mh.query_forward(x, qpos)
    assert isinstance(c, torch.Tensor) and isinstance(d, torch.Tensor)
    assert torch.equal(c, d), "eval must be deterministic (dropout off)"


# ---------------------------------------------------------------------------
# Invalid config raises
# ---------------------------------------------------------------------------


def test_multihead_invalid_args_raise() -> None:
    cfg = _cfg()
    with pytest.raises(ValueError):
        MultiHeadCausalReasoner(cfg, n_heads=0)
    with pytest.raises(ValueError):
        MultiHeadCausalReasoner(cfg, n_heads=2, combine="bogus")
    with pytest.raises(ValueError):
        MultiHeadCausalReasoner(cfg, n_heads=2, head_specs=[WalkHeadSpec.from_cfg(cfg)])
    # Per-head spec validation fires when the head is built (steps < 1, d_key < 2).
    with pytest.raises(ValueError):
        MultiHeadCausalReasoner(cfg, n_heads=1,
                                head_specs=[WalkHeadSpec(steps=0, query_window=6,
                                                         d_key=12, d_ctrl=24)])
    with pytest.raises(ValueError):
        MultiHeadCausalReasoner(cfg, n_heads=1,
                                head_specs=[WalkHeadSpec(steps=4, query_window=6,
                                                         d_key=1, d_ctrl=24)])
