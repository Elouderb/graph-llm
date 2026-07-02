"""Tests for the tandem causal-reasoner || delta-memory gated fusion (card 2dd3400f).

The tandem adds a SECOND pathway to ``delta_memory_lm``: the lead-verified v3 WIN
causal reasoner (card e4e8a4dc) run per position and fused with the delta-memory
backbone by an unsupervised gate (card 31fe6b00).  The load-bearing correctness
checks (mirroring the ``reasoning_field`` discipline):

* **tandem_enabled=False is a byte-for-byte NO-OP**: default config, state_dict
  keys, parameter count, and forward logits are identical to the committed backbone
  under a fixed seed (``torch.equal``).  No reasoner/gate/buffer is constructed.
* **LM-LEVEL LEAK PROBE (tandem ON)**: perturbing tokens ``> t`` leaves the logits
  at positions ``<= t`` EXACTLY unchanged (0.0) — the model-level causality check,
  in both the single-window and the multi-window (segment-bounded) regimes.
* **Component causality**: the ``CausalReasoner`` output at position ``t`` depends
  only on tokens ``<= t`` (structural: unidirectional GRU + left-padded conv +
  causal address mask + per-window bounding).
* **Trains when ON**: the ``tandem_step`` gate + aux path returns the right shapes,
  the forced-mix warmup pins ``g=0.5``, finite grads reach the reasoner + gate, and
  a tiny run stays finite.
* Invalid config raises; sub-quadratic segment-bounding.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import graph_llm.models  # noqa: F401 — registers "delta_memory_lm" (+ baselines)
from graph_llm.config import Config, DataConfig, ModelConfig, TrainConfig
from graph_llm.models import build_model
from graph_llm.models.components.causal_reasoner import CausalReasoner


def _mem_cfg(**overrides: Any) -> ModelConfig:
    """A small delta_memory_lm ModelConfig that exercises the tandem on CPU."""
    base: dict[str, Any] = {
        "name": "delta_memory_lm",
        "vocab_size": 256,
        "d_model": 32,
        "delta_layers": 2,
        "delta_n_heads": 4,
        "delta_head_k_dim": 16,
        "delta_head_v_dim": 16,
        "delta_feature_map": "l2",
        "delta_ff_mult": 2,
        "delta_dropout": 0.0,
        "dropout": 0.0,
        "max_seq_len": 64,
    }
    base.update(overrides)
    return ModelConfig(**base)


def _full_cfg(model: ModelConfig) -> Config:
    return Config(
        model=model,
        data=DataConfig(source="synthetic", seq_len=model.max_seq_len, batch_size=4),
        train=TrainConfig(max_steps=3, warmup_steps=1, mixed_precision="no"),
    )


def _on_cfg(**overrides: Any) -> ModelConfig:
    """A small config with the tandem ON (tiny reasoner: K=4, key 16)."""
    base: dict[str, Any] = {
        "tandem_enabled": True,
        "reasoning_segment_len": 64,
        "causal_reasoner_steps": 4,
        "causal_reasoner_key_dim": 16,
        "causal_reasoner_gru_layers": 1,
        "causal_reasoner_query_window": 6,
    }
    base.update(overrides)
    return _mem_cfg(**base)


# ---------------------------------------------------------------------------
# tandem_enabled=False is a byte-for-byte no-op (committed backbone unchanged)
# ---------------------------------------------------------------------------


def test_tandem_disabled_is_the_default() -> None:
    assert ModelConfig().tandem_enabled is False
    model = build_model(_full_cfg(_mem_cfg()))
    assert model.causal_reasoner is None, "tandem_enabled=False must construct NO reasoner"
    assert model.tandem_gate is None


def test_tandem_off_state_dict_identical_to_baseline() -> None:
    """tandem OFF adds zero parameters / state — same keys + count, no buffer."""
    default_model = build_model(_full_cfg(_mem_cfg()))
    explicit_off = build_model(_full_cfg(_mem_cfg(tandem_enabled=False)))
    assert set(default_model.state_dict()) == set(explicit_off.state_dict())
    assert not any(k.startswith("causal_reasoner") for k in default_model.state_dict())
    assert not any(k.startswith("tandem_gate") for k in default_model.state_dict())
    assert "_tandem_step" not in default_model.state_dict()
    assert default_model.num_parameters() == explicit_off.num_parameters()


def test_tandem_off_forward_is_bit_identical_under_fixed_seed() -> None:
    """With a fixed seed, tandem OFF forward == the no-tandem baseline forward."""
    torch.manual_seed(123)
    baseline = build_model(_full_cfg(_mem_cfg()))
    torch.manual_seed(123)
    explicit_off = build_model(_full_cfg(_mem_cfg(tandem_enabled=False)))
    baseline.eval()
    explicit_off.eval()
    x = torch.randint(0, 256, (2, 16))
    with torch.no_grad():
        _, logits_a = baseline(x)
        _, logits_b = explicit_off(x)
    assert torch.equal(logits_a, logits_b), (
        "tandem_enabled=False is not a byte-for-byte no-op (logits differ under the "
        "same seed)."
    )


def test_tandem_off_states_contract_bit_identical() -> None:
    """The (loss, logits, states) carry contract is byte-for-byte identical OFF."""
    torch.manual_seed(7)
    baseline = build_model(_full_cfg(_mem_cfg()))
    torch.manual_seed(7)
    explicit_off = build_model(_full_cfg(_mem_cfg(tandem_enabled=False)))
    baseline.eval()
    explicit_off.eval()
    x = torch.randint(0, 256, (2, 16))
    with torch.no_grad():
        loss_a, logits_a, states_a = baseline(x, None, None, True)
        loss_b, logits_b, states_b = explicit_off(x, None, None, True)
    assert torch.equal(logits_a, logits_b)
    assert torch.equal(loss_a, loss_b)
    assert len(states_a) == len(states_b)


def test_tandem_on_adds_params_over_off() -> None:
    off = build_model(_full_cfg(_mem_cfg(tandem_enabled=False)))
    on = build_model(_full_cfg(_on_cfg()))
    assert on.num_parameters() > off.num_parameters()
    assert any(k.startswith("causal_reasoner") for k in on.state_dict())
    assert any(k.startswith("tandem_gate") for k in on.state_dict())


# ---------------------------------------------------------------------------
# LM-LEVEL LEAK PROBE — the load-bearing causality check (tandem ON)
# ---------------------------------------------------------------------------


def _lm_leak_maxdiff(model: torch.nn.Module, x: torch.Tensor, t_pert: int, reps: int = 6) -> float:
    """Max |Δ logits| at positions <= t_pert under perturbation of tokens > t_pert."""
    b, t = x.shape
    model.eval()
    with torch.no_grad():
        _, l0 = model(x)
    g = torch.Generator().manual_seed(1234)
    max_diff = 0.0
    for _ in range(reps):
        xp = x.clone()
        rnd = torch.randint(1, 256, (b, t), generator=g)
        xp[:, t_pert + 1 :] = rnd[:, t_pert + 1 :]  # perturb ONLY tokens > t_pert
        with torch.no_grad():
            _, l1 = model(xp)
        d = (l1[:, : t_pert + 1] - l0[:, : t_pert + 1]).abs().max().item()
        max_diff = max(max_diff, d)
    return max_diff


def test_lm_leak_probe_single_window_exactly_zero() -> None:
    """Tandem ON, single window: perturbing tokens > t leaves logits <= t EXACTLY 0."""
    torch.manual_seed(0)
    on = build_model(_full_cfg(_on_cfg(reasoning_segment_len=64)))
    x = torch.randint(1, 256, (2, 48))
    max_diff = _lm_leak_maxdiff(on, x, t_pert=19)
    assert max_diff == 0.0, f"LM leak (single window): max|Δlogit| at <=t = {max_diff:.3e} != 0.0"


def test_lm_leak_probe_multi_window_exactly_zero() -> None:
    """Tandem ON, multiple bounded windows: still exactly causal at window boundaries."""
    torch.manual_seed(0)
    on = build_model(_full_cfg(_mem_cfg(
        tandem_enabled=True, reasoning_segment_len=16, causal_reasoner_steps=4,
        causal_reasoner_key_dim=16, causal_reasoner_query_window=6, max_seq_len=128,
    )))
    x = torch.randint(1, 256, (2, 100))
    # Probe both mid-window and just-before a window boundary.
    for t_pert in (50, 47, 31):
        max_diff = _lm_leak_maxdiff(on, x, t_pert=t_pert)
        assert max_diff == 0.0, (
            f"LM leak (multi window) at t={t_pert}: max|Δlogit| at <=t = {max_diff:.3e} != 0.0"
        )


def test_answer_prediction_position_is_leak_free() -> None:
    """Copy-leak guard (harness convention): to predict the token at position p, use
    logits[p-1] — NOT logits[p].  logits[p] depends on the input token AT p (via the
    residual/embedding), so gathering there lets the model trivially COPY its own input
    (the answer byte sits in the input at answer_pos).  Assert: perturbing token p
    leaves logits[p-1] EXACTLY unchanged (the correct, leak-free prediction position),
    while it DOES change logits[p] (proving the guard is meaningful — that is the trap)."""
    torch.manual_seed(0)
    on = build_model(_full_cfg(_on_cfg(reasoning_segment_len=64)))
    on.eval()
    b, t = 2, 40
    x = torch.randint(1, 256, (b, t))
    p = 25  # stand-in for answer_pos
    with torch.no_grad():
        _, logits = on(x)
    x_pert = x.clone()
    x_pert[:, p] = (x[:, p] + 7) % 256  # perturb the token AT p (the "answer byte")
    with torch.no_grad():
        _, logits_pert = on(x_pert)
    leak_free = (logits[:, p - 1] - logits_pert[:, p - 1]).abs().max().item()
    trap = (logits[:, p] - logits_pert[:, p]).abs().max().item()
    assert leak_free == 0.0, (
        f"prediction position answer_pos-1 leaked: perturbing token {p} changed "
        f"logits[{p - 1}] by {leak_free:.3e} (must be 0.0)."
    )
    assert trap > 0.0, (
        "logits[answer_pos] did NOT depend on the answer byte — the copy-leak guard is "
        "toothless (this position is exactly the one the trainer must NOT gather at)."
    )


def test_component_reasoner_is_causal_per_position() -> None:
    """CausalReasoner: perturbing token t changes the reasoning hidden ONLY at >= t."""
    torch.manual_seed(0)
    cfg = _on_cfg(d_model=24, reasoning_segment_len=32)
    reasoner = CausalReasoner(cfg)
    reasoner.eval()
    b, t = 2, 28
    x = torch.randint(1, 256, (b, t))
    with torch.no_grad():
        out = reasoner(x)
    assert isinstance(out, torch.Tensor)
    t_pert = 12
    x_pert = x.clone()
    x_pert[:, t_pert] = (x[:, t_pert] + 7) % 256
    with torch.no_grad():
        out_pert = reasoner(x_pert)
    max_diff = (out[:, :t_pert] - out_pert[:, :t_pert]).abs().max().item()
    assert max_diff < 1e-6, (
        f"CausalReasoner leaked: perturbing token {t_pert} changed the hidden at "
        f"positions < {t_pert} by {max_diff:.2e} (must be < 1e-6)."
    )
    changed = (out[:, t_pert] - out_pert[:, t_pert]).abs().max().item()
    assert changed > 1e-6, "the reasoner ignored the position it was perturbed at"


# ---------------------------------------------------------------------------
# Trains when ON — gate, aux, forced-mix, finite grads
# ---------------------------------------------------------------------------


def test_tandem_step_shapes_and_forced_mix() -> None:
    """tandem_step returns per-position gate + answer-gathered aux; forced-mix pins g."""
    torch.manual_seed(0)
    on = build_model(_full_cfg(_on_cfg(gate_mix_warmup_steps=5)))
    on.train()
    b, t = 3, 32
    x = torch.randint(1, 256, (b, t))
    aux_pos = torch.tensor([t - 1, 20, 10])
    tf = torch.tensor([5, 8, 3])
    out = on.tandem_step(x, aux_query_pos=aux_pos, tf_seed=tf)
    assert out["gate"].shape == (b, t)
    assert out["aux"]["seed_logits"].shape == (b, t)
    assert out["aux"]["walk_w"].shape == (b, on.causal_reasoner.steps, t)
    # Forced-mix warmup: the FUSION is pinned to g=0.5 (gate_mix), so both pathways
    # contribute equally regardless of the learned gate.  ``out["gate"]`` reports the
    # SOFT routing PREFERENCE (used for the gate losses / routing inspection), not the
    # forced fusion value.
    assert out["gate_mix"] == 0.5
    assert int(on._tandem_step.item()) == 1


def test_force_gate_scalar_routes_each_pathway() -> None:
    """force_gate=0.0 routes the fusion to PURE memory, 1.0 to PURE reasoner (the
    dissociation-probe mechanism).  The two must give different logits, and force_gate
    must dominate the learned gate."""
    torch.manual_seed(0)
    on = build_model(_full_cfg(_on_cfg()))
    on.eval()
    b, t = 2, 20
    x = torch.randint(1, 256, (b, t))
    aqp = torch.tensor([t - 1, 10])
    with torch.no_grad():
        mem = on.tandem_step(x, aux_query_pos=aqp, force_gate=0.0)["logits"]
        rea = on.tandem_step(x, aux_query_pos=aqp, force_gate=1.0)["logits"]
    assert (mem - rea).abs().max().item() > 1e-3, "force_gate 0 vs 1 gave identical logits"


def test_gate_mix_tensor_per_row_routing() -> None:
    """A per-row Tensor gate_mix (curriculum routing) routes each row to its specialist:
    row with force 0.0 matches the memory route, row with force 1.0 matches the reasoner
    route (whole position, broadcast over channels)."""
    torch.manual_seed(0)
    on = build_model(_full_cfg(_on_cfg()))
    on.eval()
    b, t = 2, 20
    x = torch.randint(1, 256, (b, t))
    aqp = torch.tensor([t - 1, 10])
    with torch.no_grad():
        mem = on.tandem_step(x, aux_query_pos=aqp, force_gate=torch.tensor([0.0, 0.0]))["logits"]
        rea = on.tandem_step(x, aux_query_pos=aqp, force_gate=torch.tensor([1.0, 1.0]))["logits"]
        mix = on.tandem_step(x, aux_query_pos=aqp, force_gate=torch.tensor([0.0, 1.0]))["logits"]
    assert torch.allclose(mix[0], mem[0], atol=1e-5), "row0 (force 0) did not match the memory route"
    assert torch.allclose(mix[1], rea[1], atol=1e-5), "row1 (force 1) did not match the reasoner route"


def test_tandem_step_releases_forced_mix_after_warmup() -> None:
    """After gate_mix_warmup steps the learned gate takes over (not pinned to 0.5)."""
    torch.manual_seed(0)
    on = build_model(_full_cfg(_on_cfg(gate_mix_warmup_steps=2)))
    on.train()
    x = torch.randint(1, 256, (2, 32))
    aux_pos = torch.tensor([31, 20])
    for _ in range(3):  # exhaust the 2-step warmup
        on.tandem_step(x, aux_query_pos=aux_pos)
    out = on.tandem_step(x, aux_query_pos=aux_pos)
    assert out["gate_mix"] is None
    assert not torch.allclose(out["gate"], torch.full_like(out["gate"], 0.5))


def test_tandem_on_finite_grads_reach_reasoner_and_gate() -> None:
    """A tandem_step answer-CE loss sends finite non-zero grads to reasoner + gate."""
    torch.manual_seed(0)
    on = build_model(_full_cfg(_on_cfg(gate_mix_warmup_steps=0)))
    on.train()
    b, t = 4, 32
    x = torch.randint(1, 256, (b, t))
    aux_pos = torch.full((b,), t - 1)
    tgt = torch.randint(0, 256, (b,))
    out = on.tandem_step(x, aux_query_pos=aux_pos)
    ans_logits = out["logits"][torch.arange(b), aux_pos]
    loss = torch.nn.functional.cross_entropy(ans_logits, tgt)
    # Locate-CE + walk-aux exercise the aux path too.
    loss = loss + torch.nn.functional.cross_entropy(out["aux"]["seed_logits"], aux_pos.clamp(0, t - 1))
    loss.backward()
    reasoner_grads = [p.grad for p in on.causal_reasoner.parameters() if p.requires_grad]
    gate_grads = [p.grad for p in on.tandem_gate.parameters() if p.requires_grad]
    assert any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in reasoner_grads)
    assert any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in gate_grads)


def test_tandem_on_trains_a_tiny_step_finite() -> None:
    """A few optimiser steps on a fixed batch keep the loss finite (no instability)."""
    torch.manual_seed(0)
    on = build_model(_full_cfg(_on_cfg(gate_mix_warmup_steps=2)))
    opt = torch.optim.AdamW(on.parameters(), lr=1e-3)
    on.train()
    b, t = 4, 32
    x = torch.randint(1, 256, (b, t))
    aux_pos = torch.full((b,), t - 1)
    tgt = torch.randint(0, 256, (b,))
    losses = []
    for _ in range(6):
        opt.zero_grad()
        out = on.tandem_step(x, aux_query_pos=aux_pos, tf_seed=aux_pos)
        ans = out["logits"][torch.arange(b), aux_pos]
        loss = torch.nn.functional.cross_entropy(ans, tgt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(on.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
    assert all(math.isfinite(x_) for x_ in losses)


def test_tandem_on_states_contract_holds() -> None:
    """The (loss, logits, states) carry path still works with the tandem ON."""
    cfg = _full_cfg(_on_cfg())
    model = build_model(cfg)
    model.eval()
    x = torch.randint(0, cfg.model.vocab_size, (2, 16))
    with torch.no_grad():
        out = model(x, None, None, True)
    assert len(out) == 3
    _, logits, states = out
    assert logits.shape == (2, 16, cfg.model.vocab_size)
    assert len(states) == cfg.model.delta_layers


# ---------------------------------------------------------------------------
# Validation + sub-quadratic segment-bounding
# ---------------------------------------------------------------------------


def test_causal_reasoner_gamma_floor_below_one_raises() -> None:
    with pytest.raises(ValueError):
        CausalReasoner(_on_cfg(causal_reasoner_gamma_floor=0.5))


def test_causal_reasoner_bad_steps_raises() -> None:
    with pytest.raises(ValueError):
        CausalReasoner(_on_cfg(causal_reasoner_steps=0))


def test_reasoner_addressing_is_segment_bounded() -> None:
    """A position never addresses across a window boundary: perturbing a token in an
    EARLIER window (but still < t) does NOT change position t's reasoning hidden —
    the segment-bounded (sub-quadratic) property.  (Contrast: an unbounded reasoner
    WOULD change.)"""
    torch.manual_seed(0)
    cfg = _on_cfg(d_model=24, reasoning_segment_len=16)
    reasoner = CausalReasoner(cfg)
    reasoner.eval()
    b, t = 2, 48
    x = torch.randint(1, 256, (b, t))
    with torch.no_grad():
        out = reasoner(x)
    # Position 40 is in window [32,48); perturb token 5 (window [0,16), earlier window).
    x_pert = x.clone()
    x_pert[:, 5] = (x[:, 5] + 3) % 256
    with torch.no_grad():
        out_pert = reasoner(x_pert)
    diff_at_40 = (out[:, 40] - out_pert[:, 40]).abs().max().item()
    assert diff_at_40 < 1e-6, (
        f"reasoner at position 40 depended on token 5 in an EARLIER window "
        f"(Δ={diff_at_40:.2e}) — not segment-bounded."
    )
