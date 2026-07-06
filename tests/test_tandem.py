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
from graph_llm.train.tandem import (
    TandemConfig,
    _build_run_config,
    _ramp_depth,
    _resolve_ramp_delay,
    _train_one,
)


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
# EVAL-TIME per-hop precision knobs (card 0a98292b) — walk_hard / gamma_add
# ---------------------------------------------------------------------------


def test_reasoner_eval_knobs_default_is_noop() -> None:
    """walk_hard=False + gamma_add=0.0 (the defaults) are byte-for-byte the soft walk
    the tandem trains/evals with — so the shipped delta_memory_lm path is unchanged."""
    torch.manual_seed(0)
    cfg = _on_cfg(d_model=24, reasoning_segment_len=32)
    reasoner = CausalReasoner(cfg)
    reasoner.eval()
    x = torch.randint(1, 256, (2, 28))
    with torch.no_grad():
        base = reasoner(x)
        same = reasoner(x, walk_hard=False, gamma_add=0.0)
    assert isinstance(base, torch.Tensor)
    assert torch.equal(base, same), "the eval-knob defaults are not a byte-for-byte no-op"


def test_reasoner_walk_hard_changes_the_walk() -> None:
    """walk_hard=True straight-through top-1s the head so each hop's READ is a clean
    single position -> the walk output differs from the soft walk (the hard-addressing
    mechanism is active), while the collected aux stays a real soft distribution (so a
    walk-aux loss is well-defined) whose argmax matches the committed head."""
    torch.manual_seed(0)
    cfg = _on_cfg(d_model=24, reasoning_segment_len=32, causal_reasoner_steps=4)
    reasoner = CausalReasoner(cfg)
    reasoner.eval()
    b, t = 2, 28
    x = torch.randint(1, 256, (b, t))
    aqp = torch.tensor([t - 1, 15])
    with torch.no_grad():
        hard_out = reasoner(x, aux_query_pos=aqp, return_aux=True, walk_hard=True)
        soft_out = reasoner(x, aux_query_pos=aqp, return_aux=True)
    assert isinstance(hard_out, tuple) and isinstance(soft_out, tuple)
    hard, hard_aux = hard_out
    soft = soft_out[0]
    assert (hard - soft).abs().max().item() > 1e-5, "walk_hard did not change the walk"
    # The aux head stays a genuine (soft) distribution, not a degenerate one-hot.
    ww = hard_aux["walk_w"]                      # (b, K, t)
    assert torch.allclose(ww.sum(dim=-1), torch.ones_like(ww.sum(dim=-1)), atol=1e-5)
    assert ww.max(dim=-1).values.min().item() < 1.0 - 1e-4


def test_query_forward_matches_full_forward_at_query_position() -> None:
    """The single-query fast path (query_forward) is numerically identical to the full
    per-position forward gathered at query_pos (same locate-then-walk, one Lq row)."""
    torch.manual_seed(0)
    cfg = _on_cfg(d_model=24, reasoning_segment_len=32, causal_reasoner_steps=5)
    reasoner = CausalReasoner(cfg)
    reasoner.eval()
    b, t = 3, 28
    x = torch.randint(1, 256, (b, t))
    qpos = torch.tensor([t - 1, 15, 9])
    with torch.no_grad():
        full_out = reasoner(x, aux_query_pos=qpos, return_aux=True)
        fast_out = reasoner.query_forward(x, qpos, return_aux=True)
    assert isinstance(full_out, tuple) and isinstance(fast_out, tuple)
    full, full_aux = full_out
    fast, fast_aux = fast_out
    rows = torch.arange(b)
    assert torch.allclose(full[rows, qpos], fast, atol=1e-5), (
        "query_forward hidden differs from the full forward at the query position"
    )
    assert torch.allclose(full_aux["seed_logits"], fast_aux["seed_logits"], atol=1e-5)
    assert torch.allclose(full_aux["walk_w"], fast_aux["walk_w"], atol=1e-5)


def test_query_forward_teacher_forcing_and_hard_match_full() -> None:
    """query_forward matches the full path also with tf_seed + walk_hard engaged."""
    torch.manual_seed(1)
    cfg = _on_cfg(d_model=24, reasoning_segment_len=32, causal_reasoner_steps=4)
    reasoner = CausalReasoner(cfg)
    reasoner.eval()
    b, t = 2, 24
    x = torch.randint(1, 256, (b, t))
    qpos = torch.tensor([t - 1, 12])
    tf = torch.tensor([3, 5])
    with torch.no_grad():
        full_out = reasoner(
            x, aux_query_pos=qpos, tf_seed=tf, return_aux=True, walk_hard=True,
        )
        fast_out = reasoner.query_forward(
            x, qpos, tf_seed=tf, return_aux=True, walk_hard=True,
        )
    assert isinstance(full_out, tuple) and isinstance(fast_out, tuple)
    full, full_aux = full_out
    fast, fast_aux = fast_out
    rows = torch.arange(b)
    assert torch.allclose(full[rows, qpos], fast, atol=1e-5)
    assert torch.allclose(full_aux["walk_w"], fast_aux["walk_w"], atol=1e-5)


def test_reasoner_gamma_add_sharpens_the_walk() -> None:
    """A positive gamma_add is extra eval-time sharpening: it changes the walk output
    (and gamma_add=0.0 leaves it untouched, covered by the no-op test)."""
    torch.manual_seed(0)
    cfg = _on_cfg(d_model=24, reasoning_segment_len=32, causal_reasoner_steps=4)
    reasoner = CausalReasoner(cfg)
    reasoner.eval()
    x = torch.randint(1, 256, (2, 28))
    with torch.no_grad():
        soft = reasoner(x)
        sharp = reasoner(x, gamma_add=4.0)
    assert (soft - sharp).abs().max().item() > 1e-6, "gamma_add had no effect on the walk"


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


# ---------------------------------------------------------------------------
# MLP WORKHORSE — third pathway + 3-way softmax gate (card a7948491)
# ---------------------------------------------------------------------------


def _on3_cfg(**overrides: Any) -> ModelConfig:
    """A small config with the tandem ON and the 3-way MLP workhorse enabled."""
    base: dict[str, Any] = {"tandem_mlp_enabled": True, "tandem_mlp_ff_mult": 2}
    base.update(overrides)
    return _on_cfg(**base)


def test_mlp_workhorse_off_by_default() -> None:
    assert ModelConfig().tandem_mlp_enabled is False
    # tandem ON but MLP flag OFF -> the shipped 2-way tandem (no MLP pathway).
    model = build_model(_full_cfg(_on_cfg()))
    assert model.tandem_mlp is None, "tandem_mlp_enabled=False must construct NO MLP pathway"
    assert model.tandem_gate is not None
    assert model.tandem_gate.in_features == 3 * model.d_model  # 2-way gate: [mem, reason, ctx]


def test_mlp_flag_is_noop_when_tandem_off() -> None:
    """tandem_mlp_enabled=True with tandem_enabled=False builds NOTHING (still no-op)."""
    torch.manual_seed(5)
    baseline = build_model(_full_cfg(_mem_cfg()))
    torch.manual_seed(5)
    mlp_flag_off = build_model(_full_cfg(_mem_cfg(tandem_mlp_enabled=True)))
    assert mlp_flag_off.causal_reasoner is None
    assert mlp_flag_off.tandem_mlp is None
    assert mlp_flag_off.tandem_gate is None
    assert set(baseline.state_dict()) == set(mlp_flag_off.state_dict())
    assert baseline.num_parameters() == mlp_flag_off.num_parameters()
    baseline.eval()
    mlp_flag_off.eval()
    x = torch.randint(0, 256, (2, 16))
    with torch.no_grad():
        _, la = baseline(x)
        _, lb = mlp_flag_off(x)
    assert torch.equal(la, lb), "the MLP flag is not a no-op when the tandem is OFF"


def test_two_way_gate_unchanged_when_mlp_disabled() -> None:
    """With the MLP flag OFF the 2-way tandem is byte-for-byte identical under a seed."""
    torch.manual_seed(11)
    a = build_model(_full_cfg(_on_cfg()))
    torch.manual_seed(11)
    b = build_model(_full_cfg(_on_cfg(tandem_mlp_enabled=False)))
    a.eval()
    b.eval()
    x = torch.randint(1, 256, (2, 24))
    with torch.no_grad():
        _, la = a(x)
        _, lb = b(x)
    assert torch.equal(la, lb)


def test_three_way_builds_mlp_and_gate() -> None:
    on3 = build_model(_full_cfg(_on3_cfg()))
    assert on3.tandem_mlp is not None, "3-way must construct the MLP workhorse"
    assert on3.causal_reasoner is not None
    assert on3.tandem_gate is not None
    d = on3.d_model
    assert on3.tandem_gate.in_features == 4 * d  # [mem, reason, mlp, ctx]
    assert on3.tandem_gate.out_features == 3 * d  # per-channel: 3 experts x d_model
    assert any(k.startswith("tandem_mlp") for k in on3.state_dict())
    assert on3.num_parameters() > build_model(_full_cfg(_on_cfg())).num_parameters()


def test_three_way_scalar_gate_builds() -> None:
    on3 = build_model(_full_cfg(_on3_cfg(tandem_gate_scalar=True)))
    assert on3.tandem_gate is not None
    assert on3.tandem_gate.out_features == 3  # scalar 3-way: one weight per expert
    on3.train()
    x = torch.randint(1, 256, (2, 20))
    out = on3.tandem_step(x, aux_query_pos=torch.tensor([19, 10]))
    assert out["gate"].shape == (2, 20, 3)


def test_three_way_gate_is_softmax_over_experts() -> None:
    """The reported 3-way gate is a per-position distribution over {mem, reason, mlp}."""
    torch.manual_seed(0)
    on3 = build_model(_full_cfg(_on3_cfg(gate_mix_warmup_steps=0)))
    on3.train()
    b, t = 3, 24
    x = torch.randint(1, 256, (b, t))
    out = on3.tandem_step(x, aux_query_pos=torch.tensor([t - 1, 12, 5]))
    gate = out["gate"]
    assert gate.shape == (b, t, 3)
    sums = gate.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), "gate does not sum to 1 over experts"
    assert bool((gate >= 0).all())


def test_three_way_force_expert_routes_each_pathway() -> None:
    """force_gate = per-row expert index in {0,1,2} routes wholly to mem / reason / mlp;
    the three routes must give pairwise-different logits (distinct pathways)."""
    torch.manual_seed(0)
    on3 = build_model(_full_cfg(_on3_cfg()))
    on3.eval()
    b, t = 2, 20
    x = torch.randint(1, 256, (b, t))
    aqp = torch.tensor([t - 1, 10])
    with torch.no_grad():
        mem = on3.tandem_step(x, aux_query_pos=aqp, force_gate=torch.full((b,), 0))["logits"]
        rea = on3.tandem_step(x, aux_query_pos=aqp, force_gate=torch.full((b,), 1))["logits"]
        mlp = on3.tandem_step(x, aux_query_pos=aqp, force_gate=torch.full((b,), 2))["logits"]
    assert (mem - rea).abs().max().item() > 1e-4, "memory vs reasoner route identical"
    assert (mem - mlp).abs().max().item() > 1e-4, "memory vs mlp route identical"
    assert (rea - mlp).abs().max().item() > 1e-4, "reasoner vs mlp route identical"


def test_three_way_per_row_curriculum_routing() -> None:
    """A per-row expert-index gate_mix routes each row to its specialist: row i's fused
    logits match the force-to-expert route for its assigned expert."""
    torch.manual_seed(0)
    on3 = build_model(_full_cfg(_on3_cfg()))
    on3.eval()
    b, t = 3, 20
    x = torch.randint(1, 256, (b, t))
    aqp = torch.tensor([t - 1, 10, 15])
    with torch.no_grad():
        mem = on3.tandem_step(x, aux_query_pos=aqp, force_gate=torch.full((b,), 0))["logits"]
        rea = on3.tandem_step(x, aux_query_pos=aqp, force_gate=torch.full((b,), 1))["logits"]
        mlp = on3.tandem_step(x, aux_query_pos=aqp, force_gate=torch.full((b,), 2))["logits"]
        # Route row0->mem(0), row1->reason(1), row2->mlp(2) in ONE call.
        mix = on3.tandem_step(x, aux_query_pos=aqp, force_gate=torch.tensor([0, 1, 2]))["logits"]
    assert torch.allclose(mix[0], mem[0], atol=1e-5), "row0 did not match the memory route"
    assert torch.allclose(mix[1], rea[1], atol=1e-5), "row1 did not match the reasoner route"
    assert torch.allclose(mix[2], mlp[2], atol=1e-5), "row2 did not match the mlp route"


def test_three_way_lm_leak_probe_exactly_zero() -> None:
    """3-way ON: perturbing tokens > t leaves logits <= t EXACTLY 0 (the MLP is
    per-token on h_embed -> adds no future dependence)."""
    torch.manual_seed(0)
    on3 = build_model(_full_cfg(_on3_cfg(reasoning_segment_len=64)))
    x = torch.randint(1, 256, (2, 48))
    max_diff = _lm_leak_maxdiff(on3, x, t_pert=19)
    assert max_diff == 0.0, f"3-way LM leak: max|Δlogit| at <=t = {max_diff:.3e} != 0.0"


def test_three_way_answer_prediction_is_leak_free() -> None:
    """3-way copy-leak guard: perturbing token p leaves logits[p-1] unchanged; p is the trap."""
    torch.manual_seed(0)
    on3 = build_model(_full_cfg(_on3_cfg(reasoning_segment_len=64)))
    on3.eval()
    b, t, p = 2, 40, 25
    x = torch.randint(1, 256, (b, t))
    with torch.no_grad():
        _, logits = on3(x)
    x_pert = x.clone()
    x_pert[:, p] = (x[:, p] + 7) % 256
    with torch.no_grad():
        _, logits_pert = on3(x_pert)
    leak_free = (logits[:, p - 1] - logits_pert[:, p - 1]).abs().max().item()
    trap = (logits[:, p] - logits_pert[:, p]).abs().max().item()
    assert leak_free == 0.0, f"3-way prediction position p-1 leaked by {leak_free:.3e}"
    assert trap > 0.0, "logits[p] did not depend on the answer byte (guard toothless)"


def test_three_way_finite_grads_reach_all_pathways() -> None:
    """A 3-way answer-CE loss sends finite non-zero grads to reasoner, gate, AND mlp."""
    torch.manual_seed(0)
    on3 = build_model(_full_cfg(_on3_cfg(gate_mix_warmup_steps=0)))
    on3.train()
    b, t = 4, 32
    x = torch.randint(1, 256, (b, t))
    aux_pos = torch.full((b,), t - 1)
    tgt = torch.randint(0, 256, (b,))
    out = on3.tandem_step(x, aux_query_pos=aux_pos)
    ans_logits = out["logits"][torch.arange(b), aux_pos]
    loss = torch.nn.functional.cross_entropy(ans_logits, tgt)
    loss.backward()

    def _has_grad(mod: torch.nn.Module) -> bool:
        return any(
            p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
            for p in mod.parameters()
            if p.requires_grad
        )

    assert _has_grad(on3.causal_reasoner), "no finite grad reached the reasoner"
    assert _has_grad(on3.tandem_gate), "no finite grad reached the gate"
    assert _has_grad(on3.tandem_mlp), "no finite grad reached the MLP workhorse"


def test_three_way_forced_mix_then_release() -> None:
    """Forced-mix warmup pins the reported gate near uniform (1/3); after warmup the
    learned softmax gate takes over (not uniform)."""
    torch.manual_seed(0)
    on3 = build_model(_full_cfg(_on3_cfg(gate_mix_warmup_steps=2)))
    on3.train()
    x = torch.randint(1, 256, (2, 32))
    aux_pos = torch.tensor([31, 20])
    warm = on3.tandem_step(x, aux_query_pos=aux_pos)
    assert warm["gate_mix"] == 0.5  # the forced-mix sentinel (interpreted as uniform 1/3)
    on3.tandem_step(x, aux_query_pos=aux_pos)  # exhaust the 2-step warmup
    out = on3.tandem_step(x, aux_query_pos=aux_pos)
    assert out["gate_mix"] is None
    uniform = torch.full_like(out["gate"], 1.0 / 3.0)
    assert not torch.allclose(out["gate"], uniform), "learned gate stayed uniform after release"


def test_three_way_states_contract_holds() -> None:
    """The (loss, logits, states) carry path still works with the 3-way tandem ON."""
    cfg = _full_cfg(_on3_cfg())
    model = build_model(cfg)
    model.eval()
    x = torch.randint(0, cfg.model.vocab_size, (2, 16))
    with torch.no_grad():
        out = model(x, None, None, True)
    assert len(out) == 3
    _, logits, states = out
    assert logits.shape == (2, 16, cfg.model.vocab_size)
    assert len(states) == cfg.model.delta_layers


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


# ---------------------------------------------------------------------------
# query_forward fast path wired into tandem_step (card cff8f5ee)
# ---------------------------------------------------------------------------
# The tandem trainer reads every loss + aux at the single answer position, so running the
# reasoner ONLY there (query_forward, O(K*L)) is loss/grad-identical to the full O(K*L^2)
# per-position walk but cheap enough to train the deep k_train walk at the routing-stable
# batch.  Default OFF => the shipped full-path forward is byte-for-byte unchanged.


def test_tandem_step_reasoner_query_only_matches_full_path() -> None:
    """``tandem_step(reasoner_query_only=True)`` is numerically IDENTICAL to the full
    per-position walk AT the query position — logits, gate, AND aux (seed_logits, walk_w) —
    through the full 3-way gate fusion.  This is the correctness guarantee that lets the
    trainer use the O(K*L) fast path for deep-K training loss/grad-identically."""
    torch.manual_seed(0)
    model = build_model(_full_cfg(_on3_cfg(reasoning_segment_len=32, causal_reasoner_steps=6)))
    model.eval()  # deterministic: no gate noise / forced-mix / hard-gate / dropout
    b, t = 3, 32
    x = torch.randint(1, 256, (b, t))
    qpos = torch.tensor([5, 17, 28])
    tf = torch.tensor([2, 9, 20])
    common: dict[str, Any] = {"aux_query_pos": qpos, "tf_seed": tf, "steps": 6, "collect_aux": True}
    full = model.tandem_step(x, **common, reasoner_query_only=False)
    fast = model.tandem_step(x, **common, reasoner_query_only=True)
    rows = torch.arange(b)
    assert torch.allclose(full["logits"][rows, qpos], fast["logits"][rows, qpos], atol=1e-5)
    assert torch.allclose(full["gate"][rows, qpos], fast["gate"][rows, qpos], atol=1e-5)
    assert torch.allclose(full["aux"]["seed_logits"], fast["aux"]["seed_logits"], atol=1e-5)
    assert torch.allclose(full["aux"]["walk_w"], fast["aux"]["walk_w"], atol=1e-5)


def test_tandem_step_reasoner_query_only_is_leak_free() -> None:
    """The fast-path answer at the query position depends only on tokens <= query_pos:
    perturbing a token AFTER the query position leaves the query-position logits unchanged
    (query_forward uses the same causal address window as the full path)."""
    torch.manual_seed(1)
    model = build_model(_full_cfg(_on3_cfg(reasoning_segment_len=32, causal_reasoner_steps=5)))
    model.eval()
    b, t = 2, 32
    x = torch.randint(1, 256, (b, t))
    qpos = torch.tensor([10, 15])
    with torch.no_grad():
        base = model.tandem_step(x, aux_query_pos=qpos, steps=5, collect_aux=False,
                                 reasoner_query_only=True)
        xp = x.clone()
        xp[:, 20] = (x[:, 20] + 7) % 256  # perturb a token AFTER both query positions
        pert = model.tandem_step(xp, aux_query_pos=qpos, steps=5, collect_aux=False,
                                 reasoner_query_only=True)
    rows = torch.arange(b)
    diff = (base["logits"][rows, qpos] - pert["logits"][rows, qpos]).abs().max().item()
    assert diff < 1e-6, f"fast-path query-position logits leaked from a future token (Δ={diff:.2e})"


def test_tandem_step_reasoner_query_only_requires_query_pos() -> None:
    """reasoner_query_only=True with aux_query_pos=None RAISES (does not silently fall back to
    the full O(K*L^2) walk) — the documented fast-path contract (card cff8f5ee review)."""
    torch.manual_seed(0)
    model = build_model(_full_cfg(_on3_cfg(reasoning_segment_len=16, causal_reasoner_steps=4)))
    model.eval()
    x = torch.randint(1, 256, (2, 16))
    with pytest.raises(ValueError, match="aux_query_pos"):
        model.tandem_step(x, aux_query_pos=None, reasoner_query_only=True)


# ---------------------------------------------------------------------------
# Progressive R-depth curriculum (card cff8f5ee) — the per-step depth-ramp hook
# ---------------------------------------------------------------------------
# The ramp deepens the reasoner's TRAINING chain depth (r_depth_start -> r_depth_max)
# so the weight-tied walk controller groks the deep walk (lifts acc_R@32 off the
# trained-walk-budget cliff — card 0a98292b).  It is additive + default-OFF; the OFF
# branch must consume the RNG IDENTICALLY to the shipped rng.choice(train_r_depths).


def test_r_depth_ramp_defaults_off() -> None:
    """The progressive ramp is OFF by default -> the shipped fixed train_r_depths."""
    cfg = TandemConfig()
    assert cfg.r_depth_ramp is False
    assert cfg.train_r_depths == (4, 5, 6)


def test_r_depth_ramp_off_is_rng_preserving() -> None:
    """Ramp OFF: ``_ramp_depth`` draws IDENTICALLY to the shipped
    ``rng.choice(train_r_depths)`` — same depth sequence AND same downstream RNG state,
    so the 2-way / non-ramp reproduction is byte-for-byte unchanged."""
    import random

    cfg = TandemConfig()  # r_depth_ramp=False
    r1, r2 = random.Random(123), random.Random(123)
    ramp_draws = [_ramp_depth(cfg, s, r1) for s in range(200)]
    choice_draws = [r2.choice(cfg.train_r_depths) for _ in range(200)]
    assert ramp_draws == choice_draws
    assert r1.getstate() == r2.getstate()  # no extra RNG consumed


def test_r_depth_ramp_schedule_and_bounds() -> None:
    """Ramp ON: hold at the floor during the delay; a WIDENING window ``[start, cur_max]``
    (never below the floor, never above ``r_depth_max``); the max reaches ``r_depth_max``
    after ``delay + ramp_steps``; shallow depths stay in the mix throughout (anti-forget)."""
    import random

    cfg = TandemConfig(
        r_depth_ramp=True, r_depth_start=4, r_depth_max=28,
        r_depth_ramp_steps=3500, r_depth_ramp_delay=1500,
    )
    rng = random.Random(0)
    # Delay hold: exactly the floor.
    assert all(_ramp_depth(cfg, s, rng) == 4 for s in range(1500) for _ in range(30))
    # Global bounds over the whole run.
    alld = [_ramp_depth(cfg, s, rng) for s in range(7000) for _ in range(3)]
    assert min(alld) == cfg.r_depth_start and max(alld) == cfg.r_depth_max
    # After delay + ramp_steps the max reaches the ceiling, but the WIDENING window still
    # samples the floor (shallow walks stay in the mix — no forgetting).
    late = [_ramp_depth(cfg, s, rng) for s in range(1500 + 3500, 7000) for _ in range(50)]
    assert max(late) == cfg.r_depth_max and min(late) == cfg.r_depth_start
    # The per-step max is monotone non-decreasing as the ramp climbs.
    maxes = [max(_ramp_depth(cfg, s, rng) for _ in range(300)) for s in range(1500, 5000, 350)]
    assert all(b >= a for a, b in zip(maxes, maxes[1:]))


def test_r_depth_ramp_option_b_starts_at_step_zero() -> None:
    """Option B (delay 0) ramps from step 0 (still the floor at step 0); by the end the
    widening window spans the full ``[start, max]`` range."""
    import random

    cfg = TandemConfig(r_depth_ramp=True, r_depth_ramp_delay=0, r_depth_ramp_steps=100)
    assert _ramp_depth(cfg, 0, random.Random(1)) == cfg.r_depth_start
    rng = random.Random(1)
    late = [_ramp_depth(cfg, s, rng) for s in range(100, 400) for _ in range(30)]
    assert max(late) == cfg.r_depth_max and min(late) == cfg.r_depth_start


@pytest.mark.parametrize("delay", [0, 3])
def test_r_depth_ramp_trains_end_to_end_2way(delay: int) -> None:
    """The ramp integrates into the real training loop end-to-end (2-way M+R — no text8
    dependency): a tiny ramped run trains + evals without error and returns per-depth
    accuracies.  Exercises ``_ramp_depth`` inside ``_train_one`` + ``make_stream_batch``
    at ramped depth (both Option A ``delay>0`` and Option B ``delay=0``)."""
    cfg = TandemConfig(
        r_depth_ramp=True, r_depth_start=2, r_depth_max=6, r_depth_ramp_steps=6,
        r_depth_ramp_delay=delay, train_r_depths=(2, 6),
        k_train=8, train_steps=12, batch_size=8, seg_len=128, d_model=32,
        delta_n_heads=2, delta_head_dim=16, eval_batches=1, eval_batch=8,
        test_r_depths=(2, 6), locate_warmup=4, gate_mix_warmup=4, type_warmup=4,
        gate_commit_anneal=4, lr_warmup=2, log_every=100,
    )
    import warnings

    with warnings.catch_warnings():
        # These tiny delays (0/3) are below the safe lock point and intentionally warn (that
        # contract has its own test); here we only exercise the ramp mechanics end-to-end.
        warnings.simplefilter("ignore")
        out = _train_one(cfg, seed=0, device=torch.device("cpu"), verbose=False)
    assert "acc_R" in out and set(out["acc_R"]) == {"2", "6"}
    assert 0.0 <= out["acc_M"] <= 1.0
    assert all(0.0 <= v <= 1.0 for v in out["acc_R"].values())


def test_r_depth_ramp_requires_k_train_covers_max_depth() -> None:
    """The ramp requires ``k_train >= r_depth_max`` (the walk must reach the deepest
    chain's root, else the per-hop walk-aux never sees the answer clause) — a too-small
    walk budget raises a clear error rather than silently training a broken walk."""
    cfg = TandemConfig(r_depth_ramp=True, r_depth_max=28, k_train=8)
    with pytest.raises(ValueError, match="k_train"):
        _train_one(cfg, seed=0, device=torch.device("cpu"), verbose=False)


# --- routing-locks-before-ramp invariant (card cff8f5ee review MAJOR) -------------------
# The invariant must hold at the RUNTIME path, not just the CLI: a raw TandemConfig must NOT
# silently use the known-broken concurrent (delay-0) schedule.


def test_r_depth_ramp_delay_default_is_sentinel_and_resolves_safe() -> None:
    """A raw TandemConfig(r_depth_ramp=True) defaults to the SENTINEL (-1), which resolves to
    the safe lock-before-ramp point (release + gate_commit_anneal) — NOT the broken delay 0."""
    cfg = TandemConfig(r_depth_ramp=True, mlp_enabled=True, type_warmup=1200, gate_commit_anneal=900)
    assert cfg.r_depth_ramp_delay == -1  # sentinel default
    assert _resolve_ramp_delay(cfg) == 1200 + 900  # 2100
    # type_warmup=0 (flat-mix / 2-way) falls back to the model's gate_mix_warmup as the release.
    cfg2 = TandemConfig(r_depth_ramp=True, type_warmup=0, gate_mix_warmup=600, gate_commit_anneal=900)
    assert _resolve_ramp_delay(cfg2) == 600 + 900


def test_resolve_ramp_delay_explicit_safe_does_not_warn() -> None:
    """An explicit delay at/above the safe point is honoured with no warning."""
    import warnings

    cfg = TandemConfig(r_depth_ramp=True, type_warmup=600, gate_commit_anneal=900,
                       r_depth_ramp_delay=2000)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning becomes a test failure
        assert _resolve_ramp_delay(cfg) == 2000


def test_resolve_ramp_delay_too_short_warns_but_honours() -> None:
    """An explicit delay shorter than the safe lock point WARNS (footgun never silent) yet is
    honoured — e.g. the documented Option-B concurrent ablation (delay 0)."""
    cfg = TandemConfig(r_depth_ramp=True, type_warmup=600, gate_commit_anneal=900,
                       r_depth_ramp_delay=0)
    with pytest.warns(UserWarning, match="routing basin commits"):
        assert _resolve_ramp_delay(cfg) == 0


def test_train_one_resolves_sentinel_delay_before_ramp() -> None:
    """_train_one resolves the sentinel to the safe delay for EVERY entry point (a raw config
    that never touched the CLI): a tiny ramped run with delay=-1 records the resolved delay."""
    cfg = TandemConfig(
        r_depth_ramp=True, r_depth_start=2, r_depth_max=6, r_depth_ramp_steps=4,
        r_depth_ramp_delay=-1, train_r_depths=(2, 6), k_train=8, train_steps=12, batch_size=8,
        seg_len=128, d_model=32, delta_n_heads=2, delta_head_dim=16, eval_batches=1, eval_batch=8,
        test_r_depths=(2, 6), locate_warmup=4, gate_mix_warmup=4, type_warmup=4,
        gate_commit_anneal=4, lr_warmup=2, log_every=100,
    )
    _train_one(cfg, seed=0, device=torch.device("cpu"), verbose=False)
    assert cfg.r_depth_ramp_delay == 4 + 4  # release(type_warmup=4) + gate_commit_anneal(4)


# --- CLI derived-defaults are unit-testable via _build_run_config (review MAJOR part b) ---


def test_cli_mlp_rramp_derives_robust_recipe() -> None:
    """--mlp --r-ramp derives the shipped robust recipe (the CLI derived-defaults logic)."""
    cfg, seeds, out = _build_run_config(["--mlp", "--r-ramp", "--seeds", "0,1,2", "--out", "x.json"])
    assert cfg.mlp_enabled is True and cfg.r_depth_ramp is True
    assert cfg.type_warmup == 1200
    assert cfg.k_train == 30
    assert cfg.batch_size == 40
    assert cfg.train_steps == 8500
    assert cfg.train_r_depths == (4, 28)
    assert cfg.r_depth_ramp_delay == -1  # sentinel -> resolved to 2100 at runtime
    assert _resolve_ramp_delay(cfg) == 2100
    assert seeds == (0, 1, 2) and out == "x.json"


def test_cli_overrides_type_warmup_and_ramp_delay() -> None:
    """Explicit --type-warmup / --ramp-delay override the derived defaults."""
    cfg, _, _ = _build_run_config(
        ["--mlp", "--r-ramp", "--type-warmup", "800", "--ramp-delay", "1500"]
    )
    assert cfg.type_warmup == 800
    assert cfg.r_depth_ramp_delay == 1500  # explicit, not the sentinel


def test_cli_no_ramp_leaves_shipped_defaults() -> None:
    """Without --r-ramp / --mlp the shipped 2-way defaults are unchanged (additive flags)."""
    cfg, _, _ = _build_run_config([])
    assert cfg.r_depth_ramp is False and cfg.mlp_enabled is False
    assert cfg.train_r_depths == (4, 5, 6) and cfg.k_train == 8
