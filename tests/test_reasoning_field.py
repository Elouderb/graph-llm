"""Tests for the transient reasoning field (R-LM step 1, card 9907dc9e).

The reasoning field ports the R2-validated iterated 2-D reasoner into
``delta_memory_lm`` as a config-flagged, input-seeded, transient, CAUSAL block.
The load-bearing correctness checks:

* **reasoning_enabled=False is a byte-for-byte NO-OP**: default config, state_dict
  keys, parameter count, and forward logits are identical to the committed
  backbone under a fixed seed (``torch.equal``).  The (loss, logits) /
  (loss, logits, states) Trainer contracts are unchanged, so the existing suite
  stays green.
* **CAUSALITY** by a perturbation probe (the #1 correctness risk — future leakage
  in an autoregressive LM would silently invalidate every result): perturbing
  token ``t``'s input leaves the reasoning contribution at positions ``!= t``
  (in particular ``< t``) unchanged (< 1e-6).  The reasoner runs INDEPENDENTLY per
  position (field seeded from ``h_t`` alone, no cross-position mixing), so this is
  structural — proven both at the module level and end-to-end on the full LM.
* **Trains when ON**: the (loss, logits) contract holds, finite non-zero grads
  reach the reasoner, and a tiny synthetic train run stays finite & non-increasing
  (no NaN / instability with the conservative R2 recipe).
* The mechanism is the R2 one (mandatory per-step sharpen, delta-write off);
  invalid config raises; shapes; transient (no carry) interaction.
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
from graph_llm.models.baselines import count_params, match_params
from graph_llm.models.components.reasoning_field import (
    VALID_REASONING_FIELDS,
    ReasoningField,
)


def _mem_cfg(**overrides: Any) -> ModelConfig:
    """A small delta_memory_lm ModelConfig that exercises the reasoner on CPU."""
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
        "max_seq_len": 32,
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
    """A small config with the reasoner ON (kept tiny: 4x4 grid, K=4)."""
    base: dict[str, Any] = {
        "reasoning_enabled": True,
        "reasoning_rows": 4,
        "reasoning_cols": 4,
        "reasoning_steps": 4,
        "reasoning_d_cell": 16,
        "reasoning_d_ctrl": 32,
        "reasoning_key_dim": 16,
    }
    base.update(overrides)
    return _mem_cfg(**base)


# ---------------------------------------------------------------------------
# reasoning_enabled=False is a byte-for-byte no-op (existing backbone unchanged)
# ---------------------------------------------------------------------------


def test_reasoning_disabled_is_the_default() -> None:
    """The additive config default is OFF (preserves the committed backbone)."""
    assert ModelConfig().reasoning_enabled is False
    model = build_model(_full_cfg(_mem_cfg()))
    assert model.reasoner is None, "reasoning_enabled=False must construct NO module"


def test_reasoning_off_state_dict_identical_to_baseline() -> None:
    """reasoning_enabled=False adds zero parameters / state — same keys + count.

    Builds the SAME config with the field explicitly False and asserts the model
    has no reasoner params (no `reasoner.*` keys) and the same total param count
    as a model that never set the field.
    """
    default_model = build_model(_full_cfg(_mem_cfg()))
    explicit_off = build_model(_full_cfg(_mem_cfg(reasoning_enabled=False)))
    assert set(default_model.state_dict()) == set(explicit_off.state_dict())
    assert not any(k.startswith("reasoner") for k in default_model.state_dict())
    assert default_model.num_parameters() == explicit_off.num_parameters()


def test_reasoning_off_forward_is_bit_identical_under_fixed_seed() -> None:
    """With a fixed seed, reasoning OFF forward == the no-field baseline forward.

    Same seed -> same init -> same RNG consumption (no extra module is built), so
    the logits must be bit-identical: the no-op guarantee, proven numerically.
    """
    torch.manual_seed(123)
    baseline = build_model(_full_cfg(_mem_cfg()))
    torch.manual_seed(123)
    explicit_off = build_model(_full_cfg(_mem_cfg(reasoning_enabled=False)))

    baseline.eval()
    explicit_off.eval()
    x = torch.randint(0, 256, (2, 16))
    with torch.no_grad():
        _, logits_a = baseline(x)
        _, logits_b = explicit_off(x)
    assert torch.equal(logits_a, logits_b), (
        "reasoning_enabled=False is not a byte-for-byte no-op (logits differ from "
        "the baseline under the same seed)."
    )


def test_reasoning_off_states_contract_bit_identical() -> None:
    """The (loss, logits, states) carry contract is byte-for-byte identical OFF.

    Exercises the return_states path (cross-segment carry) to confirm the no-op
    holds there too — the reasoner sits outside the carry plumbing.
    """
    torch.manual_seed(7)
    baseline = build_model(_full_cfg(_mem_cfg()))
    torch.manual_seed(7)
    explicit_off = build_model(_full_cfg(_mem_cfg(reasoning_enabled=False)))
    baseline.eval()
    explicit_off.eval()
    x = torch.randint(0, 256, (2, 16))
    with torch.no_grad():
        out_a = baseline(x, None, None, True)
        out_b = explicit_off(x, None, None, True)
    loss_a, logits_a, states_a = out_a
    loss_b, logits_b, states_b = out_b
    assert torch.equal(logits_a, logits_b)
    assert torch.equal(loss_a, loss_b)
    assert len(states_a) == len(states_b)


def test_reasoning_on_adds_params_over_off() -> None:
    """Turning the reasoner ON strictly increases the parameter count."""
    off = build_model(_full_cfg(_mem_cfg(reasoning_enabled=False)))
    on = build_model(_full_cfg(_on_cfg()))
    assert on.num_parameters() > off.num_parameters()
    assert any(k.startswith("reasoner") for k in on.state_dict())


# ---------------------------------------------------------------------------
# CAUSALITY — perturbation probe (load-bearing; the whole game)
# ---------------------------------------------------------------------------


def test_reasoning_contribution_is_per_position_independent() -> None:
    """Module-level: perturbing h at position t changes the reasoning output ONLY
    at position t (every other position unchanged < 1e-6).

    The reasoner runs independently per position (field seeded from h_t alone, no
    cross-position mixing).  This is the structural causality guarantee tested in
    the cleanest possible form: continuous input, every other position fixed.
    """
    torch.manual_seed(0)
    cfg = _on_cfg(d_model=16)
    reasoner = ReasoningField(cfg)
    reasoner.eval()

    b, t = 2, 12
    h = torch.randn(b, t, cfg.d_model)
    with torch.no_grad():
        out = reasoner(h)

    pos = 5
    h_pert = h.clone()
    h_pert[:, pos] += torch.randn_like(h_pert[:, pos])
    with torch.no_grad():
        out_pert = reasoner(h_pert)

    mask = [p for p in range(t) if p != pos]
    max_diff = (out[:, mask] - out_pert[:, mask]).abs().max().item()
    assert max_diff < 1e-6, (
        f"reasoning field is NOT per-position independent: perturbing h at "
        f"position {pos} changed the reasoning contribution at other positions by "
        f"{max_diff:.2e} (must be < 1e-6)."
    )
    # And it DOES change the perturbed position (it is actually doing something).
    changed = (out[:, pos] - out_pert[:, pos]).abs().max().item()
    assert changed > 1e-6, "the reasoner ignored the position it was seeded from"


def test_delta_memory_lm_reasoning_is_causal_by_perturbation() -> None:
    """End-to-end: perturb token t's input; the reasoning contribution at positions
    < t is unchanged (< 1e-6).  No future leakage.

    Isolates the reasoning contribution (logits_on - logits_off on the SAME shared
    backbone weights) so the probe measures the reasoner's effect specifically, not
    the already-causal delta backbone.  This is the card's exact acceptance probe.
    """
    torch.manual_seed(0)
    # Build ON, then build OFF sharing the ON model's backbone weights so the
    # difference is PURELY the reasoner's contribution.
    on = build_model(_full_cfg(_on_cfg()))
    off = build_model(_full_cfg(_mem_cfg(reasoning_enabled=False)))
    # Copy every shared (non-reasoner) parameter from `on` into `off`.
    off_state = off.state_dict()
    for k, v in on.state_dict().items():
        if not k.startswith("reasoner") and k in off_state:
            off_state[k].copy_(v)
    off.load_state_dict(off_state)
    on.eval()
    off.eval()

    b, t = 2, 20
    x = torch.randint(0, 256, (b, t))

    def reasoning_contribution(tokens: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            _, logits_on = on(tokens)
            _, logits_off = off(tokens)
        return logits_on - logits_off  # the reasoner's effect on the logits

    contrib = reasoning_contribution(x)

    t_pert = 9
    x_pert = x.clone()
    x_pert[:, t_pert] = (x[:, t_pert] + 1) % 256  # change token t's INPUT
    contrib_pert = reasoning_contribution(x_pert)

    # The reasoning contribution at positions < t must be unchanged.
    max_diff = (contrib[:, :t_pert] - contrib_pert[:, :t_pert]).abs().max().item()
    assert max_diff < 1e-6, (
        f"reasoning field leaked future info: perturbing token {t_pert} changed the "
        f"reasoning contribution at positions < {t_pert} by {max_diff:.2e} "
        f"(must be < 1e-6)."
    )


def test_delta_memory_lm_reasoning_on_whole_model_is_causal() -> None:
    """End-to-end on the full ON model: perturb token t; logits at < t unchanged.

    The composite (causal delta backbone + per-position reasoner) must itself be
    causal — perturbing token t cannot change ANY logit at positions < t.
    """
    torch.manual_seed(1)
    on = build_model(_full_cfg(_on_cfg()))
    on.eval()
    b, t = 2, 18
    x = torch.randint(0, 256, (b, t))
    with torch.no_grad():
        _, logits = on(x)
    t_pert = 7
    x_pert = x.clone()
    x_pert[:, t_pert] = (x[:, t_pert] + 1) % 256
    with torch.no_grad():
        _, logits_pert = on(x_pert)
    max_diff = (logits[:, :t_pert] - logits_pert[:, :t_pert]).abs().max().item()
    assert max_diff < 1e-6, (
        f"full ON model leaked future info: perturbing token {t_pert} changed "
        f"logits at positions < {t_pert} by {max_diff:.2e} (must be < 1e-6)."
    )


# ---------------------------------------------------------------------------
# Trains when ON — contract, gradients, finite training
# ---------------------------------------------------------------------------


def test_reasoning_on_forward_backward_contract() -> None:
    """(loss, logits) contract holds with the reasoner on; finite grads flow to it."""
    cfg = _full_cfg(_on_cfg())
    model = build_model(cfg)
    b, t = 2, cfg.model.max_seq_len
    x = torch.randint(0, cfg.model.vocab_size, (b, t))
    targets = torch.randint(0, cfg.model.vocab_size, (b, t))
    loss, logits = model(x, targets)
    assert loss.ndim == 0 and math.isfinite(loss.item())
    assert logits.shape == (b, t, cfg.model.vocab_size)
    loss.backward()
    assert model.reasoner is not None
    reasoner_grads = [
        p.grad for p in model.reasoner.parameters() if p.requires_grad
    ]
    assert any(
        g is not None and torch.isfinite(g).all() and g.abs().sum() > 0
        for g in reasoner_grads
    ), "no finite non-zero gradient reached the reasoning field"


def test_reasoning_on_trains_a_tiny_synthetic_step() -> None:
    """A few optimiser steps on a fixed synthetic batch keep the loss finite."""
    cfg = _full_cfg(_on_cfg())
    model = build_model(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    b, t = 4, cfg.model.max_seq_len
    torch.manual_seed(0)
    x = torch.randint(0, cfg.model.vocab_size, (b, t))
    y = torch.randint(0, cfg.model.vocab_size, (b, t))
    losses = []
    for _ in range(8):
        opt.zero_grad()
        loss, _ = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
    assert all(math.isfinite(loss) for loss in losses)
    assert losses[-1] <= losses[0] + 1e-3


def test_reasoning_on_states_contract_holds() -> None:
    """The (loss, logits, states) carry path still works with the reasoner ON.

    The reasoner is transient (per-position, no carry), so it composes with the
    return_states path without changing its shape contract.
    """
    cfg = _full_cfg(_on_cfg())
    model = build_model(cfg)
    model.eval()
    x = torch.randint(0, cfg.model.vocab_size, (2, 16))
    with torch.no_grad():
        out = model(x, None, None, True)
    assert len(out) == 3
    loss, logits, states = out
    assert logits.shape == (2, 16, cfg.model.vocab_size)
    assert len(states) == cfg.model.delta_layers


# ---------------------------------------------------------------------------
# Mechanism — the R2 recipe properties
# ---------------------------------------------------------------------------


def test_field_shapes_and_output() -> None:
    """``(B, T, d) -> (B, T, d)`` reasoning contribution, finite."""
    cfg = _on_cfg(d_model=24)
    reasoner = ReasoningField(cfg)
    reasoner.eval()
    h = torch.randn(3, 11, cfg.d_model)
    out = reasoner(h)
    assert out.shape == (3, 11, cfg.d_model)
    assert torch.isfinite(out).all()
    assert reasoner.n_cells == cfg.reasoning_rows * cfg.reasoning_cols


def test_head_stays_a_normalised_distribution_each_step() -> None:
    """Mandatory sharpen + softmax keep the soft head a valid distribution (sums
    to 1, non-negative) at every iterated step — it never disperses to NaN/blows up.

    Re-runs the internal loop exposing the head each step to assert the R2
    invariant the mandatory sharpening protects.
    """
    torch.manual_seed(0)
    cfg = _on_cfg(d_model=16)
    r = ReasoningField(cfg)
    r.eval()
    p = 5
    h = torch.randn(p, cfg.d_model)
    with torch.no_grad():
        field = r.seed(h).view(p, r.n_cells, r.d_cell)
        keys = r._cell_keys(h.device)
        keys_n = torch.nn.functional.normalize(keys, dim=1).unsqueeze(0)
        ctrl = torch.zeros(p, r.d_ctrl)
        w = torch.zeros(p, r.n_cells)
        w[:, 0] = 1.0
        scale = torch.nn.functional.softplus(r.key_scale)
        for step in range(r.steps):
            phi = 2.0 * math.pi * step / max(1, r.steps)
            clk = torch.tensor([math.sin(phi), math.cos(phi)]).expand(p, 2)
            rd = r._soft_read(field, w)
            ctrl = r.gru(torch.cat([rd, clk], dim=1), ctrl)
            key = torch.nn.functional.normalize(r.to_key(ctrl), dim=1)
            betak = torch.nn.functional.softplus(r.to_betak(ctrl))
            g = torch.sigmoid(r.to_gate(ctrl))
            s = torch.nn.functional.softmax(r.to_shift(ctrl), dim=1)
            gamma = r.gamma_floor + torch.nn.functional.softplus(r.to_gamma(ctrl))
            cos = torch.einsum("pd,bnd->pn", key, keys_n)
            w_c = torch.nn.functional.softmax(scale * betak * cos, dim=1)
            w = g * w_c + (1.0 - g) * w
            w = r._shift2d(w, s)
            w = r._sharpen(w, gamma)
            assert torch.allclose(w.sum(1), torch.ones(p), atol=1e-5)
            assert (w >= 0).all()
            assert torch.isfinite(w).all()
            # gamma respects the mandatory floor every step.
            assert (gamma >= r.gamma_floor - 1e-6).all()


def test_match_params_sizes_a_baseline_to_reasoning_target() -> None:
    """A delta_memory_lm WITH the reasoner is a valid match_params target."""
    target_cfg = _on_cfg(
        d_model=128, delta_layers=4, delta_n_heads=8, delta_head_k_dim=32,
        delta_head_v_dim=32, vocab_size=4096, max_seq_len=256,
    )
    target = count_params(build_model(_full_cfg(target_cfg)))
    assert target > 0
    base = Config(
        model=ModelConfig(
            name="transformer", vocab_size=4096, n_heads=8, n_layers=6,
            d_ff=1024, max_seq_len=256, dropout=0.0,
        )
    )
    matched = match_params(target, base, tolerance=0.05)
    achieved = count_params(build_model(matched))
    rel_err = abs(achieved - target) / target
    assert rel_err <= 0.05, (
        f"match to reasoning target={target:,} achieved={achieved:,} err={rel_err:.3f}"
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_valid_reasoning_fields_constant() -> None:
    assert "grid" in VALID_REASONING_FIELDS


def test_invalid_reasoning_field_raises() -> None:
    with pytest.raises(ValueError):
        ReasoningField(_on_cfg(reasoning_field="nonsense"))


def test_invalid_reasoning_steps_raises() -> None:
    with pytest.raises(ValueError):
        ReasoningField(_on_cfg(reasoning_steps=0))


def test_invalid_reasoning_grid_raises() -> None:
    with pytest.raises(ValueError):
        ReasoningField(_on_cfg(reasoning_rows=0))


def test_gamma_floor_below_one_raises() -> None:
    """A sharpen floor < 1 would BLUR the head each step — reject it (mandatory
    sharpening is the load-bearing R2 trick)."""
    with pytest.raises(ValueError):
        ReasoningField(_on_cfg(reasoning_gamma_floor=0.5))
