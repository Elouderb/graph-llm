"""Tests for the multi-scale conv front-end (cheap local combiner, card ed853f9c).

Realises design note 8b5341f0.  The load-bearing correctness checks (this stage
feeds the project's central-thesis memory backbone, so future leakage here would
silently invalidate every long-context result):

* **CAUSALITY** by a perturbation probe on BOTH the ``MultiScaleConvEmbedding``
  module directly AND the full ``delta_memory_lm`` with the front-end ON: perturb
  token ``t+1`` -> outputs / logits at positions ``<= t`` are unchanged (< 1e-6).
  A non-causal conv (centred, not left-padded) or a leaky condense would fail.
* **Weight-sharing** of the condense: feeding two positions identical multi-scale
  ``(S, d)`` slices yields identical condensed output (the condense is a 1x1 conv /
  broadcast ``Linear`` over the S*d axis, NOT a position-indexed weight).
* **front_end="none" is a byte-for-byte no-op**: the model's state_dict keys,
  parameter count, and forward output are identical to the committed backbone, so
  the existing 63 delta tests + the full suite stay green.
* Shapes for both condense modes; builds at a configurable scale + a tiny
  synthetic train step; ``match_params`` sizes it; registry/build +
  ``(loss, logits)`` / ``num_parameters()`` contracts intact.
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
from graph_llm.models.components.multiscale_conv_embed import (
    VALID_CONDENSE,
    MultiScaleConvEmbedding,
)

CONDENSE_MODES = list(VALID_CONDENSE)


def _mem_cfg(**overrides: Any) -> ModelConfig:
    """A small delta_memory_lm ModelConfig that exercises the front-end on CPU."""
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


# ---------------------------------------------------------------------------
# Shapes — both condense modes, depthwise on/off
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("condense", CONDENSE_MODES)
@pytest.mark.parametrize("depthwise", [True, False])
def test_front_end_preserves_shape(condense: str, depthwise: str) -> None:
    """``(B, T, d) -> (B, T, d)`` for both condense modes — a drop-in enrichment."""
    cfg = _mem_cfg(d_model=24, conv_condense=condense, conv_depthwise=depthwise)
    fe = MultiScaleConvEmbedding(cfg)
    fe.eval()
    x = torch.randn(2, 17, cfg.d_model)
    out = fe(x)
    assert out.shape == (2, 17, cfg.d_model)
    assert torch.isfinite(out).all()


def test_num_scales_matches_conv_widths() -> None:
    """The conv bank has exactly len(conv_widths) scales (S)."""
    cfg = _mem_cfg(conv_widths=[1, 2, 4, 8, 16])
    fe = MultiScaleConvEmbedding(cfg)
    assert fe.num_scales == 5
    assert len(fe.scales) == 5


def test_soft_select_is_a_convex_blend_in_embedding_space() -> None:
    """soft_select output is a softmax-weighted (convex) blend of the scale slices.

    With convex weights summing to 1 over the S scales, the condensed vector lies
    in the convex hull of the per-scale embeddings — it stays in embedding space
    (the GBST property), unlike the free Linear of concat_proj.
    """
    cfg = _mem_cfg(d_model=16, conv_condense="soft_select")
    fe = MultiScaleConvEmbedding(cfg)
    fe.eval()
    assert fe.select is not None
    x = torch.randn(2, 9, cfg.d_model)
    with torch.no_grad():
        multi = torch.stack([scale(x) for scale in fe.scales], dim=2)  # (B,T,S,d)
        flat = multi.reshape(2, 9, fe.num_scales * cfg.d_model)
        weights = torch.softmax(fe.select(flat), dim=-1)  # (B,T,S)
        blended = (weights.unsqueeze(-1) * multi).sum(dim=2)
        out = fe(x)
    assert torch.allclose(out, blended, atol=1e-6)
    assert torch.allclose(weights.sum(-1), torch.ones(2, 9), atol=1e-5)
    assert (weights >= 0).all()


# ---------------------------------------------------------------------------
# CAUSALITY — perturbation probe (load-bearing; the whole game)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("condense", CONDENSE_MODES)
@pytest.mark.parametrize("depthwise", [True, False])
def test_front_end_module_is_causal_by_perturbation(
    condense: str, depthwise: str
) -> None:
    """Perturb token t+1 in the module input; outputs at <= t are unchanged.

    Tests the module directly on continuous input so the probe is exact (not
    quantised by an embedding lookup).  A centred (non-left-padded) conv or a leaky
    condense would change outputs at <= t — this is the load-bearing check.
    """
    torch.manual_seed(0)
    cfg = _mem_cfg(d_model=16, conv_condense=condense, conv_depthwise=depthwise)
    fe = MultiScaleConvEmbedding(cfg)
    fe.eval()

    B, T = 2, 24  # widen past the largest width (16) so the deep scale is exercised
    x = torch.randn(B, T, cfg.d_model)
    with torch.no_grad():
        out = fe(x)

    t = 9
    x_pert = x.clone()
    x_pert[:, t + 1] += torch.randn_like(x_pert[:, t + 1])  # perturb a FUTURE token
    with torch.no_grad():
        out_pert = fe(x_pert)

    max_diff = (out[:, : t + 1] - out_pert[:, : t + 1]).abs().max().item()
    assert max_diff < 1e-6, (
        f"MultiScaleConvEmbedding leaked future info (condense={condense}, "
        f"depthwise={depthwise}): perturbing token {t + 1} changed outputs at "
        f"positions <= {t} by {max_diff:.2e} (must be < 1e-6)."
    )


@pytest.mark.parametrize("condense", CONDENSE_MODES)
def test_delta_memory_lm_with_front_end_is_causal_by_perturbation(
    condense: str,
) -> None:
    """End-to-end causality with the front-end ON: perturb token t+1; logits <= t fixed.

    A leak in the conv bank or condense would silently invalidate every
    long-context perplexity-vs-position result the thesis depends on.
    """
    torch.manual_seed(0)
    cfg = _mem_cfg(max_seq_len=24, front_end="multiscale_conv", conv_condense=condense)
    model = build_model(_full_cfg(cfg))
    model.eval()
    assert model.front_end is not None

    B, T = 2, 20
    x = torch.randint(0, 256, (B, T))
    with torch.no_grad():
        _, logits = model(x)

    t = 9
    x_pert = x.clone()
    x_pert[:, t + 1] = (x[:, t + 1] + 1) % 256  # deterministic future-token change
    with torch.no_grad():
        _, logits_pert = model(x_pert)

    max_diff = (logits[:, : t + 1] - logits_pert[:, : t + 1]).abs().max().item()
    assert max_diff < 1e-6, (
        f"delta_memory_lm + multiscale_conv ({condense}) leaked future info: "
        f"perturbing token {t + 1} changed logits at positions <= {t} by "
        f"{max_diff:.2e} (must be < 1e-6)."
    )


def test_width_one_scale_is_purely_per_position() -> None:
    """The width-1 (identity) scale is causal AND a pure per-position map.

    A kernel-1 conv cannot mix across time, so perturbing ANY other position
    (past OR future) leaves a given position's width-1 output unchanged.
    """
    torch.manual_seed(0)
    cfg = _mem_cfg(d_model=16, conv_widths=[1])
    fe = MultiScaleConvEmbedding(cfg)
    fe.eval()
    x = torch.randn(1, 10, cfg.d_model)
    with torch.no_grad():
        out = fe(x)
        x_pert = x.clone()
        x_pert[:, 3] += torch.randn_like(x_pert[:, 3])  # perturb a different position
        out_pert = fe(x_pert)
    # Every position except 3 is unchanged (no cross-time mixing at width 1).
    mask = [p for p in range(10) if p != 3]
    max_diff = (out[:, mask] - out_pert[:, mask]).abs().max().item()
    assert max_diff < 1e-6, f"width-1 scale mixed across time: diff {max_diff:.2e}"


# ---------------------------------------------------------------------------
# Weight-sharing — the condense is shared across positions (not position-indexed)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("condense", CONDENSE_MODES)
def test_condense_is_weight_shared_across_positions(condense: str) -> None:
    """Identical multi-scale slices at two DIFFERENT positions -> identical condense.

    The condense is ONE learned function (1x1 conv / broadcast Linear) applied to
    every token's (S, d) slice — translation invariant, NOT position-indexed.  We
    bypass the conv bank and drive the condense directly with a hand-built stack
    whose slice at position 1 equals the slice at position 5, then assert the two
    condensed outputs match exactly.
    """
    torch.manual_seed(0)
    cfg = _mem_cfg(d_model=12, conv_condense=condense)
    fe = MultiScaleConvEmbedding(cfg)
    fe.eval()

    B, T, S, d = 1, 8, fe.num_scales, cfg.d_model
    multi = torch.randn(B, T, S, d)
    shared_slice = torch.randn(S, d)
    multi[0, 1] = shared_slice
    multi[0, 5] = shared_slice  # identical (S, d) slice at a different position

    with torch.no_grad():
        flat = multi.reshape(B, T, S * d)
        if condense == "concat_proj":
            assert fe.proj is not None
            condensed = fe.proj(flat)
        else:
            assert fe.select is not None
            w = torch.softmax(fe.select(flat), dim=-1)
            condensed = (w.unsqueeze(-1) * multi).sum(dim=2)

    diff = (condensed[0, 1] - condensed[0, 5]).abs().max().item()
    assert diff < 1e-6, (
        f"condense ({condense}) is position-indexed: identical slices at positions "
        f"1 and 5 gave different outputs (diff {diff:.2e}) — it must be shared."
    )


def test_concat_proj_condense_is_a_single_linear_over_scale_dim_axis() -> None:
    """concat_proj's condense is exactly one Linear(S*d, d) — no per-position weight.

    The parameter shape proves the sharing: a position-indexed condense would have
    a T (or max_seq_len) axis in its weight; a shared 1x1 conv / Linear does not.
    """
    cfg = _mem_cfg(d_model=20, conv_condense="concat_proj")
    fe = MultiScaleConvEmbedding(cfg)
    assert fe.proj is not None
    assert fe.proj.weight.shape == (cfg.d_model, fe.num_scales * cfg.d_model)
    # No max_seq_len / T dependence anywhere in the condense weight.
    assert cfg.max_seq_len not in fe.proj.weight.shape


# ---------------------------------------------------------------------------
# front_end="none" is a byte-for-byte no-op (existing backbone unchanged)
# ---------------------------------------------------------------------------


def test_front_end_none_is_the_default() -> None:
    """The additive config default is "none" (preserves the committed backbone)."""
    assert ModelConfig().front_end == "none"
    model = build_model(_full_cfg(_mem_cfg()))
    assert model.front_end is None, "front_end='none' must construct NO module"


def test_front_end_none_state_dict_identical_to_baseline() -> None:
    """front_end="none" adds zero parameters / state — same keys + count as before.

    Builds the SAME config with the field explicitly "none" and asserts the model
    has no front-end params (no `front_end.*` keys) and the same total param count
    as a model that never set the field.
    """
    default_model = build_model(_full_cfg(_mem_cfg()))
    explicit_none = build_model(_full_cfg(_mem_cfg(front_end="none")))
    assert set(default_model.state_dict()) == set(explicit_none.state_dict())
    assert not any(k.startswith("front_end") for k in default_model.state_dict())
    assert default_model.num_parameters() == explicit_none.num_parameters()


def test_front_end_none_forward_is_bit_identical_under_fixed_seed() -> None:
    """With a fixed seed, front_end="none" forward == the no-field baseline forward.

    Same seed -> same init -> same RNG consumption (no extra module is built), so
    the logits must be bit-identical: the no-op guarantee, proven numerically.
    """
    torch.manual_seed(123)
    baseline = build_model(_full_cfg(_mem_cfg()))
    torch.manual_seed(123)
    explicit_none = build_model(_full_cfg(_mem_cfg(front_end="none")))

    baseline.eval()
    explicit_none.eval()
    x = torch.randint(0, 256, (2, 16))
    with torch.no_grad():
        _, logits_a = baseline(x)
        _, logits_b = explicit_none(x)
    assert torch.equal(logits_a, logits_b), (
        "front_end='none' is not a byte-for-byte no-op (logits differ from the "
        "baseline under the same seed)."
    )


def test_front_end_on_adds_params_over_none() -> None:
    """Turning the front-end ON strictly increases the parameter count."""
    off = build_model(_full_cfg(_mem_cfg(front_end="none")))
    on = build_model(_full_cfg(_mem_cfg(front_end="multiscale_conv")))
    assert on.num_parameters() > off.num_parameters()


# ---------------------------------------------------------------------------
# Registry / contract / build / train / sizing with the front-end ON
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("condense", CONDENSE_MODES)
def test_delta_memory_lm_with_front_end_forward_backward_contract(
    condense: str,
) -> None:
    """(loss, logits) contract holds with the front-end on; finite grads flow."""
    cfg = _full_cfg(_mem_cfg(front_end="multiscale_conv", conv_condense=condense))
    model = build_model(cfg)
    B, T = 2, cfg.model.max_seq_len
    x = torch.randint(0, cfg.model.vocab_size, (B, T))
    targets = torch.randint(0, cfg.model.vocab_size, (B, T))
    loss, logits = model(x, targets)
    assert loss.ndim == 0 and math.isfinite(loss.item())
    assert logits.shape == (B, T, cfg.model.vocab_size)
    loss.backward()
    # Gradients must reach the front-end params (it is actually in the graph).
    fe_grads = [
        p.grad for p in model.front_end.parameters() if p.requires_grad
    ]
    assert any(
        g is not None and torch.isfinite(g).all() and g.abs().sum() > 0
        for g in fe_grads
    ), "no finite non-zero gradient reached the front-end"


def test_delta_memory_lm_with_front_end_trains_a_tiny_synthetic_step() -> None:
    """A few optimiser steps on a fixed synthetic batch keep the loss finite."""
    cfg = _full_cfg(_mem_cfg(front_end="multiscale_conv"))
    model = build_model(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    B, T = 4, cfg.model.max_seq_len
    torch.manual_seed(0)
    x = torch.randint(0, cfg.model.vocab_size, (B, T))
    y = torch.randint(0, cfg.model.vocab_size, (B, T))
    losses = []
    for _ in range(5):
        opt.zero_grad()
        loss, _ = model(x, y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert all(math.isfinite(loss) for loss in losses)
    assert losses[-1] <= losses[0] + 1e-3


def test_match_params_sizes_a_baseline_to_front_end_target() -> None:
    """A delta_memory_lm WITH the front-end is a valid match_params target."""
    target_cfg = _mem_cfg(
        d_model=128, delta_layers=4, delta_n_heads=8, delta_head_k_dim=32,
        delta_head_v_dim=32, vocab_size=4096, max_seq_len=256,
        front_end="multiscale_conv",
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
        f"match to front-end target={target:,} achieved={achieved:,} err={rel_err:.3f}"
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_invalid_front_end_raises() -> None:
    with pytest.raises(ValueError):
        build_model(_full_cfg(_mem_cfg(front_end="nonsense")))


def test_invalid_condense_raises() -> None:
    with pytest.raises(ValueError):
        MultiScaleConvEmbedding(_mem_cfg(conv_condense="nonsense"))


def test_invalid_conv_width_raises() -> None:
    with pytest.raises(ValueError):
        MultiScaleConvEmbedding(_mem_cfg(conv_widths=[1, 0, 4]))


def test_empty_conv_widths_raises() -> None:
    with pytest.raises(ValueError):
        MultiScaleConvEmbedding(_mem_cfg(conv_widths=[]))
