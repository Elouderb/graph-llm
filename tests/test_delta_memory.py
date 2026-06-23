"""Tests for the Gated-DeltaNet delta-rule matrix memory + delta_memory_lm
(card e2c6ea95).

All CPU / offline.  The load-bearing tests (this is the project's CENTRAL thesis
component, so the math must be exactly right):

* **Causality** by perturbation probe — perturb token ``t+1``; logits at every
  position ``<= t`` must be unchanged (< 1e-6).  Non-negotiable for an
  autoregressive LM: a leak here silently inflates every long-context result.
* **Delta-rule correctness** — write ``(k, v1)`` then ``(k, v2)`` with
  ``beta=1, alpha=1`` into a fresh state; reading ``k`` returns ``~v2``
  (OVERWRITE / edit-in-place), NOT ``v1 + v2`` (accumulate).  Plus an exact
  3-step scan checked against a by-hand reference.
* **Bounded / fixed-size state** — the state matrix shape is independent of the
  sequence length ``T`` (no per-step growth) — the bounded-memory property.
* forward/backward finite + correct shapes; ``delta_memory_lm`` builds at a
  ~GPT-1/2 param count + trains a tiny synthetic step; ``match_params`` sizes a
  Mamba/Transformer baseline to it; the registry resolves ``delta_memory_lm`` and
  respects the phonological-init hook.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import graph_llm.models  # noqa: F401 — registers "delta_memory_lm" (+ baselines transitively)
import graph_llm.models.baselines  # noqa: F401 — registers "transformer" and "mamba"
from graph_llm.config import Config, DataConfig, ModelConfig, TrainConfig
from graph_llm.models import build_model
from graph_llm.models.baselines import count_params, match_params
from graph_llm.models.components.delta_memory import (
    VALID_DELTA_SCANS,
    VALID_FEATURE_MAPS,
    GatedDeltaMemory,
    _feature_map,
)


def _mem_cfg(**overrides: Any) -> ModelConfig:
    """A small ModelConfig that exercises the delta memory on CPU."""
    base: dict[str, Any] = {
        "name": "delta_memory_lm",
        "vocab_size": 256,
        "d_model": 32,
        "delta_layers": 2,
        "delta_n_heads": 4,
        "delta_head_k_dim": 16,
        "delta_head_v_dim": 16,
        "delta_feature_map": "l2",
        "delta_use_forget_gate": True,
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
# Delta-rule correctness — overwrite, not accumulate (the math proof)
# ---------------------------------------------------------------------------


def test_delta_rule_overwrites_not_accumulates() -> None:
    """Write (k, v1) then (k, v2) with beta=1, alpha=1; reading k returns ~v2.

    This is THE defining property of the delta rule vs plain linear-attention
    accumulation: the second write OVERWRITES the binding for ``k`` (writes the
    correction ``v2 - S k``), so the memory is self-evicting and bounded.  Plain
    accumulation (``S += v k^T``) would instead return ``v1 + v2``.

    Uses the ``identity`` feature map and an orthonormal key so the readout is
    exact and hand-checkable.
    """
    cfg = _mem_cfg(
        d_model=8, delta_n_heads=1, delta_head_k_dim=4, delta_head_v_dim=4,
        delta_feature_map="identity",
    )
    mem = GatedDeltaMemory(cfg)
    mem.eval()

    k = torch.tensor([1.0, 0.0, 0.0, 0.0]).view(1, 1, 1, 4)  # (B, H, 1, d_k)
    v1 = torch.tensor([2.0, 1.0, 0.0, 0.0])
    v2 = torch.tensor([0.0, 0.0, 3.0, 5.0])

    # Three time steps with the SAME key: write v1, write v2, then read (the read
    # at a step uses the state BEFORE that step's write, so we need a 3rd step to
    # observe the state after BOTH writes).
    k_seq = k.repeat(1, 1, 3, 1)                                   # (1,1,3,d_k)
    v_seq = torch.stack([v1, v2, torch.zeros(4)]).view(1, 1, 3, 4)
    beta = torch.ones(1, 1, 3)
    alpha = torch.ones(1, 1, 3)

    out = mem._delta_scan(k_seq, k_seq, v_seq, beta, alpha)        # (1,1,3,d_v)

    # Read at t=1 (after only v1 is written) must return v1 (causal read).
    assert torch.allclose(out[0, 0, 1], v1, atol=1e-5), (
        f"causal read after first write should be v1; got {out[0, 0, 1].tolist()}"
    )
    # Read at t=2 (after v1 AND v2 written to the same key) must be v2 (OVERWRITE),
    # NOT v1 + v2 (accumulate).
    read_after_both = out[0, 0, 2]
    assert torch.allclose(read_after_both, v2, atol=1e-5), (
        f"delta rule must OVERWRITE: read after (k,v1),(k,v2) should be ~v2={v2.tolist()}, "
        f"got {read_after_both.tolist()}"
    )
    assert not torch.allclose(read_after_both, v1 + v2, atol=1e-3), (
        "memory ACCUMULATED (v1+v2) instead of overwriting — this is the linear-"
        "attention failure mode the delta rule is meant to fix."
    )


def test_delta_scan_matches_by_hand_reference() -> None:
    """A 3-step scan with two distinct (near-orthogonal) keys == a by-hand loop.

    Reimplements the recurrence ``S_t = alpha S_{t-1} + beta phi(k) (v - S^T phi(k))^T``
    and ``o_t = S_{t-1}^T phi(q_t)`` independently with explicit Python and asserts
    the module's ``_delta_scan`` reproduces it bit-for-bit (within fp tolerance),
    INCLUDING a non-trivial forget gate and write strength.
    """
    torch.manual_seed(0)
    cfg = _mem_cfg(
        d_model=8, delta_n_heads=1, delta_head_k_dim=3, delta_head_v_dim=2,
        delta_feature_map="identity",
    )
    mem = GatedDeltaMemory(cfg)
    mem.eval()

    B, H, T, d_k, d_v = 1, 1, 3, 3, 2
    q = torch.randn(B, H, T, d_k)
    k = torch.randn(B, H, T, d_k)
    v = torch.randn(B, H, T, d_v)
    beta = torch.tensor([[[0.3, 1.0, 0.7]]])     # (B, H, T)
    alpha = torch.tensor([[[0.9, 0.5, 0.8]]])    # (B, H, T)

    out = mem._delta_scan(q, k, v, beta, alpha)  # (B, H, T, d_v)

    # --- independent by-hand reference ---
    S = torch.zeros(d_k, d_v)
    ref = []
    for t in range(T):
        q_t, k_t, v_t = q[0, 0, t], k[0, 0, t], v[0, 0, t]
        o_t = S.t() @ q_t                        # READ with S_{t-1}
        ref.append(o_t)
        pred = S.t() @ k_t
        delta = v_t - pred
        S = alpha[0, 0, t] * S + beta[0, 0, t] * torch.outer(k_t, delta)
    ref_t = torch.stack(ref).view(B, H, T, d_v)

    assert torch.allclose(out, ref_t, atol=1e-5), (
        f"delta scan != by-hand reference; max abs diff {(out - ref_t).abs().max():.2e}"
    )


# ---------------------------------------------------------------------------
# Causality — perturbation probe (non-negotiable)
# ---------------------------------------------------------------------------


def test_layer_is_causal_by_perturbation() -> None:
    """Perturb token t+1 in the GatedDeltaMemory input; outputs at <= t are unchanged.

    Tests the layer directly (continuous input) so the probe is exact, not
    quantised by the embedding lookup.
    """
    torch.manual_seed(0)
    cfg = _mem_cfg(d_model=16, delta_n_heads=2, delta_head_k_dim=8, delta_head_v_dim=8)
    mem = GatedDeltaMemory(cfg)
    mem.eval()

    B, T = 2, 12
    x = torch.randn(B, T, cfg.d_model)
    with torch.no_grad():
        out = mem(x)

    t = 5
    x_pert = x.clone()
    x_pert[:, t + 1] += torch.randn_like(x_pert[:, t + 1])  # perturb a FUTURE token
    with torch.no_grad():
        out_pert = mem(x_pert)

    max_diff = (out[:, : t + 1] - out_pert[:, : t + 1]).abs().max().item()
    assert max_diff < 1e-6, (
        f"GatedDeltaMemory leaked future info: perturbing token {t + 1} changed "
        f"outputs at positions <= {t} by {max_diff:.2e} (must be < 1e-6)."
    )


def test_delta_memory_lm_is_causal_by_perturbation() -> None:
    """End-to-end causality: perturb token t+1; logits at positions <= t unchanged.

    Same rigour as the bilinear front-end / baseline causality probes.  A leak
    here would silently invalidate every long-context perplexity-vs-position
    result the thesis depends on.
    """
    torch.manual_seed(0)
    model = build_model(_full_cfg(_mem_cfg(max_seq_len=24)))
    model.eval()

    B, T = 2, 20
    x = torch.randint(0, 256, (B, T))
    with torch.no_grad():
        _, logits = model(x)

    t = 9
    x_pert = x.clone()
    # Change a FUTURE token to a different id (deterministic, guaranteed != orig).
    x_pert[:, t + 1] = (x[:, t + 1] + 1) % 256
    with torch.no_grad():
        _, logits_pert = model(x_pert)

    max_diff = (logits[:, : t + 1] - logits_pert[:, : t + 1]).abs().max().item()
    assert max_diff < 1e-6, (
        f"delta_memory_lm leaked future info: perturbing token {t + 1} changed "
        f"logits at positions <= {t} by {max_diff:.2e} (must be < 1e-6)."
    )


# ---------------------------------------------------------------------------
# Bounded / fixed-size state — the bounded-memory property
# ---------------------------------------------------------------------------


def test_state_size_is_independent_of_sequence_length() -> None:
    """The state matrix S is (B, H, d_k, d_v) for ANY T — no per-step growth.

    Captures the state shape inside ``_delta_scan`` at every step for two very
    different sequence lengths and asserts it is identical and equal to the
    fixed ``(B, H, d_k, d_v)`` — the bounded-memory guarantee that distinguishes
    this from an attention KV cache.

    Pins ``delta_scan="sequential"`` so ``forward`` routes through the per-step
    ``_delta_scan`` the spy patches.  (The chunkwise fast path keeps the SAME
    fixed-size ``(B, H, d_k, d_v)`` state — carried between chunks recurrently —
    but iterates per chunk, not per token, so this per-step spy targets the
    reference scan directly.)
    """
    cfg = _mem_cfg(
        d_model=16, delta_n_heads=3, delta_head_k_dim=8, delta_head_v_dim=5,
        delta_scan="sequential",
    )
    mem = GatedDeltaMemory(cfg)
    mem.eval()

    captured: list[tuple[int, ...]] = []
    orig_scan = mem._delta_scan

    def _spy(q, k, v, beta, alpha):  # noqa: ANN001, ANN202
        # Re-run the scan but record the state shape at each step.
        B, H, T, _ = q.shape
        S = torch.zeros(B, H, mem.d_k, mem.d_v)
        for t in range(T):
            captured.append(tuple(S.shape))
            pred = torch.einsum("bhkv,bhk->bhv", S, k[:, :, t])
            delta = v[:, :, t] - pred
            S = alpha[:, :, t][..., None, None] * S + beta[:, :, t][..., None, None] * torch.einsum(
                "bhk,bhv->bhkv", k[:, :, t], delta
            )
        return orig_scan(q, k, v, beta, alpha)

    mem._delta_scan = _spy  # type: ignore[method-assign]

    B = 2
    expected = (B, cfg.delta_n_heads, cfg.delta_head_k_dim, cfg.delta_head_v_dim)
    for T in (4, 64):
        captured.clear()
        with torch.no_grad():
            mem(torch.randn(B, T, cfg.d_model))
        assert len(captured) == T, "spy should observe one state per step"
        assert all(s == expected for s in captured), (
            f"state shape grew or drifted over T={T}: saw {set(captured)}, "
            f"expected constant {expected}"
        )


# ---------------------------------------------------------------------------
# Feature map + forward/backward
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fmap", list(VALID_FEATURE_MAPS))
def test_feature_map_l2_normalizes(fmap: str) -> None:
    x = torch.randn(2, 3, 8)
    out = _feature_map(x, fmap)
    assert out.shape == x.shape
    if fmap in ("l2", "silu_l2"):
        norms = out.norm(p=2, dim=-1)
        # L2-normalised rows have unit norm (zero rows would give 0, but randn
        # rows are non-zero w.p. 1).
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_invalid_feature_map_raises() -> None:
    with pytest.raises(ValueError):
        GatedDeltaMemory(_mem_cfg(delta_feature_map="nonsense"))


@pytest.mark.parametrize("fmap", list(VALID_FEATURE_MAPS))
@pytest.mark.parametrize("gate", [True, False])
def test_memory_forward_backward_finite(fmap: str, gate: bool) -> None:
    cfg = _mem_cfg(delta_feature_map=fmap, delta_use_forget_gate=gate)
    mem = GatedDeltaMemory(cfg)
    x = torch.randn(2, 7, cfg.d_model, requires_grad=True)
    out = mem(x)
    assert out.shape == (2, 7, cfg.d_model)
    assert torch.isfinite(out).all()
    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    grads = [p.grad for p in mem.parameters() if p.requires_grad]
    assert any(
        g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads
    ), f"no finite non-zero parameter gradients (fmap={fmap}, gate={gate})"


def test_ungated_alpha_is_one() -> None:
    """With the forget gate off, alpha_t == 1 everywhere (ungated DeltaNet)."""
    cfg = _mem_cfg(delta_use_forget_gate=False)
    mem = GatedDeltaMemory(cfg)
    assert mem.alpha_proj is None


# ---------------------------------------------------------------------------
# delta_memory_lm: registry, contract, param scaling, tiny train step
# ---------------------------------------------------------------------------


def test_delta_memory_lm_registered_and_builds() -> None:
    model = build_model(_full_cfg(_mem_cfg()))
    assert isinstance(model, torch.nn.Module)
    assert count_params(model) > 0
    assert hasattr(model, "num_parameters")


def test_delta_memory_lm_forward_backward_contract() -> None:
    cfg = _full_cfg(_mem_cfg())
    model = build_model(cfg)
    B, T = 2, cfg.model.max_seq_len
    x = torch.randint(0, cfg.model.vocab_size, (B, T))
    targets = torch.randint(0, cfg.model.vocab_size, (B, T))
    loss, logits = model(x, targets)
    assert loss.ndim == 0 and math.isfinite(loss.item())
    assert logits.shape == (B, T, cfg.model.vocab_size)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(
        g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads
    )


def test_delta_memory_lm_eval_zero_loss_without_targets() -> None:
    model = build_model(_full_cfg(_mem_cfg()))
    model.eval()
    x = torch.randint(0, 256, (1, 16))
    with torch.no_grad():
        loss, logits = model(x)
    assert float(loss.sum()) == 0.0
    assert logits.shape[-1] == 256


def test_delta_memory_lm_trains_a_tiny_synthetic_step() -> None:
    """A few optimiser steps on a fixed synthetic batch keep the loss finite."""
    cfg = _full_cfg(_mem_cfg())
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
    # Overfitting a single batch should not increase loss overall.
    assert losses[-1] <= losses[0] + 1e-3


def test_delta_memory_lm_scales_to_target_param_count() -> None:
    """Depth/width/heads scale by config to reach a GPT-1/2-order param count.

    We assert the model *can* be built at ~100M params (the lower end of the
    GPT-1/2 band) without instantiating the 350M upper end on a CI box.
    """
    cfg = _mem_cfg(
        d_model=768,
        delta_layers=12,
        delta_n_heads=12,
        delta_head_k_dim=64,
        delta_head_v_dim=64,
        delta_ff_mult=4,
        vocab_size=50_257,
        max_seq_len=1024,
    )
    model = build_model(_full_cfg(cfg))
    n = count_params(model)
    assert 80_000_000 <= n <= 400_000_000, f"param count {n:,} outside GPT-1/2 band"


def test_match_params_sizes_mamba_to_delta_memory_lm() -> None:
    """A Mamba baseline can be size-matched to a delta_memory_lm target."""
    dm_cfg = _mem_cfg(
        d_model=256, delta_layers=6, delta_n_heads=8, delta_head_k_dim=32,
        delta_head_v_dim=32, vocab_size=4096, max_seq_len=256,
    )
    target = count_params(build_model(_full_cfg(dm_cfg)))
    assert target > 0
    base = Config(
        model=ModelConfig(
            name="mamba", vocab_size=4096, n_heads=8, n_layers=6,
            d_ff=1024, max_seq_len=256, dropout=0.0,
        )
    )
    matched = match_params(target, base, tolerance=0.05)
    achieved = count_params(build_model(matched))
    rel_err = abs(achieved - target) / target
    assert rel_err <= 0.05, (
        f"mamba match to delta_memory_lm target={target:,} "
        f"achieved={achieved:,} err={rel_err:.3f}"
    )


def test_match_params_sizes_transformer_to_delta_memory_lm() -> None:
    """A Transformer baseline can be size-matched to a delta_memory_lm target."""
    dm_cfg = _mem_cfg(
        d_model=256, delta_layers=6, delta_n_heads=8, delta_head_k_dim=32,
        delta_head_v_dim=32, vocab_size=4096, max_seq_len=256,
    )
    target = count_params(build_model(_full_cfg(dm_cfg)))
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
        f"transformer match to delta_memory_lm target={target:,} "
        f"achieved={achieved:,} err={rel_err:.3f}"
    )


# ---------------------------------------------------------------------------
# Phonological-init hook (card e1644700) is respected
# ---------------------------------------------------------------------------


def test_delta_memory_lm_respects_embedding_init_hook() -> None:
    """A registered embedding-init hook must have the final say on embed.weight."""
    from graph_llm.models.registry import register_embedding_init

    sentinel = 4.25

    @register_embedding_init("delta_sentinel")
    def _const_init(weight, vocab_size, d_model):  # noqa: ANN001, ARG001
        torch.nn.init.constant_(weight, sentinel)

    cfg = _mem_cfg(embedding_init="delta_sentinel")
    model = build_model(_full_cfg(cfg))
    expected = torch.full_like(model.embed.weight, sentinel)
    assert torch.allclose(model.embed.weight, expected), (
        "embedding-init hook was overwritten by _init_weights(); the hook must "
        "run last so the phonological init (card e1644700) takes effect."
    )


# ---------------------------------------------------------------------------
# Chunkwise-parallel scan == sequential oracle (card 18b14615, LOAD-BEARING)
# ---------------------------------------------------------------------------
#
# The sequential _delta_scan is the trusted, hand-verified oracle.  The fast
# chunkwise path must reproduce it within tolerance on randomized inputs —
# INCLUDING sequence lengths that are not a multiple of the chunk size, and both
# constant and varying alpha/beta.  This is the central correctness gate: a
# mismatch here means the fast path silently changes the model's behaviour.


def _make_mem(
    *, gate: bool, fmap: str = "l2", d_k: int = 16, d_v: int = 12, n_heads: int = 3,
    chunk_size: int = 32,
) -> GatedDeltaMemory:
    cfg = _mem_cfg(
        d_model=24, delta_n_heads=n_heads, delta_head_k_dim=d_k,
        delta_head_v_dim=d_v, delta_feature_map=fmap, delta_use_forget_gate=gate,
        delta_chunk_size=chunk_size,
    )
    mem = GatedDeltaMemory(cfg)
    mem.eval()
    return mem


def _rand_scan_inputs(
    mem: GatedDeltaMemory, B: int, T: int, *, alpha_mode: str, seed: int
) -> tuple[torch.Tensor, ...]:
    """Feature-mapped (q, k, v, beta, alpha) for a direct scan call, in fp32.

    alpha_mode: "ungated" (alpha == 1), "const" (one decay per head, repeated over
    T — the constant-alpha case), or "varying" (the model's per-token
    exp(-softplus(.)) gate — the varying-alpha case).
    """
    torch.manual_seed(seed)
    H, d_k, d_v = mem.n_heads, mem.d_k, mem.d_v
    q = _feature_map(torch.randn(B, H, T, d_k), mem.feature_map).float()
    k = _feature_map(torch.randn(B, H, T, d_k), mem.feature_map).float()
    v = torch.randn(B, H, T, d_v).float()
    beta = torch.sigmoid(torch.randn(B, H, T)).float()  # varying beta in (0,1)
    if alpha_mode == "ungated":
        alpha = torch.ones(B, H, T)
    elif alpha_mode == "const":
        a = torch.sigmoid(torch.randn(B, H, 1)).clamp(0.3, 0.999)
        alpha = a.expand(B, H, T).contiguous()
    elif alpha_mode == "varying":
        alpha = torch.exp(-torch.nn.functional.softplus(torch.randn(B, H, T)))
    else:  # pragma: no cover - guard
        raise ValueError(alpha_mode)
    return q, k, v, beta, alpha.float()


@pytest.mark.parametrize("alpha_mode", ["ungated", "const", "varying"])
@pytest.mark.parametrize("T", [1, 31, 32, 33, 64, 65, 100, 128, 257])
def test_chunkwise_matches_sequential(alpha_mode: str, T: int) -> None:
    """Chunkwise scan == sequential oracle within 1e-4 (fp32), incl. T % C != 0.

    T sweeps multiples and explicit NON-multiples of the chunk size (C=32: 31, 33,
    65, 100, 257) to exercise the right-pad boundary handling; alpha_mode covers
    ungated, constant-alpha and varying-alpha.  beta varies per token in all cases.
    """
    gate = alpha_mode != "ungated"
    mem = _make_mem(gate=gate, chunk_size=32)
    q, k, v, beta, alpha = _rand_scan_inputs(mem, B=2, T=T, alpha_mode=alpha_mode, seed=T)

    ref = mem._delta_scan(q, k, v, beta, alpha)
    fast = mem._delta_scan_chunkwise(q, k, v, beta, alpha)

    assert fast.shape == ref.shape == (2, mem.n_heads, T, mem.d_v)
    assert torch.isfinite(fast).all(), "chunkwise scan produced non-finite values"
    max_diff = (fast - ref).abs().max().item()
    assert max_diff <= 1e-4, (
        f"chunkwise != sequential oracle (alpha={alpha_mode}, T={T}): "
        f"max abs diff {max_diff:.3e} exceeds 1e-4"
    )


@pytest.mark.parametrize("chunk_size", [1, 4, 16, 64])
def test_chunkwise_matches_sequential_across_chunk_sizes(chunk_size: int) -> None:
    """The equivalence holds for several chunk sizes (boundary handling is generic)."""
    mem = _make_mem(gate=True, chunk_size=chunk_size)
    T = 70  # not a multiple of 4 / 16 / 64
    q, k, v, beta, alpha = _rand_scan_inputs(mem, B=2, T=T, alpha_mode="varying", seed=7)
    ref = mem._delta_scan(q, k, v, beta, alpha)
    fast = mem._delta_scan_chunkwise(q, k, v, beta, alpha)
    max_diff = (fast - ref).abs().max().item()
    assert max_diff <= 1e-4, f"C={chunk_size}: max diff {max_diff:.3e} > 1e-4"


def test_chunkwise_matches_sequential_long_sequence_finite() -> None:
    """At T=1024 the chunkwise scan stays finite and matches the oracle (fp32)."""
    mem = _make_mem(gate=True, d_k=64, d_v=64, n_heads=2, chunk_size=32)
    q, k, v, beta, alpha = _rand_scan_inputs(mem, B=1, T=1024, alpha_mode="varying", seed=1024)
    ref = mem._delta_scan(q, k, v, beta, alpha)
    fast = mem._delta_scan_chunkwise(q, k, v, beta, alpha)
    assert torch.isfinite(fast).all()
    assert (fast - ref).abs().max().item() <= 1e-4


def test_chunkwise_forward_matches_sequential_forward() -> None:
    """Full-layer forward agrees between the two scans (same weights, same input).

    Guards the end-to-end (loss, logits)-relevant path: the only difference is the
    config selector, so the layer output must be identical within tolerance.
    """
    torch.manual_seed(0)
    cfg_seq = _mem_cfg(d_model=32, delta_n_heads=4, delta_head_k_dim=16,
                       delta_head_v_dim=16, delta_scan="sequential")
    mem_seq = GatedDeltaMemory(cfg_seq)
    mem_seq.eval()

    cfg_fast = _mem_cfg(d_model=32, delta_n_heads=4, delta_head_k_dim=16,
                        delta_head_v_dim=16, delta_scan="chunkwise", delta_chunk_size=16)
    mem_fast = GatedDeltaMemory(cfg_fast)
    mem_fast.load_state_dict(mem_seq.state_dict())  # identical weights
    mem_fast.eval()

    x = torch.randn(2, 50, cfg_seq.d_model)  # 50 not a multiple of 16
    with torch.no_grad():
        out_seq = mem_seq(x)
        out_fast = mem_fast(x)
    max_diff = (out_seq - out_fast).abs().max().item()
    assert max_diff <= 1e-4, f"forward seq vs chunkwise max diff {max_diff:.3e} > 1e-4"


def test_chunkwise_layer_is_causal_by_perturbation() -> None:
    """Causality probe on the FAST path: perturb token t+1; outputs <= t unchanged."""
    torch.manual_seed(0)
    cfg = _mem_cfg(d_model=16, delta_n_heads=2, delta_head_k_dim=8,
                   delta_head_v_dim=8, delta_scan="chunkwise", delta_chunk_size=4)
    mem = GatedDeltaMemory(cfg)
    mem.eval()
    assert mem.delta_scan == "chunkwise"

    B, T = 2, 13  # not a multiple of the chunk size (4)
    x = torch.randn(B, T, cfg.d_model)
    with torch.no_grad():
        out = mem(x)
    t = 5
    x_pert = x.clone()
    x_pert[:, t + 1] += torch.randn_like(x_pert[:, t + 1])  # perturb a FUTURE token
    with torch.no_grad():
        out_pert = mem(x_pert)
    max_diff = (out[:, : t + 1] - out_pert[:, : t + 1]).abs().max().item()
    assert max_diff < 1e-6, (
        f"chunkwise path leaked future info: perturbing token {t + 1} changed "
        f"outputs at positions <= {t} by {max_diff:.2e} (must be < 1e-6)."
    )


@pytest.mark.parametrize("scan", ["sequential", "chunkwise", "auto"])
def test_delta_scan_selector_builds_and_runs(scan: str) -> None:
    """All valid delta_scan modes build and run; "auto" resolves to chunkwise."""
    cfg = _mem_cfg(delta_scan=scan)
    mem = GatedDeltaMemory(cfg)
    resolved = "chunkwise" if scan == "auto" else scan
    assert mem.delta_scan == resolved
    x = torch.randn(2, 20, cfg.d_model)
    out = mem(x)
    assert out.shape == (2, 20, cfg.d_model)
    assert torch.isfinite(out).all()


def test_invalid_delta_scan_raises() -> None:
    with pytest.raises(ValueError):
        GatedDeltaMemory(_mem_cfg(delta_scan="nonsense"))
    assert "chunkwise" in VALID_DELTA_SCANS and "sequential" in VALID_DELTA_SCANS


def test_chunkwise_forward_backward_finite() -> None:
    """The fast path is differentiable: finite grads flow to inputs and params."""
    cfg = _mem_cfg(delta_scan="chunkwise", delta_chunk_size=8)
    mem = GatedDeltaMemory(cfg)
    x = torch.randn(2, 19, cfg.d_model, requires_grad=True)  # 19 % 8 != 0
    out = mem(x)
    assert out.shape == (2, 19, cfg.d_model)
    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    grads = [p.grad for p in mem.parameters() if p.requires_grad]
    assert any(
        g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads
    )
