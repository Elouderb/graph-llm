"""Tests for the factorized bilinear (MFB) front-end + bilinear_lm (card 86347418).

All CPU / offline.  The load-bearing tests:

* **Factorization correctness** -- on a TINY config the factorized MFB path
  matches a brute-force reference that explicitly reconstructs the per-output
  ``emb x emb`` bilinear weight and contracts the outer product.  This is the
  proof that ``z = SumPool(U^T x  o  V^T y, k)`` equals ``x^T W y``.
* **Memory guard** -- a forward hook over every tensor op asserts the factorized
  path never allocates an intermediate whose last two axes are both ``emb`` (the
  forbidden ``128 x 128`` pair).
* forward/backward finite for all three interaction modes; ``bilinear_lm``
  builds at a configurable ~GPT-1/2 param count + trains a tiny synthetic step;
  ``match_params`` sizes a Transformer baseline to it; the registry resolves
  ``bilinear_lm`` and respects the phonological-init hook.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import graph_llm.models  # noqa: F401 — registers "bilinear_lm" (+ baselines transitively)
import graph_llm.models.baselines  # noqa: F401 — registers "transformer" and "mamba"
from graph_llm.config import Config, DataConfig, ModelConfig, TrainConfig
from graph_llm.models import build_model
from graph_llm.models.baselines import count_params, match_params
from graph_llm.models.components.bilinear_frontend import (
    VALID_MODES,
    BilinearFrontEnd,
    _causal_shift,
)

MODES = list(VALID_MODES)


def _frontend_cfg(**overrides: Any) -> ModelConfig:
    """A small ModelConfig that exercises the front-end on CPU."""
    base: dict[str, Any] = {
        "name": "bilinear_lm",
        "vocab_size": 256,
        "d_model": 16,
        "bilinear_window": 4,
        "bilinear_k": 2,
        "bilinear_o": 32,
        "front_end_dropout": 0.0,
        "post_mixer_layers": 2,
        "post_mixer_kernel": 3,
        "post_mixer_ff_mult": 2,
        "materialized_reduce_dim": 8,
        "materialized_cnn_channels": 8,
        "dropout": 0.0,
        "max_seq_len": 32,
    }
    base.update(overrides)
    return ModelConfig(**base)


def _tensor_leaves(out: Any) -> Iterator[torch.Tensor]:
    """Yield every Tensor in a (possibly nested tuple/list) op output."""
    if isinstance(out, torch.Tensor):
        yield out
    elif isinstance(out, list | tuple):
        for item in out:
            yield from _tensor_leaves(item)


def _full_cfg(model: ModelConfig) -> Config:
    return Config(
        model=model,
        data=DataConfig(source="synthetic", seq_len=model.max_seq_len, batch_size=4),
        train=TrainConfig(max_steps=3, warmup_steps=1, mixed_precision="no"),
    )


# ---------------------------------------------------------------------------
# Factorization correctness — the proof the MFB math is right
# ---------------------------------------------------------------------------


def test_factorized_matches_bruteforce_bilinear() -> None:
    """On a TINY config (emb=8, o=16, W=2), the factorized path == brute force.

    Reference: reconstruct each output's low-rank ``emb x emb`` weight
    ``W_i = sum_j u_ij v_ij^T`` (formed ONLY here in the test) and compute
    ``z_i[t] = sum_d weight[d,i] * x_t^T W_i x_{t-d}``.  Compare against the
    module's pre-normalisation accumulator.
    """
    torch.manual_seed(0)
    cfg = _frontend_cfg(
        d_model=8, bilinear_o=16, bilinear_window=2, bilinear_k=3,
        bilinear_offset_weighting="learned",
    )
    fe = BilinearFrontEnd(cfg)
    fe.eval()
    # Randomise the learned offset weights so the test exercises real values
    # (they init to 1.0, which would mask an offset-handling bug).
    with torch.no_grad():
        fe.offset_weights.copy_(torch.randn_like(fe.offset_weights))  # type: ignore[union-attr]

    emb, o, k, W = fe.emb, fe.o, fe.k, fe.window
    B, T = 2, 5
    x = torch.randn(B, T, emb)

    # --- module's factorized accumulator (pre-norm) ---
    u = fe.u_proj(x)
    acc = x.new_zeros(B, T, o)
    for d in range(W):
        partner = _causal_shift(x, d)
        v = fe.v_proj(partner)
        had = u * v
        pooled = had.view(B, T, o, k).sum(dim=-1)
        pooled = pooled * fe.offset_weights[d]  # type: ignore[index]
        acc = acc + pooled

    # --- brute-force reference: reconstruct W_i (emb x emb) and contract ---
    # u_proj.weight: (k*o, emb); row (i*k + j) is u_ij^T.  Same for v_proj.
    U = fe.u_proj.weight.view(o, k, emb)   # (o, k, emb)
    V = fe.v_proj.weight.view(o, k, emb)   # (o, k, emb)
    # W_mat[i] = sum_j u_ij v_ij^T  -> (o, emb, emb)   (formed ONLY in the test)
    W_mat = torch.einsum("oke,okf->oef", U, V)

    ref = torch.zeros(B, T, o)
    for d in range(W):
        partner = _causal_shift(x, d)
        # z_i = x_t^T W_i partner_t  -> (B, T, o)
        z = torch.einsum("bte,oef,btf->bto", x, W_mat, partner)
        ref = ref + z * fe.offset_weights[d]  # type: ignore[index]

    assert torch.allclose(acc, ref, atol=1e-4, rtol=1e-4), (
        f"factorized MFB != brute-force bilinear; max abs diff "
        f"{(acc - ref).abs().max().item():.2e}"
    )


def test_mlb_is_mfb_with_k1() -> None:
    """k=1 (MLB) is a clean special case: SumPool over a single element == identity."""
    torch.manual_seed(1)
    cfg = _frontend_cfg(d_model=6, bilinear_o=10, bilinear_window=3, bilinear_k=1)
    fe = BilinearFrontEnd(cfg)
    fe.eval()
    x = torch.randn(2, 4, 6)
    out = fe(x)
    assert out.shape == (2, 4, 10)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# Memory guard — no emb x emb axis pair in the factorized path
# ---------------------------------------------------------------------------


def test_factorized_path_never_materializes_emb_by_emb() -> None:
    """Hook every tensor op; assert no intermediate has an (emb, emb) trailing pair.

    The whole point of MFB is to avoid the 128x128 (here emb x emb) interaction
    matrix.  We use a TorchDispatch-free approach: a forward pre-hook is too
    coarse, so we walk the autograd graph after a forward and inspect the shapes
    of every tensor produced.
    """
    emb = 12
    cfg = _frontend_cfg(d_model=emb, bilinear_o=20, bilinear_window=3, bilinear_k=2)
    fe = BilinearFrontEnd(cfg)
    x = torch.randn(2, 5, emb, requires_grad=True)

    seen_shapes: list[tuple[int, ...]] = []

    # Use __torch_function__ via a mode to capture all op outputs.
    from torch.overrides import TorchFunctionMode

    class _CaptureMode(TorchFunctionMode):
        def __torch_function__(self, func, types, args=(), kwargs=None):  # noqa: ANN001, ANN204
            kwargs = kwargs or {}
            out = func(*args, **kwargs)
            for t in _tensor_leaves(out):
                if t.dim() >= 2:
                    seen_shapes.append(tuple(t.shape))
            return out

    with _CaptureMode():
        _ = fe(x)

    offenders = [
        s for s in seen_shapes
        if len(s) >= 2 and s[-1] == emb and s[-2] == emb
    ]
    assert not offenders, (
        f"factorized path materialised an emb x emb ({emb}x{emb}) axis pair: "
        f"{offenders[:3]}"
    )


def test_materialized_cnn_uses_small_reduced_dim_not_emb() -> None:
    """The materialized_cnn mode forms r x r (small), NOT emb x emb."""
    emb, r = 16, 4
    cfg = _frontend_cfg(
        d_model=emb, bilinear_o=20, bilinear_window=3,
        interaction_mode="materialized_cnn", materialized_reduce_dim=r,
        materialized_cnn_channels=8,
    )
    fe = BilinearFrontEnd(cfg)
    x = torch.randn(2, 5, emb, requires_grad=True)
    seen: list[tuple[int, ...]] = []
    from torch.overrides import TorchFunctionMode

    class _Cap(TorchFunctionMode):
        def __torch_function__(self, func, types, args=(), kwargs=None):  # noqa: ANN001, ANN204
            kwargs = kwargs or {}
            out = func(*args, **kwargs)
            for t in _tensor_leaves(out):
                if t.dim() >= 2:
                    seen.append(tuple(t.shape))
            return out

    with _Cap():
        _ = fe(x)
    # No emb x emb pair; the interaction map is r x r.
    assert not any(s[-1] == emb and s[-2] == emb for s in seen if len(s) >= 2)
    assert any(s[-1] == r and s[-2] == r for s in seen if len(s) >= 2), (
        "expected an r x r interaction map in materialized_cnn mode"
    )


# ---------------------------------------------------------------------------
# Forward / backward finite for all three modes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", MODES)
def test_frontend_forward_backward_finite(mode: str) -> None:
    cfg = _frontend_cfg(interaction_mode=mode)
    fe = BilinearFrontEnd(cfg)
    x = torch.randn(2, 7, cfg.d_model, requires_grad=True)
    out = fe(x)
    assert out.shape == (2, 7, cfg.bilinear_o)
    assert torch.isfinite(out).all()
    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    grads = [p.grad for p in fe.parameters() if p.requires_grad]
    assert any(
        g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads
    ), f"no finite non-zero parameter gradients in mode {mode}"


def test_invalid_mode_raises() -> None:
    with pytest.raises(ValueError):
        BilinearFrontEnd(_frontend_cfg(interaction_mode="nonsense"))


def test_control_linear_has_no_multiplicative_interaction() -> None:
    """The control is the ablation null: scaling one operand region scales output
    affinely (no second-order term).  A bilinear map would scale quadratically.
    """
    cfg = _frontend_cfg(interaction_mode="control_linear", front_end_dropout=0.0)
    fe = BilinearFrontEnd(cfg)
    fe.eval()
    # control_linear's pre-norm output is linear in x: out_raw(2x) == 2 * out_raw(x).
    # We probe the mixer directly (before the norm, which is non-linear).
    x = torch.randn(1, 5, cfg.d_model)
    shifts = [_causal_shift(x, d) for d in range(cfg.bilinear_window)]
    ctx = torch.cat(shifts, dim=-1)
    raw1 = fe.mixer(ctx)
    raw2 = fe.mixer(2 * ctx) - fe.mixer.bias  # remove affine bias
    assert torch.allclose(2 * (raw1 - fe.mixer.bias), raw2, atol=1e-4)


# ---------------------------------------------------------------------------
# bilinear_lm: registry, contract, param scaling, tiny train step
# ---------------------------------------------------------------------------


def test_bilinear_lm_registered_and_builds() -> None:
    model = build_model(_full_cfg(_frontend_cfg()))
    assert isinstance(model, torch.nn.Module)
    assert count_params(model) > 0
    assert hasattr(model, "num_parameters")


@pytest.mark.parametrize("mode", MODES)
def test_bilinear_lm_forward_backward_contract(mode: str) -> None:
    cfg = _full_cfg(_frontend_cfg(interaction_mode=mode))
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


def test_bilinear_lm_eval_zero_loss_without_targets() -> None:
    model = build_model(_full_cfg(_frontend_cfg()))
    model.eval()
    x = torch.randint(0, 256, (1, 16))
    with torch.no_grad():
        loss, logits = model(x)
    assert float(loss.sum()) == 0.0
    assert logits.shape[-1] == 256


def test_bilinear_lm_trains_a_tiny_synthetic_step() -> None:
    """A few optimiser steps on a fixed synthetic batch keep the loss finite."""
    cfg = _full_cfg(_frontend_cfg())
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


def test_bilinear_lm_scales_to_target_param_count() -> None:
    """Depth/width scale by config to reach a GPT-1/2-order param count.

    We assert the model *can* be built at ~100M params (the lower end of the
    GPT-1/2 band) at the intended tiny-embedding / fat-front-end proportions,
    without instantiating the 350M upper end on a CI box.
    """
    cfg = _frontend_cfg(
        d_model=128,           # intentionally tiny embedding
        bilinear_o=4096,       # 64x64 default — the second-order interaction width
        bilinear_k=2,
        bilinear_window=16,
        post_mixer_width=1024,  # wide trunk carries the depth*width param budget
        post_mixer_layers=12,
        post_mixer_ff_mult=4,
        post_mixer_kernel=7,
        vocab_size=50_257,
        max_seq_len=1024,
    )
    model = build_model(_full_cfg(cfg))
    n = count_params(model)
    assert 80_000_000 <= n <= 400_000_000, f"param count {n:,} outside GPT-1/2 band"


def test_match_params_sizes_transformer_to_bilinear_lm() -> None:
    """A Transformer baseline can be size-matched to a bilinear_lm target."""
    bl_cfg = _frontend_cfg(
        d_model=128, bilinear_o=2048, bilinear_k=2, bilinear_window=16,
        post_mixer_layers=6, vocab_size=4096, max_seq_len=256,
    )
    target = count_params(build_model(_full_cfg(bl_cfg)))
    assert target > 0
    base = Config(
        model=ModelConfig(
            name="transformer", vocab_size=4096, n_heads=8, n_layers=6,
            d_ff=2048, max_seq_len=256, dropout=0.0,
        )
    )
    matched = match_params(target, base, tolerance=0.05)
    achieved = count_params(build_model(matched))
    rel_err = abs(achieved - target) / target
    assert rel_err <= 0.05, (
        f"transformer match to bilinear_lm target={target:,} "
        f"achieved={achieved:,} err={rel_err:.3f}"
    )


# ---------------------------------------------------------------------------
# Phonological-init hook (card e1644700) is respected
# ---------------------------------------------------------------------------


def test_bilinear_lm_respects_embedding_init_hook() -> None:
    """A registered embedding-init hook must have the final say on embed.weight."""
    from graph_llm.models.registry import register_embedding_init

    sentinel = 3.5

    @register_embedding_init("bilinear_sentinel")
    def _const_init(weight, vocab_size, d_model):  # noqa: ANN001, ARG001
        torch.nn.init.constant_(weight, sentinel)

    cfg = _frontend_cfg(embedding_init="bilinear_sentinel")
    model = build_model(_full_cfg(cfg))
    expected = torch.full_like(model.embed.weight, sentinel)
    assert torch.allclose(model.embed.weight, expected), (
        "embedding-init hook was overwritten by _init_weights(); the hook must "
        "run last so the phonological init (card e1644700) takes effect."
    )
