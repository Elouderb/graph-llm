"""Parameter-counting and size-matching utilities (card 424e3a8e).

The whole point of the baseline suite is an *apples-to-apples* comparison: the
Transformer baseline, the Mamba baseline, and (later) the project's own model
must all carry roughly the same parameter budget, or any quality difference is
confounded by capacity.  :func:`match_params` searches a baseline's width/depth
to land within a tolerance of a target parameter count.

The search builds candidate models and counts their parameters directly rather
than relying on a hand-derived param formula — the formula would silently drift
the moment either baseline's architecture changes, whereas a build-and-count is
always exact.  Construction is cheap (CPU, no forward pass), so this is fast
enough to run once at experiment-setup time.
"""

from __future__ import annotations

import copy
from dataclasses import replace
from typing import TYPE_CHECKING, cast

import torch.nn as nn

from graph_llm.models.registry import build_model

if TYPE_CHECKING:
    from graph_llm.config import Config


def count_params(model: nn.Module, trainable_only: bool = True) -> int:
    """Return the (trainable) parameter count of *model*.

    Prefers the model's own ``num_parameters`` if present (both baselines expose
    it), otherwise sums ``parameters()`` directly.

    Args:
        model: Any ``nn.Module``.
        trainable_only: Count only parameters with ``requires_grad=True``.

    Returns:
        Total parameter count as an ``int``.
    """
    num_parameters = getattr(model, "num_parameters", None)
    if callable(num_parameters):
        return int(cast("int", num_parameters(trainable_only=trainable_only)))
    return sum(
        p.numel() for p in model.parameters() if (not trainable_only or p.requires_grad)
    )


def count_params_for_config(cfg: Config) -> int:
    """Build the model named by ``cfg.model.name`` and return its param count."""
    return count_params(build_model(cfg))


def _round_d_model(d_model: int, n_heads: int) -> int:
    """Round *d_model* to the nearest positive multiple of ``2 * n_heads``.

    The Transformer baseline asserts ``d_model % n_heads == 0``, AND its RoPE
    needs an EVEN head_dim (= d_model / n_heads) because it rotates dimension
    pairs. Snapping to a multiple of ``2 * n_heads`` guarantees both, so every
    candidate the search proposes is buildable (an odd head_dim crashes RoPE).
    """
    step = 2 * n_heads
    return max(step, round(d_model / step) * step)


def _best_width_for_depth(
    target_params: int,
    base: Config,
    n_layers: int,
    n_heads: int,
    ff_ratio: float,
    max_iters: int,
) -> tuple[Config, float]:
    """Bisect the width ``d_model`` at a fixed depth; return (best_cfg, rel_err).

    Parameter count grows monotonically with width at fixed depth, so a clean
    bisection converges.  ``d_ff`` tracks the template's ``d_ff / d_model`` ratio
    and width is snapped to a multiple of ``n_heads`` so every candidate is
    buildable (the Transformer asserts head-divisibility).
    """

    # Only the attention baseline requires d_model % n_heads == 0; snapping the
    # width grid to n_heads for head-free models (e.g. Mamba) would bias the
    # param match away from the true closest config.
    snap_to_heads = base.model.name == "transformer"

    def make(d_model: int) -> Config:
        d_model = (
            _round_d_model(d_model, n_heads) if snap_to_heads else max(1, round(d_model))
        )
        d_ff = max(1, round(d_model * ff_ratio))
        return replace(
            base,
            model=replace(base.model, d_model=d_model, d_ff=d_ff, n_layers=n_layers),
        )

    # Bracket [lo, hi] in d_model such that params(lo) <= target <= params(hi).
    lo = n_heads
    hi = max(n_heads * 2, base.model.d_model)
    hi_cfg = make(hi)
    hi_params = count_params_for_config(hi_cfg)

    grow_guard = 0
    while hi_params < target_params and grow_guard < max_iters:
        hi *= 2
        hi_cfg = make(hi)
        hi_params = count_params_for_config(hi_cfg)
        grow_guard += 1

    best_cfg = hi_cfg
    best_err = abs(hi_params - target_params) / target_params

    # Also seed with the narrowest model (covers targets below the bracket so we
    # still return the closest achievable candidate rather than nothing).
    lo_cfg = make(lo)
    lo_err = abs(count_params_for_config(lo_cfg) - target_params) / target_params
    if lo_err < best_err:
        best_cfg, best_err = lo_cfg, lo_err

    for _ in range(max_iters):
        mid = (lo + hi) // 2
        mid_cfg = make(mid)
        mid_params = count_params_for_config(mid_cfg)
        err = abs(mid_params - target_params) / target_params
        if err < best_err:
            best_cfg, best_err = mid_cfg, err
        if best_err <= 0.0:
            break
        if mid_params < target_params:
            lo = mid_cfg.model.d_model
        else:
            hi = mid_cfg.model.d_model
        if hi - lo <= n_heads:  # bracket collapsed to one grid step
            break
    return best_cfg, best_err


def match_params(
    target_params: int,
    base_cfg: Config,
    tolerance: float = 0.05,
    max_iters: int = 64,
    depth_search: int = 3,
) -> Config:
    """Return a config whose model lands within ``±tolerance`` of *target_params*.

    Two-dimensional search over **width and depth**.  For each candidate depth
    ``n_layers`` in a small window around ``base_cfg.model.n_layers``, the width
    ``d_model`` is binary-searched (param count is monotonic in width at fixed
    depth); the overall closest candidate is returned.  Searching depth as well
    as width closes the granularity gaps that a width-only search leaves when the
    ``n_heads`` grid is coarse relative to the target.

    ``d_ff`` is kept at the template's ``d_ff / d_model`` ratio (default ``4x``,
    the GPT convention) so the FFN scales with the width.  The returned config's
    ``model.name`` is whatever ``base_cfg`` carried, so the same call
    size-matches either baseline (``"transformer"`` or ``"mamba"``).

    Args:
        target_params: Desired trainable parameter count.
        base_cfg: Template config; its ``model.name``, ``n_heads``,
            ``vocab_size``, and the ``d_ff / d_model`` ratio are preserved.
            ``n_layers`` is the *centre* of the depth search.
        tolerance: Allowed relative deviation (``0.05`` = ±5 %).
        max_iters: Width-bisection iteration cap.
        depth_search: How many layers to probe on each side of the base depth
            (``0`` = width-only at the base depth).

    Returns:
        A new :class:`~graph_llm.config.Config` (deep-copied; ``base_cfg`` is
        never mutated) whose model is size-matched to ``target_params``.

    Raises:
        ValueError: If ``target_params`` is not positive, or if no
            ``(depth, width)`` candidate reaches the tolerance band (e.g. the
            target is below the irreducible embedding-table cost even at the
            minimum width and depth — widen the band or shrink ``vocab_size``).
    """
    if target_params <= 0:
        raise ValueError(f"target_params must be positive, got {target_params}")

    base = copy.deepcopy(base_cfg)
    n_heads = base.model.n_heads
    ff_ratio = base.model.d_ff / base.model.d_model if base.model.d_model else 4.0

    base_layers = base.model.n_layers
    candidate_depths = sorted(
        {max(1, base_layers + d) for d in range(-depth_search, depth_search + 1)}
    )

    best_cfg: Config | None = None
    best_err = float("inf")
    for n_layers in candidate_depths:
        cfg, err = _best_width_for_depth(
            target_params, base, n_layers, n_heads, ff_ratio, max_iters
        )
        if err < best_err:
            best_cfg, best_err = cfg, err
        if best_err <= tolerance:
            break

    assert best_cfg is not None  # candidate_depths is non-empty
    if best_err > tolerance:
        achieved = count_params_for_config(best_cfg)
        raise ValueError(
            f"Could not match target_params={target_params:,} within "
            f"±{tolerance:.0%}: best candidate had {achieved:,} params "
            f"(d_model={best_cfg.model.d_model}, n_layers={best_cfg.model.n_layers}, "
            f"err={best_err:.1%}). Try widening tolerance or depth_search, or "
            f"shrinking vocab_size."
        )
    return best_cfg
