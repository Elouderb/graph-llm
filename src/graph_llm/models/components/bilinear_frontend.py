"""Windowed factorized bilinear (MFB) front-end (card 86347418).

This is the first *novel* architectural component.  Each token's embedding
interacts (a second-order, multiplicative "outer-product" interaction) with
itself and its ``W-1`` trailing neighbours, producing a per-token feature of
dimension ``o`` (default ``4096 == 64x64``) that later phases feed to the GNNs.

The crux: memory safety
-----------------------
The naive realisation forms, for every position ``t`` and offset ``d``, the
outer product ``emb_t emb_{t-d}^T`` -- a ``128 x 128`` matrix -- and stacks them
into a ``(T, 128, 128, W)`` tensor.  At our scale that is ~1 MiB **per token**
and infeasible on a 12 GB GPU.  The prior-art pass found this full bilinear
interaction is ~98% redundant (it is a degree-2 polynomial kernel), and a
low-rank Hadamard factorisation (MLB / MFB) recovers the pairwise multiplicative
signal *without ever forming the matrix*.

Factorized MFB identity (Yu et al., arXiv:1708.01471)
-----------------------------------------------------
For two operands ``x, y`` and a low-rank factorisation of the per-output bilinear
weight ``W_i = sum_{j=1..k} u_{ij} v_{ij}^T``::

    z_i = x^T W_i y
        = x^T ( sum_{j=1..k} u_{ij} v_{ij}^T ) y
        = sum_{j=1..k} (x^T u_{ij}) (v_{ij}^T y)
        = sum_{j=1..k} ( U_i^T x  o  V_i^T y )_j           ( o == Hadamard )

Stacking all ``o`` outputs and writing ``U-tilde, V-tilde`` in
``R^{emb x (k*o)}`` for the concatenated factors::

    z = SumPool( (U-tilde^T x)  o  (V-tilde^T y),  k )      ( z in R^o )

where ``SumPool(., k)`` reshapes the ``k*o`` vector to ``(o, k)`` and sums over
the ``k`` axis.  Only ``(k*o)``-sized intermediates ever exist -- no ``emb x emb``
axis pair is materialised.  MLB (Kim et al., arXiv:1610.04325) is the ``k = 1``
special case.  After pooling, MFB applies power normalisation (signed sqrt),
L2 normalisation, and dropout.

Three interaction modes (the ablation machinery)
------------------------------------------------
* ``factorized_mfb``  (default) -- the contribution above.
* ``control_linear``  -- a matched-parameter linear/conv mixer over the same
  windowed context with **no multiplicative interaction**.  This is the ablation
  *null*: if MFB does not beat it, the second-order interaction is not pulling
  its weight.
* ``materialized_cnn`` -- reduce ``emb`` to a small ``r`` (default 32), form the
  *small* ``(r, r, W)`` interaction explicitly, and run a 2-D CNN over it.  This
  is the honest test of the original "CNN over the interaction matrix" idea
  (an inversion of Bilinear-CNN, arXiv:1504.07889) at ``~16x`` lower memory than
  the naive ``128 x 128`` map -- treated as an ablation variant, not the default.

All three emit ``(B, T, o)`` and share the windowing/causal-padding machinery,
so the surrounding LM is mode-agnostic.

Shared projection weights across offsets
----------------------------------------
``U-tilde`` and ``V-tilde`` are shared across all ``W`` offsets, keeping the
parameter count bounded at ``O(emb * k * o)`` rather than ``O(W * emb * k * o)``.
The per-offset specialisation lives in the (cheap) aggregation weights, not in
the (expensive) projections.  The self term (``d = 0``, ``x o x``) is kept --
it is a quadratic feature of a single token -- but note it carries less
*interaction* information than the cross terms ``d > 0``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

if TYPE_CHECKING:
    from graph_llm.config import ModelConfig

VALID_MODES = ("factorized_mfb", "control_linear", "materialized_cnn")


def _causal_shift(x: Tensor, d: int) -> Tensor:
    """Return the partner sequence where position ``t`` holds ``x[t - d]``.

    Causal: positions ``t < d`` (no valid trailing neighbour) are zero-padded.
    Implemented by left-padding the *time* axis and slicing; the embedding axis
    is never touched, so no ``emb x emb`` intermediate is created.

    Args:
        x: ``(B, T, E)`` sequence of embeddings.
        d: Non-negative trailing offset (``d = 0`` is the identity / self term).

    Returns:
        ``(B, T, E)`` shifted sequence.
    """
    if d == 0:
        return x
    B, T, E = x.shape
    pad = x.new_zeros(B, d, E)
    return torch.cat([pad, x[:, : T - d, :]], dim=1)


class BilinearFrontEnd(nn.Module):
    """Windowed factorized-bilinear (MFB) per-token feature extractor.

    Consumes a ``(B, T, emb)`` embedding sequence and emits ``(B, T, o)``.
    The interaction mode is selected by ``cfg.interaction_mode``; all modes
    share the causal windowing so the surrounding model is mode-agnostic.

    Args:
        cfg: A :class:`~graph_llm.config.ModelConfig`.  Reads ``d_model``
            (== ``emb``), ``bilinear_window`` (W), ``bilinear_k`` (k),
            ``bilinear_o`` (o), ``interaction_mode``, ``front_end_dropout``,
            ``bilinear_offset_weighting``, and (for ``materialized_cnn``)
            ``materialized_reduce_dim`` / ``materialized_cnn_channels``.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.emb = cfg.d_model
        self.window = cfg.bilinear_window
        self.k = cfg.bilinear_k
        self.o = cfg.bilinear_o
        self.mode = cfg.interaction_mode
        self.offset_weighting = cfg.bilinear_offset_weighting

        if self.mode not in VALID_MODES:
            raise ValueError(
                f"interaction_mode={self.mode!r} not in {VALID_MODES}."
            )
        if self.window < 1:
            raise ValueError(f"bilinear_window must be >= 1, got {self.window}")
        if self.k < 1:
            raise ValueError(f"bilinear_k must be >= 1, got {self.k}")
        if self.o < 1:
            raise ValueError(f"bilinear_o must be >= 1, got {self.o}")

        self.dropout = nn.Dropout(cfg.front_end_dropout)
        self.out_dim = self.o

        if self.mode == "factorized_mfb":
            self._init_factorized_mfb()
        elif self.mode == "control_linear":
            self._init_control_linear()
        else:  # materialized_cnn
            self._init_materialized_cnn(cfg)

    # ------------------------------------------------------------------
    # Mode constructors
    # ------------------------------------------------------------------

    def _init_factorized_mfb(self) -> None:
        """Shared U-tilde / V-tilde projections + per-offset aggregation weights.

        Projections map ``emb -> k*o`` and are shared across all ``W`` offsets;
        the only per-offset parameters are the cheap aggregation weights.
        """
        self.u_proj = nn.Linear(self.emb, self.k * self.o, bias=False)
        self.v_proj = nn.Linear(self.emb, self.k * self.o, bias=False)
        if self.offset_weighting == "learned":
            # One learned scalar per (offset, output) — initialised to 1 so the
            # untrained module reduces to a plain sum over offsets.
            self.offset_weights: nn.Parameter | None = nn.Parameter(
                torch.ones(self.window, self.o)
            )
        elif self.offset_weighting == "sum":
            self.offset_weights = None
        else:
            raise ValueError(
                f"bilinear_offset_weighting must be 'sum' or 'learned', "
                f"got {self.offset_weighting!r}"
            )

    def _init_control_linear(self) -> None:
        """Matched-param ablation null: a windowed linear mixer, NO multiply.

        Concatenates the ``W`` causally-shifted neighbour embeddings and applies
        a single linear map to ``o`` dims.  Parameter count ``W*emb*o`` is the
        same order as the MFB projections' ``2*emb*k*o`` for typical ``k = 2``,
        ``W = 16`` (``16 == 2*2*4``), so the comparison is roughly capacity-matched.
        The defining property: it contains no Hadamard / second-order term.
        """
        self.mixer = nn.Linear(self.window * self.emb, self.o, bias=True)

    def _init_materialized_cnn(self, cfg: ModelConfig) -> None:
        """Reduce emb -> r, form the SMALL (r, r, W) interaction, 2-D CNN over it.

        The interaction map is ``r x r`` (default ``32 x 32``), i.e. ``~16x``
        smaller than the naive ``128 x 128``.  A 2-D CNN treats the W offsets as
        input channels and pools the ``r x r`` map down to the ``o`` output.
        """
        self.reduce_dim = cfg.materialized_reduce_dim
        ch = cfg.materialized_cnn_channels
        self.reduce = nn.Linear(self.emb, self.reduce_dim, bias=False)
        # W offsets -> channels; a small conv stack over the (r, r) map.
        self.cnn = nn.Sequential(
            nn.Conv2d(self.window, ch, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(ch, ch, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),  # (B*T, ch, 1, 1)
        )
        self.cnn_proj = nn.Linear(ch, self.o, bias=True)

    # ------------------------------------------------------------------
    # MFB normalisation
    # ------------------------------------------------------------------

    @staticmethod
    def _power_norm(z: Tensor) -> Tensor:
        """Signed square-root power normalisation: ``sign(z) * sqrt(|z|)``."""
        return torch.sign(z) * torch.sqrt(torch.abs(z) + 1e-12)

    def _mfb_normalize(self, z: Tensor) -> Tensor:
        """MFB output normalisation: power-norm -> L2-norm -> dropout."""
        z = self._power_norm(z)
        z = F.normalize(z, p=2.0, dim=-1)
        return self.dropout(z)

    # ------------------------------------------------------------------
    # Forward (dispatches by mode)
    # ------------------------------------------------------------------

    def forward(self, x: Tensor) -> Tensor:
        """Apply the windowed interaction.

        Args:
            x: ``(B, T, emb)`` embedding sequence.

        Returns:
            ``(B, T, o)`` per-token feature.
        """
        if self.mode == "factorized_mfb":
            return self._forward_factorized(x)
        if self.mode == "control_linear":
            return self._forward_control(x)
        return self._forward_materialized(x)

    def _forward_factorized(self, x: Tensor) -> Tensor:
        """Factorized MFB over the causal window.

        For each offset ``d`` the partner is ``x[t - d]``.  We project the query
        ``x`` once and re-use it across offsets (the projection is offset-shared);
        only the partner projection is recomputed per offset.  The largest
        intermediate is ``(B, T, k*o)`` -- never an ``emb x emb`` pair.
        """
        # Query projection (shared across offsets): (B, T, k*o)
        u = self.u_proj(x)
        B, T, _ = x.shape

        acc = x.new_zeros(B, T, self.o)
        for d in range(self.window):
            partner = _causal_shift(x, d)            # (B, T, emb)
            v = self.v_proj(partner)                 # (B, T, k*o)
            had = u * v                              # (B, T, k*o)  Hadamard
            # SumPool over the k axis: (B, T, o, k) -> (B, T, o)
            pooled = had.view(B, T, self.o, self.k).sum(dim=-1)
            if self.offset_weights is not None:
                pooled = pooled * self.offset_weights[d]
            acc = acc + pooled

        return self._mfb_normalize(acc)

    def _forward_control(self, x: Tensor) -> Tensor:
        """Matched-param windowed linear mixer (the no-interaction ablation null)."""
        B, T, _ = x.shape
        shifts = [_causal_shift(x, d) for d in range(self.window)]
        ctx = torch.cat(shifts, dim=-1)              # (B, T, W*emb)
        z = self.mixer(ctx)                          # (B, T, o)
        # Same post-norm so the only architectural difference vs MFB is the
        # presence/absence of the multiplicative term.
        return self._mfb_normalize(z)

    def _forward_materialized(self, x: Tensor) -> Tensor:
        """Reduce -> small (r, r) interaction per offset -> 2-D CNN -> o.

        The interaction map is ``r x r`` (small by construction), with the ``W``
        offsets as CNN input channels.  This preserves the "CNN over the
        interaction matrix" idea at ``~16x`` lower memory than the naive map.
        """
        B, T, _ = x.shape
        r = self.reduce_dim
        xr = self.reduce(x)                          # (B, T, r)

        # Build the (B, T, W, r, r) interaction.  r is small (default 32), so the
        # r x r axis pair is ~16x smaller than the forbidden 128 x 128.
        maps = []
        for d in range(self.window):
            partner = self.reduce(_causal_shift(x, d))   # (B, T, r)
            # outer product over the small reduced dim: (B, T, r, r)
            maps.append(torch.einsum("bti,btj->btij", xr, partner))
        inter = torch.stack(maps, dim=2)             # (B, T, W, r, r)

        # Fold (B, T) into the batch axis; W offsets become CNN channels.
        inter = inter.reshape(B * T, self.window, r, r)
        feat = self.cnn(inter).flatten(1)            # (B*T, ch)
        z = self.cnn_proj(feat).view(B, T, self.o)   # (B, T, o)
        return self._mfb_normalize(z)
