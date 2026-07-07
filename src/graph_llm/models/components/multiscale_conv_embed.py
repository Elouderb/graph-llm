"""Multi-scale causal conv front-end (cheap local combiner) — card ed853f9c.

Realises design note 8b5341f0: the *cheap, additive* version of the bilinear
front-end's instinct.  Instead of embedding lone tokens, embed local *context*
at multiple scales, condense to ONE enriched ``d_model`` vector per token, and
feed the result to the committed ``delta_memory_lm`` memory backbone as better
per-position features (richer keys/values).

Pipeline (per position ``t``)::

    token embeddings (B, T, d)
      -> bank of CAUSAL depthwise-separable 1-D convs at dyadic widths {1,2,4,8,16}
         (width-1 == pointwise/identity scale; each width summarises the trailing
          window ending at t)                                  -> S tensors (B, T, d)
      -> stack over the scale axis                              -> (B, T, S, d)
      -> condense (a 1x1 conv == a per-position projection SHARED across all T
         positions) collapses the S scales back to one vector  -> (B, T, d)

Why a sliding conv (not a table)
--------------------------------
You can NOT table-lookup an n-gram (``vocab**k`` entries); a *sliding kernel*
turns it into a learned FUNCTION of the trailing window.  This is the proven
n-gram-CNN-embedding family (Kim 2014; Charformer/GBST multi-block-size
embeddings; dilated-conv input stacks ByteNet/TCN).

Two load-bearing correctness properties
---------------------------------------
* **Causality.**  Each conv is *left-padded* by ``width - 1`` (and never right-
  padded), so the output at position ``t`` depends only on inputs ``<= t`` — no
  future leakage.  The condense is purely per-position over the ``(S, d)`` slice,
  so a 1x1 conv / broadcast ``Linear`` cannot see the future either.  Verified by
  a perturbation probe (perturb token ``t+1`` -> outputs at ``<= t`` unchanged).
* **Weight sharing.**  The condense kernel is ONE learned function applied
  identically to every token's ``(S, d)`` slice — the defining property of a 1x1
  conv / ``nn.Linear`` broadcast over the sequence.  It is NOT position-indexed;
  sharing is what gives translation invariance + length generalisation.

Cheapness
---------
Dyadic (not contiguous) widths give ``~log(W)`` kernels instead of ``W``.
Depthwise-separable convs (depthwise over the window + pointwise channel mix)
keep the per-scale cost low; ``conv_depthwise=False`` falls back to a single full
1-D conv per scale.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

if TYPE_CHECKING:
    from graph_llm.config import ModelConfig

VALID_CONDENSE = ("concat_proj", "soft_select")


class _CausalScaleConv(nn.Module):
    """One causal scale of the bank: a depthwise-separable (or full) 1-D conv.

    Consumes ``(B, T, d)`` and emits ``(B, T, d)`` of the same length, with the
    output at position ``t`` depending only on inputs ``<= t`` (left-pad only, no
    right pad).  ``width == 1`` is the pointwise / identity scale: a kernel-1 conv
    is a pure per-position channel mix (depthwise k=1 is per-channel; the pointwise
    then mixes channels), so the width-1 "k-gram" is just the (re-projected) token.

    Args:
        d_model: Channel dim ``d`` (== embedding dim).
        width: Causal window width ``k`` (>= 1).
        depthwise: If True, depthwise conv (``groups=d``) + pointwise 1x1 mix; if
            False, a single full 1-D conv (``groups=1``).
    """

    def __init__(self, d_model: int, width: int, depthwise: bool) -> None:
        super().__init__()
        if width < 1:
            raise ValueError(f"conv width must be >= 1, got {width}")
        self.width = width
        self.depthwise = depthwise
        if depthwise:
            # Depthwise over the trailing window (mixes within a channel across
            # time), then a pointwise 1x1 conv (mixes channels per position).
            self.conv = nn.Conv1d(
                d_model, d_model, kernel_size=width, groups=d_model, bias=False
            )
            self.pointwise: nn.Conv1d | None = nn.Conv1d(
                d_model, d_model, kernel_size=1, bias=True
            )
        else:
            # A single full 1-D conv mixes both time and channels at once.
            self.conv = nn.Conv1d(d_model, d_model, kernel_size=width, bias=True)
            self.pointwise = None

    def forward(self, x: Tensor) -> Tensor:
        """``(B, T, d) -> (B, T, d)``, causal (no future leakage)."""
        h = x.transpose(1, 2)                       # (B, d, T)
        # Left-pad by width-1 and never right-pad: output[t] sees inputs [t-(width-1) .. t].
        h = F.pad(h, (self.width - 1, 0))
        h = self.conv(h)
        if self.pointwise is not None:
            h = self.pointwise(h)
        return h.transpose(1, 2)                     # (B, T, d)


class MultiScaleConvEmbedding(nn.Module):
    """Multi-scale causal conv front-end + 1x1 condense to one ``d_model`` vector.

    Takes token embeddings ``(B, T, d)`` and returns enriched per-position
    features ``(B, T, d)`` (same shape — a drop-in input-stage enrichment).  A
    bank of causal depthwise-separable convs at the configured dyadic ``widths``
    produces ``S = len(widths)`` scale tensors that are stacked to ``(B, T, S, d)``
    and condensed back to ``(B, T, d)``.

    Args:
        cfg: A :class:`~graph_llm.config.ModelConfig`.  Reads ``d_model``,
            ``conv_widths`` (the dyadic widths; each >= 1), ``conv_condense``
            (``"concat_proj"`` | ``"soft_select"``), and ``conv_depthwise``.

    Condense modes (both weight-shared across all T positions):
        * ``concat_proj`` — reshape ``(B, T, S, d) -> (B, T, S*d)`` then a shared
          ``Linear(S*d, d)``.  Mixes across both scales and dims (the literal
          1x1-conv collapse).
        * ``soft_select`` — a shared ``Linear(S*d, S) -> softmax`` over the ``S``
          scales -> convex blend of the scale-embeddings (stays in embedding
          space; the per-position granularity is readable from the weights).
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        d_model = cfg.d_model
        widths = list(cfg.conv_widths)
        condense = cfg.conv_condense

        if not widths:
            raise ValueError("conv_widths must be non-empty")
        if any(w < 1 for w in widths):
            raise ValueError(f"every conv width must be >= 1, got {widths}")
        if condense not in VALID_CONDENSE:
            raise ValueError(
                f"conv_condense={condense!r} not in {VALID_CONDENSE}."
            )

        self.d_model = d_model
        self.widths = widths
        self.num_scales = len(widths)
        self.condense = condense

        # The causal conv bank — one scale per width.  width-1 is the identity
        # scale (the per-token embedding, re-projected).
        self.scales = nn.ModuleList(
            [_CausalScaleConv(d_model, w, cfg.conv_depthwise) for w in widths]
        )

        # Condense the (S, d) slice per position with a weight SHARED across all T
        # positions (a 1x1 conv / broadcast Linear) — translation invariance +
        # length generalisation, and trivially causal (per-position only).
        flat_dim = self.num_scales * d_model
        if condense == "concat_proj":
            self.proj = nn.Linear(flat_dim, d_model, bias=True)
            self.select: nn.Linear | None = None
        else:  # soft_select
            self.proj = None  # type: ignore[assignment]
            self.select = nn.Linear(flat_dim, self.num_scales, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        """Enrich token embeddings with multi-scale causal local context.

        Args:
            x: Token embeddings, shape ``(B, T, d_model)``.

        Returns:
            ``(B, T, d_model)`` enriched per-position features.
        """
        # Bank of causal scales -> stack to (B, T, S, d).
        multi = torch.stack([scale(x) for scale in self.scales], dim=2)
        B, T, S, d = multi.shape

        if self.condense == "concat_proj":
            assert self.proj is not None
            flat = multi.reshape(B, T, S * d)            # (B, T, S*d)
            return self.proj(flat)                        # (B, T, d) — shared proj
        # soft_select: GBST-style convex blend over the S scales.
        assert self.select is not None
        flat = multi.reshape(B, T, S * d)                # (B, T, S*d)
        weights = F.softmax(self.select(flat), dim=-1)   # (B, T, S)
        return (weights.unsqueeze(-1) * multi).sum(dim=2)  # (B, T, d)

    @torch.no_grad()
    def identity_init(self) -> None:
        """Init so the front-end is an EXACT pass-through of the raw embedding at step 0
        (card b1926d5d, coordinator; the zero-init-superset pattern from the stacked probe's
        ``seed_ctx_proj``).

        ``front_end(x) == x`` at init, so enabling the conv starts from the SAME point as the
        no-front-end backbone — the optimizer then OPENS the multi-scale mixing only where it
        helps, instead of paying the up-front re-learning cost of a random condense projection
        (which scrambles the embedding and regressed text8 bpb at a fixed budget).

        Construction: scale 0 is set to an exact identity (a causal delta at the CURRENT tap
        + an identity channel mix), and the condense is set to read ONLY scale 0.  The other
        scales keep their random init but contribute NOTHING at step 0 (``concat_proj`` zeros
        their blocks; ``soft_select`` routes ~all mass to scale 0), so training grows them via
        the condense gradient.  Called AFTER the model's generic weight init so it overrides.
        """
        d = self.d_model
        s0 = self.scales[0]
        assert isinstance(s0, _CausalScaleConv)
        # scale 0 -> exact identity: a causal delta at the CURRENT position (the last tap,
        # since the conv is left-padded by width-1) + an identity pointwise channel mix.
        nn.init.zeros_(s0.conv.weight)
        if s0.depthwise:
            s0.conv.weight[:, 0, -1] = 1.0                       # (d, 1, width) delta
            assert s0.pointwise is not None
            s0.pointwise.weight.copy_(torch.eye(d).view(d, d, 1))
            assert s0.pointwise.bias is not None
            nn.init.zeros_(s0.pointwise.bias)
        else:
            s0.conv.weight[:, :, -1] = torch.eye(d)              # (d, d, width) identity tap
            if s0.conv.bias is not None:
                nn.init.zeros_(s0.conv.bias)
        # condense -> select ONLY scale 0 (the first d columns of the flattened (S, d) slice).
        if self.condense == "concat_proj":
            assert self.proj is not None
            nn.init.zeros_(self.proj.weight)
            self.proj.weight[:, :d] = torch.eye(d)
            nn.init.zeros_(self.proj.bias)
        else:  # soft_select: input-independent, near one-hot on scale 0.
            assert self.select is not None
            nn.init.zeros_(self.select.weight)
            nn.init.zeros_(self.select.bias)
            self.select.bias[0] = 20.0
