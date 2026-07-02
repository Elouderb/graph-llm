"""Transient iterated reasoning field (R-LM step 1) — card 9907dc9e.

Ports the VALIDATED iterated reasoner (cards 64ea347f / 46aa2292 / 11ab96e7 — the
R1 / R1b / R2 mechanism) into ``delta_memory_lm`` as a config-flagged transient
reasoning block.  The reasoner is the R2 "2-D Turing machine": a small toroidal
``R x C`` grid (the *field*) with a soft read/write *head* driven by a weight-tied
controller, iterated ``K`` steps.  Each step the controller does (the proven
recipe, carried verbatim from ``reasoner_r2.py``):

    content-address  -> NTM interpolate gate -> 2-D shift {stay,N,S,E,W} (torus
    rolls) -> MANDATORY per-step SHARPEN (gamma >= floor)

The mandatory per-step sharpen is the non-obvious, load-bearing trick from the
research (iterated soft-shifts DISPERSE the pointer; the exponent + renormalise
keeps it peaked).  Delta-write is OFF here (read-style traversal of an
input-seeded field), matching the R2 traversal recipe.

How it differs from the standalone R2 probe (and WHY)
-----------------------------------------------------
* **Input-seeded, per position.**  In the probe the field content was a *given*
  grid (the synthetic task).  Here there is no external grid: the field content
  for position ``t`` is PROJECTED from that position's hidden state ``h_t`` alone
  (``Linear: d_model -> N * d_cell``, reshaped to the ``N`` cells).  This is the
  card's "input-seeded from the current token hidden states (local context), NOT
  the delta-net memory matrix" constraint (memory->scratchpad is R3b, later).
* **Provably causal by construction.**  The reasoner runs INDEPENDENTLY at every
  position ``t`` — its field, head, and controller state are seeded ONLY from
  ``h_t``, with NO mixing across positions inside this module.  ``(B, T)``
  positions are simply flattened into one big batch of ``B*T`` independent
  reasoning problems.  So the contribution at position ``t`` is a pure function of
  ``h_t`` and cannot depend on any other position; since ``h_t`` is itself
  produced causally by the delta-memory stack, perturbing a future (or any other)
  token cannot change the reasoning contribution at ``t``.  This sidesteps the
  "reinvent attention" trap entirely: there is no cross-position interaction here.
* **Transient.**  The field is rebuilt from scratch on every forward at every
  position; nothing is carried across positions or segments.  It is completely
  separate from :class:`DeltaMemoryState` and the cross-segment carry plumbing.

The readout (final soft-read -> ``Linear -> d_model``) is added (residual) into
the hidden state by the caller, refining the next-token prediction.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

if TYPE_CHECKING:
    from graph_llm.config import ModelConfig

# Supported reasoning-field forms.  "grid" == the R2-validated 2-D field/head.
VALID_REASONING_FIELDS = ("grid",)

# Head shift directions on the torus, as (drow, dcol).  Order MUST match the
# 2-D shift mix in ``_shift2d``: 0=stay 1=N(up,row-1) 2=S(down,row+1)
# 3=E(right,col+1) 4=W(left,col-1).  (Identical to reasoner_r2.py's DIR_VECS.)
_DIR_VECS = ((0, 0), (-1, 0), (1, 0), (0, 1), (0, -1))
_N_SHIFT = len(_DIR_VECS)


class ReasoningField(nn.Module):
    """Per-position transient iterated 2-D reasoning field (the R2 mechanism).

    Consumes hidden states ``(B, T, d_model)`` and returns a reasoning
    contribution ``(B, T, d_model)`` of the same shape — a residual to add into
    the hidden state before the LM head.  Each of the ``B*T`` positions is an
    INDEPENDENT reasoning problem: the field content is seeded from ``h_t`` only,
    so there is no cross-position mixing (the causality guarantee) and no carried
    state (transient).

    Args:
        cfg: A :class:`~graph_llm.config.ModelConfig`.  Reads ``d_model`` and the
            ``reasoning_*`` fields: ``reasoning_field`` (must be ``"grid"``),
            ``reasoning_rows`` / ``reasoning_cols`` (the toroidal grid),
            ``reasoning_steps`` (iterated depth K), ``reasoning_d_cell``
            (per-cell content width), ``reasoning_d_ctrl`` (controller width),
            ``reasoning_gamma_floor`` (mandatory sharpen floor >= the value), and
            ``reasoning_key_dim`` (content-address key width).
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        if cfg.reasoning_field not in VALID_REASONING_FIELDS:
            raise ValueError(
                f"reasoning_field={cfg.reasoning_field!r} not in "
                f"{VALID_REASONING_FIELDS}."
            )
        rows, cols = cfg.reasoning_rows, cfg.reasoning_cols
        if rows < 1 or cols < 1:
            raise ValueError(
                f"reasoning_rows/cols must be >= 1, got {rows}x{cols}."
            )
        if cfg.reasoning_steps < 1:
            raise ValueError(
                f"reasoning_steps must be >= 1, got {cfg.reasoning_steps}."
            )
        if cfg.reasoning_d_cell < 1 or cfg.reasoning_d_ctrl < 1:
            raise ValueError("reasoning_d_cell/d_ctrl must be >= 1.")
        if cfg.reasoning_key_dim < 2:
            raise ValueError("reasoning_key_dim must be >= 2 (row+col halves).")
        if cfg.reasoning_gamma_floor < 1.0:
            raise ValueError(
                "reasoning_gamma_floor must be >= 1.0 (mandatory sharpening; a "
                "floor < 1 would BLUR the soft head each step instead of peaking "
                "it — the dispersion failure the research flagged)."
            )

        self.d_model = cfg.d_model
        self.rows = rows
        self.cols = cols
        self.n_cells = rows * cols
        self.steps = cfg.reasoning_steps
        self.d_cell = cfg.reasoning_d_cell
        self.d_ctrl = cfg.reasoning_d_ctrl
        self.d_key = cfg.reasoning_key_dim
        self.gamma_floor = float(cfg.reasoning_gamma_floor)

        # INPUT SEED: project the per-position hidden state h_t into the field
        # content (N cells x d_cell).  This is the ONLY place the input enters the
        # reasoner; it is per-position (a broadcast Linear), so no cross-position
        # mixing.  Seeded from h_t alone => input-seeded + causal.
        self.seed = nn.Linear(self.d_model, self.n_cells * self.d_cell)

        # Cell (row,col)-factored address keys (learned, shared across positions):
        # the content-addressing target space.  Embedding row and col separately
        # and concatenating gives a (row,col)-factored key so cosine addressing can
        # land on a specific cell (the R2 recipe).
        self.row_embed = nn.Embedding(rows, self.d_key // 2)
        self.col_embed = nn.Embedding(cols, self.d_key - self.d_key // 2)

        # Weight-tied controller (one GRUCell, iterated K steps) + heads.  Input =
        # current soft-read (d_cell) ++ step-clock (sin, cos).
        self.gru = nn.GRUCell(self.d_cell + 2, self.d_ctrl)
        self.to_key = nn.Linear(self.d_ctrl, self.d_key)     # content key
        self.to_betak = nn.Linear(self.d_ctrl, 1)            # key strength (>=0)
        self.to_gate = nn.Linear(self.d_ctrl, 1)             # NTM interp gate g
        self.to_shift = nn.Linear(self.d_ctrl, _N_SHIFT)     # 2-D shift {stay,N,S,E,W}
        self.to_gamma = nn.Linear(self.d_ctrl, 1)            # sharpen exponent
        # Final field-read -> reasoning contribution in d_model space (residual).
        self.readout = nn.Linear(self.d_cell, self.d_model)
        # Learnable positive scale on the cosine logits (decisive content match).
        self.key_scale = nn.Parameter(torch.tensor(5.0))

        self._reset_head_biases()

    def _reset_head_biases(self) -> None:
        """R2 head-bias priors: trust a content match (betak positive), neutral
        interpolation gate (free to lean on the 2-D shift — the natural solution),
        neutral shift, a mild sharpen prior (the floor enforces >= 1)."""
        with torch.no_grad():
            self.to_gate.bias.fill_(0.0)
            self.to_betak.bias.fill_(1.0)
            self.to_shift.bias.zero_()
            self.to_gamma.bias.fill_(0.5)

    # ------------------------------------------------------------------ head ops
    @staticmethod
    def _soft_read(field: Tensor, w: Tensor) -> Tensor:
        """``(P, N, d_cell), (P, N) -> (P, d_cell)`` weighted read."""
        return torch.einsum("pn,pnd->pd", w, field)

    def _shift2d(self, w: Tensor, s: Tensor) -> Tensor:
        """2-D circular shift of a flat head ``w (P, N)`` by a 5-way soft mix
        ``s (P, 5)`` over {stay,N,S,E,W} — N/S roll the ROW axis, E/W the COL
        axis (the 2-D analog of the 1-D ``torch.roll`` shift).  Roll directions
        verified against the R2 oracle (on-path rate ~1.0 only when correct)."""
        p = w.shape[0]
        wg = w.view(p, self.rows, self.cols)
        stay = wg
        north = torch.roll(wg, shifts=-1, dims=1)   # head up (row-1)
        south = torch.roll(wg, shifts=1, dims=1)     # head down (row+1)
        east = torch.roll(wg, shifts=1, dims=2)      # head right (col+1)
        west = torch.roll(wg, shifts=-1, dims=2)     # head left (col-1)
        out = (
            s[:, 0].view(p, 1, 1) * stay
            + s[:, 1].view(p, 1, 1) * north
            + s[:, 2].view(p, 1, 1) * south
            + s[:, 3].view(p, 1, 1) * east
            + s[:, 4].view(p, 1, 1) * west
        )
        return out.reshape(p, self.n_cells)

    @staticmethod
    def _sharpen(w: Tensor, gamma: Tensor) -> Tensor:
        """MANDATORY per-step sharpen: ``w^gamma`` renormalised (gamma >= floor)."""
        wp = w.clamp_min(1e-12) ** gamma
        return wp / wp.sum(1, keepdim=True).clamp_min(1e-12)

    def _cell_keys(self, device: torch.device) -> Tensor:
        """``(N, d_key)``: the (row,col)-factored address key of every cell."""
        rr = torch.arange(self.n_cells, device=device) // self.cols
        cc = torch.arange(self.n_cells, device=device) % self.cols
        return torch.cat([self.row_embed(rr), self.col_embed(cc)], dim=1)

    # --------------------------------------------------------------------- main
    def forward(self, h: Tensor) -> Tensor:
        """Run the iterated reasoner independently at every position.

        Args:
            h: Hidden states ``(B, T, d_model)``.

        Returns:
            ``(B, T, d_model)`` reasoning contribution (a residual to add into
            ``h``).  Each position is independent (causal + transient).
        """
        b, t, _ = h.shape
        p = b * t                                       # flattened positions
        device = h.device
        hp = h.reshape(p, self.d_model)

        # INPUT SEED: field content from h_t alone (per-position, no cross mixing).
        field = self.seed(hp).view(p, self.n_cells, self.d_cell)

        keys = self._cell_keys(device)                  # (N, d_key)
        keys_n = F.normalize(keys, dim=1).unsqueeze(0)   # (1, N, d_key)

        ctrl = torch.zeros(p, self.d_ctrl, device=device, dtype=h.dtype)
        # Head starts on flat cell 0 == grid (0,0) (the fixed R2 path start).
        w = torch.zeros(p, self.n_cells, device=device, dtype=h.dtype)
        w[:, 0] = 1.0
        scale = F.softplus(self.key_scale)

        for step in range(self.steps):
            phi = 2.0 * math.pi * step / max(1, self.steps)
            clk = torch.tensor(
                [math.sin(phi), math.cos(phi)], device=device, dtype=h.dtype
            ).expand(p, 2)
            r = self._soft_read(field, w)                       # (P, d_cell)
            ctrl = self.gru(torch.cat([r, clk], dim=1), ctrl)

            key = self.to_key(ctrl)                             # (P, d_key)
            betak = F.softplus(self.to_betak(ctrl))             # (P, 1) >= 0
            g = torch.sigmoid(self.to_gate(ctrl))               # (P, 1)
            s = F.softmax(self.to_shift(ctrl), dim=1)           # (P, 5)
            gamma = self.gamma_floor + F.softplus(self.to_gamma(ctrl))  # >= floor

            # CONTENT ADDRESS (cosine to learned cell keys, learnable scale).
            key_n = F.normalize(key, dim=1)
            cos = torch.einsum("pd,bnd->pn", key_n, keys_n)     # (P, N) in [-1,1]
            w_c = F.softmax(scale * betak * cos, dim=1)         # (P, N)
            # NTM INTERPOLATE with previous head.
            w = g * w_c + (1.0 - g) * w
            # 2-D SHIFT then MANDATORY SHARPEN (delta-write OFF: read-style).
            w = self._shift2d(w, s)
            w = self._sharpen(w, gamma)

        r_final = self._soft_read(field, w)                     # (P, d_cell)
        out = self.readout(r_final)                             # (P, d_model)
        return out.view(b, t, self.d_model)
