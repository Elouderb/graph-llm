"""Per-position, segment-bounded CAUSAL reasoner (tandem card 2dd3400f).

Ports the lead-verified v3 WIN reasoner (card e4e8a4dc — the causal multi-chain
solve) into ``delta_memory_lm`` as the *reasoner pathway* of the gated tandem.
The scratchpad reasoner (``scratchpad/reasoner_v2_causal.py``) answered ONE query
per sequence (the print-argument at ``answer_pos``); a language model needs a
reasoning hidden at EVERY position (to feed the per-position gate + fusion + head),
so this module generalises the SAME mechanism with a query-position axis, run over
bounded windows.

The WIN mechanism (carried faithfully from card e4e8a4dc)
--------------------------------------------------------
* :class:`CausalGRUEncoder` — embed -> a small LEFT-padded (strictly causal)
  depthwise conv -> a UNIDIRECTIONAL GRU scan -> LayerNorm.  The recurrence is the
  load-bearing piece: it holds a clause's LHS name in state until the clause
  completes, so the clause-END representation carries BOTH names cleanly — the
  variable-offset carry a fixed-tap conv provably cannot do (the 3a20c143 wall).
  Unidirectional => position ``t`` depends only on tokens ``<= t`` (zero future
  leak, verified by the LM-level leak probe).
* LOCATE-THEN-WALK with a SEPARATE locate key head (``seed_name_key``): a pooled
  read of the window ending at the query position content-addresses a DEDICATED key
  space to seed the walk; keeping it separate from the walk's ``name_key`` is the
  residual dead-seed fix (card f10a7c08).
* Soft K-step walk: cosine content-address -> NTM interpolate gate -> MANDATORY
  per-step sharpen (``gamma >= gamma_floor``, default 2.0) + ``direct_ptr`` (read
  the RHS descriptor straight into the move query).  The walk stays SOFT
  (a hard walk disperses the multi-chain head).

Segment-bounded => sub-quadratic (the card's HARD CONSTRAINT)
------------------------------------------------------------
The input ``(B, T)`` is chunked into windows of ``reasoning_segment_len`` (``L``).
The causal encoder RESETS at each window boundary and every query position
addresses ONLY positions ``<= t`` WITHIN ITS OWN WINDOW — never unbounded history
(that is the delta-memory's job).  Cost is O(K * L) per position -> LINEAR in T for
a bounded ``L``.  The per-query walk is a batched matmul
``(B, Lq, Lk) @ (B, Lk, d)`` — no ``(B, L, L, d)`` blow-up.

Aux supervision (single-window regime only)
-------------------------------------------
``forward(..., return_aux=True, aux_query_pos=...)`` gathers the locate
``seed_logits`` and the per-hop walk distribution ``walk_w`` ONLY at the supplied
answer positions (one per row), so the reasoning-synthetic locate-CE + walk-aux
losses (card 2dd3400f) can be applied without materialising a per-position
trajectory.  Requires a single window (``T <= reasoning_segment_len``); the mixed
M+R reproduction feeds one segment per forward, so this always holds there.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

if TYPE_CHECKING:
    from graph_llm.config import ModelConfig


def _st_top1(w: Tensor) -> Tensor:
    """Straight-through top-1 over the LAST dim: forward one-hot(argmax), backward
    identity.  Commits the head to a single position in the forward pass (killing
    the cross-chain blend that disperses the walk) while keeping it differentiable.
    """
    idx = w.argmax(dim=-1, keepdim=True)
    hard = torch.zeros_like(w).scatter_(-1, idx, 1.0)
    return hard + (w - w.detach())


class CausalGRUEncoder(nn.Module):
    """Byte embed + a small causal conv + a unidirectional (CAUSAL) GRU scan.

    Candidate B (card e4e8a4dc): a content-carrying recurrence.  Position ``t``
    depends only on tokens ``<= t`` (the GRU is unidirectional and the conv is
    LEFT-padded), so the encoder is a legal autoregressive front-end.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        conv_kernel: int = 5,
        conv_layers: int = 1,
        gru_layers: int = 1,
    ) -> None:
        super().__init__()
        if conv_kernel < 1:
            raise ValueError("conv_kernel must be >= 1.")
        self.embed = nn.Embedding(vocab_size, d_model)
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(d_model, d_model, kernel_size=conv_kernel, groups=d_model, padding=0)
                for _ in range(conv_layers)
            ]
        )
        self.pointwise = nn.ModuleList(
            [nn.Linear(d_model, d_model) for _ in range(conv_layers)]
        )
        self.gru = nn.GRU(d_model, d_model, num_layers=gru_layers, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.conv_kernel = conv_kernel

    def forward(self, x: Tensor) -> Tensor:
        """``(N, L) int -> (N, L, d_model)`` with strictly-causal carried context."""
        h = self.embed(x)
        pad = self.conv_kernel - 1
        for conv, pw in zip(self.convs, self.pointwise):
            hc = h.transpose(1, 2)
            hc = F.pad(hc, (pad, 0))  # LEFT-ONLY causal conv
            hc = conv(hc).transpose(1, 2)
            h = h + pw(F.gelu(hc))
        out, _ = self.gru(h)  # unidirectional GRU -> strictly causal carry
        return self.norm(out)


class CausalReasoner(nn.Module):
    """Per-position, segment-bounded causal locate-then-walk reasoner.

    Consumes token ids ``(B, T)`` and returns a per-position reasoning hidden
    ``(B, T, d_model)`` for the tandem gated fusion.  Each position ``t`` runs the
    locate-then-walk seeded from its own context, addressing only positions ``<= t``
    within its window (``reasoning_segment_len``) — segment-bounded and causal.

    Args:
        cfg: A :class:`~graph_llm.config.ModelConfig`.  Reads ``d_model``,
            ``vocab_size``, ``reasoning_segment_len`` and the ``causal_reasoner_*``
            fields (steps, gamma_floor, key_dim, conv_kernel, gru_layers,
            query_window, direct_ptr, hard_seed).
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        d_model = cfg.d_model
        d_key = cfg.causal_reasoner_key_dim
        if cfg.causal_reasoner_gamma_floor < 1.0:
            raise ValueError(
                "causal_reasoner_gamma_floor must be >= 1.0 (mandatory sharpen; a "
                "floor < 1 would BLUR the soft head each step)."
            )
        if cfg.reasoning_segment_len < 1:
            raise ValueError("reasoning_segment_len must be >= 1.")
        if cfg.causal_reasoner_steps < 1:
            raise ValueError("causal_reasoner_steps must be >= 1.")
        if d_key < 2:
            raise ValueError("causal_reasoner_key_dim must be >= 2.")
        if cfg.causal_reasoner_query_window < 1:
            raise ValueError("causal_reasoner_query_window must be >= 1.")

        self.d_model = d_model
        self.d_key = d_key
        self.d_ctrl = d_model
        self.steps = cfg.causal_reasoner_steps
        self.gamma_floor = float(cfg.causal_reasoner_gamma_floor)
        self.segment_len = cfg.reasoning_segment_len
        self.query_window = cfg.causal_reasoner_query_window
        self.direct_ptr = cfg.causal_reasoner_direct_ptr
        self.hard_seed = cfg.causal_reasoner_hard_seed

        self.encoder = CausalGRUEncoder(
            cfg.vocab_size,
            d_model,
            conv_kernel=cfg.causal_reasoner_conv_kernel,
            gru_layers=cfg.causal_reasoner_gru_layers,
        )

        # Per-position addressable projections (verbatim v2 recipe).
        self.name_key = nn.Linear(d_model, d_key)   # LHS name this clause DEFINES
        self.ptr_key = nn.Linear(d_model, d_key)    # RHS name this clause POINTS TO
        self.val = nn.Linear(d_model, d_model)      # read payload
        self.seed_name_key = nn.Linear(d_model, d_key)  # DEDICATED locate key space
        self.query_pool = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_key),
        )
        self.move_gain = nn.Parameter(torch.tensor(1.0))

        # Weight-tied walk controller.
        self.gru = nn.GRUCell(d_model + 2, self.d_ctrl)
        self.to_query = nn.Linear(self.d_ctrl, d_key)
        self.to_beta = nn.Linear(self.d_ctrl, 1)
        self.to_gate = nn.Linear(self.d_ctrl, 1)
        self.to_gamma = nn.Linear(self.d_ctrl, 1)
        self.key_scale = nn.Parameter(torch.tensor(5.0))

        # Fusion projection: pre-readout hidden cat([r_final, ctrl]) -> d_model.
        self.proj = nn.Linear(d_model + self.d_ctrl, d_model)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Re-apply the reasoner's own (faithful) init + head-bias priors.

        Called AFTER the host model's generic ``_init_weights`` (which re-inits all
        ``nn.Linear`` to std 0.02 and zeros biases) so the reasoner keeps the
        scratchpad-validated default init for its Linears and its load-bearing bias
        priors (beta positive, neutral gate, mild sharpen, key-scale 5).  Mirrors
        how ``alpha_proj`` / ``ReasoningField._reset_head_biases`` are re-applied.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()
        with torch.no_grad():
            self.to_beta.bias.fill_(1.0)
            self.to_gate.bias.fill_(0.0)
            self.to_gamma.bias.fill_(0.5)
            self.key_scale.fill_(5.0)
            self.move_gain.fill_(1.0)

    # ------------------------------------------------------------------ head ops
    @staticmethod
    def _sharpen(w: Tensor, gamma: Tensor) -> Tensor:
        """MANDATORY per-step sharpen over the LAST dim: ``w^gamma`` renormalised."""
        wp = w.clamp_min(1e-12) ** gamma
        return wp / wp.sum(-1, keepdim=True).clamp_min(1e-12)

    def _causal_pool(self, h: Tensor) -> Tensor:
        """Causal mean-pool of width ``query_window`` ending at each position.

        ``pooled[:, t] = mean(h[:, t-W+1 : t+1])`` (clamped at 0) — the per-position
        analogue of the scratchpad's print-argument window read.  ``(N, L, d)``.
        """
        n, length, d = h.shape
        acc = torch.zeros_like(h)
        cnt = torch.zeros(1, length, 1, device=h.device, dtype=h.dtype)
        ones = torch.ones(1, length, 1, device=h.device, dtype=h.dtype)
        for off in range(self.query_window):
            if off == 0:
                acc = acc + h
                cnt = cnt + ones
            else:
                acc[:, off:] = acc[:, off:] + h[:, : length - off]
                cnt[:, off:] = cnt[:, off:] + ones[:, : length - off]
        return acc / cnt

    def _address(
        self, query: Tensor, keys_n: Tensor, addr_mask: Tensor, scale: Tensor,
        beta: Tensor | None, neg_inf: float,
    ) -> Tensor:
        """Cosine content-address -> masked softmax head.  ``query (N,Lq,dk)``,
        ``keys_n (N,Lk,dk)``, ``addr_mask (N,Lq,Lk)`` -> ``(N,Lq,Lk)``."""
        q_n = F.normalize(query, dim=-1)
        cos = torch.einsum("nqd,nkd->nqk", q_n, keys_n)
        cos = cos.masked_fill(~addr_mask, neg_inf)
        logit = scale * cos if beta is None else scale * beta * cos
        return F.softmax(logit, dim=-1)

    # --------------------------------------------------------------------- walk
    def _walk_window(
        self,
        x: Tensor,
        steps: int,
        commit_seed: bool,
        aux_query_pos: Tensor | None,
        tf_seed: Tensor | None,
        collect_aux: bool,
        walk_hard: bool = False,
        gamma_add: float = 0.0,
        clock_period: int | None = None,
        query_pos: Tensor | None = None,
    ) -> tuple[Tensor, dict | None]:
        """Run the per-position locate-then-walk over ONE batch of windows.

        Args:
            x: ``(N, L)`` token ids (N = B * n_windows).
            steps: walk depth K for this call.
            commit_seed: gate the ``hard_seed`` straight-through commit.
            aux_query_pos: ``(N,)`` per-row answer position to collect aux at, or
                ``None``.  Only valid when N == the original batch (single window).
            tf_seed: ``(N,)`` teacher-forced walk-seed index applied at
                ``aux_query_pos`` rows during training, or ``None``.
            collect_aux: gather ``seed_logits`` + per-hop ``walk_w`` at
                ``aux_query_pos``.
            query_pos: ``(N,)`` SINGLE query position per row.  When given, the walk
                is run only at that one position per row (``Lq = 1``) instead of at
                every position (``Lq = L``) — the O(K*L) single-query fast path used
                when the caller needs the reasoning hidden at ONE position (e.g. the
                answer position for the R-synthetic aux training / eval).  The returned
                hidden is ``(N, 1, d_model)`` and, with ``collect_aux``, the aux is that
                single row directly.  ``None`` (default) is the full per-position path
                (byte-for-byte the shipped LM behaviour).
            walk_hard: EVAL-TIME per-hop hard addressing (card 0a98292b).  When
                ``True``, straight-through top-1 the walk head AFTER each step's
                sharpen so the next read/query is a CLEAN single-position read — this
                stops the residual off-chain probability mass from accumulating over
                deep walks (the ~0.98/hop soft-blend compounding tax).  ``False``
                (default) is the byte-for-byte soft walk the tandem trains/evals with.
            gamma_add: EVAL-TIME extra sharpening added to every per-step ``gamma``
                (card 0a98292b, candidate 2).  ``0.0`` (default) leaves the trained
                sharpen untouched.
            clock_period: EVAL-TIME period for the walk step-clock (card 0a98292b).
                The clock is ``phi = 2*pi*step / period``; ``None`` (default) uses
                ``period = steps`` (the shipped behaviour — the clock stretches to fill
                the walk).  Setting it to the TRAINED walk depth makes the clock REPEAT
                its training cycle every ``clock_period`` hops, so a deep-extrapolation
                walk keeps seeing an in-distribution clock instead of a stretched one
                the controller never trained on.  A no-op when ``steps <= clock_period``
                (then ``step/period`` matches the default over the trained range).
        Returns:
            ``(h_reason (N, L, d_model), aux | None)``.
        """
        n, length = x.shape
        device = x.device
        rows = torch.arange(n, device=device)
        qmode = query_pos is not None                    # single-query fast path
        h = self.encoder(x)                              # (N, L, d)
        keys_n = F.normalize(self.name_key(h), dim=-1)   # (N, L, dk)
        ptr_keys = self.ptr_key(h)                       # (N, L, dk)
        vals = self.val(h)                               # (N, L, d)
        seed_keys_n = F.normalize(self.seed_name_key(h), dim=-1)

        # Causal mask: query q addresses keys <= q_pos.  Full path: every position is a
        # query (Lq=L).  Query path: one query per row (Lq=1) at ``query_pos``.
        pos = torch.arange(length, device=device)
        if qmode:
            assert query_pos is not None
            lq = 1
            addr_mask = pos.view(1, 1, length) <= query_pos.view(n, 1, 1)  # (N,1,Lk)
        else:
            lq = length
            addr_mask = (pos.unsqueeze(0) <= pos.unsqueeze(1)).unsqueeze(0)  # (1,Lq,Lk)
            addr_mask = addr_mask.expand(n, length, length)
        neg_inf = torch.finfo(h.dtype).min
        scale = F.softplus(self.key_scale)

        # --- Stage 1: locate (seed) ---
        # Pool the window ending at each query position -> a seed query, matched against
        # the DEDICATED locate keys.  ``seed_pre`` is the pre-softmax address logit
        # (cosine * scale, masked) kept for the aux locate-CE.
        pooled = self._causal_pool(h)                    # (N, L, d)
        if qmode:
            assert query_pos is not None
            pooled = pooled[rows, query_pos].unsqueeze(1)  # (N, 1, d)
        q0 = self.query_pool(pooled)                     # (N, Lq, dk)
        q0_n = F.normalize(q0, dim=-1)
        seed_cos = torch.einsum("nqd,nkd->nqk", q0_n, seed_keys_n)
        seed_pre = (scale * seed_cos).masked_fill(~addr_mask, neg_inf)  # (N,Lq,Lk)
        w = F.softmax(seed_pre, dim=-1)
        gamma_seed = torch.full((n, lq, 1), self.gamma_floor + 1.0, device=device, dtype=h.dtype)
        w = self._sharpen(w, gamma_seed)
        if self.hard_seed and commit_seed:
            w = _st_top1(w)

        # Teacher-force the walk seed at the supervised answer positions (train only).
        # Only rows with a VALID seed index (``tf_seed >= 0``) are overridden, so in a
        # mixed batch the reasoning rows get the known-head one-hot start while the
        # memory rows (``tf_seed = -1``) keep the learned locate.
        if tf_seed is not None and aux_query_pos is not None:
            valid = tf_seed >= 0
            if bool(valid.any()):
                one_hot = torch.zeros(n, length, device=device, dtype=h.dtype)
                one_hot.scatter_(1, tf_seed.clamp_min(0).unsqueeze(1), 1.0)
                w = w.clone()
                vr = rows[valid]
                if qmode:
                    w[vr, 0] = one_hot[valid]            # the single Lq row IS the query
                else:
                    w[vr, aux_query_pos[valid]] = one_hot[valid]

        aux_seed_logits = None
        walk_w_list: list[Tensor] = []
        if collect_aux and aux_query_pos is not None:
            aux_seed_logits = seed_pre[:, 0] if qmode else seed_pre[rows, aux_query_pos]  # (N, Lk)

        # --- Stage 2: walk ---
        ctrl = torch.zeros(n, lq, self.d_ctrl, device=device, dtype=h.dtype)
        clock_norm = steps if clock_period is None else clock_period
        for step in range(steps):
            phi = 2.0 * math.pi * step / max(1, clock_norm)
            clk = torch.tensor([math.sin(phi), math.cos(phi)], device=device, dtype=h.dtype)
            clk = clk.view(1, 1, 2).expand(n, lq, 2)
            r = torch.einsum("nqk,nkd->nqd", w, vals)          # (N, Lq, d)
            read_ptr = torch.einsum("nqk,nkd->nqd", w, ptr_keys)  # (N, Lq, dk)
            gru_in = torch.cat([r, clk], dim=-1).reshape(n * lq, -1)
            ctrl = self.gru(gru_in, ctrl.reshape(n * lq, -1)).view(n, lq, self.d_ctrl)

            query = self.to_query(ctrl)
            if self.direct_ptr:
                query = query + self.move_gain * read_ptr
            beta = F.softplus(self.to_beta(ctrl))
            g = torch.sigmoid(self.to_gate(ctrl))
            gamma = self.gamma_floor + F.softplus(self.to_gamma(ctrl))
            if gamma_add:
                gamma = gamma + gamma_add

            w_c = self._address(query, keys_n, addr_mask, scale, beta, neg_inf)
            w = g * w_c + (1.0 - g) * w
            w = w.masked_fill(~addr_mask, 0.0)
            w = w / w.sum(-1, keepdim=True).clamp_min(1e-12)
            w = self._sharpen(w, gamma)
            # Collect the SOFT head for the aux (walk-aux NLL needs a real distribution;
            # ``argmax`` is unchanged by the hard commit below, so the eval on-target
            # trace is identical either way).
            if collect_aux and aux_query_pos is not None:
                walk_w_list.append(w[:, 0] if qmode else w[rows, aux_query_pos])  # (N, Lk)
            if walk_hard:
                # HARD addressing: straight-through top-1 the head so the NEXT step's
                # read/query is a CLEAN single-position read (no compounding off-chain
                # blur).  Eval lever (card 0a98292b); also a valid TRAINING regime
                # (straight-through is differentiable) that sharpens per-hop precision
                # while the walk-aux above still supervises the underlying soft head.
                w = _st_top1(w)

        r_final = torch.einsum("nqk,nkd->nqd", w, vals)         # (N, Lq, d)
        fusion_hidden = torch.cat([r_final, ctrl], dim=-1)      # (N, Lq, d + d_ctrl)
        out = self.proj(fusion_hidden)                          # (N, Lq, d_model)

        aux: dict | None = None
        if collect_aux and aux_query_pos is not None:
            aux = {
                "seed_logits": aux_seed_logits,                 # (N, Lk)
                "walk_w": torch.stack(walk_w_list, dim=1) if walk_w_list else None,  # (N, K, Lk)
            }
        return out, aux

    def forward(
        self,
        x: Tensor,
        steps: int | None = None,
        commit_seed: bool = True,
        return_aux: bool = False,
        aux_query_pos: Tensor | None = None,
        tf_seed: Tensor | None = None,
        walk_hard: bool = False,
        gamma_add: float = 0.0,
        clock_period: int | None = None,
    ) -> Tensor | tuple[Tensor, dict]:
        """Per-position reasoning hidden over segment-bounded windows.

        Args:
            x: ``(B, T)`` token ids.
            steps: walk depth K for this forward (defaults to ``self.steps``).
            commit_seed: gate the ``hard_seed`` straight-through commit.
            return_aux: also return ``{"seed_logits", "walk_w"}`` gathered at
                ``aux_query_pos`` (requires a single window: ``T <= segment_len``).
            aux_query_pos: ``(B,)`` answer position per row (for the aux gather /
                teacher forcing).
            tf_seed: ``(B,)`` teacher-forced walk-seed index at ``aux_query_pos``.
            walk_hard: EVAL-TIME per-hop hard addressing (card 0a98292b) — straight-
                through top-1 the walk head after each step so deep walks do not
                accumulate off-chain blur.  ``False`` (default) = the soft walk the
                tandem trains/evals with (no behavioural change to the shipped path).
            gamma_add: EVAL-TIME extra per-step sharpening added to ``gamma``.  ``0.0``
                (default) leaves the trained sharpen untouched.
            clock_period: EVAL-TIME walk step-clock period (see ``_walk_window``).
                ``None`` (default) = the shipped stretch-to-fill clock.

        Returns:
            ``h_reason (B, T, d_model)`` or, when ``return_aux``, ``(h_reason, aux)``.
        """
        b, t = x.shape
        k = self.steps if steps is None else steps
        length = self.segment_len

        if return_aux or aux_query_pos is not None or tf_seed is not None:
            # Single-window regime (the mixed M+R reproduction feeds one segment).
            if t > length:
                raise ValueError(
                    f"aux/teacher-forcing require a single window (T={t} <= "
                    f"reasoning_segment_len={length})."
                )
            out, aux = self._walk_window(
                x, k, commit_seed, aux_query_pos, tf_seed, collect_aux=return_aux,
                walk_hard=walk_hard, gamma_add=gamma_add, clock_period=clock_period,
            )
            if return_aux:
                assert aux is not None
                return out, aux
            return out

        # General windowed path (text stream / free-running LM): chunk into windows
        # of length L, run each independently (encoder resets, addressing bounded to
        # the window) -> O(K*L)/position, linear in T.
        n_win = (t + length - 1) // length
        pad = n_win * length - t
        if pad:
            x = F.pad(x, (0, pad))
        xw = x.view(b, n_win, length).reshape(b * n_win, length)
        out, _ = self._walk_window(
            xw, k, commit_seed, aux_query_pos=None, tf_seed=None, collect_aux=False,
            walk_hard=walk_hard, gamma_add=gamma_add, clock_period=clock_period,
        )
        out = out.view(b, n_win, length, self.d_model).reshape(b, n_win * length, self.d_model)
        return out[:, :t]

    def query_forward(
        self,
        x: Tensor,
        query_pos: Tensor,
        steps: int | None = None,
        commit_seed: bool = True,
        return_aux: bool = False,
        tf_seed: Tensor | None = None,
        walk_hard: bool = False,
        gamma_add: float = 0.0,
        clock_period: int | None = None,
    ) -> Tensor | tuple[Tensor, dict]:
        """Single-query fast path: the reasoning hidden AT ``query_pos`` only.

        Runs the identical locate-then-walk as :meth:`forward` but only for the one
        query position per row, so cost is ``O(K * L)`` (not ``O(K * L^2)``) — the
        efficient route for R-synthetic aux training/eval, where only the answer
        position is supervised/read.  It is numerically identical to
        ``forward(x, ..., return_aux=True)`` gathered at ``query_pos`` (pinned by a test).

        Args:
            x: ``(B, T)`` token ids (single window: ``T <= segment_len``).
            query_pos: ``(B,)`` the one position per row to run the walk at.
            (Remaining args mirror :meth:`forward`.)

        Returns:
            ``h_reason (B, d_model)`` or, when ``return_aux``, ``(h_reason, aux)`` with
            ``aux = {"seed_logits" (B, T), "walk_w" (B, K, T)}``.
        """
        b, t = x.shape
        k = self.steps if steps is None else steps
        if t > self.segment_len:
            raise ValueError(
                f"query_forward requires a single window (T={t} <= "
                f"reasoning_segment_len={self.segment_len})."
            )
        out, aux = self._walk_window(
            x, k, commit_seed, aux_query_pos=query_pos, tf_seed=tf_seed,
            collect_aux=return_aux, walk_hard=walk_hard, gamma_add=gamma_add,
            clock_period=clock_period, query_pos=query_pos,
        )
        out = out[:, 0]  # (B, d_model) — the single query row
        if return_aux:
            assert aux is not None
            return out, aux
        return out


# =====================================================================================
# MULTI-HEAD WALK (card 22acac98) — N parallel weight-independent locate-then-walk heads
# over a SHARED causal encoder, combined on their FINAL reads.  Entirely additive and
# default-off: the shipped ``CausalReasoner`` above is untouched, and ``n_heads=1`` with
# the default ``combine="mean"`` reproduces a single ``CausalReasoner`` walk exactly
# (pinned by a faithfulness test).  The probe (card 22acac98) asks whether redundancy
# buys per-hop precision (independent per-hop errors ε -> joint ~ε^H under agreement) or
# whether the heads collapse to identical walks.
# =====================================================================================


def _mh_causal_pool(h: Tensor, window: int) -> Tensor:
    """Causal mean-pool of width ``window`` ending at each position (free-function twin of
    :meth:`CausalReasoner._causal_pool`, so each head can pool its OWN window without
    touching the shipped class).  ``(N, L, d)`` -> ``(N, L, d)``."""
    _, length, _ = h.shape
    acc = torch.zeros_like(h)
    cnt = torch.zeros(1, length, 1, device=h.device, dtype=h.dtype)
    ones = torch.ones(1, length, 1, device=h.device, dtype=h.dtype)
    for off in range(window):
        if off == 0:
            acc = acc + h
            cnt = cnt + ones
        else:
            acc[:, off:] = acc[:, off:] + h[:, : length - off]
            cnt[:, off:] = cnt[:, off:] + ones[:, : length - off]
    return acc / cnt


def _mh_sharpen(w: Tensor, gamma: Tensor) -> Tensor:
    """``w^gamma`` renormalised over the last dim (twin of ``CausalReasoner._sharpen``)."""
    wp = w.clamp_min(1e-12) ** gamma
    return wp / wp.sum(-1, keepdim=True).clamp_min(1e-12)


def _mh_address(
    query: Tensor, keys_n: Tensor, addr_mask: Tensor, scale: Tensor,
    beta: Tensor | None, neg_inf: float,
) -> Tensor:
    """Cosine content-address -> masked softmax head (twin of ``CausalReasoner._address``)."""
    q_n = F.normalize(query, dim=-1)
    cos = torch.einsum("nqd,nkd->nqk", q_n, keys_n)
    cos = cos.masked_fill(~addr_mask, neg_inf)
    logit = scale * cos if beta is None else scale * beta * cos
    return F.softmax(logit, dim=-1)


@dataclass(frozen=True)
class WalkHeadSpec:
    """Per-head architecture for :class:`MultiHeadCausalReasoner`.

    Homogeneous heads share one spec (identical architecture, INDEPENDENT parameters =
    diversity via init/dropout).  Heterogeneous heads take DIFFERENT specs — different
    ``steps`` (walk depth K), ``query_window`` (locate pool = short vs long addressing),
    ``d_ctrl`` / ``d_key`` (width) — so they cannot collapse to the same walk by
    construction (structural decorrelation, the card's "diversity for free").

    Fields:
        steps: walk depth K (native/trained budget for this head).
        query_window: causal locate-pool width for this head.
        d_key: addressing key dimension for this head.
        d_ctrl: walk-controller hidden dimension for this head.
        gamma_floor: mandatory per-step sharpen floor (>= 1.0).
        direct_ptr: read the RHS descriptor straight into the move query.
        hard_seed: straight-through commit the locate seed.
        read_dropout: dropout on the per-hop value read (train-only DIVERSITY pressure —
            independent masks per head decorrelate homogeneous trajectories; 0.0 = off).
        addr_range: if set, band-limit the walk addressing to keys within ``addr_range``
            positions back of the query (a "short-range" head); ``None`` = full causal
            window (the default / long-range head).
    """

    steps: int
    query_window: int
    d_key: int
    d_ctrl: int
    gamma_floor: float = 2.0
    direct_ptr: bool = True
    hard_seed: bool = True
    read_dropout: float = 0.0
    addr_range: int | None = None

    @classmethod
    def from_cfg(cls, cfg: ModelConfig) -> WalkHeadSpec:
        """The homogeneous default: a head matching the shipped ``CausalReasoner`` cfg."""
        return cls(
            steps=cfg.causal_reasoner_steps,
            query_window=cfg.causal_reasoner_query_window,
            d_key=cfg.causal_reasoner_key_dim,
            d_ctrl=cfg.d_model,
            gamma_floor=float(cfg.causal_reasoner_gamma_floor),
            direct_ptr=cfg.causal_reasoner_direct_ptr,
            hard_seed=cfg.causal_reasoner_hard_seed,
        )


class _WalkHead(nn.Module):
    """One locate-then-walk head over a SHARED encoder output ``h`` (see
    :class:`MultiHeadCausalReasoner`).  Holds its own copy of every walk parameter and
    a per-head fusion ``head_proj`` mapping ``cat([r_final, ctrl]) -> d_model`` (so the
    combine step is width-uniform even when heads differ in ``d_ctrl``).

    The walk math is byte-faithful to ``CausalReasoner._walk_window`` (a single head): a
    dedicated-seed locate, then K weight-tied soft hops with mandatory per-step sharpen,
    ``direct_ptr``, optional eval hard-addressing.  ``forward`` returns the fused per-head
    hidden, the per-row commit confidence (final head peak), and — when ``collect_aux`` —
    the seed logits, per-hop soft distribution, and final position at the query row.
    """

    def __init__(self, d_model: int, spec: WalkHeadSpec) -> None:
        super().__init__()
        if spec.gamma_floor < 1.0:
            raise ValueError("WalkHeadSpec.gamma_floor must be >= 1.0 (mandatory sharpen).")
        if spec.d_key < 2:
            raise ValueError("WalkHeadSpec.d_key must be >= 2.")
        if spec.query_window < 1:
            raise ValueError("WalkHeadSpec.query_window must be >= 1.")
        if spec.steps < 1:
            raise ValueError("WalkHeadSpec.steps must be >= 1.")
        self.spec = spec
        self.d_model = d_model
        d_key = spec.d_key
        d_ctrl = spec.d_ctrl

        self.name_key = nn.Linear(d_model, d_key)
        self.ptr_key = nn.Linear(d_model, d_key)
        self.val = nn.Linear(d_model, d_model)
        self.seed_name_key = nn.Linear(d_model, d_key)
        self.query_pool = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_key),
        )
        self.move_gain = nn.Parameter(torch.tensor(1.0))

        self.gru = nn.GRUCell(d_model + 2, d_ctrl)
        self.to_query = nn.Linear(d_ctrl, d_key)
        self.to_beta = nn.Linear(d_ctrl, 1)
        self.to_gate = nn.Linear(d_ctrl, 1)
        self.to_gamma = nn.Linear(d_ctrl, 1)
        self.key_scale = nn.Parameter(torch.tensor(5.0))
        self.read_drop = nn.Dropout(spec.read_dropout)
        self.head_proj = nn.Linear(d_model + d_ctrl, d_model)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Faithful init + the load-bearing head-bias priors (mirrors ``CausalReasoner``)."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()
        with torch.no_grad():
            self.to_beta.bias.fill_(1.0)
            self.to_gate.bias.fill_(0.0)
            self.to_gamma.bias.fill_(0.5)
            self.key_scale.fill_(5.0)
            self.move_gain.fill_(1.0)

    def forward(
        self,
        h: Tensor,
        addr_mask: Tensor,
        pos: Tensor,
        query_pos: Tensor | None,
        aux_query_pos: Tensor | None,
        tf_seed: Tensor | None,
        commit_seed: bool,
        steps: int,
        walk_hard: bool,
        gamma_add: float,
        clock_period: int | None,
        collect_aux: bool,
    ) -> tuple[Tensor, Tensor, dict | None]:
        n, length, _ = h.shape
        device = h.device
        rows = torch.arange(n, device=device)
        qmode = query_pos is not None
        lq = 1 if qmode else length
        spec = self.spec

        keys_n = F.normalize(self.name_key(h), dim=-1)
        ptr_keys = self.ptr_key(h)
        vals = self.val(h)
        seed_keys_n = F.normalize(self.seed_name_key(h), dim=-1)
        neg_inf = torch.finfo(h.dtype).min
        scale = F.softplus(self.key_scale)

        # Optional band-limit (a short-range head): key k is addressable by query q only
        # when q - addr_range <= k (<= q is already the causal mask).
        mask = addr_mask
        if spec.addr_range is not None:
            if qmode:
                assert query_pos is not None
                qp = query_pos.view(n, 1, 1)
            else:
                qp = pos.view(1, length, 1)
            band = pos.view(1, 1, length) >= (qp - spec.addr_range)
            mask = addr_mask & band

        # --- Stage 1: locate (seed) ---
        pooled = _mh_causal_pool(h, spec.query_window)
        if qmode:
            assert query_pos is not None
            pooled = pooled[rows, query_pos].unsqueeze(1)
        q0 = self.query_pool(pooled)
        q0_n = F.normalize(q0, dim=-1)
        seed_cos = torch.einsum("nqd,nkd->nqk", q0_n, seed_keys_n)
        seed_pre = (scale * seed_cos).masked_fill(~mask, neg_inf)
        w = F.softmax(seed_pre, dim=-1)
        gamma_seed = torch.full((n, lq, 1), spec.gamma_floor + 1.0, device=device, dtype=h.dtype)
        w = _mh_sharpen(w, gamma_seed)
        if spec.hard_seed and commit_seed:
            w = _st_top1(w)

        if tf_seed is not None and aux_query_pos is not None:
            valid = tf_seed >= 0
            if bool(valid.any()):
                one_hot = torch.zeros(n, length, device=device, dtype=h.dtype)
                one_hot.scatter_(1, tf_seed.clamp_min(0).unsqueeze(1), 1.0)
                w = w.clone()
                vr = rows[valid]
                if qmode:
                    w[vr, 0] = one_hot[valid]
                else:
                    w[vr, aux_query_pos[valid]] = one_hot[valid]

        aux_seed_logits = None
        walk_w_list: list[Tensor] = []
        if collect_aux and aux_query_pos is not None:
            aux_seed_logits = seed_pre[:, 0] if qmode else seed_pre[rows, aux_query_pos]

        # --- Stage 2: walk ---
        ctrl = torch.zeros(n, lq, spec.d_ctrl, device=device, dtype=h.dtype)
        clock_norm = steps if clock_period is None else clock_period
        for step in range(steps):
            phi = 2.0 * math.pi * step / max(1, clock_norm)
            clk = torch.tensor([math.sin(phi), math.cos(phi)], device=device, dtype=h.dtype)
            clk = clk.view(1, 1, 2).expand(n, lq, 2)
            r = torch.einsum("nqk,nkd->nqd", w, vals)
            r = self.read_drop(r)  # train-only diversity pressure (off at eval / p=0)
            read_ptr = torch.einsum("nqk,nkd->nqd", w, ptr_keys)
            gru_in = torch.cat([r, clk], dim=-1).reshape(n * lq, -1)
            ctrl = self.gru(gru_in, ctrl.reshape(n * lq, -1)).view(n, lq, spec.d_ctrl)

            query = self.to_query(ctrl)
            if spec.direct_ptr:
                query = query + self.move_gain * read_ptr
            beta = F.softplus(self.to_beta(ctrl))
            g = torch.sigmoid(self.to_gate(ctrl))
            gamma = spec.gamma_floor + F.softplus(self.to_gamma(ctrl))
            if gamma_add:
                gamma = gamma + gamma_add

            w_c = _mh_address(query, keys_n, mask, scale, beta, neg_inf)
            w = g * w_c + (1.0 - g) * w
            w = w.masked_fill(~mask, 0.0)
            w = w / w.sum(-1, keepdim=True).clamp_min(1e-12)
            w = _mh_sharpen(w, gamma)
            if collect_aux and aux_query_pos is not None:
                walk_w_list.append(w[:, 0] if qmode else w[rows, aux_query_pos])
            if walk_hard:
                w = _st_top1(w)

        r_final = torch.einsum("nqk,nkd->nqd", w, vals)
        conf = w.max(dim=-1).values                       # (N, Lq) commit confidence
        head_out = self.head_proj(torch.cat([r_final, ctrl], dim=-1))  # (N, Lq, d_model)

        aux: dict | None = None
        if collect_aux and aux_query_pos is not None:
            w_at = w[:, 0] if qmode else w[rows, aux_query_pos]        # (N, Lk)
            aux = {
                "seed_logits": aux_seed_logits,
                "walk_w": torch.stack(walk_w_list, dim=1) if walk_w_list else None,
                "final_pos": w_at.argmax(dim=-1),                     # (N,)
                "conf": w_at.max(dim=-1).values,                     # (N,)
            }
        return head_out, conf, aux


class MultiHeadCausalReasoner(nn.Module):
    """Multi-head causal locate-then-walk reasoner (card 22acac98).

    ``n_heads`` parallel :class:`_WalkHead` walks over ONE shared
    :class:`CausalGRUEncoder`, fused per-head to ``d_model`` and then COMBINED on those
    final reads.  Default-off and interface-stable: ``n_heads=1`` + ``combine="mean"``
    reproduces a single :class:`CausalReasoner` walk exactly.

    Args:
        cfg: a :class:`~graph_llm.config.ModelConfig` (supplies ``d_model``,
            ``vocab_size``, ``reasoning_segment_len`` and the ``causal_reasoner_*``
            defaults used for homogeneous heads / the shared encoder).
        n_heads: number of parallel walk heads (default 1).
        head_specs: optional per-head :class:`WalkHeadSpec` list of length ``n_heads``
            (HETEROGENEOUS heads).  ``None`` (default) = HOMOGENEOUS: every head uses
            ``WalkHeadSpec.from_cfg(cfg)`` (identical architecture, independent params).
        combine: how to reduce the ``n_heads`` per-head hiddens ->
            ``"mean"`` (equal average), ``"confidence"`` (agreement-gated: softmax over
            heads of the final commit confidence), or ``"concat"`` (learned projection of
            the concatenation).
        combine_temp_init: initial temperature for the ``"confidence"`` softmax.
    """

    def __init__(
        self,
        cfg: ModelConfig,
        *,
        n_heads: int = 1,
        head_specs: Sequence[WalkHeadSpec] | None = None,
        combine: str = "mean",
        combine_temp_init: float = 1.0,
    ) -> None:
        super().__init__()
        if n_heads < 1:
            raise ValueError("n_heads must be >= 1.")
        if combine not in ("mean", "confidence", "concat"):
            raise ValueError(f"combine must be mean|confidence|concat, got {combine!r}.")
        if cfg.reasoning_segment_len < 1:
            raise ValueError("reasoning_segment_len must be >= 1.")
        if head_specs is not None and len(head_specs) != n_heads:
            raise ValueError(
                f"head_specs length {len(head_specs)} != n_heads {n_heads}."
            )

        self.d_model = cfg.d_model
        self.segment_len = cfg.reasoning_segment_len
        self.n_heads = n_heads
        self.combine = combine
        if head_specs is None:
            base = WalkHeadSpec.from_cfg(cfg)
            specs: list[WalkHeadSpec] = [replace(base) for _ in range(n_heads)]
        else:
            specs = list(head_specs)
        self.specs = tuple(specs)
        # Default single-head walk depth (mirrors CausalReasoner.steps for the API).
        self.steps = specs[0].steps

        self.encoder = CausalGRUEncoder(
            cfg.vocab_size,
            cfg.d_model,
            conv_kernel=cfg.causal_reasoner_conv_kernel,
            gru_layers=cfg.causal_reasoner_gru_layers,
        )
        self.heads = nn.ModuleList(_WalkHead(cfg.d_model, s) for s in specs)
        self.combine_temp = nn.Parameter(torch.tensor(float(combine_temp_init)))
        self.final_proj = (
            nn.Linear(n_heads * cfg.d_model, cfg.d_model) if combine == "concat" else None
        )

    # ------------------------------------------------------------------- helpers
    def _resolve_steps(self, steps: int | Sequence[int] | None) -> list[int]:
        """Per-head effective walk depth.  ``None`` -> each head's native ``spec.steps``;
        an int -> uniform override (deep-extrapolation eval); a sequence -> per head."""
        if steps is None:
            return [s.steps for s in self.specs]
        if isinstance(steps, int):
            return [steps] * self.n_heads
        steps = list(steps)
        if len(steps) != self.n_heads:
            raise ValueError(f"per-head steps length {len(steps)} != n_heads {self.n_heads}.")
        return steps

    def _combine(self, outs: Tensor, confs: Tensor) -> tuple[Tensor, Tensor | None]:
        """Reduce ``outs (N,Lq,H,d)`` over the head axis -> ``(N,Lq,d)`` + combine weights."""
        if self.combine == "mean":
            return outs.mean(dim=2), None
        if self.combine == "confidence":
            alpha = F.softmax(self.combine_temp * confs, dim=2)      # (N,Lq,H)
            combined = (alpha.unsqueeze(-1) * outs).sum(dim=2)
            return combined, alpha
        # concat
        n, lq, hh, d = outs.shape
        assert self.final_proj is not None
        return self.final_proj(outs.reshape(n, lq, hh * d)), None

    def _run(
        self,
        x: Tensor,
        steps_per_head: list[int],
        commit_seed: bool,
        aux_query_pos: Tensor | None,
        tf_seed: Tensor | None,
        collect_aux: bool,
        walk_hard: bool,
        gamma_add: float,
        clock_period: int | None,
        query_pos: Tensor | None,
    ) -> tuple[Tensor, dict | None]:
        """Encode once (shared), run every head, combine on the final reads."""
        n, length = x.shape
        device = x.device
        qmode = query_pos is not None
        h = self.encoder(x)
        pos = torch.arange(length, device=device)
        if qmode:
            assert query_pos is not None
            addr_mask = pos.view(1, 1, length) <= query_pos.view(n, 1, 1)
        else:
            addr_mask = (pos.unsqueeze(0) <= pos.unsqueeze(1)).unsqueeze(0)
            addr_mask = addr_mask.expand(n, length, length)

        head_outs: list[Tensor] = []
        confs: list[Tensor] = []
        per_head_aux: list[dict | None] = []
        for i, head in enumerate(self.heads):
            ho, conf, aux_h = head(
                h, addr_mask, pos, query_pos, aux_query_pos, tf_seed, commit_seed,
                steps_per_head[i], walk_hard, gamma_add, clock_period, collect_aux,
            )
            head_outs.append(ho)
            confs.append(conf)
            per_head_aux.append(aux_h)

        outs = torch.stack(head_outs, dim=2)      # (N, Lq, H, d)
        cf = torch.stack(confs, dim=2)            # (N, Lq, H)
        combined, cw = self._combine(outs, cf)    # (N, Lq, d)

        aux: dict | None = None
        if collect_aux and aux_query_pos is not None:
            aux = {
                "per_head": per_head_aux,
                "combine_weights": None if cw is None else (cw[:, 0] if qmode else cw),
                # Convenience mirrors of head-0 aux (single-head-compatible callers).
                "seed_logits": per_head_aux[0]["seed_logits"] if per_head_aux[0] else None,
                "walk_w": per_head_aux[0]["walk_w"] if per_head_aux[0] else None,
            }
        return combined, aux

    # -------------------------------------------------------------------- public
    def forward(
        self,
        x: Tensor,
        steps: int | Sequence[int] | None = None,
        commit_seed: bool = True,
        return_aux: bool = False,
        aux_query_pos: Tensor | None = None,
        tf_seed: Tensor | None = None,
        walk_hard: bool = False,
        gamma_add: float = 0.0,
        clock_period: int | None = None,
    ) -> Tensor | tuple[Tensor, dict]:
        """Per-position combined reasoning hidden over segment-bounded windows.

        Mirrors :meth:`CausalReasoner.forward`; ``steps`` may be an int (uniform), a
        per-head sequence, or ``None`` (each head's native ``spec.steps``).  ``return_aux``
        gathers per-head ``{seed_logits, walk_w, final_pos, conf}`` at ``aux_query_pos``
        (requires a single window).
        """
        _, t = x.shape
        length = self.segment_len
        steps_per_head = self._resolve_steps(steps)

        if return_aux or aux_query_pos is not None or tf_seed is not None:
            if t > length:
                raise ValueError(
                    f"aux/teacher-forcing require a single window (T={t} <= "
                    f"reasoning_segment_len={length})."
                )
            out, aux = self._run(
                x, steps_per_head, commit_seed, aux_query_pos, tf_seed,
                collect_aux=return_aux, walk_hard=walk_hard, gamma_add=gamma_add,
                clock_period=clock_period, query_pos=None,
            )
            if return_aux:
                assert aux is not None
                return out, aux
            return out

        b = x.shape[0]
        n_win = (t + length - 1) // length
        pad = n_win * length - t
        if pad:
            x = F.pad(x, (0, pad))
        xw = x.view(b, n_win, length).reshape(b * n_win, length)
        out, _ = self._run(
            xw, steps_per_head, commit_seed, aux_query_pos=None, tf_seed=None,
            collect_aux=False, walk_hard=walk_hard, gamma_add=gamma_add,
            clock_period=clock_period, query_pos=None,
        )
        out = out.view(b, n_win, length, self.d_model).reshape(b, n_win * length, self.d_model)
        return out[:, :t]

    def query_forward(
        self,
        x: Tensor,
        query_pos: Tensor,
        steps: int | Sequence[int] | None = None,
        commit_seed: bool = True,
        return_aux: bool = False,
        tf_seed: Tensor | None = None,
        walk_hard: bool = False,
        gamma_add: float = 0.0,
        clock_period: int | None = None,
    ) -> Tensor | tuple[Tensor, dict]:
        """Single-query fast path: the combined reasoning hidden AT ``query_pos`` only —
        ``O(sum_h K_h * L)`` (mirrors :meth:`CausalReasoner.query_forward`)."""
        _, t = x.shape
        if t > self.segment_len:
            raise ValueError(
                f"query_forward requires a single window (T={t} <= "
                f"reasoning_segment_len={self.segment_len})."
            )
        steps_per_head = self._resolve_steps(steps)
        out, aux = self._run(
            x, steps_per_head, commit_seed, aux_query_pos=query_pos, tf_seed=tf_seed,
            collect_aux=return_aux, walk_hard=walk_hard, gamma_add=gamma_add,
            clock_period=clock_period, query_pos=query_pos,
        )
        out = out[:, 0]
        if return_aux:
            assert aux is not None
            return out, aux
        return out


__all__ = [
    "CausalGRUEncoder",
    "CausalReasoner",
    "MultiHeadCausalReasoner",
    "WalkHeadSpec",
]
