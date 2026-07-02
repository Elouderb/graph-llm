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

        Returns:
            ``(h_reason (N, L, d_model), aux | None)``.
        """
        n, length = x.shape
        device = x.device
        h = self.encoder(x)                              # (N, L, d)
        keys_n = F.normalize(self.name_key(h), dim=-1)   # (N, L, dk)
        ptr_keys = self.ptr_key(h)                       # (N, L, dk)
        vals = self.val(h)                               # (N, L, d)
        seed_keys_n = F.normalize(self.seed_name_key(h), dim=-1)

        # Causal mask: query t addresses keys <= t (predicting token t+1).
        pos = torch.arange(length, device=device)
        addr_mask = (pos.unsqueeze(0) <= pos.unsqueeze(1)).unsqueeze(0)  # (1,Lq,Lk)
        addr_mask = addr_mask.expand(n, length, length)
        neg_inf = torch.finfo(h.dtype).min
        scale = F.softplus(self.key_scale)

        # --- Stage 1: locate (seed) ---
        # Pool the window ending at each query position -> a seed query, matched against
        # the DEDICATED locate keys.  ``seed_pre`` is the pre-softmax address logit
        # (cosine * scale, masked) kept for the aux locate-CE.
        pooled = self._causal_pool(h)                    # (N, L, d)
        q0 = self.query_pool(pooled)                     # (N, L, dk)
        q0_n = F.normalize(q0, dim=-1)
        seed_cos = torch.einsum("nqd,nkd->nqk", q0_n, seed_keys_n)
        seed_pre = (scale * seed_cos).masked_fill(~addr_mask, neg_inf)  # (N,Lq,Lk)
        w = F.softmax(seed_pre, dim=-1)
        gamma_seed = torch.full((n, length, 1), self.gamma_floor + 1.0, device=device, dtype=h.dtype)
        w = self._sharpen(w, gamma_seed)
        if self.hard_seed and commit_seed:
            w = _st_top1(w)

        # Teacher-force the walk seed at the supervised answer positions (train only).
        # Only rows with a VALID seed index (``tf_seed >= 0``) are overridden, so in a
        # mixed batch the reasoning rows get the known-head one-hot start while the
        # memory rows (``tf_seed = -1``) keep the learned locate.
        if tf_seed is not None and aux_query_pos is not None:
            rows = torch.arange(n, device=device)
            valid = tf_seed >= 0
            if bool(valid.any()):
                one_hot = torch.zeros(n, length, device=device, dtype=h.dtype)
                one_hot.scatter_(1, tf_seed.clamp_min(0).unsqueeze(1), 1.0)
                w = w.clone()
                vr = rows[valid]
                w[vr, aux_query_pos[valid]] = one_hot[valid]

        aux_seed_logits = None
        walk_w_list: list[Tensor] = []
        if collect_aux and aux_query_pos is not None:
            rows = torch.arange(n, device=device)
            aux_seed_logits = seed_pre[rows, aux_query_pos]  # (N, Lk)

        # --- Stage 2: walk ---
        ctrl = torch.zeros(n, length, self.d_ctrl, device=device, dtype=h.dtype)
        for step in range(steps):
            phi = 2.0 * math.pi * step / max(1, steps)
            clk = torch.tensor([math.sin(phi), math.cos(phi)], device=device, dtype=h.dtype)
            clk = clk.view(1, 1, 2).expand(n, length, 2)
            r = torch.einsum("nqk,nkd->nqd", w, vals)          # (N, Lq, d)
            read_ptr = torch.einsum("nqk,nkd->nqd", w, ptr_keys)  # (N, Lq, dk)
            gru_in = torch.cat([r, clk], dim=-1).reshape(n * length, -1)
            ctrl = self.gru(gru_in, ctrl.reshape(n * length, -1)).view(n, length, self.d_ctrl)

            query = self.to_query(ctrl)
            if self.direct_ptr:
                query = query + self.move_gain * read_ptr
            beta = F.softplus(self.to_beta(ctrl))
            g = torch.sigmoid(self.to_gate(ctrl))
            gamma = self.gamma_floor + F.softplus(self.to_gamma(ctrl))

            w_c = self._address(query, keys_n, addr_mask, scale, beta, neg_inf)
            w = g * w_c + (1.0 - g) * w
            w = w.masked_fill(~addr_mask, 0.0)
            w = w / w.sum(-1, keepdim=True).clamp_min(1e-12)
            w = self._sharpen(w, gamma)
            if collect_aux and aux_query_pos is not None:
                rows = torch.arange(n, device=device)
                walk_w_list.append(w[rows, aux_query_pos])  # (N, Lk)

        r_final = torch.einsum("nqk,nkd->nqd", w, vals)         # (N, Lq, d)
        fusion_hidden = torch.cat([r_final, ctrl], dim=-1)      # (N, Lq, d + d_ctrl)
        out = self.proj(fusion_hidden)                          # (N, L, d_model)

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
                x, k, commit_seed, aux_query_pos, tf_seed, collect_aux=return_aux
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
            xw, k, commit_seed, aux_query_pos=None, tf_seed=None, collect_aux=False
        )
        out = out.view(b, n_win, length, self.d_model).reshape(b, n_win * length, self.d_model)
        return out[:, :t]


__all__ = ["CausalGRUEncoder", "CausalReasoner"]
