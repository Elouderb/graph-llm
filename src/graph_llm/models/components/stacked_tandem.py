"""Stacked-tandem topology components (card 3ac77deb).

Ports the validated stacked-tandem probe (scratchpad/stacked_{tandem,readback}.py,
3/3 seeds compositional) into the repo.  The repeating unit is a WHOLE tandem block

    {delta memory || causal reasoner || per-position workhorse} -> 3-way softmax
    gate -> fused per-position hidden

where block N's full-sequence fused output is block N+1's input stream and only the
FINAL block feeds the (untied) LM head.  A single fusion point runs memory and reasoner
in PARALLEL on the same input, so a single block PROVABLY fails a two-stage
compositional task (retrieve-then-reason); a SECOND block reasoning over the FIRST's
retrieval composes it.

Validated wiring facts mirrored here (see the card; do not re-derive):

* Each block's reasoner reads the RAW answer-segment TOKENS through its OWN
  :class:`CausalGRUEncoder` (the recurrence carries a clause's LHS name to the clause
  END — a conv-only front-end provably cannot).  It is run ONLY at the leak-free
  prediction position ``pred_pos = answer_pos - 1`` via the O(K*L)
  ``query_forward_composed``; its readout is written (residual) into the block fusion
  at that position, IDENTITY elsewhere.
* The locate SEED at ``pred_pos`` is blended with a ``seed_ctx`` vector taken from the
  BLOCK HIDDEN at that position (for the compositional type, block-0's memory writes
  the retrieved head there so block-1 can decode WHICH chain to walk).  The
  ``seed_ctx_proj`` is ZERO-INIT (the identity-init doctrine): at step 0 the composed
  seed == the raw-token pool (an exact superset of the working single-stage reasoner),
  so the optimizer OPENS the hidden channel only where it helps.
* Each block carries its OWN cross-segment :class:`DeltaMemoryState` (one memory per
  block).  Memory, workhorse, and gate of block N+1 consume block N's fused hidden.
* Optional bounded cross-attention READBACK between block 0 and block 1: Q = block-0
  fused output, K/V = the bottom (front-end) embedding stream, window-relative RoPE,
  strictly-causal bounded window (O(T*W)), 1-2 heads, ZERO-INIT output projection so
  readback-on == readback-off at step 0.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from graph_llm.models.components.causal_reasoner import CausalReasoner
from graph_llm.models.components.delta_memory import DeltaMemoryState, GatedDeltaMemory

if TYPE_CHECKING:
    from graph_llm.config import ModelConfig

# Suppress an unused expert's gate logit (asym inner blocks): softmax([., NEG, .]) ~= 0
# mass on that expert.
_NEG = -1e9


def gather_positions(h: Tensor, pos: Tensor) -> Tensor:
    """Gather ``h[b, pos[b]]`` -> ``(B, D)`` (the leak-free per-row prediction position)."""
    b = h.shape[0]
    return h[torch.arange(b, device=h.device), pos]


class StackedWorkhorseMLP(nn.Module):
    """Per-position workhorse: LN -> Linear -> GELU -> Linear, residual.

    The plain 'neither memory nor reasoner' pathway of a stacked block (the probe's
    ``GatedMLP``).  Stateless + per-position, so it never adds future dependence — the
    LM-leak guarantee is preserved.
    """

    def __init__(self, d_model: int, ff_mult: int = 2) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, ff_mult * d_model)
        self.fc2 = nn.Linear(ff_mult * d_model, d_model)

    def forward(self, h: Tensor) -> Tensor:
        return h + self.fc2(F.gelu(self.fc1(self.norm(h))))


class ComposableCausalReasoner(CausalReasoner):
    """:class:`CausalReasoner` + the stacked-tandem COMPOSITION channel (card 3ac77deb).

    Identical to the shipped reasoner — raw-token causal GRU encoder, clause-name keys,
    locate-then-walk, aux, teacher-forcing — EXCEPT the locate SEED query is blended with
    a ``seed_ctx`` vector taken from the BLOCK HIDDEN at the query position.  For a
    compositional (retrieve-then-reason) type, an EARLIER block's memory writes the
    retrieved head into that hidden, so a LATER block's reasoner can decode WHICH chain
    to walk (the head is NOT in the raw answer segment — only the query marker is).
    ``seed_ctx_proj`` is ZERO-INIT so at step 0 the composed seed == the raw-token pool
    (an exact superset of the working single-stage reasoner); addressing/walk keys stay
    on the RAW tokens (the surface name structure the walk needs).
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__(cfg)
        # Interposed into the working locate seed -> IDENTITY-INIT (zero-init), the
        # ``seed_ctx_proj`` doctrine: opens the hidden channel only where it helps.
        self.seed_ctx_proj = nn.Linear(self.d_model, self.d_key)
        nn.init.zeros_(self.seed_ctx_proj.weight)
        nn.init.zeros_(self.seed_ctx_proj.bias)

    def query_forward_composed(
        self,
        x: Tensor,
        query_pos: Tensor,
        seed_ctx: Tensor,
        steps: int | None = None,
        tf_seed: Tensor | None = None,
        commit_seed: bool = True,
        return_aux: bool = True,
        hidden_input: bool = False,
    ) -> tuple[Tensor, dict]:
        """Single-query locate-then-walk with the hidden-derived composition seed.

        A THIN wrapper over the shipped :meth:`CausalReasoner.query_forward` — it reuses the
        exact SAME locate-then-walk (no duplicated walk body -> no drift, and the eval levers
        ``walk_hard``/``gamma_add``/``clock_period`` stay available on the shared path).  The
        ONLY addition is the composition seed: ``q0 += seed_ctx_proj(seed_ctx)`` threaded via
        the base ``seed_ctx_add`` hook.  ``seed_ctx_proj`` is ZERO-INIT, so at step 0 the
        added term is 0 and this is numerically identical to ``query_forward`` (pinned by a
        test).  ``hidden_input`` addresses a GROUNDED hidden stream ``x (N, L, D)`` instead of
        raw token ids ``x (N, L)`` via the base ``hidden_input`` hook.

        Returns ``(h_reason (B, d), aux {seed_logits (B, L), walk_w (B, K, L)})``.
        """
        seed_ctx_add = self.seed_ctx_proj(seed_ctx).unsqueeze(1)  # (N, 1, d_key), 0 at init
        result = self.query_forward(
            x, query_pos, steps=steps, commit_seed=commit_seed, return_aux=return_aux,
            tf_seed=tf_seed, hidden_input=hidden_input, seed_ctx_add=seed_ctx_add,
        )
        if return_aux:
            assert isinstance(result, tuple)
            return result
        assert isinstance(result, Tensor)
        return result, {"seed_logits": None, "walk_w": None}

    def forward_composed(
        self,
        x: Tensor,
        seed_ctx: Tensor,
        steps: int | None = None,
        hidden_input: bool = False,
    ) -> Tensor:
        """Full per-position locate-then-walk with the hidden-derived composition seed (LM path).

        The full-sequence twin of :meth:`query_forward_composed` for the plain LM forward
        (card d05da2db): EVERY position predicts, so every position needs a reasoning hidden.
        ``seed_ctx`` is the BLOCK HIDDEN ``(B, T, d)`` (an earlier block's memory writes the
        retrieved head there for the compositional type); it is projected PER POSITION to
        ``(B, T, d_key)`` by the ZERO-INIT ``seed_ctx_proj`` and threaded into the base
        :meth:`CausalReasoner.forward` seed via the ``seed_ctx_add`` hook — additive and inert
        at step 0, so at init the composed walk == the shipped windowed walk (pinned by a test).
        ``hidden_input`` addresses the grounded block hidden ``x (B, T, d)`` instead of raw
        token ids (the ``reason_reads_hidden`` variant).

        Returns ``h_reason (B, T, d_model)``.
        """
        seed_ctx_add = self.seed_ctx_proj(seed_ctx)  # (B, T, d_key), 0 at init
        out = self.forward(
            x, steps=steps, commit_seed=True, hidden_input=hidden_input,
            seed_ctx_add=seed_ctx_add,
        )
        assert isinstance(out, Tensor)
        return out


class StackedTandemBlock(nn.Module):
    """{memory || reasoner(raw tokens) || workhorse -> 3-way gated fusion}.

    Consumes a per-segment list of hiddens and returns the full-sequence residual fused
    output (so blocks compose), the per-example gate at ``pred_pos``, and the final
    block's reasoner aux.  Each block owns its cross-segment :class:`DeltaMemoryState`.
    """

    def __init__(
        self,
        cfg: ModelConfig,
        *,
        reason_enabled: bool = True,
        gate_bias_init: float = 0.0,
        mlp_ff_mult: int = 2,
        reason_reads_hidden: bool = False,
        gate_per_channel: bool = False,
        gate_input_routed: bool = False,
        reason_threshold: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_model = cfg.d_model
        self.reason_enabled = reason_enabled
        # structural-necessity variant: address the block's (readback-grounded) HIDDEN
        # stream instead of the raw-token tap.
        self.reason_reads_hidden = reason_reads_hidden
        self.mem = GatedDeltaMemory(cfg)
        self.mem_norm = nn.LayerNorm(cfg.d_model)
        if reason_enabled:
            self.reasoner: ComposableCausalReasoner | None = ComposableCausalReasoner(cfg)
        else:
            self.reasoner = None
        self.mlp = StackedWorkhorseMLP(cfg.d_model, mlp_ff_mult)
        # Per-channel gate option (card 1cfe0cb8): False (default) == the shipped SCALAR gate,
        # one 3-way softmax per position (``Linear(4d, 3)``) — byte-for-byte the committed
        # stacked recipe.  True builds a PER-CHANNEL gate (``Linear(4d, 3*d)``, reshaped to
        # ``(..., d, 3)`` in ``forward``/``lm_forward`` — an independent 3-way softmax per
        # output channel).  Applied consistently to BOTH stacked entry points.
        self.gate_per_channel = gate_per_channel
        # INPUT-ROUTED gate (card 75ada834): route BEFORE the experts (MoE routing).  False
        # (default) == the shipped OUTPUT-mixture gate reading ``[h_mem ; h_reason ; h_mlp ; ctx]``
        # (``Linear(4d, ...)``).  True narrows the gate to ``[h_mem ; h_mlp ; ctx]``
        # (``Linear(3d, 3)``) so routing is decided WITHOUT the reasoner output — the precondition
        # for skipping the walk (``reason_threshold``).  SCALAR-only (the locked ladder gate): it
        # is incompatible with the per-channel gate.
        if gate_input_routed and gate_per_channel:
            raise ValueError(
                "stacked_gate_input_routed is scalar-only (the locked ladder gate); it cannot "
                "be combined with stacked_gate_per_channel."
            )
        self.gate_input_routed = gate_input_routed
        # EVAL/INFERENCE-ONLY sparse-invocation threshold (card 75ada834): positions whose
        # reasoner gate weight is ``< tau`` skip the walk on the LM path.  Requires the
        # input-routed gate (the gate must route WITHOUT the reasoner output); a negative tau is
        # invalid.  0.0 (default) == OFF (dense everywhere).
        if reason_threshold < 0.0:
            raise ValueError(f"stacked_reason_threshold must be >= 0, got {reason_threshold}.")
        if reason_threshold > 0.0 and not gate_input_routed:
            raise ValueError(
                "stacked_reason_threshold > 0 requires stacked_gate_input_routed=True (the gate "
                "must decide the reasoner weight WITHOUT its output so the walk can be skipped)."
            )
        self.reason_threshold = reason_threshold
        gate_in = 3 if gate_input_routed else 4
        gate_out = 3 * cfg.d_model if gate_per_channel else 3
        self.gate = nn.Linear(gate_in * cfg.d_model, gate_out)
        with torch.no_grad():
            self.gate.bias.fill_(gate_bias_init)

    def _pathways(
        self,
        seg_hidden: list[Tensor],
        raw_ans: Tensor,
        answer_seg: int,
        pred_pos: Tensor,
        steps: int | None,
        tf_seed: Tensor | None,
    ) -> tuple[list[Tensor], list[Tensor], list[Tensor], dict | None]:
        state: DeltaMemoryState | None = None
        mem_seq: list[Tensor] = []
        for h in seg_hidden:
            o, state = self.mem(self.mem_norm(h), state_in=state, return_state=True)
            mem_seq.append(o)
        mlp_seq = [self.mlp(h) for h in seg_hidden]
        rea_seq = list(seg_hidden)
        aux: dict | None = None
        if self.reasoner is not None:
            h_ans = seg_hidden[answer_seg]
            b = h_ans.shape[0]
            ar = torch.arange(b, device=h_ans.device)
            # The locate seed also reads the BLOCK HIDDEN at pred_pos (an earlier block's
            # memory writes the retrieved head there for the compositional type);
            # addressing on raw tokens (default) OR on the grounded block hidden.
            seed_ctx = gather_positions(h_ans, pred_pos)  # (B, D)
            rin = h_ans if self.reason_reads_hidden else raw_ans
            r_vec, aux = self.reasoner.query_forward_composed(
                rin, query_pos=pred_pos, seed_ctx=seed_ctx, steps=steps, tf_seed=tf_seed,
                commit_seed=True, return_aux=True, hidden_input=self.reason_reads_hidden,
            )
            add = torch.zeros_like(h_ans)
            add[ar, pred_pos] = r_vec
            rea_ans = h_ans + add
            rea_seq = [rea_ans if si == answer_seg else h for si, h in enumerate(seg_hidden)]
        return mem_seq, rea_seq, mlp_seq, aux

    def forward(
        self,
        seg_hidden: list[Tensor],
        raw_ans: Tensor,
        answer_seg: int,
        pred_pos: Tensor,
        steps: int | None = None,
        force_gate: Tensor | None = None,
        gate_mix: bool = False,
        gate_noise: float = 0.0,
        tf_seed: Tensor | None = None,
    ) -> tuple[list[Tensor], Tensor, dict | None]:
        """Returns ``(fused_seg list, gate_pred (B, 3), aux {seed_logits, walk_w} | None)``."""
        mem_seq, rea_seq, mlp_seq, aux = self._pathways(
            seg_hidden, raw_ans, answer_seg, pred_pos, steps, tf_seed
        )
        b = seg_hidden[0].shape[0]
        ar = torch.arange(b, device=seg_hidden[0].device)
        fused_seg: list[Tensor] = []
        gate_pred: Tensor | None = None
        for si in range(len(seg_hidden)):
            if self.gate_input_routed:
                # INPUT-ROUTED (card 75ada834): route BEFORE the reasoner — the gate omits
                # ``rea_seq`` (h_reason) and reads only the cheap ``[h_mem ; h_mlp ; ctx]``.
                gin = torch.cat([mem_seq[si], mlp_seq[si], seg_hidden[si]], dim=-1)
            else:
                gin = torch.cat([mem_seq[si], rea_seq[si], mlp_seq[si], seg_hidden[si]], dim=-1)
            logit = self.gate(gin)  # (B, L, 3) scalar, or (B, L, 3*d_model) per-channel
            if self.gate_per_channel:
                logit = logit.view(*logit.shape[:-1], self.d_model, 3)  # (B, L, d, 3)
            if not self.reason_enabled:
                logit = logit + torch.tensor([0.0, _NEG, 0.0], device=logit.device)
            if gate_noise > 0.0 and self.training:
                logit = logit + gate_noise * torch.randn_like(logit)
            w = F.softmax(logit, dim=-1)
            if gate_mix:
                if self.reason_enabled:
                    w = torch.full_like(w, 1.0 / 3.0)
                else:
                    w = torch.tensor([0.5, 0.0, 0.5], device=w.device).expand_as(w)
            if force_gate is not None:
                oh = F.one_hot(force_gate.clamp(0, 2), num_classes=3).to(w.dtype)  # (B, 3)
                oh_b = oh.unsqueeze(1).unsqueeze(1) if self.gate_per_channel else oh.unsqueeze(1)
                w = oh_b.expand_as(w)
            if self.gate_per_channel:
                # Independent per-channel softmax weights (B, L, d, 3): combine via an
                # explicit stack + weighted sum.  Summation order is immaterial here — there
                # is no prior byte-for-byte baseline for this NEW mode, unlike the scalar path
                # below (card 1cfe0cb8).
                paths = torch.stack([mem_seq[si], rea_seq[si], mlp_seq[si]], dim=-1)
                fused = (paths * w).sum(dim=-1)
            else:
                # SHIPPED scalar path — kept EXACTLY as committed (identical op order) so the
                # default stays byte-for-byte ``torch.equal`` against HEAD.
                fused = (
                    w[..., 0:1] * mem_seq[si]
                    + w[..., 1:2] * rea_seq[si]
                    + w[..., 2:3] * mlp_seq[si]
                )
            fused_seg.append(fused)
            if si == answer_seg:
                g = w[ar, pred_pos]
                # Gate-fraction reporting is shape-stable ((B, 3)) regardless of the flag: the
                # per-channel gate is reduced by an equal-weight mean over channels.
                gate_pred = g.mean(dim=-2) if self.gate_per_channel else g
        assert gate_pred is not None
        return fused_seg, gate_pred, aux

    def lm_forward(
        self,
        h_in: Tensor,
        raw_tokens: Tensor,
        state_in: DeltaMemoryState | None = None,
        steps: int | None = None,
    ) -> tuple[Tensor, Tensor, DeltaMemoryState]:
        """Full-sequence, per-position LM forward of ONE stacked block (card d05da2db).

        Mirrors the single-fusion LM semantics PER BLOCK: memory over the block input
        (carrying this block's OWN cross-segment :class:`DeltaMemoryState`) || the full
        per-position causal reasoner (residual) || the per-position workhorse -> a
        per-position 3-way softmax gate over ``[h_mem ; rea ; h_mlp ; h_in]`` -> the fused
        per-position hidden that feeds the next block.  Unlike the segment/task-structured
        :meth:`forward` (reasoner ONLY at ``pred_pos``, identity elsewhere), EVERY position
        predicts in an LM, so the reasoner runs at every position via the windowed
        :meth:`CausalReasoner.forward`.  Strictly causal at every position (memory recurrence
        + windowed causal reasoner + per-position workhorse/gate), so the LM-leak guarantee
        holds.  DISTINCT from :meth:`forward`/stacked_step, which are left byte-for-byte.

        Args:
            h_in: ``(B, T, d)`` block input stream (block 0 = the bottom embedding; block
                i>0 = block i-1's fused output).
            raw_tokens: ``(B, T)`` raw token ids for the reasoner's own encoder (the RAW
                surface name structure the walk addresses).
            state_in: this block's carried :class:`DeltaMemoryState` (matrix + conv tail)
                from the previous segment, or ``None`` to reset it.
            steps: walk depth K override (defaults to the reasoner's own ``self.steps``).

        Returns:
            ``(fused (B, T, d), gate_probs (B, T, 3), state_out)`` — ``state_out`` is this
            block's final memory state for the next segment; ``gate_probs`` are the soft
            per-expert weights [mem, reason, mlp] (routing-health reporting).
        """
        # MEMORY pathway over the block input, carrying THIS block's cross-segment state
        # (matrix + conv tail) — the delta contract that makes segmented == full.
        h_mem, state_out = self.mem(self.mem_norm(h_in), state_in=state_in, return_state=True)
        # WORKHORSE pathway (per position -> leak-safe).
        h_mlp = self.mlp(h_in)
        # INPUT-ROUTED gate (card 75ada834): route BEFORE the reasoner so the (expensive) walk
        # can be skipped at low-weight positions.  DISTINCT code path (scalar gate, no h_reason
        # in the gate input); the shipped output-mixture path below is left byte-for-byte.
        if self.gate_input_routed:
            return self._lm_forward_input_routed(h_in, raw_tokens, h_mem, h_mlp, state_out, steps)
        # REASONER pathway (per position, full windowed walk), added as a residual to the
        # block input — the LM analogue of ``_pathways``' at-``pred_pos`` residual.
        if self.reasoner is not None:
            rin = h_in if self.reason_reads_hidden else raw_tokens
            h_reason = self.reasoner.forward_composed(
                rin, seed_ctx=h_in, steps=steps, hidden_input=self.reason_reads_hidden,
            )
            rea = h_in + h_reason
        else:
            rea = h_in
        # Per-position 3-way gate + fusion (mirrors ``forward`` WITHOUT the pred_pos gather).
        logit = self.gate(
            torch.cat([h_mem, rea, h_mlp, h_in], dim=-1)
        )  # (B, T, 3) scalar, or (B, T, 3*d_model) per-channel
        if self.gate_per_channel:
            logit = logit.view(*logit.shape[:-1], self.d_model, 3)  # (B, T, d, 3)
        if not self.reason_enabled:
            logit = logit + torch.tensor([0.0, _NEG, 0.0], device=logit.device)
        w = F.softmax(logit, dim=-1)
        if self.gate_per_channel:
            # See ``forward`` — summation order is immaterial for this new mode.
            paths = torch.stack([h_mem, rea, h_mlp], dim=-1)
            fused = (paths * w).sum(dim=-1)
            # Gate-fraction reporting is shape-stable ((B, T, 3)) regardless of the flag: the
            # per-channel gate is reduced by an equal-weight mean over channels.
            w_report = w.mean(dim=-2)
        else:
            # SHIPPED scalar path — kept EXACTLY as committed (identical op order) so the
            # default stays byte-for-byte ``torch.equal`` against HEAD.
            fused = (
                w[..., 0:1] * h_mem
                + w[..., 1:2] * rea
                + w[..., 2:3] * h_mlp
            )
            w_report = w
        return fused, w_report, state_out

    def _lm_forward_input_routed(
        self,
        h_in: Tensor,
        raw_tokens: Tensor,
        h_mem: Tensor,
        h_mlp: Tensor,
        state_out: DeltaMemoryState,
        steps: int | None,
    ) -> tuple[Tensor, Tensor, DeltaMemoryState]:
        """INPUT-ROUTED per-position gate + (optionally sparse) reasoner fusion (card 75ada834).

        The gate reads only ``[h_mem ; h_mlp ; h_in]`` (scalar, 3*d_model -> 3), so the routing
        weights are known BEFORE the reasoner runs.  The reasoner pathway is then DENSE (training,
        or ``reason_threshold == 0``) or SPARSE (eval + ``reason_threshold > 0``: positions whose
        reasoner weight is ``< tau`` skip the walk, ``h_reason == 0`` there).  Fusion + reporting
        match the shipped scalar gate ((B, T, 3) weights), so ``_run_stacked_blocks`` /
        ``stacked_lm_gate_fractions`` need no per-mode branch.
        """
        logit = self.gate(torch.cat([h_mem, h_mlp, h_in], dim=-1))  # (B, T, 3) scalar
        if not self.reason_enabled:
            logit = logit + torch.tensor([0.0, _NEG, 0.0], device=logit.device)
        w = F.softmax(logit, dim=-1)  # (B, T, 3): [mem, reason, mlp]
        if self.reasoner is None:
            rea = h_in
        elif self.reason_threshold > 0.0 and not self.training:
            # EVAL sparsity: run the walk ONLY where the reasoner weight >= tau; identity
            # (h_reason == 0) elsewhere.  The walk cost scales with the invoked count.
            rea = self._sparse_reason(h_in, raw_tokens, w[..., 1], steps)
        else:
            # DENSE (training / tau == 0): full per-position windowed walk (residual).
            rin = h_in if self.reason_reads_hidden else raw_tokens
            h_reason = self.reasoner.forward_composed(
                rin, seed_ctx=h_in, steps=steps, hidden_input=self.reason_reads_hidden,
            )
            rea = h_in + h_reason
        fused = (
            w[..., 0:1] * h_mem
            + w[..., 1:2] * rea
            + w[..., 2:3] * h_mlp
        )
        return fused, w, state_out

    def _sparse_reason(
        self, h_in: Tensor, raw_tokens: Tensor, w_reason: Tensor, steps: int | None
    ) -> Tensor:
        """Reasoner residual ``rea = h_in + h_reason`` with the walk run ONLY at positions whose
        reasoner gate weight ``w_reason >= tau`` (card 75ada834).

        Skipped positions keep ``rea == h_in`` (``h_reason`` treated as 0 -> the fused deviation
        vs the dense forward is ``w_reason * h_reason`` with ``w_reason < tau``, i.e. tau-scaled).
        Invoked positions run the EXACT dense walk: each is a single-query walk over the SAME
        FIXED window the dense windowed path chunks into (``[w*L : (w+1)*L]``) at its own
        within-window position, addressing only positions ``<= t_local`` — so the tail pad (beyond
        T) is masked out and the output matches the dense per-position walk.  Cost scales with the
        invoked-position count (gathered ``Lq = 1`` walks via ``query_forward_composed``), NOT
        with T.  Eval/inference only + ``self.reasoner`` non-None (both guaranteed by the caller).
        """
        assert self.reasoner is not None
        _b, t, _d = h_in.shape
        length = self.reasoner.segment_len
        device = h_in.device
        rea = h_in.clone()  # identity where the walk is skipped
        invoked = w_reason >= self.reason_threshold  # (B, T)
        idx = invoked.nonzero(as_tuple=False)  # (N, 2): [row, pos]
        if idx.numel() == 0:
            return rea
        rows, pos = idx[:, 0], idx[:, 1]
        win_start = (pos // length) * length              # window the dense path chunks into
        t_local = pos - win_start                          # query position within that window
        gather = win_start.unsqueeze(1) + torch.arange(length, device=device)  # (N, L)
        keep = gather < t                                  # positions beyond T are the dense pad
        gather = gather.clamp_max(t - 1)
        if self.reason_reads_hidden:
            xw = h_in[rows.unsqueeze(1), gather] * keep.unsqueeze(-1)  # (N, L, d), zero pad tail
        else:
            xw = raw_tokens[rows.unsqueeze(1), gather] * keep          # (N, L), token-id-0 pad
        seed_ctx = h_in[rows, pos]  # (N, d): the block hidden at the query position
        h_q, _aux = self.reasoner.query_forward_composed(
            xw, query_pos=t_local, seed_ctx=seed_ctx, steps=steps,
            commit_seed=True, return_aux=False, hidden_input=self.reason_reads_hidden,
        )  # (N, d)
        rea[rows, pos] = h_in[rows, pos] + h_q
        return rea


def _rope_cache(
    seq_len: int, head_dim: int, device: torch.device, base: float = 10000.0
) -> tuple[Tensor, Tensor]:
    half = head_dim // 2
    inv = 1.0 / (base ** (torch.arange(0, half, device=device).float() / half))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv)  # (L, half)
    return torch.cos(freqs), torch.sin(freqs)


def _apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """``x: (B, H, L, Dh)``.  Rotary on the last dim using per-position cos/sin (L, Dh/2)."""
    x1, x2 = x[..., 0::2], x[..., 1::2]  # (B, H, L, half)
    cos = cos[None, None]
    sin = sin[None, None]
    o1 = x1 * cos - x2 * sin
    o2 = x1 * sin + x2 * cos
    out = torch.empty_like(x)
    out[..., 0::2] = o1
    out[..., 1::2] = o2
    return out


class ReadbackAttention(nn.Module):
    """Bounded strictly-causal cross-attention readback (card 3ac77deb).

    Q = block-0 fused output, K/V = the bottom (front-end) embedding stream, window-
    relative RoPE, a causal band mask (query ``t`` attends only to ``j`` with
    ``t - window < j <= t``) so cost is O(T*W) and the prediction-position output depends
    only on tokens ``<= t`` (leak-free — the memory keeps the long range).  A learned
    residual scale ``alpha`` starts at 0, so ``alpha * o_proj(o) == 0`` at init (EXACT
    identity: readback-on == readback-off at step 0) — but ``o_proj`` keeps its normal
    random init, so ``alpha``'s gradient ``= o_proj(o)`` is NONZERO and the channel can
    OPEN under training.  (Zero-initialising ``o_proj`` too would make BOTH factors of the
    product start at exactly zero -> every gradient in the module is a fixed point at zero
    forever, a dead-gradient trap; the validated reference scales only ``alpha``.)
    """

    def __init__(self, d_model: int, window: int = 32, n_heads: int = 2) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} not divisible by readback n_heads={n_heads}.")
        if window < 1:
            raise ValueError(f"readback window must be >= 1, got {window}.")
        self.d_model, self.window, self.n_heads = d_model, window, n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)  # NORMAL init -> keeps alpha's gradient live
        self.q_norm = nn.LayerNorm(d_model)
        self.kv_norm = nn.LayerNorm(d_model)
        # IDENTITY-INIT doctrine: scale the WHOLE readback by a learned residual gain init 0,
        # so readback(x) == alpha * o_proj(o) == 0 at step 0 regardless of o_proj — an exact
        # no-op — while alpha (grad = o_proj(o), nonzero) keeps the pathway learnable.
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, query_seq: Tensor, kv_seq: Tensor) -> Tensor:
        """``query_seq, kv_seq: (B, L, D)``.  Returns ``(B, L, D)`` residual delta."""
        b, length, _ = query_seq.shape
        q = self.q_proj(self.q_norm(query_seq))
        k = self.k_proj(self.kv_norm(kv_seq))
        v = self.v_proj(self.kv_norm(kv_seq))
        q, k, v = (
            z.view(b, length, self.n_heads, self.head_dim).transpose(1, 2) for z in (q, k, v)
        )
        cos, sin = _rope_cache(length, self.head_dim, query_seq.device)
        q, k = _apply_rope(q, cos, sin), _apply_rope(k, cos, sin)
        scores = torch.matmul(q, k.transpose(-1, -2)) / (self.head_dim**0.5)  # (B,H,L,L)
        idx = torch.arange(length, device=query_seq.device)
        causal = idx[None, :] <= idx[:, None]  # j <= t
        banded = (idx[:, None] - idx[None, :]) < self.window  # t - j < window
        mask = causal & banded
        scores = scores.masked_fill(~mask[None, None], float("-inf"))
        w = F.softmax(scores, dim=-1)
        o = torch.matmul(w, v)  # (B, H, L, Dh)
        o = o.transpose(1, 2).reshape(b, length, self.d_model)
        return self.alpha * self.o_proj(o)


__all__ = [
    "ComposableCausalReasoner",
    "ReadbackAttention",
    "StackedTandemBlock",
    "StackedWorkhorseMLP",
    "gather_positions",
]
