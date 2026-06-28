"""Gated-DeltaNet delta-rule matrix associative memory (card e2c6ea95).

This is the project's **central thesis component**: a *persistent, bounded,
fixed-size matrix memory* that replaces the context window.  Unlike attention
(which keeps an unbounded KV cache that grows with the sequence) and unlike a
graph (which grows nodes/edges), the *entire* memory is a small matrix ``S`` per
head that is **edited in place** token by token.  Its size is independent of the
sequence length ``T`` — that is the bounded-memory property the thesis rests on.

The math (Gated-DeltaNet; Yang et al., arXiv:2412.06464; DeltaNet, Yang et al.
arXiv:2406.06484; Fast Weight Programmers, Schlag/Irie/Schmidhuber
arXiv:2102.11174)
-------------------------------------------------------------------------------
Per head, ``S`` is a ``(d_k, d_v)`` matrix (a key -> value associative store).
With feature map ``phi`` (here L2-normalisation), forget gate
``alpha_t in (0, 1)`` and write strength ``beta_t in (0, 1)`` (both via bounded
gates; ``beta_t = 1`` is the full-replacement limit used by the math tests):

* **READ** (before the write — this is what makes the layer *strictly causal*)::

      o_t = S_{t-1}^T phi(q_t)            # (d_v,)   matrix-vector, the memory's
                                          #          output for this token

* **WRITE** (the delta rule with a forget gate)::

      S_t = alpha_t * S_{t-1}
            + beta_t * phi(k_t) (v_t - S_{t-1}^T phi(k_t))^T

  The term ``(v_t - S_{t-1}^T phi(k_t))`` is the **delta / prediction error**:
  it is the *correction* between the value already bound to key ``k`` in the
  memory and the new value ``v``.  Writing the correction (not ``v`` itself)
  means the memory **overwrites / edits** the binding for ``k`` in place instead
  of accumulating — this is exactly what keeps it bounded and *self-evicting*.

Why "one gradient-descent step"
-------------------------------
The write is literally one step of online gradient descent on the per-token
associative loss ``L_t(S) = 1/2 || S^T phi(k_t) - v_t ||^2`` with step size
``beta_t``::

      grad_S L_t = phi(k_t) (S^T phi(k_t) - v_t)^T
      S_t = alpha_t S_{t-1} - beta_t grad_S L_t |_{S = S_{t-1}}
          = alpha_t S_{t-1} + beta_t phi(k_t) (v_t - S_{t-1}^T phi(k_t))^T

So the memory performs in-context *test-time learning*: each token takes one SGD
step to better recall its own value, and ``alpha_t`` slowly forgets stale keys.

Capacity bound
--------------
A ``(d_k, d_v)`` matrix can store at most ``d_k`` mutually-orthogonal key->value
bindings without interference (rank argument).  Hence per-head capacity is
``<= d_k``; multi-head with ``H`` heads gives ``H`` independent stores.  Budget
``d_k`` conservatively; scale the model via ``heads x layers x width`` instead of
inflating a single head's ``d_k``.

Implementation note (sequential reference + chunkwise-parallel fast path)
------------------------------------------------------------------------
The recurrence is implemented two ways, selected by ``cfg.delta_scan``:

* ``"sequential"`` — a **clean sequential scan in plain PyTorch** (``O(T)``
  Python iterations over the time axis, exactly like the Mamba baseline's
  ``_selective_scan``).  This is the **trusted oracle**: the delta-rule math is
  proven exact against a by-hand reference, so it is the ground truth the fast
  path is validated against.  Kept as a config-selectable fallback.

* ``"chunkwise"`` (default) — the **chunkwise-parallel DeltaNet scan** (Yang et
  al., arXiv:2406.06484; gated variant arXiv:2412.06464): split the sequence into
  chunks of size ``C``; do the intra-chunk delta corrections in parallel via the
  WY / UT-transform (batched triangular solve + matmuls) and carry the state
  matrix ``S`` between chunks recurrently — ``T/C`` sequential steps instead of
  ``T``.  The per-step forget gate ``alpha`` is folded in via its cumulative
  product within each chunk, kept entirely as **bounded ``(0, 1]`` decay ratios**
  so no ``1/cumprod`` term ever overflows fp32.  This path is proven equivalent
  to the sequential oracle within tolerance (see
  ``tests/test_delta_memory.py::test_chunkwise_matches_sequential*``), so the two
  produce the **same ``(loss, logits)``** — ``delta_scan`` only selects the
  internal scan.

We deliberately do **not** depend on the ``fla-hub`` /
``flash-linear-attention`` Triton/CUDA kernels: those carry the same
build/toolchain risk on the 3060 that made us hand-roll Mamba.  The chunkwise
path here is a **self-contained pure-PyTorch** implementation kept in fp32.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast, overload

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

if TYPE_CHECKING:
    from graph_llm.config import ModelConfig

VALID_FEATURE_MAPS = ("l2", "silu_l2", "identity")
VALID_DELTA_SCANS = ("sequential", "chunkwise", "auto")


@dataclass
class DeltaMemoryState:
    """The full cross-segment carried state of one :class:`GatedDeltaMemory` layer.

    The bare delta-memory matrix ``S`` is no longer the *whole* carried state once
    a short causal conv is added (card 571d50ec): a width-``W`` causal conv at a
    segment boundary needs the previous segment's last ``W-1`` input rows to
    reproduce a single full-sequence forward, so that conv tail is carried
    alongside ``S``.

    Attributes:
        memory: The delta-memory state matrix ``S`` of shape ``(B, H, d_k, d_v)``
            (fp32), exactly the per-head associative store the scan threads.
        conv_tail: The last ``W-1`` rows of this layer's conv input, shape
            ``(B, W-1, d_model)`` — the strictly-past context the next segment's
            causal conv must see at its left edge.  ``None`` when the conv is
            disabled (``delta_conv_width == 1``), in which case the state degrades
            to just ``memory`` and the carry matches the pre-conv behaviour.
    """

    memory: Tensor
    conv_tail: Tensor | None = None


def _feature_map(x: Tensor, kind: str) -> Tensor:
    """Apply the key/query feature map ``phi``.

    Args:
        x: ``(..., d_k)`` keys or queries.
        kind: ``"l2"`` (L2-normalise — the Gated-DeltaNet choice; bounds
            ``||phi(x)|| = 1`` so the delta step is a well-scaled GD step),
            ``"silu_l2"`` (SiLU then L2-normalise), or ``"identity"`` (no map;
            used by the delta-rule math tests where an exact hand reference is
            checked).

    Returns:
        ``(..., d_k)`` mapped tensor.
    """
    if kind == "identity":
        return x
    if kind == "silu_l2":
        x = F.silu(x)
    elif kind != "l2":
        raise ValueError(
            f"delta_feature_map={kind!r} not in {VALID_FEATURE_MAPS}."
        )
    return F.normalize(x, p=2.0, dim=-1)


class GatedDeltaMemory(nn.Module):
    """Multi-head delta-rule matrix associative memory with a forget gate.

    A token-mixing layer (drop-in alongside attention / the Mamba block): it
    consumes ``(B, T, d_model)`` and returns ``(B, T, d_model)``.  Internally,
    per head, it maintains a fixed-size ``(d_k, d_v)`` state matrix ``S`` updated
    by the Gated-DeltaNet recurrence above.  **Causal by construction**: the read
    at step ``t`` uses ``S_{t-1}`` (the state *before* token ``t``'s write), so
    output ``t`` depends only on tokens ``<= t``.

    The state is allocated fresh per ``forward`` (size independent of ``T``), so
    the layer is stateless across calls — the same train/eval contract as the
    Transformer and Mamba baselines.

    Args:
        cfg: A :class:`~graph_llm.config.ModelConfig`.  Reads ``d_model``,
            ``delta_n_heads`` (H), ``delta_head_k_dim`` (d_k),
            ``delta_head_v_dim`` (d_v), ``delta_feature_map``,
            ``delta_use_forget_gate``, and ``delta_dropout``.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.d_model = cfg.d_model
        self.n_heads = cfg.delta_n_heads
        self.d_k = cfg.delta_head_k_dim
        self.d_v = cfg.delta_head_v_dim
        self.feature_map = cfg.delta_feature_map
        self.use_forget_gate = cfg.delta_use_forget_gate
        # Scan implementation selector (card 18b14615).  "auto" resolves to the
        # fast chunkwise path (it is bit-equivalent to the sequential oracle).
        scan = cfg.delta_scan
        if scan not in VALID_DELTA_SCANS:
            raise ValueError(
                f"delta_scan={scan!r} not in {VALID_DELTA_SCANS}."
            )
        self.delta_scan = "chunkwise" if scan == "auto" else scan
        if cfg.delta_chunk_size < 1:
            raise ValueError(
                f"delta_chunk_size must be >= 1, got {cfg.delta_chunk_size}"
            )
        self.chunk_size = cfg.delta_chunk_size

        if self.n_heads < 1:
            raise ValueError(f"delta_n_heads must be >= 1, got {self.n_heads}")
        if self.d_k < 1 or self.d_v < 1:
            raise ValueError(
                f"delta_head_k_dim/delta_head_v_dim must be >= 1, "
                f"got d_k={self.d_k}, d_v={self.d_v}"
            )
        if self.feature_map not in VALID_FEATURE_MAPS:
            raise ValueError(
                f"delta_feature_map={self.feature_map!r} not in {VALID_FEATURE_MAPS}."
            )

        qk_dim = self.n_heads * self.d_k
        v_dim = self.n_heads * self.d_v

        # q, k -> (H * d_k); v -> (H * d_v).  No bias (keys/queries are L2-mapped;
        # values feed a linear readout).
        self.q_proj = nn.Linear(self.d_model, qk_dim, bias=False)
        self.k_proj = nn.Linear(self.d_model, qk_dim, bias=False)
        self.v_proj = nn.Linear(self.d_model, v_dim, bias=False)

        # Write strength beta_t = sigmoid(.) in (0, 1), one scalar per (token, head).
        self.beta_proj = nn.Linear(self.d_model, self.n_heads, bias=True)
        # Forget gate alpha_t in (0, 1): scalar per (token, head).  Parameterised
        # in log space, alpha = exp(-softplus(.)) in (0, 1), like the SSM decay,
        # so it is numerically stable and strictly < 1 (a true contraction).  At
        # zero input alpha = exp(-softplus(0)) = exp(-log 2) ~= 0.5 (moderate
        # forgetting at init); the learned bias moves the per-head decay from there.
        if self.use_forget_gate:
            self.alpha_proj: nn.Linear | None = nn.Linear(self.d_model, self.n_heads, bias=True)
        else:
            self.alpha_proj = None

        # Output: per-head readout (d_v) -> back to d_model.
        self.out_proj = nn.Linear(v_dim, self.d_model, bias=False)
        self.dropout = nn.Dropout(cfg.delta_dropout)

        # Short causal depthwise conv on the memory input (card 571d50ec).  The
        # delta write is token-LOCAL (k_t, v_t both from x_t); without local mixing
        # a value position never sees its key, so the memory cannot BIND k_i->v_i
        # (MQAR: ~0.23 recall capped at width 1 vs ~1.0 at width >= 2).  A RESIDUAL
        # causal depthwise(+pointwise) conv supplies that mixing before the q/k/v
        # projections.  width == 1 builds NOTHING -> no module, no params, no RNG
        # draws -> byte-for-byte the committed backbone (back-compat / ablation).
        conv_width = cfg.delta_conv_width
        if conv_width < 1:
            raise ValueError(
                f"delta_conv_width must be >= 1, got {conv_width}"
            )
        self.conv_width = conv_width
        if conv_width > 1:
            # Depthwise over the trailing window (per-channel, mixes across time)
            # then a pointwise 1x1 channel mix — the same cheap separable form as
            # the multi-scale conv front-end's scales.  Left-pad only (no future
            # leakage); the residual keeps width=1 behaviour reachable at init.
            self.conv_dw: nn.Conv1d | None = nn.Conv1d(
                self.d_model, self.d_model, kernel_size=conv_width,
                groups=self.d_model, bias=False,
            )
            self.conv_pw: nn.Conv1d | None = nn.Conv1d(
                self.d_model, self.d_model, kernel_size=1, bias=True
            )
        else:
            self.conv_dw = None
            self.conv_pw = None

    # ------------------------------------------------------------------
    # The sequential delta-rule scan (the pure-PyTorch core)
    # ------------------------------------------------------------------

    @overload
    def _delta_scan(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        beta: Tensor,
        alpha: Tensor,
        state_in: Tensor | None = ...,
        return_state: Literal[False] = ...,
    ) -> Tensor: ...

    @overload
    def _delta_scan(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        beta: Tensor,
        alpha: Tensor,
        state_in: Tensor | None,
        return_state: Literal[True],
    ) -> tuple[Tensor, Tensor]: ...

    def _delta_scan(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        beta: Tensor,
        alpha: Tensor,
        state_in: Tensor | None = None,
        return_state: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Sequential Gated-DeltaNet recurrence (the load-bearing math).

        Per head, maintaining ``S`` of shape ``(d_k, d_v)``::

            o_t = S_{t-1}^T phi(q_t)                              # READ (causal)
            u_t = v_t - S_{t-1}^T phi(k_t)                        # delta / error
            S_t = alpha_t * S_{t-1} + beta_t * phi(k_t) u_t^T     # WRITE

        Implemented batched over ``(B, H)`` with ``einsum``; the only Python loop
        is over the time axis ``T`` (like Mamba's selective scan).

        **Cross-segment state carry (card 61f900ca).**  By default the state is
        seeded at zero — the standard within-sequence DeltaNet, identical to the
        original behaviour.  Passing ``state_in`` seeds the recurrence from a
        *carried* state representing strictly-past tokens (e.g. the final state of
        the previous segment), so the read at ``t = 0`` already sees that history.
        Because the carried state is purely past, causality is preserved.  With
        ``return_state=True`` the final ``S`` (after token ``T-1``'s write) is
        returned alongside the outputs so it can be threaded into the next
        segment.

        Args:
            q: ``(B, H, T, d_k)``  feature-mapped queries phi(q).
            k: ``(B, H, T, d_k)``  feature-mapped keys phi(k).
            v: ``(B, H, T, d_v)``  values.
            beta: ``(B, H, T)``    write strengths in (0, 1].
            alpha: ``(B, H, T)``   forget gates in (0, 1] (all-ones when ungated).
            state_in: Optional initial state ``(B, H, d_k, d_v)`` to seed the
                recurrence (default: a fresh zero state — reset-per-sequence).
            return_state: When ``True`` also return the final state ``S``.

        Returns:
            ``(B, H, T, d_v)`` per-head memory outputs, or — when
            ``return_state`` — a ``(outputs, final_state)`` tuple whose
            ``final_state`` has shape ``(B, H, d_k, d_v)``.
        """
        B, H, T, _ = q.shape
        # Fixed-size state: (B, H, d_k, d_v) — INDEPENDENT of T (bounded memory).
        # Seed from the carried state when given (cross-segment persistence);
        # otherwise start at zero (the reset-per-sequence default).
        if state_in is None:
            S = torch.zeros(B, H, self.d_k, self.d_v, device=q.device, dtype=q.dtype)
        else:
            S = state_in

        outputs = []
        for t in range(T):
            q_t = q[:, :, t]          # (B, H, d_k)
            k_t = k[:, :, t]          # (B, H, d_k)
            v_t = v[:, :, t]          # (B, H, d_v)
            beta_t = beta[:, :, t]    # (B, H)
            alpha_t = alpha[:, :, t]  # (B, H)

            # READ with the state BEFORE this token's write => strictly causal.
            o_t = torch.einsum("bhkv,bhk->bhv", S, q_t)            # (B, H, d_v)
            outputs.append(o_t)

            # Delta rule: error between stored value for k and the new value.
            pred_t = torch.einsum("bhkv,bhk->bhv", S, k_t)         # (B, H, d_v)
            delta_t = v_t - pred_t                                 # (B, H, d_v)
            # Outer-product write, scaled by beta_t, after forgetting by alpha_t.
            write = beta_t[..., None, None] * torch.einsum(
                "bhk,bhv->bhkv", k_t, delta_t
            )
            S = alpha_t[..., None, None] * S + write

        out = torch.stack(outputs, dim=2)  # (B, H, T, d_v)
        if return_state:
            return out, S
        return out

    # ------------------------------------------------------------------
    # The chunkwise-parallel delta-rule scan (the fast path)
    # ------------------------------------------------------------------

    @overload
    def _delta_scan_chunkwise(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        beta: Tensor,
        alpha: Tensor,
        state_in: Tensor | None = ...,
        return_state: Literal[False] = ...,
    ) -> Tensor: ...

    @overload
    def _delta_scan_chunkwise(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        beta: Tensor,
        alpha: Tensor,
        state_in: Tensor | None,
        return_state: Literal[True],
    ) -> tuple[Tensor, Tensor]: ...

    def _delta_scan_chunkwise(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        beta: Tensor,
        alpha: Tensor,
        state_in: Tensor | None = None,
        return_state: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Chunkwise-parallel Gated-DeltaNet recurrence (the fast path).

        Mathematically identical to :meth:`_delta_scan` (validated bit-for-bit
        within tolerance by the equivalence tests), but instead of ``T`` Python
        iterations it runs ``T/C`` chunk steps with the intra-chunk delta
        corrections done in parallel as batched matmuls (DeltaNet WY / UT
        transform, Yang et al. arXiv:2406.06484).

        Per chunk of length ``C`` (local 1-based index ``t``), with incoming
        state ``S0`` (the state before the chunk) and the same recurrence
        ``S_t = alpha_t S_{t-1} + beta_t k_t (v_t - S_{t-1}^T k_t)^T``,
        ``o_t = S_{t-1}^T q_t``:

        Let the within-chunk cumulative forget-gate decay be
        ``D_t = prod_{j<=t} alpha_j`` (inclusive) and ``d_t = D_{t-1}``
        (exclusive, ``d_1 = 1``).  Substituting the decay-normalised state
        ``Stil_t = S_t / D_t`` turns the gated recurrence into a plain (ungated)
        delta rule whose every decay coefficient is a **bounded ratio in
        ``(0, 1]``** — ``d_t/D_s`` for ``s < t`` is ``prod_{s<j<t} alpha`` and the
        state carry ``D_C/D_t`` is ``prod_{j>t} alpha`` — so no ``1/cumprod``
        term is ever materialised and the scan is fp32-stable:

            u_t   = v_t - Stil_{t-1}^T (d_t k_t)
            (I + tril(M, -1)) U = V - (d k) S0 ,
                  M[t, s] = beta_s (d_t / D_s) <k_t, k_s>          # bounded
            o_t   = (d_t q_t)^T S0 + sum_{s<t} beta_s (d_t/D_s) <q_t,k_s> u_s
            S_end = D_C S0 + sum_t (D_C / D_t) beta_t k_t u_t^T     # bounded

        ``U`` (the per-token deltas ``u_t``) is recovered by a single batched
        unit-lower-triangular solve per chunk.  Sequences whose length is not a
        multiple of ``C`` are right-padded (keys/values/beta with zeros, alpha
        with ones so padded steps neither read, write, nor decay); the padded
        tail is sliced off the output.

        **Cross-segment state carry (card 61f900ca).**  The scan already threads
        the carried state ``S0`` between chunks recurrently (``rhs`` subtracts the
        decayed-state read, ``o`` adds it back, and ``S`` is carried out at the
        end of each chunk).  Seeding ``S0`` from a non-zero ``state_in`` is the
        *same* operation the inter-chunk carry performs, so it is correct by
        construction.  Because the right-padded tail neither writes (``beta = 0``)
        nor decays (``alpha = 1``), the post-loop state equals the true final
        state after the real tokens — so it is safe to return even when
        ``T % C != 0``.

        Args:
            q: ``(B, H, T, d_k)``  feature-mapped queries phi(q).
            k: ``(B, H, T, d_k)``  feature-mapped keys phi(k).
            v: ``(B, H, T, d_v)``  values.
            beta: ``(B, H, T)``    write strengths in (0, 1].
            alpha: ``(B, H, T)``   forget gates in (0, 1] (all-ones when ungated).
            state_in: Optional initial state ``(B, H, d_k, d_v)`` to seed the
                recurrence (default: a fresh zero state — reset-per-sequence).
            return_state: When ``True`` also return the final state ``S``.

        Returns:
            ``(B, H, T, d_v)`` per-head memory outputs (same as ``_delta_scan``),
            or — when ``return_state`` — a ``(outputs, final_state)`` tuple whose
            ``final_state`` has shape ``(B, H, d_k, d_v)``.
        """
        B, H, T, d_k = q.shape
        d_v = v.shape[-1]
        dtype, device = q.dtype, q.device
        C = self.chunk_size

        # Right-pad to a whole number of chunks.  Padded keys/values/beta are zero
        # (no read / no write) and padded alpha is 1 (no extra decay); the padded
        # output tail is discarded below.
        pad = (C - T % C) % C
        if pad:
            q = F.pad(q, (0, 0, 0, pad))
            k = F.pad(k, (0, 0, 0, pad))
            v = F.pad(v, (0, 0, 0, pad))
            beta = F.pad(beta, (0, pad))
            alpha = F.pad(alpha, (0, pad), value=1.0)
        t_pad = T + pad
        n_chunks = t_pad // C

        def _chunk(x: Tensor) -> Tensor:
            return x.reshape(B, H, n_chunks, C, *x.shape[3:])

        q, k, v, beta, alpha = (_chunk(t) for t in (q, k, v, beta, alpha))

        # Cumulative log-decay within each chunk.  log_d_incl = log D_t,
        # log_d_excl = log d_t = log D_{t-1} (d_1 = 1 -> log_d_excl[..., 0] = 0).
        log_alpha = torch.log(alpha)
        log_d_incl = torch.cumsum(log_alpha, dim=-1)          # (B,H,nc,C)
        log_d_excl = log_d_incl - log_alpha                   # (B,H,nc,C)
        chunk_total = torch.exp(log_d_incl[..., -1])          # (B,H,nc) == D_C

        eye = torch.eye(C, dtype=dtype, device=device)
        # Strict lower-triangular causal mask (s < t).
        strict_lower = torch.tril(
            torch.ones(C, C, dtype=dtype, device=device), diagonal=-1
        )
        neg_inf = torch.finfo(dtype).min

        # Seed from the carried state when given (cross-segment persistence);
        # otherwise start at zero (the reset-per-sequence default).  Seeding a
        # non-zero S0 is exactly the inter-chunk carry the loop already performs.
        if state_in is None:
            S = torch.zeros(B, H, d_k, d_v, dtype=dtype, device=device)
        else:
            S = state_in
        outputs = []
        for c in range(n_chunks):
            k_c = k[:, :, c]                                  # (B,H,C,d_k)
            q_c = q[:, :, c]
            v_c = v[:, :, c]                                  # (B,H,C,d_v)
            beta_c = beta[:, :, c]                            # (B,H,C)
            log_incl = log_d_incl[:, :, c]                    # (B,H,C)
            log_excl = log_d_excl[:, :, c]                    # (B,H,C)
            d_total = chunk_total[:, :, c]                    # (B,H)

            d_excl = torch.exp(log_excl)                      # (B,H,C) in (0,1]
            k_hat = k_c * d_excl[..., None]                   # decayed read key
            q_hat = q_c * d_excl[..., None]                   # decayed read query

            # Bounded decay ratio[t, s] = d_t / D_s for s < t (== prod_{s<j<t} a),
            # else 0.  Mask in LOG-space (set to -inf) BEFORE exp so the upper
            # triangle is exactly 0 and never overflows to inf.
            ratio_log = log_excl[..., :, None] - log_incl[..., None, :]
            ratio = torch.exp(
                torch.where(strict_lower.bool(), ratio_log, neg_inf)
            )                                                 # (B,H,C,C)

            kk = torch.einsum("bhtk,bhsk->bhts", k_c, k_c)    # <k_t, k_s>
            mat = beta_c[..., None, :] * ratio * kk           # M[t,s], strict lower
            rhs = v_c - torch.einsum("bhtk,bhkv->bhtv", k_hat, S)
            # u_t solve: (I + tril(M,-1)) U = V - (d k) S0 .  Unit lower triangular.
            u = torch.linalg.solve_triangular(
                eye + mat, rhs, upper=False, unitriangular=True
            )                                                 # (B,H,C,d_v)

            # READ: inter-chunk (decayed S0 via q_hat) + intra-chunk (strict lower).
            qk = torch.einsum("bhtk,bhsk->bhts", q_c, k_c)    # <q_t, k_s>
            mat_o = beta_c[..., None, :] * ratio * qk
            o = torch.einsum("bhtk,bhkv->bhtv", q_hat, S) + torch.einsum(
                "bhts,bhsv->bhtv", mat_o, u
            )
            outputs.append(o)

            # STATE carry: S_end = D_C S0 + sum_t (D_C/D_t) beta_t k_t u_t^T.
            # carry[t] = D_C / D_t = prod_{j>t} alpha in (0,1] (bounded).
            carry = torch.exp(
                torch.log(d_total[..., None].clamp_min(1e-30)) - log_incl
            )                                                 # (B,H,C)
            k_write = k_c * (beta_c * carry)[..., None]       # (B,H,C,d_k)
            S = d_total[..., None, None] * S + torch.einsum(
                "bhtk,bhtv->bhkv", k_write, u
            )

        out = torch.stack(outputs, dim=2)                    # (B,H,nc,C,d_v)
        out = out.reshape(B, H, t_pad, d_v)[:, :, :T]        # drop padded tail
        if return_state:
            return out, S
        return out

    # ------------------------------------------------------------------
    # Short causal conv on the memory input (card 571d50ec)
    # ------------------------------------------------------------------

    def _causal_conv(
        self, x: Tensor, conv_tail_in: Tensor | None
    ) -> tuple[Tensor, Tensor | None]:
        """Residual causal depthwise(+pointwise) conv over the memory input.

        Computes ``xc = x + pointwise(depthwise_causal(x))`` so each position is
        locally mixed with its trailing ``W-1`` tokens — the binding mechanism the
        token-local delta write lacks.  **Causal**: the conv is left-padded by
        ``W-1`` and never right-padded, so position ``t`` sees only inputs ``<= t``.

        Cross-segment / cross-chunk carry: at a boundary the left edge of ``x``
        must see the previous segment's last ``W-1`` rows, or a segmented forward
        would not match a single full-sequence forward.  Those rows arrive as
        ``conv_tail_in``; they are prepended before the conv and the leading
        ``W-1`` outputs they produce are sliced off so the output length stays
        ``T``.  The returned tail is the last ``W-1`` rows of ``cat([tail, x])`` —
        the strictly-past context for the *next* segment.

        Args:
            x: ``(B, T, d_model)`` memory input (the layer's pre-projection input).
            conv_tail_in: Optional carried ``(B, W-1, d_model)`` tail from the
                previous segment (``None`` resets to a zero left-pad, the
                within-sequence default).

        Returns:
            ``(xc, conv_tail_out)`` — the locally-mixed input (same shape as ``x``)
            and the ``(B, W-1, d_model)`` tail to carry forward (``None`` if the
            conv is disabled).
        """
        if self.conv_dw is None:  # width == 1: disabled — pure pass-through.
            return x, None
        assert self.conv_pw is not None
        pad = self.conv_width - 1  # conv_dw exists only when width > 1, so pad >= 1.
        B, _, d = x.shape

        # Normalise the carried left context to EXACTLY ``pad`` rows.  A reset
        # (``None``) or a short prior history (fewer than ``pad`` accumulated rows —
        # e.g. a length-1 first segment) is left-zero-padded, which is precisely the
        # "no past tokens before the window" assumption: zeros are the absent past.
        # This guarantees ``primed`` is always length ``pad + T``, so the conv emits
        # exactly ``T`` rows and the residual add never broadcasts.
        if conv_tail_in is None:
            left_ctx = x.new_zeros(B, pad, d)
        elif conv_tail_in.shape[1] < pad:
            left_ctx = F.pad(conv_tail_in, (0, 0, pad - conv_tail_in.shape[1], 0))
        else:
            # Carry only the most recent ``pad`` rows (older context cannot reach a
            # width-``W`` conv); keeps the carried tail fixed-size and bounded.
            left_ctx = conv_tail_in[:, -pad:]

        primed_seq = torch.cat([left_ctx, x], dim=1)          # (B, pad + T, d)
        h = self.conv_dw(primed_seq.transpose(1, 2))          # (B, d, T)
        h = self.conv_pw(h)
        xc = x + h.transpose(1, 2)                            # residual local mix

        # The next segment's left context = the last ``pad`` rows of THIS conv input
        # (always exactly ``pad`` rows, since ``primed_seq`` has length ``pad + T``).
        # NOT detached: within a truncated-BPTT window the trainer threads this state
        # graph-connected and severs it via ``detach_states`` at the window boundary
        # — the same treatment the delta-memory matrix ``S`` gets, so the conv shares
        # the cross-segment gradient.
        conv_tail_out = primed_seq[:, -pad:]
        return xc, conv_tail_out

    def forward(
        self,
        x: Tensor,
        state_in: DeltaMemoryState | None = None,
        return_state: bool = False,
    ) -> Tensor | tuple[Tensor, DeltaMemoryState]:
        """Run the memory over a sequence.

        Args:
            x: ``(B, T, d_model)``.
            state_in: Optional carried :class:`DeltaMemoryState` from a previous
                segment (cross-segment persistence, cards 61f900ca + 571d50ec):
                its ``memory`` seeds the delta-rule recurrence and its ``conv_tail``
                primes the causal conv's left edge.  ``None`` (the default) resets
                both — the standard within-sequence behaviour, byte-for-byte the
                original when the conv is disabled.  The memory matrix is held in
                fp32 (the scan's native dtype); a ``memory`` of another dtype is
                cast to fp32 on entry.
            return_state: When ``True`` also return the final
                :class:`DeltaMemoryState` so it can be threaded into the next
                segment.

        Returns:
            ``(B, T, d_model)``, or — when ``return_state`` — a
            ``(out, state_out)`` tuple whose ``state_out`` is a
            :class:`DeltaMemoryState` (fp32 ``memory`` of shape
            ``(B, H, d_k, d_v)`` plus the ``(B, W-1, d_model)`` conv tail).
        """
        B, T, _ = x.shape
        H, d_k, d_v = self.n_heads, self.d_k, self.d_v

        # Local mixing FIRST (card 571d50ec): bind each value position to its
        # trailing keys before the token-local q/k/v projections.  Carries/returns
        # the conv tail so the binding is exact across segment boundaries.
        mem_in = None if state_in is None else state_in.memory
        conv_tail_in = None if state_in is None else state_in.conv_tail
        x, conv_tail_out = self._causal_conv(x, conv_tail_in)

        q = self.q_proj(x).view(B, T, H, d_k).transpose(1, 2)  # (B, H, T, d_k)
        k = self.k_proj(x).view(B, T, H, d_k).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, d_v).transpose(1, 2)  # (B, H, T, d_v)

        q = _feature_map(q, self.feature_map)
        k = _feature_map(k, self.feature_map)

        # Write strength beta_t in (0, 1) (sigmoid is strictly (0, 1)).
        beta = torch.sigmoid(self.beta_proj(x)).transpose(1, 2)  # (B, H, T)
        # Forget gate alpha_t in (0, 1): alpha = exp(-softplus(.)) (stable, < 1).
        if self.alpha_proj is not None:
            alpha = torch.exp(-F.softplus(self.alpha_proj(x))).transpose(1, 2)
        else:
            alpha = torch.ones_like(beta)

        # The recurrence is run in fp32 for numerical stability (the state can
        # accumulate over long sequences), mirroring the Mamba scan; cast back
        # to the input dtype afterwards so bf16/amp paths see the right dtype.
        # The chunkwise fast path is the default; the sequential scan is the
        # validated reference/fallback.  Both return the SAME (B, H, T, d_v).
        # The carried state is kept in fp32 throughout for the same reason.
        scan = (
            self._delta_scan_chunkwise
            if self.delta_scan == "chunkwise"
            else self._delta_scan
        )
        # The carried memory matrix enters the scan in fp32 (its native dtype).
        # Note: this casts dtype only, not device — a carried state is assumed to
        # be on the same device as ``x`` (true for all current call sites, which
        # keep the state alongside the running segment).
        mem_in_f = None if mem_in is None else mem_in.float()

        def _readout(out_raw: Tensor) -> Tensor:
            """Cast back to the input dtype + per-head readout projection."""
            out_raw = out_raw.to(x.dtype)
            out_raw = out_raw.transpose(1, 2).reshape(B, T, H * d_v)  # (B,T,H*d_v)
            return self.dropout(self.out_proj(out_raw))

        if return_state:
            raw, mem_out = cast(
                "tuple[Tensor, Tensor]",
                scan(
                    q.float(), k.float(), v.float(), beta.float(), alpha.float(),
                    state_in=mem_in_f, return_state=True,
                ),
            )
            # Bundle the delta-memory matrix with the conv tail so both pieces of
            # the layer's strictly-past context cross the segment boundary together.
            return _readout(raw), DeltaMemoryState(memory=mem_out, conv_tail=conv_tail_out)
        raw = cast(
            Tensor,
            scan(
                q.float(), k.float(), v.float(), beta.float(), alpha.float(),
                state_in=mem_in_f, return_state=False,
            ),
        )
        return _readout(raw)
