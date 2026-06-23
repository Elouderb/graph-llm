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

Implementation note (v1: sequential pure-PyTorch, validate-first)
-----------------------------------------------------------------
v1 implements the recurrence as a **clean sequential scan in plain PyTorch** —
correctness over speed, exactly like the Mamba baseline's ``_selective_scan``.
We deliberately do **not** depend on the ``fla-hub`` /
``flash-linear-attention`` Triton/CUDA kernels for v1: those carry the same
build/toolchain risk on the 3060 that made us hand-roll Mamba.  The scan is
``O(T)`` Python iterations and materialises the per-token state, so it is slower
than the chunkwise-parallel DeltaNet kernel — but it is mathematically the same
recurrence.  The chunkwise-parallel kernel is a documented follow-up
optimisation (see ``docs/memory_stage.md``); a vectorised path, if added, must
match this sequential reference within tolerance.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

if TYPE_CHECKING:
    from graph_llm.config import ModelConfig

VALID_FEATURE_MAPS = ("l2", "silu_l2", "identity")


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

    # ------------------------------------------------------------------
    # The sequential delta-rule scan (the pure-PyTorch core)
    # ------------------------------------------------------------------

    def _delta_scan(
        self, q: Tensor, k: Tensor, v: Tensor, beta: Tensor, alpha: Tensor
    ) -> Tensor:
        """Sequential Gated-DeltaNet recurrence (the load-bearing math).

        Per head, maintaining ``S`` of shape ``(d_k, d_v)``::

            o_t = S_{t-1}^T phi(q_t)                              # READ (causal)
            u_t = v_t - S_{t-1}^T phi(k_t)                        # delta / error
            S_t = alpha_t * S_{t-1} + beta_t * phi(k_t) u_t^T     # WRITE

        Implemented batched over ``(B, H)`` with ``einsum``; the only Python loop
        is over the time axis ``T`` (like Mamba's selective scan).

        Args:
            q: ``(B, H, T, d_k)``  feature-mapped queries phi(q).
            k: ``(B, H, T, d_k)``  feature-mapped keys phi(k).
            v: ``(B, H, T, d_v)``  values.
            beta: ``(B, H, T)``    write strengths in (0, 1].
            alpha: ``(B, H, T)``   forget gates in (0, 1] (all-ones when ungated).

        Returns:
            ``(B, H, T, d_v)`` per-head memory outputs.
        """
        B, H, T, _ = q.shape
        # Fixed-size state: (B, H, d_k, d_v) — INDEPENDENT of T (bounded memory).
        S = torch.zeros(B, H, self.d_k, self.d_v, device=q.device, dtype=q.dtype)

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

        return torch.stack(outputs, dim=2)  # (B, H, T, d_v)

    def forward(self, x: Tensor) -> Tensor:
        """Run the memory over a sequence.

        Args:
            x: ``(B, T, d_model)``.

        Returns:
            ``(B, T, d_model)``.
        """
        B, T, _ = x.shape
        H, d_k, d_v = self.n_heads, self.d_k, self.d_v

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
        out = self._delta_scan(q.float(), k.float(), v.float(), beta.float(), alpha.float())
        out = out.to(x.dtype)

        out = out.transpose(1, 2).reshape(B, T, H * d_v)  # (B, T, H*d_v)
        return self.dropout(self.out_proj(out))
