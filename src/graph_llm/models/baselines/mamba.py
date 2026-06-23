"""Pure-PyTorch Mamba / selective-SSM baseline (card 424e3a8e).

This is the *recurrent-state* arm of the Phase 0 matched-parameter baseline
suite (the attention arm is ``transformer.py``).  It is the load-bearing
comparison for the project's headline claim — "a memory GNN can replace the
context window" is only meaningful measured against a recurrent-state model
that already streams over arbitrary length.  Mamba (arXiv:2312.00752) is the
canonical such baseline.

Dependency choice
-----------------
We deliberately do **not** depend on the ``mamba-ssm`` package.  Its
``selective_scan`` is a hand-written fused CUDA kernel that must be compiled
against a matching CUDA toolchain; this build environment is CPU-only, so that
kernel is impractical and would make the recurrent baseline silently
unavailable.  Instead we implement the selective scan in plain PyTorch, in two
forms selected by ``cfg.mamba_scan`` (both mathematically equivalent — same
discretisation and gating — and validated bit-for-bit against each other):

* ``"sequential"`` — the readable ``O(T)`` Python-loop recurrence; the
  **trusted reference / oracle**.
* ``"chunkwise"`` (default) — a **chunked parallel scan** (card 18b14615): the
  selective SSM is a diagonal linear recurrence, so within each chunk the
  contributions combine via cumulative-decay matmuls and only the chunk *state*
  is carried recurrently — ``T/C`` sequential steps instead of ``T``.  This is
  the effective throughput fix: ``torch.compile`` on the raw ``T``-loop is
  impractical (it unrolls the data-dependent loop, giving pathological compile
  time), whereas the chunked form cuts the Python loop length directly.

The throughput caveat / fix is documented in ``docs/baselines.md``.  Correctness
and architecture parity over speed — that is the right trade for a research
baseline.

Block structure (matches ``mamba_ssm.Mamba``, confirmed against the reference):

    in_proj:  d_model            -> 2 * d_inner          (split into x, z)
    conv1d:   depthwise causal conv over x               (width d_conv)
    x_proj:   d_inner            -> dt_rank + 2 * d_state (split into dt, B, C)
    dt_proj:  dt_rank            -> d_inner               (then softplus)
    selective scan (A, B, C, D, dt) -> y
    gate:     y = y * silu(z)
    out_proj: d_inner            -> d_model

The model exposes the same ``forward(x, targets) -> (loss, logits)`` contract
as the Transformer baseline, so the model-agnostic Trainer needs zero changes.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import Tensor

from graph_llm.models.baselines.transformer import RMSNorm
from graph_llm.models.registry import get_embedding_init, register_model

if TYPE_CHECKING:
    from graph_llm.config import Config, ModelConfig


def _resolve_dt_rank(dt_rank: int | str, d_model: int) -> int:
    """Resolve the ``"auto"`` dt-rank sentinel to ``ceil(d_model / 16)``."""
    if isinstance(dt_rank, str):
        if dt_rank.lower() != "auto":
            raise ValueError(f"dt_rank must be an int or 'auto', got {dt_rank!r}")
        return max(1, math.ceil(d_model / 16))
    return int(dt_rank)


class MambaBlock(nn.Module):
    """A single selective-SSM (Mamba) block.

    Args:
        d_model: Residual-stream width.
        d_state: SSM state dimension ``N``.
        d_conv: Depthwise causal convolution kernel width.
        expand: Inner expansion factor; ``d_inner = expand * d_model``.
        dt_rank: Rank of the ``dt`` (selective time-step) projection, or
            ``"auto"`` for ``ceil(d_model / 16)``.
        dt_min: Lower bound for the softplus-space ``dt`` bias initialisation.
        dt_max: Upper bound for the softplus-space ``dt`` bias initialisation.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int,
        d_conv: int,
        expand: int,
        dt_rank: int | str,
        dt_min: float,
        dt_max: float,
        scan: str = "chunkwise",
        chunk_size: int = 32,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = expand * d_model
        self.dt_rank = _resolve_dt_rank(dt_rank, d_model)
        if scan not in ("sequential", "chunkwise"):
            raise ValueError(
                f"mamba_scan={scan!r} not in ('sequential', 'chunkwise')."
            )
        self.scan = scan
        if chunk_size < 1:
            raise ValueError(f"mamba_chunk_size must be >= 1, got {chunk_size}")
        self.chunk_size = chunk_size

        # in_proj produces the SSM input branch (x) and the gate branch (z).
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)

        # Depthwise causal 1-D convolution over the x branch.  We pad on the
        # left only (``d_conv - 1``) and crop the right so position t never sees
        # the future — the causal-conv equivalent of a left-shifted FIR filter.
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            bias=True,
        )

        # x_proj maps the (post-conv, activated) x to the selective parameters:
        # dt (rank dt_rank), B (d_state), C (d_state).
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        # dt_proj lifts dt back to per-channel and is the only branch with a bias.
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # A is stored in log space (A = -exp(A_log)) so it stays strictly
        # negative (stable decay) under unconstrained optimisation.  Initialised
        # to S4D-real: A_n = -(n+1), broadcast across channels.
        a = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(a))
        # D is the per-channel skip (input -> output) connection.
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        self._init_dt_bias(dt_min, dt_max)

    def _init_dt_bias(self, dt_min: float, dt_max: float) -> None:
        """Initialise dt_proj bias so softplus(bias) lands in [dt_min, dt_max].

        Mirrors the reference Mamba init: sample dt uniformly in log space over
        ``[dt_min, dt_max]`` then invert softplus to recover the bias.
        """
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=1e-4)
        # inverse of softplus: bias = dt + log(-expm1(-dt))
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

    def _causal_conv(self, x: Tensor) -> Tensor:
        """Depthwise causal conv over the time axis.

        Args:
            x: ``(B, T, d_inner)``.

        Returns:
            ``(B, T, d_inner)`` — same length, no future leakage.
        """
        T = x.shape[1]
        x = x.transpose(1, 2)  # (B, d_inner, T)
        x = self.conv1d(x)[..., :T]  # crop the right padding -> causal
        return x.transpose(1, 2)  # (B, T, d_inner)

    def _selective_scan(
        self, u: Tensor, delta: Tensor, B: Tensor, C: Tensor
    ) -> Tensor:
        """Sequential selective-scan recurrence (the pure-PyTorch core).

        Implements, per channel and per state n::

            dA = exp(delta * A)            # discretised state transition (ZOH)
            dB = delta * B                 # discretised input matrix
            h_t = dA * h_{t-1} + dB * u_t
            y_t = sum_n C * h_t + D * u_t

        Args:
            u:     ``(B, T, d_inner)``  input sequence (post-conv, activated).
            delta: ``(B, T, d_inner)``  positive time-steps (already softplus'd).
            B:     ``(B, T, d_state)``  selective input matrix.
            C:     ``(B, T, d_state)``  selective output matrix.

        Returns:
            ``(B, T, d_inner)`` output sequence.
        """
        batch, seq_len, d_inner = u.shape
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state), strictly < 0

        # Discretise.  Shapes: (B, T, d_inner, d_state).
        dA = torch.exp(delta.unsqueeze(-1) * A)
        dB_u = delta.unsqueeze(-1) * B.unsqueeze(2) * u.unsqueeze(-1)

        h = torch.zeros(batch, d_inner, self.d_state, device=u.device, dtype=dA.dtype)
        ys = []
        for t in range(seq_len):
            h = dA[:, t] * h + dB_u[:, t]
            # y_t = C_t · h_t  (contract over the state dim)
            y_t = torch.einsum("bdn,bn->bd", h, C[:, t])
            ys.append(y_t)
        y = torch.stack(ys, dim=1)  # (B, T, d_inner)
        y = y + u * self.D  # per-channel skip connection
        return y

    def _selective_scan_chunkwise(
        self, u: Tensor, delta: Tensor, B: Tensor, C: Tensor
    ) -> Tensor:
        """Chunked parallel selective-scan (the fast path).

        Mathematically identical to :meth:`_selective_scan` (validated against it
        within tolerance), but runs ``T/CH`` chunk steps instead of ``T`` Python
        iterations.  The selective SSM is a **diagonal linear recurrence**
        ``h_t = dA_t * h_{t-1} + dB_u_t`` (``dA_t = exp(delta_t A)`` with
        ``A < 0``, so ``dA_t in (0, 1)``), which has a clean chunked closed form:
        within a chunk, with inclusive cumulative log-decay ``lP_t = sum_{j<=t}
        delta_j A``,

            h_t   = exp(lP_t) h0 + sum_{s<=t} exp(lP_t - lP_s) dB_u_s
            y_t   = sum_n C_t * h_t

        Every decay coefficient ``exp(lP_t - lP_s)`` for ``s <= t`` is a product
        of ``dA in (0, 1]`` and so is bounded — fp32-safe.  ``lP`` is accumulated
        from ``delta * A`` directly (never round-tripped through ``exp`` then
        ``log``), so a strongly-decayed step cannot underflow ``dA`` to ``0`` and
        poison the cumulative sum with ``-inf``.  The masked ``s > t`` region is
        set to ``-inf`` in log-space before ``exp`` (-> exactly 0, never
        ``inf * 0 = NaN``).  Sequences not a multiple of ``CH`` are right-padded
        (zero input, zero log-decay == ``dA = 1``); the tail is sliced off.

        Args:
            u:     ``(B, T, d_inner)``  input sequence (post-conv, activated).
            delta: ``(B, T, d_inner)``  positive time-steps (already softplus'd).
            B:     ``(B, T, d_state)``  selective input matrix.
            C:     ``(B, T, d_state)``  selective output matrix.

        Returns:
            ``(B, T, d_inner)`` output sequence (same as ``_selective_scan``).
        """
        bsz, seq_len, d_inner = u.shape
        n = self.d_state
        dtype, device = u.dtype, u.device
        ch = self.chunk_size
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state), strictly < 0

        # log-decay per step: log dA = delta * A  (<= 0, finite — no exp/log round-trip).
        log_dA = delta.unsqueeze(-1) * A                       # (B, T, d_inner, d_state)
        dB_u = delta.unsqueeze(-1) * B.unsqueeze(2) * u.unsqueeze(-1)

        pad = (ch - seq_len % ch) % ch
        if pad:
            log_dA = F.pad(log_dA, (0, 0, 0, 0, 0, pad))       # 0 == dA 1 on pad
            dB_u = F.pad(dB_u, (0, 0, 0, 0, 0, pad))           # zero input on pad
            C = F.pad(C, (0, 0, 0, pad))
        t_pad = seq_len + pad
        n_chunks = t_pad // ch

        log_dA = log_dA.reshape(bsz, n_chunks, ch, d_inner, n)
        dB_u = dB_u.reshape(bsz, n_chunks, ch, d_inner, n)
        C = C.reshape(bsz, n_chunks, ch, n)
        log_cum = torch.cumsum(log_dA, dim=2)                  # lP_t, (B,nc,CH,d,n)

        lower = torch.tril(torch.ones(ch, ch, dtype=dtype, device=device))  # s <= t
        neg_inf = torch.finfo(dtype).min

        h0 = torch.zeros(bsz, d_inner, n, dtype=dtype, device=device)
        ys = []
        for c in range(n_chunks):
            lp = log_cum[:, c]                                 # (B,CH,d,n)
            p = torch.exp(lp)                                  # P_t in (0,1]
            # ratio[t, s] = P_t / P_s = exp(lP_t - lP_s) for s <= t, else 0.
            ratio_log = lp.unsqueeze(2) - lp.unsqueeze(1)      # (B,CH,CH,d,n) [t,s]
            ratio = torch.exp(
                torch.where(lower.bool()[None, :, :, None, None], ratio_log, neg_inf)
            )
            intra = torch.einsum("btsdn,bsdn->btdn", ratio, dB_u[:, c])
            h = p * h0.unsqueeze(1) + intra                    # (B,CH,d,n)
            ys.append(torch.einsum("btdn,btn->btd", h, C[:, c]))
            h0 = h[:, -1]                                      # carry state to next chunk
        y = torch.stack(ys, dim=1).reshape(bsz, t_pad, d_inner)[:, :seq_len]
        return y + u * self.D                                  # per-channel skip

    def forward(self, x: Tensor) -> Tensor:
        """Run the Mamba block.

        Args:
            x: ``(B, T, d_model)``.

        Returns:
            ``(B, T, d_model)``.
        """
        xz = self.in_proj(x)  # (B, T, 2 * d_inner)
        x_in, z = xz.chunk(2, dim=-1)

        x_in = self._causal_conv(x_in)
        x_in = F.silu(x_in)

        dbl = self.x_proj(x_in)  # (B, T, dt_rank + 2 * d_state)
        dt, B, C = torch.split(dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(dt))  # (B, T, d_inner), strictly positive

        scan = (
            self._selective_scan_chunkwise
            if self.scan == "chunkwise"
            else self._selective_scan
        )
        y = scan(x_in.float(), delta.float(), B.float(), C.float())
        y = y.to(x.dtype)
        y = y * F.silu(z)  # gating branch
        return self.out_proj(y)


class MambaResidualBlock(nn.Module):
    """Pre-norm residual wrapper around a :class:`MambaBlock` (RMSNorm + skip)."""

    def __init__(self, m: ModelConfig) -> None:
        super().__init__()
        self.norm = RMSNorm(m.d_model)
        self.mixer = MambaBlock(
            d_model=m.d_model,
            d_state=m.d_state,
            d_conv=m.d_conv,
            expand=m.expand,
            dt_rank=m.dt_rank,
            dt_min=m.dt_min,
            dt_max=m.dt_max,
            scan=m.mamba_scan,
            chunk_size=m.mamba_chunk_size,
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.mixer(self.norm(x))


@register_model("mamba")
class MambaBaseline(nn.Module):
    """Decoder-only selective-SSM language model registered as ``"mamba"``.

    Same ``forward(x, targets) -> (loss, logits)`` contract as the Transformer
    baseline; the model-agnostic Trainer needs no changes.  There is no
    positional embedding — the SSM recurrence is inherently sequential, which is
    exactly the property that lets it stream past any fixed context window.
    """

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        m = cfg.model
        self.d_model = m.d_model
        self.vocab_size = m.vocab_size
        self.activation_checkpointing = m.activation_checkpointing

        self.embed = nn.Embedding(m.vocab_size, m.d_model)
        self.embed_drop = nn.Dropout(m.dropout)
        self.blocks = nn.ModuleList([MambaResidualBlock(m) for _ in range(m.n_layers)])
        self.norm_out = RMSNorm(m.d_model)

        self.lm_head = nn.Linear(m.d_model, m.vocab_size, bias=False)
        if m.tie_embeddings:
            self.lm_head.weight = self.embed.weight

        # Init weights first, then let the optional embedding-init hook (card
        # e1644700) have the final say on the embedding table — same ordering
        # contract as the Transformer baseline.
        self._init_weights(n_layers=m.n_layers)
        if m.embedding_init is not None:
            init_fn = get_embedding_init(m.embedding_init)
            init_fn(self.embed.weight, m.vocab_size, m.d_model)

    def _init_weights(self, n_layers: int) -> None:
        """GPT-2 style init with depth-aware residual-projection scaling.

        Scaling each block's output projection by ``1/sqrt(2 * n_layers)`` keeps
        residual-stream variance bounded as depth grows, mirroring the
        Transformer baseline so ``match_params`` configs (which may stack many
        layers) train stably.  Linear biases (only ``dt_proj`` has one) keep
        their custom init and are not touched here.
        """
        std = 0.02
        nn.init.normal_(self.embed.weight, mean=0.0, std=std)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=std)
        residual_std = std / (2 * n_layers) ** 0.5
        for block in self.blocks:
            nn.init.normal_(block.mixer.out_proj.weight, mean=0.0, std=residual_std)
            # Restore the reference-Mamba dt_proj weight init that the generic
            # 0.02 loop above clobbered. std = dt_rank**-0.5 keeps the
            # input-dependent part of delta meaningful at init; without it delta
            # is bias-dominated and the SSM's selectivity is weak early on.
            dt_rank = block.mixer.dt_rank
            nn.init.normal_(block.mixer.dt_proj.weight, mean=0.0, std=float(dt_rank) ** -0.5)

    def _forward_blocks(self, x: Tensor) -> Tensor:
        for block in self.blocks:
            if self.activation_checkpointing and self.training:
                x = cast(
                    Tensor,
                    torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False),
                )
            else:
                x = block(x)
        return x

    def forward(self, x: Tensor, targets: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Forward pass.

        Args:
            x: Token ids, ``(B, T)``.
            targets: Target token ids, ``(B, T)`` or ``None`` (eval/generation).

        Returns:
            ``(loss, logits)``; ``loss`` is a scalar ``Tensor`` (zero when
            ``targets is None``) and ``logits`` is ``(B, T, vocab_size)``.
        """
        h = self.embed_drop(self.embed(x))
        h = self._forward_blocks(h)
        h = self.norm_out(h)
        logits = self.lm_head(h)

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.reshape(-1))
        else:
            loss = torch.zeros(1, device=x.device)

        return loss, logits

    def num_parameters(self, trainable_only: bool = True) -> int:
        """Return total (or trainable-only) parameter count."""
        return sum(
            p.numel()
            for p in self.parameters()
            if (not trainable_only or p.requires_grad)
        )
