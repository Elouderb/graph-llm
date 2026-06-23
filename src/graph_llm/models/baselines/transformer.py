"""Decoder-only Transformer baseline (card 424e3a8e).

This is the *attention* arm of the matched-parameter Phase 0 baseline suite
(the recurrent-state arm is ``mamba.py``).  It is fully config-driven —
``depth`` (``n_layers``), ``width`` (``d_model``), ``n_heads``, ``d_ff``, and
``max_seq_len`` all come from :class:`~graph_llm.config.ModelConfig` — so the
``match_params`` utility (``sizing.py``) can scale it to a target parameter
budget without touching this file.

Architecture:
* Token embedding + optional RoPE positional encoding (or learned absolute pos)
* Stacked Transformer decoder blocks (pre-norm, no cross-attention)
* RMSNorm
* Tied input/output embeddings (optional, toggleable)
* Causal self-attention via PyTorch's ``scaled_dot_product_attention``
* Optional activation checkpointing (``torch.utils.checkpoint``)

Kept minimal and readable; correctness and apples-to-apples parameter matching
over micro-optimisation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import Tensor

from graph_llm.models.registry import get_embedding_init, register_model

if TYPE_CHECKING:
    from graph_llm.config import Config


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (no learnable bias)."""

    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: Tensor) -> Tensor:
        norm = x.float().pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return (x.float() / norm * self.weight).to(x.dtype)


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE).

    Computes sin/cos tables up to *max_seq_len* and caches them as
    non-parameter buffers so they survive ``state_dict`` serialisation.
    """

    def __init__(self, head_dim: int, max_seq_len: int, base: int = 10_000) -> None:
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cached_len = 0
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cache", emb.cos(), persistent=False)
        self.register_buffer("sin_cache", emb.sin(), persistent=False)
        self._cached_len = seq_len

    def _maybe_extend_cache(self, seq_len: int) -> None:
        """Grow the sin/cos tables if a sequence longer than the cache arrives.

        This is what lets the Transformer run on sequences *longer than the
        training window* — required by the long-context harness (passkey probe,
        position-loss curve).  RoPE extends naturally to unseen positions.
        """
        if seq_len > self._cached_len:
            self._build_cache(seq_len)

    @staticmethod
    def _rotate_half(x: Tensor) -> Tensor:
        half = x.shape[-1] // 2
        return torch.cat([-x[..., half:], x[..., :half]], dim=-1)

    def forward(self, q: Tensor, k: Tensor) -> tuple[Tensor, Tensor]:
        """Apply RoPE to queries and keys.

        Args:
            q: ``(B, H, T, head_dim)``
            k: ``(B, H, T, head_dim)``

        Returns:
            Rotated ``(q, k)`` pair.
        """
        seq_len = q.shape[2]
        self._maybe_extend_cache(seq_len)
        cos = self.cos_cache[:seq_len].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cache[:seq_len].unsqueeze(0).unsqueeze(0)
        q_rot = q * cos + self._rotate_half(q) * sin
        k_rot = k * cos + self._rotate_half(k) * sin
        return q_rot.to(q.dtype), k_rot.to(k.dtype)


# ---------------------------------------------------------------------------
# Attention and Feed-Forward
# ---------------------------------------------------------------------------


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention (no KV cache in Phase 0)."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float,
        max_seq_len: int,
        use_rope: bool,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.rope: RotaryEmbedding | None = None
        if use_rope:
            self.rope = RotaryEmbedding(self.head_dim, max_seq_len)

    def forward(self, x: Tensor) -> Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        if self.rope is not None:
            q, k = self.rope(q, k)

        # PyTorch built-in flash / mem-efficient path (causal mask applied internally)
        attn_dropout = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=attn_dropout, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


class FeedForward(nn.Module):
    """Two-layer feed-forward with GELU activation."""

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model, bias=False),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block: Attention + FFN."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        max_seq_len: int,
        use_rope: bool,
    ) -> None:
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout, max_seq_len, use_rope)
        self.norm2 = RMSNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------


@register_model("transformer")
class TransformerBaseline(nn.Module):
    """Decoder-only Transformer baseline registered as ``"transformer"``.

    ``forward(x)`` accepts token ids of shape ``(B, T)`` and returns
    ``(loss, logits)`` where *loss* is the scalar cross-entropy over all
    non-padding positions and *logits* has shape ``(B, T, vocab_size)``.

    The Trainer only depends on this ``(loss, logits)`` contract.
    """

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        m = cfg.model
        self.d_model = m.d_model
        self.vocab_size = m.vocab_size
        self.activation_checkpointing = m.activation_checkpointing

        # Token embedding
        self.embed = nn.Embedding(m.vocab_size, m.d_model)

        # Learned absolute positional embedding (used only when RoPE is off)
        self.pos_embed: nn.Embedding | None = None
        if not m.use_rope:
            self.pos_embed = nn.Embedding(m.max_seq_len, m.d_model)

        self.embed_drop = nn.Dropout(m.dropout)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=m.d_model,
                    n_heads=m.n_heads,
                    d_ff=m.d_ff,
                    dropout=m.dropout,
                    max_seq_len=m.max_seq_len,
                    use_rope=m.use_rope,
                )
                for _ in range(m.n_layers)
            ]
        )
        self.norm_out = RMSNorm(m.d_model)

        # Output head — optionally tied to the embedding weights
        self.lm_head = nn.Linear(m.d_model, m.vocab_size, bias=False)
        if m.tie_embeddings:
            self.lm_head.weight = self.embed.weight

        # Initialise all weights first, THEN let the optional embedding-init
        # hook (card e1644700) have the final say on the embedding table.
        # Order matters: _init_weights() resets embed.weight (and, when tied,
        # the lm_head), so it must run before the hook, not after.
        self._init_weights(n_layers=m.n_layers)
        if m.embedding_init is not None:
            init_fn = get_embedding_init(m.embedding_init)
            init_fn(self.embed.weight, m.vocab_size, m.d_model)

    def _init_weights(self, n_layers: int) -> None:
        """GPT-2 style init with depth-aware residual-projection scaling.

        Scaling the residual *output* projections (attention ``out_proj`` and the
        second FFN linear) by ``1/sqrt(2 * n_layers)`` keeps the variance of the
        residual stream bounded as depth grows, so configs produced by
        ``match_params`` (which may stack many layers) train stably.
        """
        std = 0.02
        nn.init.normal_(self.embed.weight, mean=0.0, std=std)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=std)
        residual_std = std / (2 * n_layers) ** 0.5
        for block in self.blocks:
            nn.init.normal_(block.attn.out_proj.weight, mean=0.0, std=residual_std)
            # block.ff.net[3] is the second (output) Linear of the FFN.
            nn.init.normal_(block.ff.net[3].weight, mean=0.0, std=residual_std)

    def _forward_blocks(self, x: Tensor) -> Tensor:
        """Run all Transformer blocks, with optional activation checkpointing."""
        for block in self.blocks:
            if self.activation_checkpointing and self.training:
                x = cast(Tensor, torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False))
            else:
                x = block(x)
        return x

    def forward(self, x: Tensor, targets: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Forward pass.

        Args:
            x: Token ids, shape ``(B, T)``.
            targets: Target token ids, shape ``(B, T)``.  If provided, the
                cross-entropy loss is computed over the full sequence and
                returned as the first element of the output tuple.
                If ``None``, a zero loss tensor is returned (eval / generation).

        Returns:
            ``(loss, logits)`` where ``loss`` is a scalar ``Tensor`` and
            ``logits`` has shape ``(B, T, vocab_size)``.
        """
        B, T = x.shape
        tok_emb = self.embed(x)

        if self.pos_embed is not None:
            if T > self.pos_embed.num_embeddings:
                raise ValueError(
                    f"Sequence length {T} exceeds the learned positional table "
                    f"({self.pos_embed.num_embeddings}); learned absolute positions "
                    "cannot extrapolate. Use use_rope=True to run the long-context "
                    "harness beyond the training window."
                )
            pos = torch.arange(T, device=x.device)
            tok_emb = tok_emb + self.pos_embed(pos)

        h = self.embed_drop(tok_emb)
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
