"""Standalone ablatable bilinear language model (card 86347418).

``bilinear_lm`` is the first LM built around a *novel* primitive: the windowed
factorized-bilinear (MFB) front-end (:class:`BilinearFrontEnd`).  It exists to
answer one question -- *does explicit second-order (multiplicative) local token
interaction help?* -- measured against the Phase 0b baselines via the
interaction-mode switch (``factorized_mfb`` vs the ``control_linear`` null).

Architecture::

    token ids (B, T)
      -> nn.Embedding                          (B, T, d_model)   # honours the
                                                                 # phonological hook
      -> BilinearFrontEnd  (windowed MFB)       (B, T, o)
      -> front-end projection  o -> d_model     (B, T, d_model)
      -> post-mixer x N
           (causal depthwise-separable 1-D CNN over the sequence + gated MLP)
      -> RMSNorm
      -> tied LM head                           (B, T, vocab)

Honesty: the front-end is *local* (window ``W``); this model tests the
multiplicative-interaction primitive, not long-range modelling (that is the
GNNs' job in later phases).

Contracts honoured (zero trainer changes):
* ``forward(x, targets) -> (loss, logits)`` (the only thing the Trainer sees).
* ``self.embed`` is the ``nn.Embedding`` the embedding-init hook writes to.
* ``num_parameters(trainable_only=True)`` for ``sizing.count_params`` /
  ``match_params``.
* Depth (``post_mixer_layers``) and width (``d_model``, ``bilinear_o``) scale by
  config to reach ~GPT-1/2 param counts; the bulk of params live in the
  front-end + post-mixer, not the (intentionally tiny) embedding.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import Tensor

from graph_llm.models.baselines.transformer import RMSNorm
from graph_llm.models.components.bilinear_frontend import BilinearFrontEnd
from graph_llm.models.registry import get_embedding_init, register_model

if TYPE_CHECKING:
    from graph_llm.config import Config, ModelConfig


# ---------------------------------------------------------------------------
# Post-mixer block: causal depthwise-separable 1-D CNN + gated MLP
# ---------------------------------------------------------------------------


class DepthwiseSeparableConv1d(nn.Module):
    """Causal depthwise-separable 1-D convolution over the sequence axis.

    Depthwise (per-channel) causal conv + pointwise (1x1) mix.  Causality is
    enforced by left-padding by ``kernel - 1`` and trimming the right tail, so
    position ``t`` only sees ``<= t`` (next-token training stays leak-free).
    """

    def __init__(self, d_model: int, kernel: int) -> None:
        super().__init__()
        self.kernel = kernel
        self.depthwise = nn.Conv1d(
            d_model, d_model, kernel_size=kernel, groups=d_model, bias=False
        )
        self.pointwise = nn.Conv1d(d_model, d_model, kernel_size=1, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, T, C) -> (B, C, T) for conv1d
        h = x.transpose(1, 2)
        h = F.pad(h, (self.kernel - 1, 0))           # left (causal) pad only
        h = self.depthwise(h)
        h = self.pointwise(h)
        return h.transpose(1, 2)                      # (B, T, C)


class GatedMLP(nn.Module):
    """GeGLU-style gated feed-forward."""

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        h = F.gelu(self.gate_proj(x)) * self.up_proj(x)
        return self.dropout(self.down_proj(h))


class PostMixerBlock(nn.Module):
    """Pre-norm block: causal depthwise-separable conv + gated MLP, both residual.

    The conv carries cross-position information *along the sequence* (the
    front-end only mixes within a local window per position); the gated MLP is
    the per-position non-linearity.  Holds the bulk of the depth-scalable params.
    """

    def __init__(self, d_model: int, kernel: int, ff_mult: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.conv = DepthwiseSeparableConv1d(d_model, kernel)
        self.norm2 = RMSNorm(d_model)
        self.mlp = GatedMLP(d_model, ff_mult * d_model, dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.conv(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------


@register_model("bilinear_lm")
class BilinearLM(nn.Module):
    """Ablatable bilinear LM registered as ``"bilinear_lm"``.

    ``forward(x, targets)`` accepts token ids ``(B, T)`` and returns
    ``(loss, logits)`` with ``logits`` of shape ``(B, T, vocab_size)``.
    The Trainer depends only on that contract.
    """

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        m = cfg.model
        self.d_model = m.d_model
        self.vocab_size = m.vocab_size
        self.activation_checkpointing = m.activation_checkpointing

        # Trunk width. The embedding (d_model) is intentionally tiny; the
        # post-mixer trunk may run wider (post_mixer_width) so depth*width scales
        # to GPT-1/2 budgets without bloating the embedding table.  0 == d_model.
        self.trunk_width = m.post_mixer_width or m.d_model

        # Token embedding (the phonological-init hook writes here).
        self.embed = nn.Embedding(m.vocab_size, m.d_model)
        self.embed_drop = nn.Dropout(m.dropout)

        # Windowed factorized-bilinear front-end -> (B, T, o).
        self.front_end = BilinearFrontEnd(m)

        # Bridge the front-end output (o) into the trunk width.
        self.front_proj = nn.Linear(self.front_end.out_dim, self.trunk_width, bias=False)

        # Depth-scalable post-mixer stack (runs at trunk_width).
        self.blocks = nn.ModuleList(
            [
                PostMixerBlock(
                    d_model=self.trunk_width,
                    kernel=m.post_mixer_kernel,
                    ff_mult=m.post_mixer_ff_mult,
                    dropout=m.dropout,
                )
                for _ in range(m.post_mixer_layers)
            ]
        )
        self.norm_out = RMSNorm(self.trunk_width)

        # Bridge the trunk back to d_model so the LM head can tie to the
        # embedding table.  Identity-shaped (and skipped) when trunk_width == d_model.
        self.trunk_to_embed: nn.Linear | None = (
            nn.Linear(self.trunk_width, m.d_model, bias=False)
            if self.trunk_width != m.d_model
            else None
        )

        # Output head -- optionally tied to the embedding table.
        self.lm_head = nn.Linear(m.d_model, m.vocab_size, bias=False)
        if m.tie_embeddings:
            self.lm_head.weight = self.embed.weight

        # Init all weights first, THEN let the optional embedding-init hook
        # (card e1644700) have the final say on the embedding table -- mirrors the
        # Transformer baseline ordering (regression SF-1).
        self._init_weights(m)
        if m.embedding_init is not None:
            init_fn = get_embedding_init(m.embedding_init)
            init_fn(self.embed.weight, m.vocab_size, m.d_model)

    def _init_weights(self, m: ModelConfig) -> None:
        """GPT-2 style init with depth-aware residual-projection scaling.

        The residual *output* projections (the conv pointwise and the gated-MLP
        down-projection) are scaled by ``1/sqrt(2 * post_mixer_layers)`` so the
        residual-stream variance stays bounded as ``match_params`` stacks depth.
        """
        std = 0.02
        nn.init.normal_(self.embed.weight, mean=0.0, std=std)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=std)
            elif isinstance(module, nn.Conv1d | nn.Conv2d):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        n_layers = max(1, m.post_mixer_layers)
        residual_std = std / (2 * n_layers) ** 0.5
        for block in self.blocks:
            nn.init.normal_(block.conv.pointwise.weight, mean=0.0, std=residual_std)
            nn.init.normal_(block.mlp.down_proj.weight, mean=0.0, std=residual_std)

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
            x: Token ids, shape ``(B, T)``.
            targets: Target token ids, shape ``(B, T)``.  If provided, the
                cross-entropy loss is computed over the full sequence; if
                ``None`` a zero loss tensor is returned (eval / generation).

        Returns:
            ``(loss, logits)`` -- scalar ``loss`` and ``(B, T, vocab_size)``
            ``logits``.
        """
        tok_emb = self.embed(x)                       # (B, T, d_model)
        h = self.embed_drop(tok_emb)
        h = self.front_end(h)                         # (B, T, o)
        h = self.front_proj(h)                        # (B, T, trunk_width)
        h = self._forward_blocks(h)
        h = self.norm_out(h)
        if self.trunk_to_embed is not None:
            h = self.trunk_to_embed(h)                # (B, T, d_model)
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
