"""Standalone delta-rule matrix-memory language model (card e2c6ea95).

``delta_memory_lm`` is the registered LM that realises the project's **central
thesis** — *"a persistent bounded memory replaces the context window"* — so it
can be validated head-to-head against the Mamba baseline on long-context probes
(perplexity-vs-context curves + passkey/recall).  The token mixer is the
:class:`GatedDeltaMemory` layer: a fixed-size, per-head matrix associative memory
updated by a delta rule + forget gate (Gated-DeltaNet).  This is **not** a graph
and **not** attention with a KV cache — the whole "memory" is a small matrix per
head, edited in place token by token, whose size is independent of sequence
length.

Architecture (mirrors the baselines + ``bilinear_lm``)::

    token ids (B, T)
      -> nn.Embedding                          (B, T, d_model)   # honours the
                                                                 # phonological hook
      -> N x DeltaMemoryBlock
           pre-norm RMSNorm -> GatedDeltaMemory -> + residual    # token mixing
           pre-norm RMSNorm -> GatedMLP         -> + residual    # per-position FFN
      -> RMSNorm
      -> tied LM head                           (B, T, vocab)

Contracts honoured (zero trainer changes):
* ``forward(x, targets) -> (loss, logits)`` (the only thing the Trainer sees).
* ``self.embed`` is the ``nn.Embedding`` the embedding-init hook writes to, and
  the hook runs **after** ``_init_weights`` (regression SF-1 ordering).
* ``num_parameters(trainable_only=True)`` for ``sizing.count_params`` /
  ``match_params``.
* Depth (``delta_layers``) and width (``d_model``, ``delta_n_heads``,
  ``delta_head_k_dim`` / ``delta_head_v_dim``) scale by config to reach
  ~GPT-1/2 param counts.  There is no positional embedding — the memory
  recurrence is inherently sequential, the property that lets it stream past any
  fixed context window.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import Tensor

from graph_llm.models.baselines.transformer import RMSNorm
from graph_llm.models.components.delta_memory import GatedDeltaMemory
from graph_llm.models.registry import get_embedding_init, register_model

if TYPE_CHECKING:
    from graph_llm.config import Config, ModelConfig


class GatedMLP(nn.Module):
    """GeGLU-style gated feed-forward (matches the ``bilinear_lm`` post-mixer MLP)."""

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        h = F.gelu(self.gate_proj(x)) * self.up_proj(x)
        return self.dropout(self.down_proj(h))


class DeltaMemoryBlock(nn.Module):
    """Pre-norm residual block: GatedDeltaMemory (token mixing) + gated MLP.

    The memory mixer carries cross-position information *along the sequence* (via
    its persistent state); the gated MLP is the per-position non-linearity.  Both
    sub-layers are pre-norm + residual, mirroring the Transformer and Mamba
    baselines so ``match_params`` configs train stably.
    """

    def __init__(self, m: ModelConfig) -> None:
        super().__init__()
        self.norm1 = RMSNorm(m.d_model)
        self.mixer = GatedDeltaMemory(m)
        self.norm2 = RMSNorm(m.d_model)
        self.mlp = GatedMLP(m.d_model, m.delta_ff_mult * m.d_model, m.delta_dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.mixer(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


@register_model("delta_memory_lm")
class DeltaMemoryLM(nn.Module):
    """Delta-rule matrix-memory LM registered as ``"delta_memory_lm"``.

    ``forward(x, targets)`` accepts token ids ``(B, T)`` and returns
    ``(loss, logits)`` with ``logits`` of shape ``(B, T, vocab_size)``.  The
    Trainer depends only on that contract.
    """

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        m = cfg.model
        self.d_model = m.d_model
        self.vocab_size = m.vocab_size
        self.activation_checkpointing = m.activation_checkpointing

        # Token embedding (the phonological-init hook writes here).
        self.embed = nn.Embedding(m.vocab_size, m.d_model)
        self.embed_drop = nn.Dropout(m.dropout)

        # Depth-scalable stack of delta-memory blocks.
        self.blocks = nn.ModuleList([DeltaMemoryBlock(m) for _ in range(m.delta_layers)])
        self.norm_out = RMSNorm(m.d_model)

        # Output head — optionally tied to the embedding table.
        self.lm_head = nn.Linear(m.d_model, m.vocab_size, bias=False)
        if m.tie_embeddings:
            self.lm_head.weight = self.embed.weight

        # Init all weights first, THEN let the optional embedding-init hook (card
        # e1644700) have the final say on the embedding table — same ordering
        # contract as the Transformer / Mamba / bilinear_lm models (regression SF-1).
        self._init_weights(m)
        if m.embedding_init is not None:
            init_fn = get_embedding_init(m.embedding_init)
            init_fn(self.embed.weight, m.vocab_size, m.d_model)

    def _init_weights(self, m: ModelConfig) -> None:
        """GPT-2 style init with depth-aware residual-projection scaling.

        The residual *output* projections (the memory ``out_proj`` and the
        gated-MLP ``down_proj``) are scaled by ``1/sqrt(2 * delta_layers)`` so the
        residual-stream variance stays bounded as ``match_params`` stacks depth —
        mirroring the baselines.
        """
        std = 0.02
        nn.init.normal_(self.embed.weight, mean=0.0, std=std)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        n_layers = max(1, m.delta_layers)
        residual_std = std / (2 * n_layers) ** 0.5
        for block in self.blocks:
            nn.init.normal_(block.mixer.out_proj.weight, mean=0.0, std=residual_std)
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
            ``(loss, logits)`` — scalar ``loss`` and ``(B, T, vocab_size)``
            ``logits``.
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
