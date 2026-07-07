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

from typing import TYPE_CHECKING, Literal, cast, overload

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import Tensor

from graph_llm.models.baselines.transformer import RMSNorm
from graph_llm.models.components.causal_reasoner import CausalReasoner
from graph_llm.models.components.delta_memory import DeltaMemoryState, GatedDeltaMemory
from graph_llm.models.components.multiscale_conv_embed import MultiScaleConvEmbedding
from graph_llm.models.components.reasoning_field import ReasoningField
from graph_llm.models.registry import get_embedding_init, register_model

if TYPE_CHECKING:
    from graph_llm.config import Config, ModelConfig

# Valid input-stage front-ends (card ed853f9c).  "none" is the byte-for-byte
# no-op default (no module is constructed, no behaviour changes); "multiscale_conv"
# inserts the cheap local combiner after the embedding, before the memory layers.
VALID_FRONT_ENDS = ("none", "multiscale_conv")


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


class _WorkhorseBlock(nn.Module):
    """Pre-norm residual GeGLU block for the stacked MLP workhorse (card b1926d5d).

    ``x + mlp(norm(x))`` — one extra depth rung stacked on top of the workhorse's
    bare input :class:`GatedMLP`, mirroring the delta block's MLP sub-layer
    (:class:`DeltaMemoryBlock`).  Per-position (no token mixing) so the workhorse
    stays LM-leak-safe at any depth.  Constructed only for ``tandem_mlp_depth >= 2``;
    at depth 1 the workhorse is the shipped single bare ``GatedMLP`` (no block here).
    """

    def __init__(self, m: ModelConfig) -> None:
        super().__init__()
        self.norm = RMSNorm(m.d_model)
        self.mlp = GatedMLP(m.d_model, m.tandem_mlp_ff_mult * m.d_model, m.delta_dropout)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.mlp(self.norm(x))


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

    def forward(
        self,
        x: Tensor,
        state_in: DeltaMemoryState | None = None,
        return_state: bool = False,
    ) -> Tensor | tuple[Tensor, DeltaMemoryState]:
        """Pre-norm residual block forward.

        Args:
            x: ``(B, T, d_model)``.
            state_in: Optional carried :class:`DeltaMemoryState` for the mixer
                (cross-segment persistence, cards 61f900ca + 571d50ec): the
                delta-memory matrix plus the conv tail.  ``None`` is the reset
                default — byte-for-byte the original behaviour when the conv is
                disabled.
            return_state: When ``True`` also return the mixer's final state.

        Returns:
            ``(B, T, d_model)``, or — when ``return_state`` — a
            ``(out, mixer_state_out)`` tuple.  Only the token-mixer carries a
            cross-segment state; the gated MLP is per-position.
        """
        if return_state:
            mixed, state_out = cast(
                "tuple[Tensor, DeltaMemoryState]",
                self.mixer(self.norm1(x), state_in=state_in, return_state=True),
            )
            x = x + mixed
            x = x + self.mlp(self.norm2(x))
            return x, state_out
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

    # Registered only when tandem_enabled (declared here for static typing).
    _tandem_step: Tensor

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        m = cfg.model
        self.d_model = m.d_model
        self.vocab_size = m.vocab_size
        self.activation_checkpointing = m.activation_checkpointing

        # Token embedding (the phonological-init hook writes here).
        self.embed = nn.Embedding(m.vocab_size, m.d_model)
        self.embed_drop = nn.Dropout(m.dropout)

        # Optional multi-scale conv front-end (card ed853f9c): a cheap causal
        # local combiner inserted AFTER the embedding (+ phonological hook), BEFORE
        # the memory layers, enriching each token with trailing local context.
        # ``front_end="none"`` (default) builds NOTHING here -> the committed
        # backbone is byte-for-byte unchanged (no extra params / RNG draws), so it
        # is a clean one-flag A/B.
        front_end = getattr(m, "front_end", "none")
        if front_end not in VALID_FRONT_ENDS:
            raise ValueError(f"front_end={front_end!r} not in {VALID_FRONT_ENDS}.")
        self.front_end: MultiScaleConvEmbedding | None = (
            MultiScaleConvEmbedding(m) if front_end == "multiscale_conv" else None
        )
        # Front-end scope (card b1926d5d): route the conv output to the MLP-workhorse ONLY
        # (leaving memory + reasoner-ctx + gate on the raw embedding) when True; feed it to
        # all pathways (the committed default) when False.
        self._front_end_workhorse_only = bool(getattr(m, "front_end_workhorse_only", False))

        # Depth-scalable stack of delta-memory blocks.
        self.blocks = nn.ModuleList([DeltaMemoryBlock(m) for _ in range(m.delta_layers)])

        # Optional transient reasoning field (card 9907dc9e): the R2-validated
        # iterated 2-D reasoner, run INDEPENDENTLY per position (seeded from h_t
        # alone -> input-seeded + provably causal + transient), inserted AFTER the
        # memory blocks and BEFORE norm_out so its readout refines the next-token
        # prediction (added as a residual in ``forward``).  It is a SEPARATE
        # scratchpad, distinct from the persistent delta-net memory + its carried
        # DeltaMemoryState.  ``reasoning_enabled=False`` (default) builds NOTHING
        # here (no module, no params, no RNG draws) -> the committed backbone is
        # byte-for-byte unchanged, so it is a clean one-flag A/B.
        self.reasoner: ReasoningField | None = (
            ReasoningField(m) if getattr(m, "reasoning_enabled", False) else None
        )

        # Optional tandem reasoner pathway (card 2dd3400f): the lead-verified v3 WIN
        # causal reasoner (card e4e8a4dc) run PER POSITION, fused with the delta-memory
        # backbone by the UNSUPERVISED gate (card 31fe6b00).  The delta stack is the
        # MEMORY pathway (unchanged); the reasoner reads the raw tokens over
        # segment-bounded windows and produces a per-position reasoning hidden; a LIGHT
        # gate ``g = sigmoid(Linear([h_mem ; h_reason ; h_ctx]))`` fuses them per
        # position: ``h = g*h_reason + (1-g)*h_mem`` -> norm -> head.  DISTINCT from the
        # legacy ``self.reasoner`` field above (separate config surface).
        #
        # ``tandem_enabled=False`` (default) builds NOTHING here (no reasoner, no gate,
        # no buffer, no params, no RNG draws) -> the committed backbone is byte-for-byte
        # unchanged, so it is a clean one-flag A/B.
        #
        # WORKHORSE (card a7948491): with ``tandem_mlp_enabled=True`` a THIRD pathway — a
        # plain per-token :class:`GatedMLP` on the same trunk input ``h_embed`` — is added
        # and the gate is generalised 2-way sigmoid {mem, reason} -> 3-way softmax {mem(0),
        # reason(1), mlp(2)}, fusing ``h = sum_e g_e * h_e``.  ``tandem_mlp_enabled=False``
        # (default) keeps the SHIPPED 2-way sigmoid path byte-for-byte (no MLP, gate stays
        # Linear(3*d_model)) so the committed recipe still reproduces exactly.
        self.causal_reasoner: CausalReasoner | None = None
        self.tandem_gate: nn.Linear | None = None
        self.tandem_mlp: GatedMLP | None = None
        # Extra stacked GeGLU rungs on top of the workhorse's bare input GatedMLP (card
        # b1926d5d).  EMPTY (no modules/params/state_dict keys/RNG draws) unless
        # tandem_mlp_depth >= 2 -> the depth-1 / 2-way / OFF states are byte-for-byte.
        self.tandem_mlp_blocks: nn.ModuleList = nn.ModuleList()
        if getattr(m, "tandem_enabled", False):
            self.causal_reasoner = CausalReasoner(m)
            self._tandem_scalar = m.tandem_gate_scalar
            if getattr(m, "tandem_mlp_enabled", False):
                # 3-way: memory / reasoner / mlp-workhorse.  Gate reads all three pathway
                # hiddens + the shared trunk context (4*d_model) and emits 3 expert logits
                # (per-channel: 3*d_model viewed (.,d_model,3); scalar: 3).
                self.tandem_mlp = GatedMLP(
                    m.d_model, m.tandem_mlp_ff_mult * m.d_model, m.delta_dropout
                )
                # WORKHORSE DEPTH (card b1926d5d): stack ``tandem_mlp_depth-1`` pre-norm
                # residual GeGLU blocks on top of the bare input GatedMLP.  depth=1 -> the
                # list stays empty (the shipped single bare workhorse, byte-for-byte).
                depth = getattr(m, "tandem_mlp_depth", 1)
                if depth < 1:
                    raise ValueError(f"tandem_mlp_depth must be >= 1, got {depth}.")
                self.tandem_mlp_blocks = nn.ModuleList(
                    [_WorkhorseBlock(m) for _ in range(depth - 1)]
                )
                gate_out = 3 if m.tandem_gate_scalar else 3 * m.d_model
                self.tandem_gate = nn.Linear(4 * m.d_model, gate_out)
            else:
                # 2-way (shipped, verified): memory / reasoner sigmoid blend (UNCHANGED).
                gate_out = 1 if m.tandem_gate_scalar else m.d_model
                self.tandem_gate = nn.Linear(3 * m.d_model, gate_out)
            # Train-step counter (card 2dd3400f): drives the forced-mix warmup + the
            # locate-first seed commit.  Registered ONLY when the tandem is on, so the
            # OFF state_dict is unchanged.
            self.register_buffer("_tandem_step", torch.zeros((), dtype=torch.long))
            self._gate_mix_warmup = m.gate_mix_warmup_steps
            self._gate_noise_std = m.gate_noise_std
            self._reasoning_locate_warmup = m.reasoning_locate_warmup
            self._tandem_hard_gate = m.tandem_hard_gate

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
            elif isinstance(module, nn.Conv1d):
                # The optional multi-scale conv front-end's bank (depthwise +
                # pointwise convs).  Only present when front_end="multiscale_conv";
                # when "none" this branch never fires, so the no-op path is
                # untouched.
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        n_layers = max(1, m.delta_layers)
        residual_std = std / (2 * n_layers) ** 0.5
        for block in self.blocks:
            nn.init.normal_(block.mixer.out_proj.weight, mean=0.0, std=residual_std)
            nn.init.normal_(block.mlp.down_proj.weight, mean=0.0, std=residual_std)

        # Stacked workhorse blocks (card b1926d5d): residual-scale their down_proj by the
        # workhorse residual depth, same discipline as the memory blocks above (keeps the
        # residual-stream variance bounded when the workhorse deepens).  The bare input
        # ``tandem_mlp`` is deliberately LEFT at the generic std=0.02 (it is not on a
        # residual stream) so the depth-1 workhorse stays byte-for-byte the shipped path.
        n_wh = len(self.tandem_mlp_blocks)
        if n_wh > 0:
            wh_std = std / (2 * n_wh) ** 0.5
            for wblock in self.tandem_mlp_blocks:
                nn.init.normal_(wblock.mlp.down_proj.weight, mean=0.0, std=wh_std)

        # Optional identity-init of the conv front-end (card b1926d5d, coordinator): make it
        # an EXACT pass-through of the raw embedding at step 0 so enabling the conv starts
        # from the no-front-end backbone and the optimizer opens the multi-scale mixing only
        # where it helps.  Applied AFTER the generic loop (which set the conv weights to
        # std=0.02) so it overrides.  No-op when the conv front-end is not built.
        if self.front_end is not None and getattr(m, "conv_identity_init", False):
            self.front_end.identity_init()

        # Forget-gate remember-by-default init (card 1e9245f4).  The generic loop above
        # zeros ALL linear biases — including alpha_proj.bias.  We re-apply the configured
        # bias HERE, after that loop, so it is not clobbered.  alpha_proj.weight was set to
        # N(0, std) above (small enough that input variation barely shifts alpha at init),
        # giving near-constant but data-dependent, fully learnable gates.
        if m.delta_use_forget_gate:
            for block in self.blocks:
                if block.mixer.alpha_proj is not None:
                    nn.init.constant_(block.mixer.alpha_proj.bias, m.delta_forget_bias_init)

        # Reasoning-field head-bias priors (card 9907dc9e).  The generic loop above
        # zeros ALL linear biases — including the reasoner's controller heads, whose
        # R2-validated priors (trust a content match, neutral gate/shift, mild
        # sharpen) are load-bearing for trainability.  Re-apply them HERE, after the
        # loop, so they survive.  When reasoning is disabled self.reasoner is None and
        # this is a no-op (the committed backbone is untouched).
        if self.reasoner is not None:
            self.reasoner._reset_head_biases()

        # Tandem causal reasoner (card 2dd3400f).  The generic loop above re-inits ALL
        # nn.Linear to std=0.02 and zeros their biases — clobbering the reasoner's
        # scratchpad-validated default init AND its load-bearing head-bias priors (beta
        # positive, neutral gate, mild sharpen, key-scale 5).  Re-apply the reasoner's
        # own faithful init HERE, after the loop, so it survives.  No-op when the tandem
        # is off (self.causal_reasoner is None).
        if self.causal_reasoner is not None:
            self.causal_reasoner.reset_parameters()
            assert self.tandem_gate is not None
            if self.tandem_mlp is not None:
                # 3-way softmax gate (card a7948491).  Suppress the UNTRAINED reasoner
                # logit at init (bias = tandem_gate_bias_init, e.g. -3 -> softmax mass
                # ~0.03) and leave mem + mlp NEUTRAL (0) -> softmax([0, bias, 0]) starts
                # the tandem ~= a mem/mlp blend with the reasoner earning its weight
                # (the 3-way analogue of the 2-way memory-favoring init; keeps text8
                # ON close to OFF).  Bias layout mirrors the forward reshape: per-channel
                # bias (3*d_model,) -> (d_model, 3), scalar bias (3,); expert index 1 =
                # reasoner.  Re-applied after the generic loop zeroed it.
                bias = self.tandem_gate.bias
                nn.init.zeros_(bias)
                with torch.no_grad():
                    bview = bias.view(3) if m.tandem_gate_scalar else bias.view(m.d_model, 3)
                    bview[..., 1] = m.tandem_gate_bias_init
            else:
                # Memory-favoring gate bias (safe default): g = sigmoid(bias) ~= 0.05 at
                # init -> the tandem starts ~= the committed memory backbone (the reasoner
                # earns its weight).  Re-applied after the generic loop zeroed it.
                nn.init.constant_(self.tandem_gate.bias, m.tandem_gate_bias_init)

    def _forward_blocks(
        self,
        x: Tensor,
        states_in: list[DeltaMemoryState] | None = None,
        return_states: bool = False,
    ) -> Tensor | tuple[Tensor, list[DeltaMemoryState]]:
        """Run the block stack, optionally threading per-layer carried states.

        The default path (``states_in=None, return_states=False``) is unchanged:
        each block resets its delta-memory state, with activation checkpointing
        when training.  When carrying state (card 61f900ca) each block is seeded
        from its entry in ``states_in`` and its final state collected into
        ``states_out``; this is an eval-time path, so checkpointing is bypassed
        (it is a no-op under ``no_grad`` anyway and complicates multi-output
        checkpointing).

        Args:
            x: ``(B, T, d_model)``.
            states_in: Optional list of per-block carried
                :class:`DeltaMemoryState` (length = ``len(self.blocks)``), or
                ``None`` for a reset block.
            return_states: When ``True`` also return the per-block final states.

        Returns:
            ``(B, T, d_model)``, or — when ``return_states`` — a
            ``(out, states_out)`` tuple with one final state per block.
        """
        if states_in is None and not return_states:
            for block in self.blocks:
                if self.activation_checkpointing and self.training:
                    x = cast(
                        Tensor,
                        torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False),
                    )
                else:
                    x = cast(Tensor, block(x))
            return x

        states_out: list[DeltaMemoryState] = []
        for i, block in enumerate(self.blocks):
            state_in = states_in[i] if states_in is not None else None
            x, state_out = cast(
                "tuple[Tensor, DeltaMemoryState]",
                block(x, state_in=state_in, return_state=True),
            )
            states_out.append(state_out)
        return x, states_out

    @overload
    def forward(
        self,
        x: Tensor,
        targets: Tensor | None = ...,
        states_in: list[DeltaMemoryState] | None = ...,
        return_states: Literal[False] = ...,
    ) -> tuple[Tensor, Tensor]: ...

    @overload
    def forward(
        self,
        x: Tensor,
        targets: Tensor | None,
        states_in: list[DeltaMemoryState] | None,
        return_states: Literal[True],
    ) -> tuple[Tensor, Tensor, list[DeltaMemoryState]]: ...

    def _forward_impl(
        self,
        x: Tensor,
        targets: Tensor | None,
        states_in: list[DeltaMemoryState] | None,
        return_states: bool,
        *,
        gate_mix: float | Tensor | None = None,
        gate_noise: float = 0.0,
        collect_aux: bool = False,
        aux_query_pos: Tensor | None = None,
        tf_seed: Tensor | None = None,
        commit_seed: bool = True,
        steps: int | None = None,
        memory_only: bool = False,
        reasoner_query_only: bool = False,
    ) -> tuple[Tensor, Tensor, list[DeltaMemoryState] | None, Tensor | None, dict | None]:
        """Shared forward body for the public ``forward`` + the tandem trainer.

        Returns ``(loss, logits, states_out, gate, aux)``.  ``gate`` /``aux`` are
        ``None`` unless the tandem is enabled; the OFF path (``causal_reasoner is
        None``) is byte-for-byte the committed backbone.
        """
        h_embed = self.embed_drop(self.embed(x))
        # Optional cheap local combiner (card ed853f9c): enrich each token with
        # multi-scale causal local context before the memory layers.  ``None`` when
        # front_end="none" -> this line is a pure pass-through (the committed
        # backbone, unchanged).
        #
        # SCOPE (card b1926d5d): by default the conv REPLACES h_embed and feeds ALL
        # pathways (memory + workhorse + gate) — the committed behaviour.  In
        # workhorse-only scope the conv feeds ONLY the MLP-workhorse input
        # (``h_wh_embed``); h_embed stays RAW so the delta-memory k->v binding, the
        # reasoner context, and the gate are on the raw bytes they were tuned on.
        h_wh_embed = h_embed
        if self.front_end is not None:
            if self._front_end_workhorse_only:
                h_wh_embed = self.front_end(h_embed)
            else:
                h_embed = self.front_end(h_embed)
                h_wh_embed = h_embed

        # MEMORY pathway = the existing delta-memory block stack (UNCHANGED).
        # ``_forward_blocks`` returns a ``(x, states)`` tuple whenever a carried state
        # is threaded in (``states_in is not None``) OR ``return_states`` — so unpack
        # in that case and only EXPOSE the carried state when the caller asked for it.
        # This keeps the two committed paths byte-for-byte (states_in=None +
        # return_states=False -> fast path; return_states=True -> unpack) and also
        # supports carrying a state WITHOUT returning it (the tandem answer segment).
        states_out: list[DeltaMemoryState] | None = None
        if return_states or states_in is not None:
            h_mem, carried_out = cast(
                "tuple[Tensor, list[DeltaMemoryState]]",
                self._forward_blocks(h_embed, states_in=states_in, return_states=True),
            )
            if return_states:
                states_out = carried_out
        else:
            h_mem = cast(Tensor, self._forward_blocks(h_embed, states_in=states_in))

        # Optional transient reasoning field (card 9907dc9e): a per-position iterated
        # 2-D reasoner seeded from h_t alone, added back as a residual.  Applied to the
        # memory pathway output (unchanged when the tandem is off).
        if self.reasoner is not None:
            h_mem = h_mem + self.reasoner(h_mem)

        gate: Tensor | None = None
        aux: dict | None = None
        if self.causal_reasoner is None or memory_only:
            # OFF path (byte-for-byte the committed backbone), or the memory-only fast
            # path for non-answer carry segments (skip the reasoner + gate — those
            # segments only build the cross-segment memory state; the delta blocks that
            # produce ``states_out`` are unaffected by the gate).
            h = h_mem
        else:
            # REASONER pathway: per-position, segment-bounded causal walk over the raw
            # tokens (card e4e8a4dc).  Sub-quadratic by construction (bounded window).
            assert self.tandem_gate is not None
            if reasoner_query_only and aux_query_pos is None:
                # Contract: the fast path needs the single query position; do not silently fall
                # back to the full O(K*L^2) walk (that would hide a caller bug — card cff8f5ee).
                raise ValueError(
                    "reasoner_query_only=True requires a per-row aux_query_pos (the single "
                    "query position to run the reasoner walk at)."
                )
            if reasoner_query_only:
                assert aux_query_pos is not None  # guaranteed by the guard above
                # FAST PATH (card cff8f5ee): run the reasoner ONLY at the query position via
                # the O(K*L) ``query_forward`` (numerically PINNED to the full per-position
                # walk gathered at ``query_pos`` — see the query_forward faithfulness tests),
                # then scatter it into a zero ``(B, T, d)``.  The tandem trainer reads every
                # loss + aux ONLY at the query position (answer-CE + locate-CE + walk-aux +
                # gate), so this is loss/grad-IDENTICAL to the full path while cutting the
                # RETAINED walk activations O(K*L^2) -> O(K*L) — deep-K training then fits the
                # routing-stable batch.  ``h_reason`` is 0 at the other positions (never read);
                # the returned logits/gate are valid ONLY at ``aux_query_pos`` in this mode.
                if collect_aux:
                    h_q, aux = cast(
                        "tuple[Tensor, dict]",
                        self.causal_reasoner.query_forward(
                            x, query_pos=aux_query_pos, steps=steps, commit_seed=commit_seed,
                            return_aux=True, tf_seed=tf_seed,
                        ),
                    )
                else:
                    h_q = cast(
                        Tensor,
                        self.causal_reasoner.query_forward(
                            x, query_pos=aux_query_pos, steps=steps, commit_seed=commit_seed,
                            tf_seed=tf_seed,
                        ),
                    )
                h_reason = h_mem.new_zeros(h_mem.shape)
                h_reason[torch.arange(x.shape[0], device=x.device), aux_query_pos] = h_q
            elif collect_aux:
                h_reason, aux = cast(
                    "tuple[Tensor, dict]",
                    self.causal_reasoner(
                        x, steps=steps, commit_seed=commit_seed, return_aux=True,
                        aux_query_pos=aux_query_pos, tf_seed=tf_seed,
                    ),
                )
            else:
                h_reason = cast(
                    Tensor,
                    self.causal_reasoner(
                        x, steps=steps, commit_seed=commit_seed,
                        aux_query_pos=aux_query_pos, tf_seed=tf_seed,
                    ),
                )
            if self.tandem_mlp is None:
                # 2-way (shipped, verified, card 2dd3400f): sigmoid blend {mem, reason}.
                # LIGHT gate over [h_mem ; h_reason ; h_ctx]; h_ctx = the shared embedding
                # / front-end hidden (the routing signal).
                gate_logit = self.tandem_gate(torch.cat([h_mem, h_reason, h_embed], dim=-1))
                if gate_noise > 0.0 and self.training:
                    gate_logit = gate_logit + gate_noise * torch.randn_like(gate_logit)
                g_soft = torch.sigmoid(gate_logit)
                if gate_mix is not None:  # forced fusion (mix warmup, or per-row curriculum).
                    if isinstance(gate_mix, Tensor):
                        # Per-row/per-position directed routing (curriculum): route each
                        # example to its specialist.  Broadcast (B,) -> (B, 1, 1) over T + d.
                        gm = gate_mix
                        while gm.dim() < g_soft.dim():
                            gm = gm.unsqueeze(-1)
                        g = gm.to(g_soft.dtype).expand_as(g_soft)
                    else:
                        g = torch.full_like(g_soft, float(gate_mix))
                elif self._tandem_hard_gate and self.training:
                    # Straight-through hard gate: forward commits to 0/1 (answer depends on
                    # the DOMINANT pathway), soft gradient in backward.
                    g = (g_soft > 0.5).to(g_soft.dtype) + (g_soft - g_soft.detach())
                else:
                    g = g_soft
                h = g * h_reason + (1.0 - g) * h_mem
                # Report/loss on the SOFT gate preference (1=reasoner, 0=memory).
                gate = g_soft.mean(dim=-1)  # (B, T)
            else:
                # 3-way WORKHORSE (card a7948491): softmax over {mem(0), reason(1), mlp(2)}
                # with a plain per-token GatedMLP on the trunk input; fusion h = sum_e
                # g_e * h_e.  h_mlp reads h_embed only (per-position) -> stays LM-leak-safe.
                assert self.tandem_mlp is not None
                # ``h_wh_embed`` == the (optionally conv-mixed) workhorse input: it equals
                # h_embed EXCEPT under workhorse-only front-end scope, where only the
                # workhorse sees the conv (card b1926d5d).
                h_mlp = self.tandem_mlp(h_wh_embed)
                # Optional deeper workhorse (card b1926d5d): stack pre-norm residual GeGLU
                # rungs.  Empty at depth 1 -> this loop is a pure pass-through (the shipped
                # single-block workhorse).  Per-position throughout -> LM-leak-safe preserved.
                for wblock in self.tandem_mlp_blocks:
                    h_mlp = wblock(h_mlp)
                gate_logit = self.tandem_gate(
                    torch.cat([h_mem, h_reason, h_mlp, h_embed], dim=-1)
                )
                if gate_noise > 0.0 and self.training:
                    gate_logit = gate_logit + gate_noise * torch.randn_like(gate_logit)
                # Reshape to expose the expert axis (last dim): scalar gate -> (B,T,3);
                # per-channel gate -> (B,T,d_model,3) (layout matches the bias init).
                if self._tandem_scalar:
                    logits3 = gate_logit
                else:
                    logits3 = gate_logit.view(*gate_logit.shape[:-1], self.d_model, 3)
                g_soft = F.softmax(logits3, dim=-1)
                if gate_mix is not None:
                    if isinstance(gate_mix, Tensor):
                        # Per-row expert INDEX in {0,1,2} -> one-hot route (curriculum /
                        # dissociation probe): row i routes wholly to expert gate_mix[i].
                        idx = gate_mix.long().view(-1)                # (B,)
                        onehot = F.one_hot(idx, 3).to(g_soft.dtype)   # (B, 3)
                        view = [onehot.shape[0]] + [1] * (g_soft.dim() - 2) + [3]
                        g = onehot.view(view).expand_as(g_soft)
                    else:
                        # Forced UNIFORM mix (warmup): every expert 1/3 (all pathways train
                        # equally regardless of the learned gate — the load-bearing rung).
                        g = torch.full_like(g_soft, 1.0 / 3.0)
                elif self._tandem_hard_gate and self.training:
                    # Straight-through: forward one-hot(argmax expert), soft in backward.
                    idx = g_soft.argmax(dim=-1, keepdim=True)
                    hard = torch.zeros_like(g_soft).scatter_(-1, idx, 1.0)
                    g = hard + (g_soft - g_soft.detach())
                else:
                    g = g_soft
                paths = torch.stack([h_mem, h_reason, h_mlp], dim=-1)  # (B, T, d, 3)
                gw = g if not self._tandem_scalar else g.unsqueeze(-2)  # -> broadcast over d
                h = (paths * gw).sum(dim=-1)                            # (B, T, d)
                # Report/loss on the SOFT per-expert weights (B, T, 3): [mem, reason, mlp].
                gate = g_soft if self._tandem_scalar else g_soft.mean(dim=-2)

        h = self.norm_out(h)
        logits = self.lm_head(h)

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.reshape(-1))
        else:
            loss = torch.zeros(1, device=x.device)

        return loss, logits, states_out, gate, aux

    def forward(
        self,
        x: Tensor,
        targets: Tensor | None = None,
        states_in: list[DeltaMemoryState] | None = None,
        return_states: bool = False,
    ) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, list[DeltaMemoryState]]:
        """Forward pass.

        Args:
            x: Token ids, shape ``(B, T)``.
            targets: Target token ids, shape ``(B, T)``.  If provided, the
                cross-entropy loss is computed over the full sequence; if
                ``None`` a zero loss tensor is returned (eval / generation).
            states_in: Optional list of per-block carried
                :class:`DeltaMemoryState` (length = ``delta_layers``) or ``None``
                to reset that block.  Enables cross-segment persistence (cards
                61f900ca + 571d50ec): seed each block's memory matrix AND its
                causal-conv tail from the previous segment's final state.  Default
                ``None`` resets every block — byte-for-byte the committed
                within-sequence behaviour when the conv is disabled.
            return_states: When ``True`` the per-block final states are returned
                as a third element so they can be threaded into the next segment.

        Returns:
            ``(loss, logits)`` by default — scalar ``loss`` and
            ``(B, T, vocab_size)`` ``logits``; the only contract the Trainer
            depends on.  When ``return_states=True``, ``(loss, logits,
            states_out)`` with one final state per block.

        Note:
            Cross-segment carry assumes ``front_end="none"`` (the committed
            backbone): the optional ``multiscale_conv`` front-end left-zero-pads
            each call, so a naive per-segment forward would not match a single
            full-sequence forward at segment boundaries.  The carried state covers
            the delta-memory layers only.  The optional reasoning field
            (``reasoning_enabled=True``) is a *transient* per-position scratchpad
            seeded from each position's hidden state alone — it carries no state
            across positions or segments, so it composes with the carry without
            affecting it (it is applied to ``h`` after the carried memory blocks).
            The optional tandem reasoner (``tandem_enabled=True``) is likewise a
            transient per-position pathway fused by the gate; the standard forward
            uses the LEARNED gate (no forced-mix / noise — those are training-only,
            applied via :meth:`tandem_step`), and it is byte-for-byte the committed
            backbone when ``tandem_enabled=False``.
        """
        loss, logits, states_out, _gate, _aux = self._forward_impl(
            x, targets, states_in, return_states
        )
        if states_out is not None:
            return loss, logits, states_out
        return loss, logits

    def memory_forward(
        self,
        x: Tensor,
        states_in: list[DeltaMemoryState] | None = None,
        return_states: bool = True,
    ) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, list[DeltaMemoryState]]:
        """Memory-only forward (skips the reasoner + gate) for carry segments.

        Used by the tandem trainer to build the cross-segment delta-memory state on
        the non-answer segments (filler / binding) without paying for the
        per-position walk — the delta blocks that produce the carried state are
        unaffected by the gate, so this yields the identical ``states_out``.  Same
        return contract as :meth:`forward`.
        """
        loss, logits, states_out, _g, _a = self._forward_impl(
            x, None, states_in, return_states, memory_only=True
        )
        if states_out is not None:
            return loss, logits, states_out
        return loss, logits

    def tandem_step(
        self,
        x: Tensor,
        targets: Tensor | None = None,
        states_in: list[DeltaMemoryState] | None = None,
        return_states: bool = False,
        *,
        aux_query_pos: Tensor | None = None,
        tf_seed: Tensor | None = None,
        steps: int | None = None,
        collect_aux: bool = True,
        force_gate: float | Tensor | None = None,
        reasoner_query_only: bool = False,
    ) -> dict:
        """Tandem training/eval forward exposing the gate + reasoner aux.

        Used by the :class:`~graph_llm.train.tandem.TandemTrainer` to apply the
        unsupervised gate losses + the reasoner locate-CE / walk-aux.  Does NOT
        touch the standard ``(loss, logits)`` Trainer contract.  In train mode it
        drives the forced-mix warmup (``g = 0.5`` while the train-step counter is
        below ``gate_mix_warmup_steps``) + gate-logit noise + the locate-first seed
        commit from the model's ``_tandem_step`` buffer.

        Returns a dict with ``logits`` ``(B, T, vocab)``, ``gate`` ``(B, T)`` (or
        ``None`` when the tandem is off), ``aux`` (``{"seed_logits", "walk_w"}``
        gathered at ``aux_query_pos``), ``states_out``, and the current ``step`` /
        ``gate_mix``.

        ``reasoner_query_only`` (card cff8f5ee): when True (requires a per-row
        ``aux_query_pos``), run the reasoner ONLY at that query position via the
        O(K*L) ``query_forward`` fast path instead of the full O(K*L^2) per-position
        walk.  Loss/grad-identical to the full path for the tandem trainer (which
        reads every signal at the query position) but cheap enough to train the deep
        walk at the routing-stable batch; the returned ``logits``/``gate`` are then
        valid ONLY at ``aux_query_pos``.  Default False = the full per-position walk
        (byte-for-byte the shipped path).
        """
        if self.causal_reasoner is None:
            raise RuntimeError("tandem_step requires tandem_enabled=True.")
        step = int(self._tandem_step.item())
        if force_gate is not None:
            # Dissociation probe: pin g to 0.0 (memory only) or 1.0 (reasoner only) to
            # measure each pathway in isolation (the interpretable-proof eval).
            gate_mix: float | Tensor | None = force_gate
        else:
            gate_mix = 0.5 if (self.training and step < self._gate_mix_warmup) else None
        gate_noise = self._gate_noise_std if self.training else 0.0
        commit_seed = step >= self._reasoning_locate_warmup
        loss, logits, states_out, gate, aux = self._forward_impl(
            x, targets, states_in, return_states,
            gate_mix=gate_mix, gate_noise=gate_noise, collect_aux=collect_aux,
            aux_query_pos=aux_query_pos, tf_seed=tf_seed, commit_seed=commit_seed,
            steps=steps, reasoner_query_only=reasoner_query_only,
        )
        if self.training:
            self._tandem_step += 1
        return {
            "loss": loss,
            "logits": logits,
            "states_out": states_out,
            "gate": gate,
            "aux": aux,
            "step": step,
            "gate_mix": gate_mix,
        }

    def num_parameters(self, trainable_only: bool = True) -> int:
        """Return total (or trainable-only) parameter count."""
        return sum(
            p.numel()
            for p in self.parameters()
            if (not trainable_only or p.requires_grad)
        )
