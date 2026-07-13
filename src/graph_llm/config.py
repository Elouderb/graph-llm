"""Typed configuration dataclasses + YAML loader.

All hyperparameters live here. No hardcoded values in code paths.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------


@dataclass
class ModelConfig:
    """Architecture hyperparameters shared across all registered models."""

    name: str = "transformer"
    vocab_size: int = 256
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 2048
    max_seq_len: int = 512
    dropout: float = 0.1
    tie_embeddings: bool = True
    use_rope: bool = True
    activation_checkpointing: bool = False

    # Hook for downstream cards: custom embedding init callable name (resolved
    # by the registry; None = default nn.Embedding init).
    embedding_init: str | None = None

    # --- Selective-SSM (Mamba) baseline hyperparameters (card 424e3a8e) ---
    # Ignored by the Transformer baseline; consumed by models/baselines/mamba.py.
    d_state: int = 16                  # SSM state dimension N
    d_conv: int = 4                    # depthwise causal conv kernel width
    expand: int = 2                    # inner expansion factor (d_inner = expand * d_model)
    dt_rank: int | str = "auto"        # rank of the dt projection ("auto" = ceil(d_model/16))
    dt_min: float = 0.001              # min init value for dt softplus bias
    dt_max: float = 0.1                # max init value for dt softplus bias
    # Selective-scan implementation (card 18b14615): "chunkwise" (default — the
    # fast chunked parallel scan: cumulative-decay matmuls within each chunk +
    # inter-chunk state recurrence, T/C sequential steps instead of T) or
    # "sequential" (the original O(T) Python-loop recurrence, kept as the
    # validated reference/oracle).  Both produce the SAME (loss, logits); this
    # only selects the internal scan.  torch.compile is impractical on the raw
    # T-loop (it unrolls the data-dependent loop -> pathological compile time),
    # so chunking is the effective fix.
    mamba_scan: str = "chunkwise"
    # Chunk size for the chunked selective scan.  Smaller than the delta scan's C
    # because the chunked SSM materialises a dense (B, C, C, d_inner, d_state)
    # decay-ratio tensor per chunk — peak VRAM grows ~linearly in C, so C=8 keeps
    # memory modest (fits an 8 GB card at seq 1024, where C=32 OOMs) while still
    # cutting the Python loop length 8x.  Raise it on larger GPUs for more speed.
    mamba_chunk_size: int = 8

    # --- Factorized bilinear (MFB) front-end hyperparameters (card 86347418) ---
    # Ignored by the baselines; consumed by models/components/bilinear_frontend.py
    # and models/bilinear_lm.py.  The 128-d embedding (d_model) is intentionally
    # small; the bulk of the parameter budget lives in the front-end + post-mixer.
    bilinear_window: int = 16          # W: each token interacts with itself + W-1 trailing neighbours
    bilinear_k: int = 2                # MFB factor k (sum-pool group size; k=1 == MLB)
    bilinear_o: int = 4096             # per-token MFB output dim o (default 4096 == 64x64)
    # Interaction mode: "factorized_mfb" (default), "control_linear" (matched-param
    # mixer with NO multiplicative interaction == the ablation null), or
    # "materialized_cnn" (reduce emb -> materialized_reduce_dim, form the small
    # reduce x reduce x W interaction, 2-D CNN over it).
    interaction_mode: str = "factorized_mfb"
    front_end_dropout: float = 0.1     # MFB dropout (applied after power+L2 norm)
    bilinear_offset_weighting: str = "learned"  # "sum" | "learned" aggregation across the W offsets
    # Post-mixer (depthwise-separable 1-D CNN over the sequence + gated MLP):
    post_mixer_layers: int = 4         # depth of the post-mixer stack
    post_mixer_kernel: int = 7         # causal depthwise conv kernel width
    post_mixer_ff_mult: int = 4        # gated-MLP inner expansion (d_ff = mult * trunk width)
    # Trunk width. The embedding (d_model) is intentionally tiny; the post-mixer
    # may run wider so depth*width can scale the model to GPT-1/2 budgets without
    # bloating the embedding table. 0 == "use d_model" (keeps small configs simple).
    post_mixer_width: int = 0
    # materialized_cnn mode only: emb is reduced to this dim before forming the
    # small dense interaction (32x32 == 16x lower memory than the 128x128 naive).
    materialized_reduce_dim: int = 32
    materialized_cnn_channels: int = 32  # 2-D CNN hidden channels over the interaction map

    # --- Gated-DeltaNet delta-rule matrix memory hyperparameters (card e2c6ea95) ---
    # Ignored by the baselines + bilinear_lm; consumed by
    # models/components/delta_memory.py and models/delta_memory_lm.py.  The memory
    # is a fixed-size per-head matrix S of shape (d_k, d_v) updated by a delta rule
    # + forget gate; its size is INDEPENDENT of sequence length (the bounded-memory
    # property).  Scale params via delta_n_heads x delta_layers x delta_head_*_dim.
    delta_layers: int = 6              # depth of the GatedDeltaMemory stack
    delta_n_heads: int = 8             # number of independent memory heads per layer
    delta_head_k_dim: int = 64         # per-head key/query dim d_k (memory capacity <= d_k)
    delta_head_v_dim: int = 64         # per-head value dim d_v (state S is d_k x d_v)
    # Feature map phi applied to keys/queries: "l2" (L2-normalise, the Gated-DeltaNet
    # choice — bounds ||phi(k)||=1 so the delta step is a well-scaled GD step) or
    # "silu_l2" (SiLU then L2-normalise) or "identity" (no map; for the math tests).
    delta_feature_map: str = "l2"
    delta_use_forget_gate: bool = True  # scalar per-head forget gate alpha_t (Gated-DeltaNet);
    #                                     False == ungated DeltaNet (alpha_t == 1).
    # Forget-gate bias initialisation (card 1e9245f4).  alpha = exp(-softplus(proj(x)));
    # at init input ~0 so alpha_init = exp(-softplus(bias)).  Default -4.0 gives
    # softplus(-4) ~= 0.018 -> alpha_init ~= 0.982 (remember-by-default; learns to
    # forget selectively).  The old implicit default was 0.0 (alpha ~= 0.5, halve every
    # token, bindings vanish in ~10-20 tokens — blocks cross-segment recall).
    delta_forget_bias_init: float = -4.0
    # Short causal depthwise conv width applied to the memory INPUT before the
    # q/k/v projections (card 571d50ec).  The delta write is token-LOCAL (k_t, v_t
    # both from x_t), so without local mixing a value position never sees its key
    # and the memory cannot BIND k_i->v_i — the MQAR testbed measured ~0.23 recall
    # capped at width 1 vs ~1.0 at width >= 2.  A residual causal depthwise(+pointwise)
    # conv supplies that local mixing.  ``1`` builds NOTHING (no module, no params,
    # no RNG draws) -> byte-for-byte the committed backbone (back-compat / ablation);
    # the default ``4`` matches Gated-DeltaNet and the validated MQAR recipe.  A
    # causal conv reaches ``width-1`` tokens into the past, so at segment/chunk
    # boundaries the conv's input tail is carried WITH the memory state (see
    # GatedDeltaMemory / DeltaMemoryState) to keep segmented==full exact.
    delta_conv_width: int = 4
    delta_ff_mult: int = 4             # gated-MLP inner expansion between memory layers
    delta_dropout: float = 0.0         # dropout inside the memory mixer + MLP
    # Scan implementation (card 18b14615): "chunkwise" (default — the fast
    # chunkwise-parallel DeltaNet scan: intra-chunk parallel matmuls + inter-chunk
    # state recurrence, T/C sequential steps instead of T), "sequential" (the
    # original O(T) Python-loop recurrence kept as the validated reference/oracle —
    # the chunkwise path is proven bit-equivalent to it within tolerance), or
    # "auto" (== "chunkwise"; alias kept for forward-compat).  Both produce the
    # SAME (loss, logits); this only selects the internal scan.
    delta_scan: str = "chunkwise"
    # Chunk size C for the chunkwise scan.  T/C sequential steps; intra-chunk work
    # is batched matmuls.  32 keeps the per-chunk cumulative forget-gate decay in a
    # safe fp32 dynamic range (the gated WY math divides by cumulative decay) while
    # cutting the sequential loop length by 32x.
    delta_chunk_size: int = 32

    # --- Multi-scale conv front-end (cheap local combiner) hyperparameters
    # (card ed853f9c, realises design note 8b5341f0) ---
    # An OPTIONAL causal multi-scale local-context input stage inserted into
    # delta_memory_lm after the embedding (+ phonological hook), before the memory
    # layers.  A bank of causal depthwise-separable 1-D convs at dyadic widths
    # enriches each token embedding with trailing local context, then a 1x1
    # condense (a per-position projection SHARED across all positions) collapses the
    # scales back to ONE d_model vector per token.  Ignored by every model except
    # delta_memory_lm; default "none" is a byte-for-byte no-op (no module is even
    # constructed), preserving the committed backbone exactly so it is a clean
    # one-flag A/B.
    #
    # "none" (default) == the committed backbone, untouched.
    # "multiscale_conv" == enable the front-end described above.
    front_end: str = "none"
    # Dyadic conv widths {1,2,4,8,16}: width-1 is the pointwise/identity scale (the
    # plain per-token embedding); larger widths each summarise a trailing window.
    # ~log(W) kernels capture multi-scale structure for a fraction of the cost of
    # all contiguous widths 1..16.  Every width must be >= 1.
    conv_widths: list[int] = field(default_factory=lambda: [1, 2, 4, 8, 16])
    # Condense mode collapsing the (B, T, S, d) scale stack to (B, T, d):
    # "concat_proj" (default — reshape to (B, T, S*d) then a shared Linear(S*d, d);
    # mixes across scales AND dims) or "soft_select" (GBST-style: a shared
    # Linear(S*d, S) -> softmax over the S scales -> convex blend of the
    # scale-embeddings, staying in embedding space; interpretable granularity).
    conv_condense: str = "concat_proj"
    # Depthwise-separable convs (depthwise over the window + pointwise channel mix)
    # for cheapness when True; a single full 1-D conv per scale when False.
    conv_depthwise: bool = True
    # Front-end SCOPE (card b1926d5d, coordinator range-separation variant).  False
    # (default) == the conv output REPLACES h_embed and feeds ALL pathways (memory +
    # MLP-workhorse + gate context) — the committed behaviour.  True == the conv feeds
    # ONLY the MLP-workhorse input; the memory pathway, the reasoner context, and the
    # gate keep the RAW embedding.  Rationale: the delta-memory k->v binding was tuned on
    # raw bytes and the reasoner already reads raw tokens, so window-mixing the shared
    # trunk can perturb the validated pathways; routing the local context to ONLY the
    # workhorse (the pathway meant to own local sequential prediction) delivers the
    # range-separation idea without touching the validated pathways.  Only meaningful
    # when the conv front-end is built AND the MLP workhorse exists (tandem_mlp_enabled);
    # a no-op otherwise (nothing to route the conv to).
    front_end_workhorse_only: bool = False
    # Identity-init the conv front-end (card b1926d5d, coordinator).  False (default) ==
    # the committed random init.  True == the front-end starts as a pass-through of the raw
    # embedding at step 0 (MultiScaleConvEmbedding.identity_init: scale 0 -> identity, the
    # condense -> select scale 0 only) — EXACT (bit-for-bit) for condense="concat_proj";
    # near-exact (a near-one-hot softmax over scales) for condense="soft_select" — so
    # enabling the conv starts from the SAME point as the no-front-end backbone and the
    # optimizer OPENS the multi-scale mixing only where it helps — instead of a random
    # condense scrambling the embedding and paying an up-front re-learning cost (which
    # regressed text8 bpb at a fixed budget).  SAFETY (card b1926d5d, cell c3): with the
    # conv feeding all pathways + tandem_mlp_depth >= 2, a RANDOM-init conv silently
    # DESTROYED cross-segment memory retrieval (acc_M -> chance) while text8 bpb still
    # looked fine — identity-init is MANDATORY in the adopted config, not a bpb nicety.
    # Only consulted when the conv front-end is built (front_end="multiscale_conv");
    # a no-op otherwise.
    conv_identity_init: bool = False

    # --- Cross-segment persistent-memory TRAINING hyperparameters (card 61f900ca,
    # piece 3) ---
    # Consumed by train/segmented.py + data/loader.py (ordered-segment stream) and
    # data/synthetic_tasks.py (cross-segment retrieval tasks).  Ignored by the
    # standard Trainer and every model forward — these only configure the OPTIONAL
    # SegmentedTrainer, which teaches delta_memory_lm to USE its carried
    # cross-segment state (the piece-1 forward(x, targets, states_in, return_states)
    # API).  The committed within-sequence training path is byte-for-byte unchanged.
    #
    # Length of each ordered contiguous segment (the truncated-BPTT unit).  The
    # stream is sliced into these in order; the per-layer delta-memory state is
    # carried across segment boundaries.
    segment_len: int = 128
    # Truncated-BPTT window K (Transformer-XL style): gradients flow within a window
    # of K consecutive segments, then the carried state is DETACHED before the next
    # window so the graph never grows past K segments.  Per arXiv 2507.02782 a SHORT
    # window suffices — the dominant lever is EXPOSURE to realistic carried-state
    # distributions (which the carry itself provides), not deep BPTT.  K=1 == detach
    # every segment (pure exposure, no through-segment gradient).
    bptt_window: int = 2
    # Reset the carried state every this many segments (0 == never reset within the
    # stream; text8 is one long stream so the default carries across the whole of it,
    # periodically detached by the K-window).  >0 simulates document boundaries.
    stream_reset_interval: int = 0
    # Probability that a truncated-BPTT window's INITIAL carried state is replaced by
    # a noise-perturbed version of the carried state (state-distribution exposure
    # augmentation, arXiv 2507.02782): broadens the attainable-state distribution the
    # model is trained to read from, so it learns to use long-accumulated states
    # robustly.  0.0 == off (no augmentation; standard carry only).
    state_noise_prob: float = 0.0
    # Std of the Gaussian perturbation added to the carried state when the state-noise
    # augmentation fires (relative to the carried state's own RMS scale).
    state_noise_std: float = 0.1
    # Fraction of training steps that draw a synthetic cross-segment retrieval task
    # (key/passkey in an EARLY segment, query in a LATER segment, answer OUTSIDE the
    # query segment's own window) instead of an ordered text-stream window.  Plain LM
    # perplexity does not force far-back use; these tasks do.  0.0 == off (LM stream
    # only).
    synthetic_task_fraction: float = 0.0
    # Number of decimal digits in the random passkey for synthetic cross-segment tasks.
    # Controls task difficulty: 5 digits = 100k possibilities (default, current
    # behaviour); 1 digit = 10 options (tractable probe for a 5 M-parameter memory).
    # Must be >= 1.  Only affects training when synthetic_task_fraction > 0 and the
    # eval harness when key_digits is forwarded to cross_segment_retrieval_nll_by_distance.
    synthetic_key_digits: int = 5

    # --- Transient reasoning field (R-LM step 1) hyperparameters (card 9907dc9e) ---
    # An OPTIONAL config-flagged transient reasoning block inserted into
    # delta_memory_lm AFTER the memory blocks, BEFORE norm_out + the LM head.  It
    # carries the VALIDATED iterated reasoner (cards 64ea347f/46aa2292/11ab96e7 — the
    # R1/R1b/R2 mechanism: a small 2-D field + head doing content-address + 2-D shift
    # + MANDATORY per-step sharpen, iterated K steps).  The field is a SEPARATE
    # transient scratchpad — DISTINCT from the persistent delta-net memory matrix and
    # the DeltaMemoryState carry (per the R3 finding: reason over a clean scratchpad,
    # NOT the raw lossy memory).  The reasoner runs INDEPENDENTLY per position, seeded
    # ONLY from that position's hidden state h_t (input-seeded local context, NOT the
    # memory matrix — memory->scratchpad is R3b, later), so its contribution at
    # position t depends on no other position => provably causal in an autoregressive
    # LM, and transient (rebuilt every forward, never carried).  Its readout is added
    # (residual) into h to refine the next-token prediction.
    #
    # reasoning_enabled=False (default) builds NOTHING (no module, no params, no RNG
    # draws) -> delta_memory_lm is byte-for-byte the committed backbone, so it is a
    # clean one-flag A/B (the (loss, logits) / (loss, logits, states) contracts and
    # the existing test suite are untouched).
    reasoning_enabled: bool = False
    # Reasoner form.  "grid" == the R2-validated 2-D field/head (content-address +
    # 2-D torus shift + sharpen).  The only supported form for now.
    reasoning_field: str = "grid"
    # The transient field is a reasoning_rows x reasoning_cols toroidal grid of cells
    # (the R2 "2-D Turing machine" geometry).  Small by default (36 cells) — this is a
    # per-position scratchpad, run once per token, so the grid stays cheap.
    reasoning_rows: int = 6
    reasoning_cols: int = 6
    # Iterated reasoning depth K (weight-tied controller steps per position).  This is
    # the iterated-computation budget; the R2 recipe ran the mechanism K steps.
    reasoning_steps: int = 6
    # Per-cell content width d_cell (each grid cell stores a d_cell vector, seeded
    # from h_t).  The controller reads/writes vectors of this width.
    reasoning_d_cell: int = 24
    # Controller (GRUCell) hidden width.
    reasoning_d_ctrl: int = 96
    # Mandatory per-step sharpening floor (gamma >= this).  >= 1.0 keeps the soft head
    # peaked against the 2-D pointer dispersion the research flagged as non-obvious +
    # load-bearing; the R2 recipe used 1.0.
    reasoning_gamma_floor: float = 1.0
    # Cell (row,col) address-key width (content-addressing space).  The controller
    # emits a key of this width; cosine-similarity to the learned per-cell keys drives
    # the soft content address.
    reasoning_key_dim: int = 32

    # --- Tandem: causal reasoner || delta-net memory -> gated fusion (card 2dd3400f) ---
    # An OPTIONAL config-flagged SECOND pathway for delta_memory_lm: the lead-verified
    # v3 WIN causal reasoner (card e4e8a4dc — CausalGRUEncoder + locate-then-walk +
    # separate locate key + gamma_floor 2.0 + walk-aux) run PER POSITION, fused with
    # the existing delta-memory backbone by the UNSUPERVISED gate (card 31fe6b00).
    # This is DISTINCT from the legacy per-position ``reasoning_*`` ReasoningField above
    # (that surface is untouched).
    #
    # tandem_enabled=False (default) builds NOTHING (no reasoner, no gate, no params, no
    # RNG draws) -> delta_memory_lm is byte-for-byte the committed backbone, a clean
    # one-flag A/B (the (loss, logits) / (loss, logits, states) contracts + the existing
    # suite are untouched).
    tandem_enabled: bool = False
    # The reasoner's bounded addressing window (the HARD sub-quadratic constraint): the
    # input is chunked into windows of this length; every position addresses ONLY
    # positions <= t WITHIN its window (never unbounded history — that is the memory's
    # job).  Cost is O(K * reasoning_segment_len) per position -> linear in T.  For the
    # mixed M+R reproduction this is set to the mixed segment length so the whole answer
    # segment is one window (aux/teacher-forcing require a single window).
    reasoning_segment_len: int = 128
    # Iterated walk depth K per position (the WIN reasoner ran K tied steps).
    causal_reasoner_steps: int = 12
    # Mandatory per-step sharpen floor (gamma >= this).  The WIN used 2.0 (deeper
    # sharpening than the legacy field's 1.0 — the extrapolation lever in card e4e8a4dc).
    causal_reasoner_gamma_floor: float = 2.0
    # Address/locate key width (cosine content-addressing space).
    causal_reasoner_key_dim: int = 48
    # CausalGRUEncoder LEFT-padded causal conv kernel width (cheap local features).
    causal_reasoner_conv_kernel: int = 5
    # CausalGRUEncoder unidirectional GRU depth (the content-carrying recurrence).
    causal_reasoner_gru_layers: int = 1
    # #positions ending at the query position the seed pools to read the local context.
    causal_reasoner_query_window: int = 12
    # Read the RHS-name descriptor directly into the move query (key-value pointer follow).
    causal_reasoner_direct_ptr: bool = True
    # Straight-through top-1 on the SEED head.  The locate-first STAGING (commit only
    # after ``reasoning_locate_warmup`` steps) is applied ONLY by ``tandem_step`` (the
    # reproduction trainer), which tracks the step via the ``_tandem_step`` buffer.  The
    # plain ``forward()`` path (e.g. a SegmentedTrainer / Trainer run with tandem on, as
    # in the text8 sanity) has no train-step signal and commits the seed from the first
    # step (``commit_seed=True``) — the committed/eval behaviour, harmless for LM use.
    causal_reasoner_hard_seed: bool = True
    # Gate bias init (memory-favoring safe default): the fusion gate bias is set so
    # ``g = sigmoid(bias) ~= 0.05`` at init -> ``h ~= h_mem``, i.e. the tandem starts
    # CLOSE to the committed memory backbone and the reasoner has to EARN its weight
    # (analogous to the forget-gate remember-by-default init).  This keeps plain-text
    # bpb close to OFF (the reasoner is untrained on real text) and gives the routing a
    # clean starting point (M already at memory; only R must climb to the reasoner).
    tandem_gate_bias_init: float = -3.0
    # Straight-through HARD gate during training (mechanism latitude, card 2dd3400f):
    # the forward fusion commits g to 0/1 (per channel) so the answer depends on the
    # DOMINANT pathway — routing an example to the wrong pathway then PRODUCES A WRONG
    # ANSWER, so the main answer loss supplies the routing DIRECTION the label-free
    # balance loss alone cannot (it is direction-agnostic).  Soft gradient in backward
    # (straight-through) keeps the gate differentiable; the reported/loss gate stays the
    # SOFT preference.  Applied post-mix-release, train-mode only.  Default False.
    tandem_hard_gate: bool = False
    # SCALAR per-position gate (mechanism latitude): when True the gate is a single
    # value per position (Linear -> 1) broadcast over channels, so the WHOLE position
    # routes to one pathway — a wrong route makes the entire fused vector wrong (a clean
    # answer-loss direction signal), and the mean-gate == the routing decision (a crisp
    # interpretable proof).  When False (default) the gate is per-channel (Linear ->
    # d_model), the scratchpad form.  ``h = g (broadcast) * h_reason + (1-g) * h_mem``.
    tandem_gate_scalar: bool = False

    # --- Tandem MLP WORKHORSE: 3rd pathway + 3-way softmax gate (card a7948491) ---
    # An OPTIONAL plain per-token MLP/FFN added as a THIRD gated pathway, generalising the
    # 2-way sigmoid blend {mem, reason} into a 3-way softmax over {mem, reason, mlp} with
    # fusion ``h = sum_e g_e * h_e``.  Motivation (card a7948491 / the conv/matched-OFF
    # findings 4d7dda62): plain per-token capacity is the strongest use of params on
    # ORDINARY prediction; the 2-way gate over-leans on the specialists (memory/reasoner)
    # because it has no default workhorse to route ordinary text to (text8 washed at
    # +0.83%).  The MLP reads the SAME trunk input the memory pathway does (``h_embed``),
    # is stateless + per-position -> LM-leak-safe (logits[t] depend only on tokens <= t).
    #
    # tandem_mlp_enabled=False (default) == the SHIPPED, verified 2-way sigmoid tandem,
    # byte-for-byte UNCHANGED (no MLP module, no gate reshape, no extra RNG draws): the
    # 3-way is a NEW MODE, opt-in, so the committed 2-way recipe still reproduces exactly.
    # Requires tandem_enabled=True (the MLP pathway + 3-way gate only exist when the tandem
    # is on); it is a no-op when tandem_enabled=False (nothing is constructed either way).
    tandem_mlp_enabled: bool = False
    # GatedMLP (GeGLU) inner-expansion for the workhorse pathway (d_ff = mult * d_model).
    tandem_mlp_ff_mult: int = 4
    # Workhorse DEPTH (card b1926d5d): how many GeGLU blocks the MLP workhorse stacks.
    # The shipped workhorse is ONE bare :class:`GatedMLP` on the raw trunk input ``h_embed``
    # (near-unigram context); a stronger, deeper workhorse gives the plain-text pathway more
    # per-token capacity (the workhorse's text win came DESPITE near-unigram input).  ``1``
    # (default) == the shipped single bare GatedMLP, BYTE-FOR-BYTE (no extra module / params /
    # RNG draws) so the depth-1 cell is a clean control.  ``depth>=2`` keeps that bare input
    # GatedMLP and stacks ``depth-1`` PRE-NORM RESIDUAL GeGLU blocks (``x + mlp(norm(x))``,
    # mirroring the delta block's MLP sub-layer) on top -> "stack 2 GeGLU blocks with
    # residual+norm" at depth 2.  Only consulted when ``tandem_mlp_enabled=True`` (a no-op
    # otherwise, like the ff_mult knob).  Every position is processed independently -> the
    # LM-leak-safe (logits[t] depend only on tokens <= t) property is preserved at any depth.
    tandem_mlp_depth: int = 1

    # --- Stacked tandem blocks: vertical stacking of the tandem block (card 3ac77deb) ---
    # An OPTIONAL VERTICAL stack of N whole tandem blocks — each block =
    # {delta memory || causal reasoner || per-position workhorse} -> 3-way softmax gate ->
    # fused per-position hidden; block N's full-sequence fused output is block N+1's input
    # stream and only the FINAL block feeds the (untied) LM head.  Because a single fusion
    # point runs memory and reasoner in PARALLEL on the same input, a single block PROVABLY
    # fails a two-stage compositional (retrieve-then-reason) task; a SECOND block reasoning
    # over the FIRST block's retrieval composes it (validated 3/3 seeds, card 3ac77deb).
    #
    # tandem_blocks=1 (default) builds the SHIPPED single-fusion tandem BYTE-FOR-BYTE (no
    # stacked module, no readback, no extra params / RNG draws / state_dict keys) -> a clean
    # default-off flag.  tandem_blocks>=2 requires tandem_enabled=True and builds the stacked
    # blocks INSTEAD of the single causal_reasoner/gate/mlp.  Two entry points drive the stack:
    # ``stacked_step`` (the segment/task-structured tandem-trainer path, reasoner only at the
    # prediction position) and the plain ``forward`` LM path (card d05da2db, per-position over
    # the full sequence) that the SegmentedTrainer / real-corpus training uses.  In stacked mode
    # the OUTER delta backbone (delta_layers ``self.blocks``) is NOT built — it is unused by both
    # stacked paths (each stacked block owns its own delta memory), so delta_layers is inert in
    # stacked mode and does not inflate the stacked param budget.
    tandem_blocks: int = 1
    # Inner-block routing mode (mitigation ladder): "mix" (the WINNING arm — every block is a
    # full 3-way tandem, unsupervised inner gate), "share" (inner blocks tie the outer gate's
    # weights), or "asym" (memory+workhorse per block, ONE reasoner in the FINAL block).
    stacked_inner_mode: str = "mix"
    # Per-block 3-way gate bias init (stacked).  0.0 (neutral) is the validated stacked
    # default (the controlled M/R/P/C repro routes cleanest from a neutral gate; distinct
    # from the single tandem's memory-favoring ``tandem_gate_bias_init``).
    stacked_gate_bias_init: float = 0.0
    # GatedMLP (GELU) inner expansion for each stacked block's per-position workhorse.
    stacked_mlp_ff_mult: int = 2
    # Structural-necessity variant: inner (block>=2) reasoners address the READBACK-grounded
    # block HIDDEN (via their own GRU re-encoder) instead of their raw-token tap.  False
    # (default) = the validated raw-token reasoner (neither a raw hidden tap nor the readback
    # is structurally required — card 3ac77deb).
    stacked_reason_reads_hidden: bool = False
    # Per-channel stacked gate (rung-1 Stage B scalar-vs-per-channel A/B, card 1cfe0cb8):
    # mirrors the single-fusion tandem's ``tandem_gate_scalar=False`` per-channel gate. False
    # (default) == the SHIPPED, validated SCALAR stacked gate (``Linear(4*d_model, 3)``, one
    # 3-way softmax per position, broadcast over channels) — byte-for-byte the committed
    # stacked recipe (same params / RNG draws / state_dict shapes). True builds a PER-CHANNEL
    # gate instead (``Linear(4*d_model, 3*d_model)`` reshaped to ``(..., d_model, 3)``, an
    # independent 3-way softmax PER CHANNEL so each output channel can route to a different
    # expert), applied identically at both stacked entry points
    # (``StackedTandemBlock.forward``/``stacked_step`` and ``StackedTandemBlock.lm_forward``).
    # Gate-fraction REPORTING always returns a ``(..., 3)`` tensor regardless of this flag (the
    # per-channel gate is reduced by an equal-weight mean over channels before being
    # returned/logged), so downstream consumers (``_stacked_gate_losses``,
    # ``stacked_lm_gate_fractions``, the eval report) need no per-channel-aware branch.
    stacked_gate_per_channel: bool = False
    # INPUT-ROUTED stacked gate (sparse-reasoner-invocation card 75ada834): route BEFORE the
    # experts (standard MoE routing) — each block's 3-way gate reads ``[h_mem ; h_mlp ; ctx]``
    # (3*d_model, SCALAR form only) instead of the shipped ``[h_mem ; h_reason ; h_mlp ; ctx]``
    # (4*d_model).  The gate then decides the reasoner's weight WITHOUT its output, so the
    # (expensive) reasoner walk can be SKIPPED at low-weight positions (see
    # ``stacked_reason_threshold``).  False (default) == the shipped output-mixture gate
    # (``Linear(4*d_model, 3)``) — byte-for-byte the committed stacked recipe (the default
    # construction still draws the identical ``nn.Linear(4*d_model, 3)``).  True narrows the
    # gate to ``Linear(3*d_model, 3)`` at BOTH stacked entry points (``StackedTandemBlock.forward``
    # /``stacked_step`` and ``StackedTandemBlock.lm_forward``).  SCALAR-ONLY: combining it with
    # ``stacked_gate_per_channel`` raises (input-routing applies to the locked SCALAR ladder gate).
    stacked_gate_input_routed: bool = False
    # SPARSE reasoner invocation threshold (sparse-reasoner-invocation card 75ada834): an
    # EVAL/INFERENCE-ONLY sparsity gate on the LM path (``StackedTandemBlock.lm_forward``).
    # Requires ``stacked_gate_input_routed=True`` (the gate must route WITHOUT the reasoner
    # output; raises otherwise, and rejects a negative tau).  At eval, positions whose reasoner
    # gate weight is ``< tau`` SKIP the walk entirely (``h_reason`` treated as 0 there — the fused
    # deviation is bounded by ``tau * ||h_reason||``); positions ``>= tau`` run the walk EXACTLY
    # via the gathered single-query fast path, so the walk cost scales with the invoked-position
    # count.  TRAINING stays DENSE regardless (the unsupervised routing recipe needs dense gates;
    # training compute is not the pain point).  0.0 (default) == OFF (dense everywhere) — a clean
    # default-off flag with no behavioural change.
    stacked_reason_threshold: float = 0.0

    # --- Per-block capacity heterogeneity: capacity ramps across the stack (card 197c6707) ---
    # OPTIONAL per-block override LISTS for the stacked-tandem stack (tandem_blocks >= 2).  The
    # depth-mechanism gate (card f7ffe653) found that only the FINAL block reliably learns the
    # clean type->expert map and that middle-block reasoners atrophy in free LM training — so the
    # user's design direction is a CAPACITY RAMP: bias MEMORY toward EARLY blocks (heads larger
    # up front, shrinking with depth) and REASONING toward LATE blocks (absent/minimal early, full
    # at the final block), with the per-position workhorse/FFN held CONSTANT per block.  These
    # lists express that ramp without touching the homogeneous default.
    #
    # Each list, when not ``None``, must have length == ``tandem_blocks`` and hold values >= 1
    # (positivity is validated in ``delta_memory_lm._build_stacked`` alongside the length check).
    # ``None`` (the default for ALL of them) == HOMOGENEOUS: every block reads the shared
    # ``delta_*``/``causal_reasoner_steps`` scalars exactly as before, so the stacked build is
    # BYTE-FOR-BYTE (``torch.equal``) the committed backbone (the per-block config object is the
    # SAME ``ModelConfig`` instance when no list applies -> identical construction + RNG).  They
    # are inert when the stack is not built (tandem off / tandem_blocks == 1), like every other
    # ``stacked_*`` flag.  ``d_model`` is NOT per-block (the residual stream + fusion + gate are
    # shared width); only the memory's internal head geometry and the reasoner's walk depth ramp.
    #
    # Per-block MEMORY head geometry — each block's ``GatedDeltaMemory`` reads these instead of the
    # shared ``delta_n_heads``/``delta_head_k_dim``/``delta_head_v_dim``.  The memory ``out_proj``
    # still maps back to ``d_model`` at every block, so heterogeneous head counts/dims never change
    # the fused residual-stream width; the per-block carried :class:`DeltaMemoryState` matrix simply
    # has a per-block ``(B, H_i, d_k_i, d_v_i)`` shape (the state carry is already shape-agnostic).
    stacked_mem_heads: list[int] | None = None    # per-block delta_n_heads (memory-heavy early)
    stacked_mem_k_dim: list[int] | None = None    # per-block delta_head_k_dim (per-head key/query)
    stacked_mem_v_dim: list[int] | None = None    # per-block delta_head_v_dim (per-head value)
    # Per-block reasoner PRESENCE.  When not ``None`` this OVERRIDES the ``stacked_inner_mode``
    # -derived reason flags (``mix`` -> all True; ``asym`` -> [False..False, True]) with an
    # EXPLICIT per-block boolean list, so an arbitrary ramp is expressible (e.g. depth-4
    # ``[True, False, False, True]`` = a minimal block-0 reasoner, no walk in the middle blocks,
    # a full walk at the final block — the HET hypothesis that middle blocks which CANNOT walk
    # remove the non-composing routing basins depth 4 fell into).  A block with reasoner disabled
    # builds NO reasoner and its 3-way gate masks the reason expert (the existing ``asym`` path).
    stacked_reason_blocks: list[bool] | None = None
    # Per-block reasoner SIZE — walk depth K (``causal_reasoner_steps``) per block, for the "small
    # early, large late" ramp.  Only meaningful for blocks that actually build a reasoner (a value
    # for a reason-disabled block is inert); ``None`` == the shared ``causal_reasoner_steps`` at
    # every block.
    stacked_reason_steps: list[int] | None = None

    # --- Optional bounded cross-attention READBACK between block 0 and block 1 (card 3ac77deb) ---
    # A small bounded strictly-causal cross-attention that re-grounds block-1's input in the
    # surface (front-end) embeddings block-0's fusion may have abstracted away: Q = block-0
    # fused output, K/V = the bottom front-end embedding stream, window-relative RoPE, a causal
    # BOUNDED window (t attends only to j in (t-window, t], O(T*W)), 1-2 heads.  A learned
    # residual scale (alpha) is init 0, so it is an EXACT no-op at step 0 (readback-on ==
    # readback-off); the output projection keeps its normal random init — zero-initing BOTH
    # factors makes every gradient in the module exactly 0 forever (a dead pathway that can
    # never learn to open; card 3ac77deb review blocker 2).  alpha's gradient is o_proj(o),
    # so the channel opens under training.  The bounded causal window keeps the prediction
    # position leak-free.  Only meaningful when tandem_blocks>=2; a no-op otherwise.
    #
    # tandem_readback=False (default) builds NOTHING (no module, no params, no state_dict keys).
    tandem_readback: bool = False
    tandem_readback_window: int = 32     # bounded causal window W (positions the query attends to)
    tandem_readback_heads: int = 2       # number of readback attention heads (d_model % heads == 0)

    # --- Unsupervised gate-routing losses (card 31fe6b00, applied by TandemTrainer) ---
    # Label-free routing: the main answer loss decides DIRECTION per example; these force
    # DIFFERENTIATION.  The load-bearing rung is the forced-mix warmup (both pathways
    # trained under g=0.5 first) — done IN the model forward via a train-step buffer.
    gate_balance_weight: float = 2.0       # (batch_mean(g) - 0.5)^2 — anti-collapse
    gate_commit_weight: float = 1.0        # mean(g*(1-g)) — per-example g -> 0/1
    gate_commit_anneal_steps: int = 900    # ramp commitment 0 -> full from mix-release
    gate_noise_std: float = 0.5            # Gaussian noise on gate logits (exploration)
    gate_mix_warmup_steps: int = 600       # force g=0.5 for N train steps (both pathways train)

    # --- R-synthetic reasoning-task mixing (card 2dd3400f) ---
    # The mixed M+R stream, the R-example shape (n_chains), and the aux-loss weights
    # (locate-CE / walk-aux) are owned by the reproduction driver's own scoped
    # ``graph_llm.train.tandem.TandemConfig`` (n_chains, locate_weight, walk_weight),
    # NOT by ModelConfig — so those knobs are intentionally absent here.  Only
    # ``reasoning_locate_warmup`` lives on ModelConfig because the MODEL reads it: it
    # gates the causal reasoner's locate-first hard-seed commit inside ``tandem_step``
    # (``commit_seed = _tandem_step >= reasoning_locate_warmup``).
    reasoning_locate_warmup: int = 600     # locate-first: commit the learned seed after N steps


@dataclass
class DataConfig:
    """Dataset and tokenization settings."""

    # source: "synthetic" | "enwik8" | "text8" | "wikitext103" | "tinystories" | "enwik9"
    # enwik8/text8 use the canonical 90M/5M/5M byte split; enwik9 (card 69776c3e, the
    # next rung above text8 for the parameter mini-ladder) uses 990M/5M/5M; wikitext103
    # is loaded byte-level (BPB) for now (token-level ppl seam documented in loader.py).
    source: str = "synthetic"
    encoder: str = "byte"              # "byte" | "bpe" (card e1644700)
    bpe_tokenizer_path: str | None = None  # path to saved BPETokenizer JSON
    data_dir: str = "data/"
    seq_len: int = 512
    batch_size: int = 8
    val_fraction: float = 0.1
    split: str = "train"               # "train" | "val" | "test" (real-corpus loaders)


@dataclass
class TrainConfig:
    """Training-loop and 12-GB toolkit settings."""

    seed: int = 42
    max_steps: int = 10_000
    grad_accumulation_steps: int = 1
    grad_clip: float = 1.0
    lr: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 400
    lr_schedule: str = "cosine"        # "cosine" | "constant"
    mixed_precision: str = "no"        # "no" | "fp16" | "bf16"
    checkpoint_dir: str = "checkpoints/"
    resume_from: str | None = None
    log_every: int = 100

    # --- Unified eval-report hook (card 69776c3e) ---
    # Consumed by SegmentedTrainer to periodically write the unified eval report
    # (val bpb + cross-segment retrieval + in-model reasoning-depth accuracy +
    # routing health -- see eval/report.py) to disk during a long run.
    # ``eval_every=0`` (default) == off, byte-for-byte the existing behaviour (no
    # extra eval passes, no report I/O).
    eval_every: int = 0
    eval_run_dir: str = "eval_reports/"

    # --- Periodic checkpoint / resume for SegmentedTrainer (card 53e55fd2) ---
    # Consumed by SegmentedTrainer only (Trainer never reads it: Trainer has no
    # automatic periodic-checkpoint call in its own train() loop, so this addition
    # is a pure no-op for it).  Reuses ``checkpoint_dir`` / ``resume_from`` above --
    # SegmentedTrainer writes checkpoints via the RNG-capturing
    # train/checkpoint.py machinery (card 69776c3e), a different (superset) format
    # from Trainer.save_checkpoint()'s own scheme.  ``checkpoint_every=0`` (default)
    # == off, byte-for-byte the existing behaviour (no checkpoint I/O, no extra RNG
    # bookkeeping); a checkpoint is written every ``checkpoint_every`` completed
    # steps when > 0 AND ``checkpoint_dir`` is set.
    checkpoint_every: int = 0


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------


@dataclass
class Config:
    """Root configuration object."""

    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


# ---------------------------------------------------------------------------
# Loader helpers
# ---------------------------------------------------------------------------


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge *override* into a copy of *base*."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _dict_to_config(d: dict[str, Any]) -> Config:
    """Populate Config dataclasses from a raw nested dict."""
    cfg = Config()
    if "model" in d:
        for k, v in d["model"].items():
            if hasattr(cfg.model, k):
                setattr(cfg.model, k, v)
    if "data" in d:
        for k, v in d["data"].items():
            if hasattr(cfg.data, k):
                setattr(cfg.data, k, v)
    if "train" in d:
        for k, v in d["train"].items():
            if hasattr(cfg.train, k):
                setattr(cfg.train, k, v)
    return cfg


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> Config:
    """Load a YAML config file and apply optional dict overrides.

    Args:
        path: Path to a YAML config file.
        overrides: Nested dict of values to override after loading.

    Returns:
        A populated :class:`Config` instance.
    """
    with open(path) as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}
    if overrides:
        raw = _deep_merge(raw, overrides)
    return _dict_to_config(raw)


def config_to_dict(cfg: Config) -> dict[str, Any]:
    """Serialize a Config to a plain nested dict."""
    return asdict(cfg)
