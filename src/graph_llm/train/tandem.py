"""Tandem reproduction trainer (card 2dd3400f) — mixed M+R stream, repo pipeline.

Drives the REPO :class:`~graph_llm.models.delta_memory_lm.DeltaMemoryLM` (tandem
ON) on the mixed memory+reasoning stream to reproduce the scratchpad tandem result
through the repo model + config, resolving the scratchpad's integration debts:

* the reasoner is the CAUSAL v3 WIN (card e4e8a4dc — CausalGRUEncoder + clause-END
  locate-then-walk), run per position and leak-free (no bidir future peek);
* routing is UNSUPERVISED (card 31fe6b00 — load-balance + commitment annealed +
  gate-logit noise + forced-mix warmup; NO type labels in training — labels used at
  EVAL only for per-type accuracy + gate-by-type);
* the reasoner is supervised only by the R-SYNTHETIC aux labels (locate-CE at the
  clause-END + per-hop walk-aux along the chain trajectory), applied ONLY on the
  reasoning rows.

Each training step: a mixed M+R batch of ``n_segments`` segments is processed left
to right — the non-answer segments build the cross-segment delta-memory state
(``memory_forward``, no reasoner), the LAST (answer) segment runs ``tandem_step``
(reasoner walk + gate fusion + forced-mix + aux collection).  Loss = answer-CE
(locate-first ramp) + locate-CE (R rows) + walk-aux (R rows) + the unsupervised
gate losses (skipped during the forced-mix warmup).

Sharding: ``--seeds`` splits the seeds across two GPU-pinned workers; ``merge``
combines the per-seed JSONs.  Result JSONs are written incrementally so a killed
run resumes from disk.
"""

from __future__ import annotations

import argparse
import json
import random
import time
import warnings
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from graph_llm.config import Config, DataConfig, ModelConfig, TrainConfig
from graph_llm.data.loader import load_text8_bytes
from graph_llm.data.reasoning_tasks import make_reasoning_example
from graph_llm.models import build_model


@dataclass
class TandemConfig:
    """Repro run configuration (mirrors the scratchpad tandem_harness knobs)."""

    train_r_depths: tuple[int, ...] = (4, 5, 6)     # SHALLOW R training depth
    test_r_depths: tuple[int, ...] = (4, 6, 16, 32)  # eval R in-dist {4,6} + extrap {16,32}
    k_train: int = 8                                 # reasoner K during training (>= max train depth)
    train_steps: int = 3000
    batch_size: int = 48
    seg_len: int = 512
    n_segments: int = 2
    n_chains: int = 2
    d_model: int = 96
    # VERIFIED Stage-3 default (card 2dd3400f comment 159/169): a SINGLE delta layer
    # (like the scratchpad's bare MemoryPathway) so the memory PROVABLY fails the
    # multi-hop chain (dissociation) while still solving the 1-hop cross-segment M
    # retrieval.  A deeper stack learns 2-chain R during the forced-mix and collapses
    # the routing pressure.  (Override with --layers only for ablations.)
    delta_layers: int = 1
    delta_n_heads: int = 4
    delta_head_dim: int = 32
    eval_batches: int = 4
    eval_batch: int = 96
    lr: float = 1e-3
    lr_warmup: int = 200
    log_every: int = 500
    # Reasoner aux + gate recipe (the model reads gate/reasoning knobs from its cfg;
    # these are the loss weights the TRAINER applies).
    locate_warmup: int = 600
    locate_weight: float = 2.0
    walk_weight: float = 1.0
    gate_balance_weight: float = 2.0
    gate_commit_weight: float = 1.0
    gate_commit_anneal: int = 900
    gate_mix_warmup: int = 600
    gate_noise_std: float = 0.5
    # Gate bias init for the REPRO: 0.0 (g~=0.5, scratchpad-faithful) so the balance
    # loss does not have to lift the mean far and pick the direction arbitrarily.  The
    # model DEFAULT (-3, memory-favoring) is the right safe default for a real LM /
    # text8, but the controlled two-type repro routes cleanest from a neutral gate.
    gate_bias_init: float = 0.0
    hard_gate: bool = False   # straight-through hard fusion gate (routing-direction fix)
    gate_scalar: bool = False  # scalar per-position gate (whole position routes to one pathway)
    # Curriculum warmup (rung-3 fix for the M-in-tandem gap): for the first
    # ``type_warmup`` steps, ROUTE each example to its specialist (M->memory, R->
    # reasoner) so each pathway reaches its isolated competence on clean, stable
    # gradient BEFORE the gate learns routing.  Uses the type labels for the WARMUP
    # fusion only (a competence scaffold); the GATE still learns to route unsupervised
    # (no supervised gate loss).  0 == off (use the label-free forced-mix instead).
    type_warmup: int = 0
    # VERIFIED Stage-3 default = FALSE (do NOT flip back to True).  card 2dd3400f
    # comments 168/169 (the TIED-HEAD-WARP finding): with a TIED head (lm_head ==
    # embed^T), the answer-CE decoding the REASONER's features on R rows REWRITES the
    # byte embeddings the MEMORY's associative binding reads -> warps M's input space
    # every batch, collapsing acc_M to ~0.08.  Untying (own head weights) is the fix
    # that reaches the accepted numbers (acc_M=1.0, gate sep +0.99, 3 seeds).  The
    # isolated memory never has reasoner features hit its head, so tying is harmless
    # there — this footgun is SPECIFIC to multi-pathway models sharing an embedding.
    tie_embeddings: bool = False
    # --- MLP WORKHORSE: 3rd pathway + 3-way softmax gate (card a7948491) ---
    # mlp_enabled=True runs the 3-way tandem (memory / reasoner / mlp) on a MIXED M+R+P
    # stream, where P = real text8 rows (the natural 'neither M nor R' plain type the
    # per-token MLP workhorse should own).  Routing target: M->memory(0), R->reasoner(1),
    # P->mlp(2).  Default False keeps the SHIPPED 2-way M+R reproduction exactly (the
    # verified recipe: delta_layers=1, tie_embeddings=False).  With mlp_enabled the memory
    # is still crippled to 1 layer (so it fails multi-hop R -> the R dissociation holds);
    # the tied-head-warp fix (tie_embeddings=False) still applies with a THIRD shared-head
    # pathway (verified: acc_M holds under untied).
    #
    # VERIFIED FINDING (card a7948491): the 3-way needs the CURRICULUM warmup (type_warmup>0),
    # NOT the flat forced-mix.  Under a FLAT forced-mix (g=uniform 1/3) the 1-layer delta
    # MEMORY — a sequence model — greedily captures the EASY plain type P (dense local
    # gradient) instead of the HARD cross-segment M (sparse gradient), so M is ORPHANED
    # (acc_M ~= chance) and the per-token MLP is starved; the balance-to-uniform loss is
    # permutation-invariant so it is satisfied by the WRONG assignment (memory<->P, mlp<->M).
    # (The 2-way avoids this because the reasoner genuinely CANNOT do M, forcing memory->M;
    # here the memory CAN do plain, so nothing forces it onto M under a flat mix.)  The
    # CURRICULUM (route each type to its specialist during type_warmup) pre-specialises the
    # memory on M (never sees P) and the MLP on P, then releases the gate into the correct
    # self-reinforcing MoE basin -> clean M->mem / R->reason / P->mlp routing (all ~1.0).
    # ``main(--mlp)`` therefore defaults type_warmup=gate_mix_warmup; ``--flat-mix`` opts
    # back into the (documented, M-orphaning) flat forced-mix for the ablation.
    mlp_enabled: bool = False
    mlp_ff_mult: int = 4  # GatedMLP inner expansion for the workhorse pathway
    # --- Workhorse strengthening A/B (card b1926d5d) ---
    # front_end: "none" (shipped) | "multiscale_conv" — enable the committed cheap causal
    # multi-scale local combiner so h_embed carries window-mixed local context for ALL
    # pathways (memory + MLP-workhorse) + the gate context.  The reasoner reads raw tokens,
    # so its name-ID locate signal is untouched.  mlp_depth: stack this many GeGLU blocks in
    # the workhorse (1 = shipped single bare GatedMLP; >=2 adds residual+norm rungs).  Both
    # default to the shipped backbone -> the front_end=none / mlp_depth=1 cells are controls.
    front_end: str = "none"
    mlp_depth: int = 1
    # Range-separation variant (card b1926d5d, coordinator): route the conv to the MLP
    # workhorse ONLY, keeping the validated memory + reasoner + gate on raw h_embed.  Only
    # meaningful with front_end != "none" + mlp_enabled.
    front_end_workhorse_only: bool = False
    # Identity-init the conv front-end (card b1926d5d, coordinator): the front-end is an
    # exact pass-through at step 0, so the optimizer opens the multi-scale mixing where it
    # helps rather than paying the random-condense re-learning cost.  front_end != "none".
    conv_identity_init: bool = False
    # --- STACKED TANDEM BLOCKS + stage-wise type-warmup (card 3ac77deb) ---
    # ``tandem_blocks>=2`` builds the VERTICAL stack of whole tandem blocks (each =
    # {memory || reasoner || workhorse -> 3-way gate}); block N's fused output feeds block
    # N+1 and only the FINAL block reaches the head.  Driven by the model's ``stacked_step``
    # (the mixed M/R/P/C stream + type-C compositional data live in the scratchpad probe /
    # card-2 data harness, so the stacked training LOOP is a scratchpad driver — this repo
    # trainer owns the model-build forwarding + the SCHEDULE helpers below).  ``1`` (default)
    # is the shipped single-fusion tandem.
    tandem_blocks: int = 1
    stacked_inner_mode: str = "mix"     # mix | share | asym (see the model)
    # STAGE-WISE type-warmup (the WIN, card 3ac77deb): a compositional routing-coordination
    # basin — the two-stage C strategy earns gradient only if BOTH gates commit at once
    # (block-0 retrieval has no payoff until block-1 uses it), so independent unsupervised
    # inner gates cannot discover it.  With stagewise, WARMUP forces the KNOWN decomposition
    # at BOTH gates (C->mem@block0, C->reason@inner; M/R/P->their expert at both) for
    # ``type_warmup`` steps, then RELEASES both together into the unsupervised commit/anneal.
    # REQUIRES ``type_warmup>0`` (the forced-decomposition window) — guarded by
    # ``_resolve_stagewise_release`` so there is no silent broken default.
    stagewise_warmup: bool = False
    # NON-stagewise inner-gate warmup (rung a, the FAILED mitigation kept for the ablation):
    # inner blocks train under a uniform forced-mix for this many steps, then release.
    inner_mix_warmup: int = 1000
    # Optional bounded cross-attention readback between block 0 and block 1 (default off).
    readback: bool = False
    readback_window: int = 32
    readback_heads: int = 2
    # --- PROGRESSIVE R-depth curriculum (card cff8f5ee; recipe from card 0a98292b) ---
    # The shipped fixed ``train_r_depths=(4,5,6)`` + ``k_train=8`` caps the reasoner at a
    # TRAINED-WALK-BUDGET CLIFF at hop==k_train: within budget the per-hop walk is fine, but
    # PAST it the weight-tied walk controller was never trained, so it drifts OOD and acc_R@16/32
    # collapse to ~0.42/0.37 in the 3-way tandem (card a7948491 comment 195).  The verified cure
    # (card 0a98292b): RAMP the training chain depth shallow-first then deepen
    # (``r_depth_start`` -> ``r_depth_max``) while ``k_train`` covers the deepest chain, so the
    # controller GROKS the deep walk (per-hop ~0.98 SUSTAINED, extrapolates 28->32).  A naive
    # WIDE curriculum (jump straight to 2..32) instead COLLAPSES pre-transition — the ramp is
    # LOAD-BEARING.  The grokking transition is stochastic per seed, so budget 5000-6000 steps and
    # a GENTLER ramp (long ``r_depth_ramp_steps``) for margin.
    #
    # ``r_depth_ramp=False`` (default) -> ``depth = rng.choice(train_r_depths)`` EXACTLY as
    # shipped (the disabled branch of ``_ramp_depth`` makes the identical ``rng.choice`` call, so
    # the 2-way / non-ramp reproduction is byte-for-byte RNG-preserving).
    r_depth_ramp: bool = False
    r_depth_start: int = 4         # ramp floor (shallowest chain depth)
    r_depth_max: int = 28          # ramp ceiling (deepest chain depth trained)
    r_depth_ramp_steps: int = 3500  # linear ramp length (gentler = more margin over the
    #                                 stochastic grokking transition — card 0a98292b note)
    # Hold at ``r_depth_start`` for this many steps BEFORE ramping, so the M/R/P routing basin
    # can COMMIT at shallow depth first (Option A: through the forced type-warmup AND the
    # unsupervised commitment anneal) — else the deep-R answer-loss destabilises the still-fluid
    # gate and P drifts onto the memory (the de-risk #1 routing crossing, card cff8f5ee).
    # ``-1`` (default, SENTINEL) = AUTO: resolved by ``_resolve_ramp_delay`` to
    # ``release + gate_commit_anneal`` (the point routing has committed) — so a raw
    # ``TandemConfig(r_depth_ramp=True)`` is SAFE by default, not the broken delay-0 schedule.
    # An explicit ``0`` ramps concurrently with the type-warmup (the Option-B ablation) and is
    # WARNED; any explicit value shorter than the safe point is also warned.
    r_depth_ramp_delay: int = -1


def _ramp_depth(cfg: TandemConfig, step: int, rng: random.Random) -> int:
    """Per-step training chain depth for the progressive R-depth curriculum (card cff8f5ee).

    When ``cfg.r_depth_ramp`` is off, returns ``rng.choice(cfg.train_r_depths)`` — the shipped
    uniform draw, with IDENTICAL RNG consumption so the non-ramp reproduction is byte-for-byte
    unchanged.  When on, ramps the max chain depth ``r_depth_start`` -> ``r_depth_max`` linearly
    over ``r_depth_ramp_steps`` (after an optional ``r_depth_ramp_delay`` hold), then samples a
    depth uniformly from the WIDENING window ``[r_depth_start, cur_max]``.

    The window WIDENS (it does not slide): every trained depth from the floor up to the current
    max stays in the mix, so the weight-tied walk controller never FORGETS the shallow walks as
    it deepens.  A sliding window collapses acc_R@4/6/16 (measured: the reasoner overfits the
    deep end and inverts the depth profile).  This matches the standalone 0.976 recipe
    (card 0a98292b: ``depth = randint(floor, cur_max)``).
    """
    if not cfg.r_depth_ramp:
        return rng.choice(cfg.train_r_depths)
    t = step - cfg.r_depth_ramp_delay
    if t <= 0:
        cur_max = cfg.r_depth_start
    else:
        frac = min(1.0, t / max(1, cfg.r_depth_ramp_steps))
        span = cfg.r_depth_max - cfg.r_depth_start
        cur_max = cfg.r_depth_start + int(round(frac * span))
    return rng.randint(cfg.r_depth_start, cur_max)


def _resolve_ramp_delay(cfg: TandemConfig) -> int:
    """Resolve/validate the depth-ramp delay against the ROUTING-LOCKS-BEFORE-RAMP invariant.

    The safe point is where the M/R/P routing basin has committed: the forced-routing release
    (``type_warmup`` under the curriculum, else the model's ``gate_mix_warmup``) PLUS the
    unsupervised commitment anneal.  Starting the depth ramp before that lets the deep-R
    answer-loss destabilise the still-fluid gate and P drifts onto the memory (the de-risk #1
    routing crossing, card cff8f5ee).

    ``r_depth_ramp_delay < 0`` (the default SENTINEL) resolves to that safe point.  An explicit
    delay shorter than it is honoured (e.g. the Option-B concurrent ablation) but WARNED, so the
    footgun is never silent — including for a raw ``TandemConfig`` that never went through the CLI.
    """
    release = cfg.type_warmup if cfg.type_warmup > 0 else cfg.gate_mix_warmup
    safe = release + cfg.gate_commit_anneal
    if cfg.r_depth_ramp_delay < 0:
        return safe
    if cfg.r_depth_ramp_delay < safe:
        warnings.warn(
            f"r_depth_ramp_delay={cfg.r_depth_ramp_delay} starts the depth ramp before the "
            f"routing basin commits (safe >= {safe} = release {release} + gate_commit_anneal "
            f"{cfg.gate_commit_anneal}); the 3-way M/R/P routing may cross (P->memory) — the "
            f"de-risk #1 failure (card cff8f5ee).  Use the auto default (r_depth_ramp_delay<0) "
            f"unless intentionally running the concurrent Option-B ablation.",
            stacklevel=2,
        )
    return cfg.r_depth_ramp_delay


# ---------------------------------------------------------------------------
# Stacked-tandem stage-wise type-warmup schedule (card 3ac77deb)
# ---------------------------------------------------------------------------
# The compositional (retrieve-then-reason) type earns gradient only if BOTH gates commit at
# once — independent unsupervised inner gates cannot bootstrap the two-stage strategy.  The
# WIN is a STAGE-WISE warmup that forces the KNOWN decomposition at BOTH gates during the
# ``type_warmup`` window (C->mem@block0, C->reason@inner; M/R/P->their expert at both), then
# RELEASES both together into the unsupervised commit/anneal.  These are pure, unit-testable
# schedule functions (the ``_ramp_depth`` / ``_resolve_ramp_delay`` discipline); the stacked
# training loop itself (mixed M/R/P/C stream) is a scratchpad driver that consumes them + the
# model's ``stacked_step``, since the type-C data generator lives with the card-2 data harness.

# Type ids on the mixed stream: M=0, R=1, P=2, C=3 (compositional).  Block-0 expert target per
# type: M->mem(0), R->reason(1), P->mlp(2), C->mem(0) (RETRIEVE); the stage-wise inner blocks
# force C->reason(1) (WALK the retrieved chain).
_TYPE_ID_C = 3
_INNER_C_EXPERT = 1  # reasoner


def _resolve_stagewise_release(cfg: TandemConfig) -> int:
    """Resolve/validate the stage-wise warmup RELEASE step (card 3ac77deb).

    Stage-wise warmup forces the known C decomposition at every gate for ``type_warmup``
    steps, then releases them together.  It REQUIRES ``type_warmup > 0`` — without the
    forced-decomposition window the two-stage strategy has no way to bootstrap (independent
    unsupervised inner gates provably cannot discover it), so this raises rather than
    silently training the known-broken schedule.  Returns the release step (``type_warmup``),
    which is also the routing-lock point the depth-ramp delay composes with
    (``_resolve_ramp_delay`` already uses ``release = type_warmup`` when it is > 0, so routing
    locks before the ramp).  A non-stagewise stacked run has no such coupling and returns the
    later of the outer (``type_warmup``) and inner (``inner_mix_warmup``) releases.
    """
    if not cfg.stagewise_warmup:
        return max(cfg.type_warmup, cfg.inner_mix_warmup)
    if cfg.type_warmup <= 0:
        raise ValueError(
            "stagewise_warmup requires type_warmup > 0 (the forced-decomposition window that "
            "commits BOTH gates at once); with type_warmup=0 the two-stage compositional "
            "strategy cannot bootstrap (card 3ac77deb).  Set a positive type_warmup."
        )
    return cfg.type_warmup


def _stacked_block_controls(
    cfg: TandemConfig, step: int, kind: torch.Tensor, type_id: torch.Tensor
) -> tuple[list[torch.Tensor | None], list[bool]]:
    """Per-block ``(force_gate, gate_mix)`` for a stacked training step (card 3ac77deb).

    STAGE-WISE: during ``type_warmup`` force the KNOWN decomposition at BOTH gates — block 0
    to ``kind`` (M->mem, R->reason, P->mlp, C->mem RETRIEVE), inner blocks to ``kind`` but
    with C->reason (WALK the retrieved chain) — then both release together.  DEFAULT (rung a):
    block 0 type-warmup; inner blocks uniform forced-mix until ``inner_mix_warmup``, then free.
    """
    n = int(cfg.tandem_blocks)
    force_gate: list[torch.Tensor | None] = [None] * n
    gate_mix: list[bool] = [False] * n
    if cfg.stagewise_warmup:
        if step < cfg.type_warmup:
            force_gate[0] = kind
            inner = kind.clone()
            inner[type_id == _TYPE_ID_C] = _INNER_C_EXPERT  # C -> reasoner at inner blocks
            for bi in range(1, n):
                force_gate[bi] = inner
    else:
        if step < cfg.type_warmup:
            force_gate[0] = kind
        for bi in range(1, n):
            if step < cfg.inner_mix_warmup:
                gate_mix[bi] = True
    return force_gate, gate_mix


def _stacked_gate_losses(
    cfg: TandemConfig, gates: list[torch.Tensor], step: int
) -> torch.Tensor:
    """Unsupervised 3-way gate losses (label-free) per gate that is OUT of its warmup.

    Load-balance to uniform (1/3) anti-collapse + annealed per-example entropy commitment.
    Stage-wise: inner gates release WITH the outer at ``type_warmup``; default: inner gates
    release at ``inner_mix_warmup``.  ``gates`` are the per-block ``(B, 3)`` gate distributions
    at the prediction position (``stacked_step``'s ``gates``).
    """
    loss = gates[0].new_zeros(())
    for bi, g in enumerate(gates):
        if bi == 0 or cfg.stagewise_warmup:
            release = cfg.type_warmup
        else:
            release = cfg.inner_mix_warmup
        if step < release:
            continue
        if cfg.gate_balance_weight > 0:
            loss = loss + cfg.gate_balance_weight * ((g.mean(0) - 1.0 / 3.0) ** 2).sum()
        if cfg.gate_commit_weight > 0:
            commit_w = cfg.gate_commit_weight
            if cfg.gate_commit_anneal > 0:
                commit_w = commit_w * min(
                    1.0, max(0, step - release) / max(1, cfg.gate_commit_anneal)
                )
            ent = -(g.clamp_min(1e-9).log() * g).sum(-1).mean()
            loss = loss + commit_w * ent
    return loss


def build_tandem_model(cfg: TandemConfig, device: torch.device) -> torch.nn.Module:
    """Build the repo DeltaMemoryLM with the tandem ON + the validated memory config.

    The MEMORY pathway uses the validated cross-segment retrieval config (forget gate
    OFF + silu_l2 feature map — card 61f900ca; with the default remember-by-default
    forget alpha~0.98 the k->v binding still decays over a long segment, so the M task
    needs no decay).  The REASONER window == ``seg_len`` (one window per segment).
    """
    m = ModelConfig(
        name="delta_memory_lm",
        vocab_size=256,
        d_model=cfg.d_model,
        delta_layers=cfg.delta_layers,
        delta_n_heads=cfg.delta_n_heads,
        delta_head_k_dim=cfg.delta_head_dim,
        delta_head_v_dim=cfg.delta_head_dim,
        delta_conv_width=4,
        delta_chunk_size=32,
        delta_feature_map="silu_l2",
        delta_use_forget_gate=False,
        delta_dropout=0.0,
        delta_scan="chunkwise",
        delta_ff_mult=4,
        dropout=0.0,
        max_seq_len=cfg.seg_len,
        tie_embeddings=cfg.tie_embeddings,
        tandem_enabled=True,
        reasoning_segment_len=cfg.seg_len,
        causal_reasoner_steps=cfg.k_train,
        causal_reasoner_gamma_floor=2.0,
        causal_reasoner_key_dim=cfg.d_model // 2,
        causal_reasoner_conv_kernel=5,
        causal_reasoner_gru_layers=1,
        causal_reasoner_query_window=12,
        causal_reasoner_hard_seed=True,
        gate_balance_weight=cfg.gate_balance_weight,
        gate_commit_weight=cfg.gate_commit_weight,
        gate_commit_anneal_steps=cfg.gate_commit_anneal,
        gate_noise_std=cfg.gate_noise_std,
        # With the curriculum, disable the model's internal label-free forced-mix
        # (the trainer drives per-type-directed routing during ``type_warmup`` instead).
        gate_mix_warmup_steps=(0 if cfg.type_warmup > 0 else cfg.gate_mix_warmup),
        reasoning_locate_warmup=cfg.locate_warmup,
        tandem_gate_bias_init=cfg.gate_bias_init,
        tandem_hard_gate=cfg.hard_gate,
        tandem_gate_scalar=cfg.gate_scalar,
        tandem_mlp_enabled=cfg.mlp_enabled,
        tandem_mlp_ff_mult=cfg.mlp_ff_mult,
        tandem_mlp_depth=cfg.mlp_depth,
        front_end=cfg.front_end,
        front_end_workhorse_only=cfg.front_end_workhorse_only,
        conv_identity_init=cfg.conv_identity_init,
        # Stacked-tandem topology (card 3ac77deb): a no-op at the default tandem_blocks=1.
        tandem_blocks=cfg.tandem_blocks,
        stacked_inner_mode=cfg.stacked_inner_mode,
        tandem_readback=cfg.readback,
        tandem_readback_window=cfg.readback_window,
        tandem_readback_heads=cfg.readback_heads,
    )
    full = Config(
        model=m,
        data=DataConfig(seq_len=cfg.seg_len, batch_size=cfg.batch_size),
        train=TrainConfig(lr=cfg.lr, max_steps=cfg.train_steps, mixed_precision="no"),
    )
    return build_model(full).to(device)


# Routing target per type: M->memory(0), R->reasoner(1), P->mlp-workhorse(2).
_KIND_TO_EXPERT: dict[str, int] = {"M": 0, "R": 1, "P": 2}


def load_text8_split(train_frac: float = 0.8) -> tuple[np.ndarray, np.ndarray]:
    """Load the cached text8 byte stream and split into (train, eval) slices.

    Used ONLY for the 3-way plain (P) type — real text8 rows are the natural 'neither M
    nor R' example the per-token MLP workhorse should own.  Train/eval are disjoint
    contiguous slices so plain-eval rows are never memorised windows.
    """
    arr = load_text8_bytes(DataConfig(source="text8"))
    if arr is None:  # pragma: no cover - requires the cached corpus
        raise RuntimeError(
            "text8 cache (data/text8.bin) required for the 3-way plain type (mlp_enabled)."
        )
    cut = int(len(arr) * train_frac)
    return arr[:cut], arr[cut:]


def _plain_example_arrays(
    text8: np.ndarray, np_rng: np.random.Generator, seg_len: int, n_segments: int, walk_steps: int
) -> tuple[np.ndarray, int, int]:
    """One PLAIN example: a contiguous text8 window reshaped into ``n_segments`` segments;
    the answer is a byte in the latter half of the LAST segment (predicted leak-free from
    ``answer_pos-1`` = ordinary next-byte prediction).  ``locate``/``walk`` are absent."""
    span = n_segments * seg_len
    start = int(np_rng.integers(0, len(text8) - span - 1))
    seg = text8[start : start + span].reshape(n_segments, seg_len).astype(np.int64)
    last = n_segments - 1
    answer_pos = int(np_rng.integers(seg_len // 2, seg_len))  # >=1 -> leak-free assert holds
    answer = int(seg[last, answer_pos])
    return seg, answer_pos, answer


def make_stream_batch(
    rng: random.Random,
    kinds: list[str],
    depth: int,
    cfg: TandemConfig,
    text8: np.ndarray | None,
    np_rng: np.random.Generator | None,
) -> dict:
    """Build a mixed batch from per-row kinds ('M'|'R'|'P') as numpy arrays.

    M/R rows reuse :func:`make_reasoning_example` (identical RNG stream to the shipped
    ``make_batch`` when no P rows are present, so the 2-way reproduction is unchanged); P
    rows draw real text8 windows via ``np_rng`` (untouched when there are no P rows).
    """
    b = len(kinds)
    seg_len, n_seg, walk_steps = cfg.seg_len, cfg.n_segments, cfg.k_train
    seg_tokens = np.zeros((b, n_seg, seg_len), dtype=np.int64)
    answer_pos = np.zeros(b, dtype=np.int64)
    answer = np.zeros(b, dtype=np.int64)
    is_r = np.zeros(b, dtype=bool)
    locate = np.full(b, -1, dtype=np.int64)
    walk = np.full((b, walk_steps + 1), -1, dtype=np.int64)
    kind = np.zeros(b, dtype=np.int64)  # expert index: 0=M, 1=R, 2=P
    for i, k in enumerate(kinds):
        if k == "P":
            if text8 is None or np_rng is None:  # pragma: no cover - guarded by caller
                raise ValueError("plain (P) rows require a text8 buffer + np_rng.")
            seg, ap, ans = _plain_example_arrays(text8, np_rng, seg_len, n_seg, walk_steps)
            seg_tokens[i] = seg
            answer_pos[i] = ap
            answer[i] = ans
            kind[i] = 2
        else:
            ex = make_reasoning_example(rng, k, depth, seg_len, n_seg, cfg.n_chains, walk_steps)
            seg_tokens[i] = ex.seg_tokens
            answer_pos[i] = ex.answer_pos
            answer[i] = ord(ex.answer)
            is_r[i] = ex.kind == "R"
            locate[i] = ex.locate_offset
            walk[i] = ex.walk_traj
            kind[i] = _KIND_TO_EXPERT[ex.kind]
    return {
        "seg_tokens": seg_tokens,
        "answer_seg": n_seg - 1,
        "answer_pos": answer_pos,
        "answer": answer,
        "is_r": is_r,
        "locate": locate,
        "walk": walk,
        "kind": kind,
    }


def _to_dev(nb: dict, device: torch.device) -> dict:
    return {
        "seg": torch.from_numpy(nb["seg_tokens"]).to(device),
        "answer_pos": torch.from_numpy(nb["answer_pos"]).to(device),
        "answer": torch.from_numpy(nb["answer"]).to(device),
        "is_r": torch.from_numpy(nb["is_r"]).to(device),
        "locate": torch.from_numpy(nb["locate"]).to(device),
        "walk": torch.from_numpy(nb["walk"]).to(device),
        "kind": torch.from_numpy(nb["kind"]).to(device),
        "answer_seg": nb["answer_seg"],
    }


def _build_state(model: torch.nn.Module, seg: torch.Tensor, answer_seg: int):
    """Carry the delta-memory state over the non-answer segments (memory only)."""
    states = None
    for si in range(answer_seg):
        _, _, states = model.memory_forward(seg[:, si], states, return_states=True)
    return states


def _answer_gather(t: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    b = t.shape[0]
    return t[torch.arange(b, device=t.device), pos]


def _train_one(cfg: TandemConfig, seed: int, device: torch.device, verbose: bool) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    # Progressive-ramp invariants (enforced at the RUNTIME path so they hold for EVERY entry
    # point, not just the CLI — card cff8f5ee review):
    # (1) the walk budget must cover the deepest trained chain so its walk_traj reaches the root
    #     (else the per-hop walk-aux never sees the answer clause) — card 0a98292b.
    # (2) routing must lock before the ramp (_resolve_ramp_delay): a raw TandemConfig gets the
    #     safe auto delay rather than the silent, known-broken delay-0 schedule.
    if cfg.r_depth_ramp:
        if cfg.k_train < cfg.r_depth_max:
            raise ValueError(
                f"r_depth_ramp requires k_train ({cfg.k_train}) >= r_depth_max "
                f"({cfg.r_depth_max}) so the walk trajectory reaches the deepest chain's root."
            )
        cfg.r_depth_ramp_delay = _resolve_ramp_delay(cfg)
    model = build_tandem_model(cfg, device)
    params = model.num_parameters()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(1, cfg.lr_warmup))
    )
    rng = random.Random(1000 + seed)
    # PLAIN (P) type = real text8 rows (3-way only).  Disjoint train/eval slices; separate
    # numpy RNGs so the M/R RNG stream (hence the 2-way reproduction) is untouched.
    text8_tr = text8_ev = None
    np_tr = np_ev = None
    if cfg.mlp_enabled:
        text8_tr, text8_ev = load_text8_split()
        np_tr = np.random.default_rng(2024 + seed)
        np_ev = np.random.default_rng(9090 + seed)
    stream_kinds = ("M", "R", "P") if cfg.mlp_enabled else ("M", "R")
    model.train()
    t0 = time.time()
    for step in range(cfg.train_steps):
        kinds = [rng.choice(stream_kinds) for _ in range(cfg.batch_size)]
        depth = _ramp_depth(cfg, step, rng)
        nb = make_stream_batch(rng, kinds, depth, cfg, text8_tr, np_tr)
        d = _to_dev(nb, device)
        states = _build_state(model, d["seg"], d["answer_seg"])
        is_r = d["is_r"]
        # LEAK FIX: predict the answer at answer_pos from PRED_POS = answer_pos-1 (the
        # answer byte is IN the input at answer_pos; gathering there would let the model
        # copy it).  Query the reasoner + gather the gate at the same PRED_POS.
        # INVARIANT: answer_pos >= 1 (the generators always place >=1 prefix byte before
        # the answer).  pred_pos MUST be answer_pos-1 (leak-free); if answer_pos were 0 the
        # clamp would gather logits[0], which depends on token[0] (the answer byte) ->
        # reintroduces the copy leak.  Assert rather than clamp-hide.
        assert bool((d["answer_pos"] >= 1).all()), "answer_pos==0 would reintroduce the copy leak"
        pred_pos = d["answer_pos"] - 1
        tf = torch.where(is_r, d["locate"], torch.full_like(d["locate"], -1))
        # Curriculum: during type_warmup, force per-row specialist routing so each pathway
        # trains on its own type cleanly.  3-way -> the per-row EXPERT INDEX (M->0, R->1,
        # P->2); 2-way -> is_r (R->reasoner g=1, M->memory g=0).
        if cfg.type_warmup > 0 and step < cfg.type_warmup:
            force = d["kind"].to(torch.long) if cfg.mlp_enabled else is_r.to(torch.float32)
        else:
            force = None
        out = model.tandem_step(
            d["seg"][:, d["answer_seg"]], None, states, return_states=False,
            aux_query_pos=pred_pos, tf_seed=tf, steps=cfg.k_train, force_gate=force,
            # FAST PATH (card cff8f5ee): run the reasoner only at the answer position (the
            # trainer reads every loss/aux there) -> O(K*L) not O(K*L^2), so the deep k_train
            # walk trains at the routing-stable batch instead of OOMing.  Loss/grad-identical
            # to the full path (query_forward is numerically pinned to it).
            reasoner_query_only=True,
        )
        gstep = out["step"]
        ans_logits = _answer_gather(out["logits"], pred_pos)
        ans_loss = F.cross_entropy(ans_logits, d["answer"])
        ans_w = min(1.0, (gstep + 1) / max(1, cfg.locate_warmup))
        loss = ans_w * ans_loss

        if bool(is_r.any()):
            seed_logits = out["aux"]["seed_logits"]        # (B, L)
            loss = loss + cfg.locate_weight * F.cross_entropy(seed_logits[is_r], d["locate"][is_r])
            walk_w = out["aux"]["walk_w"]                   # (B, K, L)
            traj = d["walk"]                               # (B, K+1)
            wr, tr = walk_w[is_r], traj[is_r]
            k = wr.shape[1]
            nll = wr.new_zeros(())
            for j in range(k):
                p = wr[:, j].gather(1, tr[:, j + 1].clamp_min(0).unsqueeze(1)).squeeze(1)
                nll = nll + (-(p.clamp_min(1e-9).log())).mean()
            loss = loss + cfg.walk_weight * (nll / max(1, k))

        # Unsupervised gate losses at the prediction position (skip during forced mix).
        gate_ans = _answer_gather(out["gate"], pred_pos)  # (B,) 2-way | (B, 3) 3-way
        if out["gate_mix"] is None:
            commit_w = cfg.gate_commit_weight
            if cfg.gate_commit_anneal > 0:
                release = cfg.type_warmup if cfg.type_warmup > 0 else cfg.gate_mix_warmup
                since = gstep - release
                commit_w = commit_w * min(1.0, max(0, since) / max(1, cfg.gate_commit_anneal))
            if cfg.mlp_enabled:
                # 3-way: load-balance to UNIFORM 1/3 (label-free anti-collapse) + low
                # per-example ENTROPY commitment (each example -> one expert).
                if cfg.gate_balance_weight > 0:
                    balance = ((gate_ans.mean(0) - 1.0 / 3.0) ** 2).sum()
                    loss = loss + cfg.gate_balance_weight * balance
                if cfg.gate_commit_weight > 0:
                    ent = -(gate_ans.clamp_min(1e-9).log() * gate_ans).sum(-1).mean()
                    loss = loss + commit_w * ent
            else:
                if cfg.gate_balance_weight > 0:
                    loss = loss + cfg.gate_balance_weight * (gate_ans.mean() - 0.5) ** 2
                if cfg.gate_commit_weight > 0:
                    loss = loss + commit_w * (gate_ans * (1.0 - gate_ans)).mean()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if verbose and (step % cfg.log_every == 0 or step == cfg.train_steps - 1):
            mixflag = "MIX" if out["gate_mix"] is not None else "   "
            mixflag = f"{mixflag} d{depth:>2d}" if cfg.r_depth_ramp else mixflag
            with torch.no_grad():
                if cfg.mlp_enabled:
                    # Per-type mass on the TARGET expert (want each near 1): M->mem(0),
                    # R->reason(1), P->mlp(2).
                    kind = d["kind"]
                    mM, mR, mP = kind == 0, kind == 1, kind == 2
                    gmm = float(gate_ans[mM][:, 0].mean()) if bool(mM.any()) else float("nan")
                    grr = float(gate_ans[mR][:, 1].mean()) if bool(mR.any()) else float("nan")
                    gpp = float(gate_ans[mP][:, 2].mean()) if bool(mP.any()) else float("nan")
                    print(f"    [tandem s{seed}] step {step:>5d}/{cfg.train_steps} {mixflag} "
                          f"loss {loss.item():.4f} (ans {ans_loss.item():.3f}) "
                          f"route[M->mem={gmm:.2f} R->rea={grr:.2f} P->mlp={gpp:.2f}]", flush=True)
                else:
                    # g -> 1 = reasoner, g -> 0 = memory; want gate on R > gate on M.
                    gr = float(gate_ans[is_r].mean()) if bool(is_r.any()) else float("nan")
                    gm = float(gate_ans[~is_r].mean()) if bool((~is_r).any()) else float("nan")
                    print(f"    [tandem s{seed}] step {step:>5d}/{cfg.train_steps} {mixflag} "
                          f"loss {loss.item():.4f} (ans {ans_loss.item():.3f}) "
                          f"gate[R={gr:.3f} M={gm:.3f} sep={gr - gm:+.3f}]", flush=True)

    erng = random.Random(7777 + seed)
    if cfg.mlp_enabled:
        return _finish_3way(model, cfg, seed, device, erng, text8_ev, np_ev, params, t0, verbose)
    accM, gM = _eval_M(model, cfg, erng, device)
    accR: dict[int, float] = {}
    gR: dict[int, float] = {}
    for depth in cfg.test_r_depths:
        a, g = _eval_R_depth(model, cfg, depth, erng, device)
        accR[depth] = a
        gR[depth] = g
    # Dissociation probe (the interpretable proof): each pathway in isolation.
    d0 = cfg.train_r_depths[0]
    mem_M, _ = _eval_M(model, cfg, erng, device, force_gate=0.0)      # memory-only on M
    mem_R, _ = _eval_R_depth(model, cfg, d0, erng, device, force_gate=0.0)   # memory-only on R
    rea_M, _ = _eval_M(model, cfg, erng, device, force_gate=1.0)      # reasoner-only on M
    rea_R, _ = _eval_R_depth(model, cfg, d0, erng, device, force_gate=1.0)   # reasoner-only on R
    dissoc = {"memory_only": {"M": mem_M, "R": mem_R}, "reasoner_only": {"M": rea_M, "R": rea_R}}
    dt = time.time() - t0
    if verbose:
        rstr = " ".join(f"R{depth}={accR[depth]:.3f}" for depth in cfg.test_r_depths)
        gstr = " ".join(f"R{depth}={gR[depth]:.3f}" for depth in cfg.test_r_depths)
        print(f"    [tandem s{seed}] params={params:,} ({dt:.0f}s) accM={accM:.3f} {rstr} "
              f"| gate[M={gM:.3f} {gstr}]", flush=True)
        print(f"      DISSOCIATION memory_only[M={mem_M:.3f} R={mem_R:.3f}] "
              f"reasoner_only[M={rea_M:.3f} R={rea_R:.3f}]", flush=True)
    return {
        "seed": seed, "params": params, "acc_M": accM,
        "acc_R": {str(depth): accR[depth] for depth in cfg.test_r_depths},
        "gate_M": gM, "gate_R": {str(depth): gR[depth] for depth in cfg.test_r_depths},
        "dissociation": dissoc, "wall_seconds": dt,
    }


def _fmt3(v) -> str:
    return f"[m={v[0]:.2f} r={v[1]:.2f} p={v[2]:.2f}]"


def _finish_3way(model, cfg, seed, device, erng, text8_ev, np_ev, params, t0, verbose) -> dict:
    """Eval + dissociation + result for the 3-way M+R+P run (routing over {mem, reason, mlp})."""
    accM, gvM_ = _eval_M(model, cfg, erng, device, text8_ev, np_ev)
    accP, gvP_ = _eval_P(model, cfg, erng, device, text8_ev, np_ev)
    gvM, gvP = np.asarray(gvM_), np.asarray(gvP_)  # (3,) expert distributions
    accR: dict[int, float] = {}
    gvR: dict[int, np.ndarray] = {}
    for depth in cfg.test_r_depths:
        a, gv = _eval_R_depth(model, cfg, depth, erng, device, text8_ev, np_ev)
        accR[depth] = a
        gvR[depth] = np.asarray(gv)
    # Dissociation probe: force each expert (memory / reasoner / mlp) on each type.
    d0 = cfg.train_r_depths[0]
    dissoc: dict[str, dict[str, float]] = {}
    for name, e in (("memory_only", 0), ("reasoner_only", 1), ("mlp_only", 2)):
        mM, _ = _eval_M(model, cfg, erng, device, text8_ev, np_ev, force_gate=e)
        mR, _ = _eval_R_depth(model, cfg, d0, erng, device, text8_ev, np_ev, force_gate=e)
        mP, _ = _eval_P(model, cfg, erng, device, text8_ev, np_ev, force_gate=e)
        dissoc[name] = {"M": mM, "R": mR, "P": mP}
    dt = time.time() - t0
    if verbose:
        rstr = " ".join(f"R{depth}={accR[depth]:.3f}" for depth in cfg.test_r_depths)
        print(f"    [tandem s{seed}] params={params:,} ({dt:.0f}s) "
              f"accM={accM:.3f} accP={accP:.3f} {rstr}", flush=True)
        print(f"      route M={_fmt3(gvM)} P={_fmt3(gvP)} "
              f"R{cfg.test_r_depths[0]}={_fmt3(gvR[cfg.test_r_depths[0]])}", flush=True)
        print("      DISSOCIATION " + " ".join(
            f"{n}[M={dissoc[n]['M']:.2f} R={dissoc[n]['R']:.2f} P={dissoc[n]['P']:.2f}]"
            for n in ("memory_only", "reasoner_only", "mlp_only")), flush=True)
    return {
        "seed": seed, "params": params, "mlp_enabled": True,
        "acc_M": accM, "acc_P": accP,
        "acc_R": {str(depth): accR[depth] for depth in cfg.test_r_depths},
        "route_M": gvM.tolist(), "route_P": gvP.tolist(),
        "route_R": {str(depth): gvR[depth].tolist() for depth in cfg.test_r_depths},
        "dissociation": dissoc, "wall_seconds": dt,
    }


@torch.no_grad()
def _eval_type(model, cfg, kinds, depth, k_eval, rng, device, text8=None, np_rng=None, force_gate=None):
    """Per-type eval.  Returns (accuracy, gate_report): a scalar reasoner-preference (2-way)
    or the mean (3,) expert distribution [mem, reason, mlp] (3-way) at the prediction pos."""
    model.eval()
    correct = total = 0
    gate_reports: list = []
    for _ in range(cfg.eval_batches):
        nb = make_stream_batch(rng, kinds, depth, cfg, text8, np_rng)
        d = _to_dev(nb, device)
        states = _build_state(model, d["seg"], d["answer_seg"])
        # answer_pos-1 is the leak-free prediction position (see _train_one); answer_pos is
        # never 0 for any exercised config (the generators always emit a prefix byte).
        assert bool((d["answer_pos"] >= 1).all()), "answer_pos==0 would reintroduce the copy leak"
        pred_pos = d["answer_pos"] - 1
        fg = force_gate
        if cfg.mlp_enabled and force_gate is not None:
            # 3-way dissociation: force_gate is an EXPERT INDEX -> per-row one-hot route.
            fg = torch.full((len(kinds),), int(force_gate), dtype=torch.long, device=device)
        out = model.tandem_step(
            d["seg"][:, d["answer_seg"]], None, states, return_states=False,
            aux_query_pos=None, tf_seed=None, steps=k_eval, collect_aux=False, force_gate=fg,
        )
        ans_logits = _answer_gather(out["logits"], pred_pos)  # LEAK FIX: predict from answer_pos-1
        g_ans = _answer_gather(out["gate"], pred_pos)  # (B,) 2-way | (B, 3) 3-way
        if cfg.mlp_enabled:
            gate_reports.append(g_ans.mean(0).cpu().numpy())  # (3,)
        else:
            gate_reports.append(float(g_ans.mean().item()))
        correct += int((ans_logits.argmax(-1) == d["answer"]).sum().item())
        total += len(kinds)
    if cfg.mlp_enabled:
        return correct / total, np.mean(np.stack(gate_reports), axis=0)  # (3,)
    return correct / total, float(np.mean(gate_reports))


def _eval_M(model, cfg, rng, device, text8=None, np_rng=None, force_gate=None):
    kinds = ["M"] * cfg.eval_batch
    return _eval_type(model, cfg, kinds, cfg.train_r_depths[0], cfg.k_train, rng, device,
                      text8, np_rng, force_gate)


def _eval_R_depth(model, cfg, depth, rng, device, text8=None, np_rng=None, force_gate=None):
    kinds = ["R"] * cfg.eval_batch
    return _eval_type(model, cfg, kinds, depth, max(depth, cfg.k_train), rng, device,
                      text8, np_rng, force_gate)


def _eval_P(model, cfg, rng, device, text8, np_rng, force_gate=None):
    kinds = ["P"] * cfg.eval_batch
    return _eval_type(model, cfg, kinds, cfg.train_r_depths[0], cfg.k_train, rng, device,
                      text8, np_rng, force_gate)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def report_3way(seed_results: list[dict], cfg: TandemConfig) -> dict:
    """Aggregate the 3-way M+R+P run: per-type accuracy + routing over {mem, reason, mlp}.

    Bars (card a7948491): acc_M / acc_R in-dist >= 0.9; M->mem, R->reason, P->mlp routing
    predominant (each type's target expert is the argmax, and P's mlp mass > 0.5).  acc_P
    has NO accuracy bar (plain text next-byte is inherently uncertain) — only ROUTING.
    """
    test_depths = [str(x) for x in cfg.test_r_depths]

    def ms(xs):
        return (float(np.mean(xs)), float(np.std(xs))) if xs else (float("nan"), float("nan"))

    accM = [r["acc_M"] for r in seed_results]
    accP = [r["acc_P"] for r in seed_results]
    accR = {d: [r["acc_R"][d] for r in seed_results] for d in test_depths}
    routeM = np.mean([r["route_M"] for r in seed_results], axis=0)  # (3,) [mem, reason, mlp]
    routeP = np.mean([r["route_P"] for r in seed_results], axis=0)
    in_dist = [d for d in test_depths if int(d) <= max(cfg.train_r_depths)]
    routeR_in = np.mean(
        [[r["route_R"][d] for d in in_dist] for r in seed_results], axis=(0, 1)
    )  # (3,)
    indist_R = float(np.mean([np.mean(accR[d]) for d in in_dist])) if in_dist else float("nan")
    mM, mP = ms(accM), ms(accP)
    both_indist = mM[0] >= 0.9 and indist_R >= 0.9
    routes = (
        int(np.argmax(routeM)) == 0
        and int(np.argmax(routeR_in)) == 1
        and int(np.argmax(routeP)) == 2
        and float(routeP[2]) > 0.5
    )

    print("\n" + "=" * 78)
    print("TANDEM 3-WAY REPRO (memory / reasoner / mlp) — per-type acc + routing")
    print(f"  seeds={[r['seed'] for r in seed_results]}  train R {cfg.train_r_depths} -> test {cfg.test_r_depths}")
    print("=" * 78)
    print(f"  acc_M = {mM[0]:.3f}+/-{mM[1]:.3f}   acc_P = {mP[0]:.3f}+/-{mP[1]:.3f} (routing-only, no bar)")
    for d in test_depths:
        m, s = ms(accR[d])
        print(f"  acc_R@{d:<3} = {m:.3f}+/-{s:.3f}")
    print(f"  route M {_fmt3(routeM)}  (mem mass {routeM[0]:.3f})")
    print(f"  route R {_fmt3(routeR_in)}  (reason mass {routeR_in[1]:.3f}, in-dist)")
    print(f"  route P {_fmt3(routeP)}  (mlp mass {routeP[2]:.3f})")
    verdict = (
        "POSITIVE: acc_M/acc_R>=0.9 in-dist AND M->mem R->reason P->mlp routing"
        if both_indist and routes else
        f"PARTIAL/REPORT: both_indist={both_indist} routes={routes}"
    )
    print(f"  VERDICT: {verdict}")
    return {
        "config": asdict(cfg), "seeds": seed_results,
        "acc_M_mean": mM[0], "acc_R_indist_mean": indist_R, "acc_P_mean": mP[0],
        "acc_R": {d: ms(accR[d]) for d in test_depths},
        "route_M": routeM.tolist(), "route_R_indist": routeR_in.tolist(), "route_P": routeP.tolist(),
        "both_indist_ge_0.9": both_indist, "routes_to_specialists": routes, "verdict": verdict,
    }


def report(seed_results: list[dict], cfg: TandemConfig) -> dict:
    """Aggregate per-seed results -> per-type table + gate routing + verdict."""
    if cfg.mlp_enabled:
        return report_3way(seed_results, cfg)
    test_depths = [str(x) for x in cfg.test_r_depths]

    def ms(xs):
        return (float(np.mean(xs)), float(np.std(xs))) if xs else (float("nan"), float("nan"))

    accM = [r["acc_M"] for r in seed_results]
    accR = {d: [r["acc_R"][d] for r in seed_results] for d in test_depths}
    gM = [r["gate_M"] for r in seed_results]
    gR = {d: [r["gate_R"][d] for r in seed_results] for d in test_depths}
    deepest = test_depths[-1]

    mM = ms(accM)
    routing = {"gate_M": ms(gM)[0], f"gate_R@{deepest}": ms(gR[deepest])[0]}
    routing["separation"] = routing[f"gate_R@{deepest}"] - routing["gate_M"]

    # In-distribution (<= max train depth) vs extrapolation.
    in_dist = [d for d in test_depths if int(d) <= max(cfg.train_r_depths)]
    indist_R = float(np.mean([np.mean(accR[d]) for d in in_dist])) if in_dist else float("nan")
    both_indist = mM[0] >= 0.9 and indist_R >= 0.9
    routes = routing["separation"] >= 0.5 and routing[f"gate_R@{deepest}"] > routing["gate_M"]

    print("\n" + "=" * 78)
    print("TANDEM REPRO (repo pipeline) — per-type accuracy (mean+/-std over seeds)")
    print(f"  seeds={[r['seed'] for r in seed_results]}  train R {cfg.train_r_depths} -> test {cfg.test_r_depths}")
    print("=" * 78)
    print(f"  acc_M = {mM[0]:.3f}+/-{mM[1]:.3f}")
    for d in test_depths:
        m, s = ms(accR[d])
        print(f"  acc_R@{d:<3} = {m:.3f}+/-{s:.3f}   gate_R@{d} = {ms(gR[d])[0]:.3f}")
    print(f"  gate_M = {routing['gate_M']:.3f}  gate_R@{deepest} = {routing[f'gate_R@{deepest}']:.3f}  "
          f"separation = {routing['separation']:+.3f}")
    verdict = (
        "POSITIVE: both types >=0.9 in-dist AND gate routes (sep>=0.5)"
        if both_indist and routes else
        f"PARTIAL/REPORT: both_indist={both_indist} routes={routes}"
    )
    print(f"  VERDICT: {verdict}")
    return {
        "config": asdict(cfg), "seeds": seed_results,
        "acc_M_mean": mM[0], "acc_R_indist_mean": indist_R,
        "acc_R": {d: ms(accR[d]) for d in test_depths},
        "gate_routing": routing, "both_indist_ge_0.9": both_indist,
        "gate_routes_ge_0.5": routes, "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@dataclass
class _Args:
    seeds: tuple[int, ...] = field(default_factory=lambda: (0, 1, 2))


def _build_run_config(argv: list[str] | None = None) -> tuple[TandemConfig, tuple[int, ...], str]:
    """Parse the (non-merge) CLI args into ``(TandemConfig, seeds, out_path)``.

    Split out of ``main`` so the derived-defaults logic — the --mlp / --r-ramp recipe
    resolution — is UNIT-TESTABLE without launching training (card cff8f5ee review).
    ``argv=None`` reads ``sys.argv``.  The ramp delay is left as the sentinel (or the explicit
    ``--ramp-delay``); it is resolved to the safe lock-before-ramp value at runtime in
    ``_train_one`` (single source of truth for every entry point).
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--out", default="")
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--seg", type=int, default=None, help="seg_len override (smaller = faster)")
    ap.add_argument("--layers", type=int, default=None, help="delta_layers (ablation; default 1)")
    ap.add_argument("--tie-embeddings", dest="tie", action="store_true",
                    help="TIE the head to the embedding (known-broken for the tandem — ablation only)")
    ap.add_argument("--mlp", action="store_true",
                    help="3-way MLP-workhorse mode (card a7948491): memory/reasoner/mlp over a "
                         "mixed M+R+P text8 stream (default OFF = the shipped 2-way M+R repro)")
    ap.add_argument("--flat-mix", action="store_true",
                    help="3-way: use the FLAT forced-mix instead of the curriculum (documented "
                         "M-orphaning ablation — the memory grabs plain, M fails; --mlp default "
                         "is the curriculum)")
    ap.add_argument("--r-ramp", dest="r_ramp", action="store_true",
                    help="progressive R-depth curriculum (card cff8f5ee): ramp the train chain "
                         "depth r_depth_start->r_depth_max so the reasoner GROKS the deep walk "
                         "(lifts acc_R@32 ~0.37->~0.9+).  Defaults k_train=30 + steps=5500 + "
                         "train_r_depths=(start,max).  Default OFF = the shipped fixed depths.")
    ap.add_argument("--k-train", dest="k_train", type=int, default=None,
                    help="reasoner walk budget K per position (must be >= max train depth)")
    ap.add_argument("--ramp-delay", dest="ramp_delay", type=int, default=None,
                    help="hold shallow for N steps before the depth ramp.  Default (omitted) = "
                         "AUTO = type_warmup + gate_commit_anneal (deepen AFTER routing commits = "
                         "Option A); 0 = ramp from step 0 = Option B ablation (warned).")
    ap.add_argument("--ramp-steps", dest="ramp_steps", type=int, default=None,
                    help="depth-ramp length in steps (gentler = more margin over the grok transition)")
    ap.add_argument("--r-depth-max", dest="r_depth_max", type=int, default=None,
                    help="ramp ceiling (deepest chain depth trained; default 28)")
    ap.add_argument("--type-warmup", dest="type_warmup", type=int, default=None,
                    help="forced per-type routing warmup steps (M->mem/R->reason/P->mlp).  Longer "
                         "FIRMS the routing basin before the depth ramp (the memory is kept off "
                         "plain text longer, so it cannot out-capture the MLP on P) — the "
                         "robustness knob for the 3-way routing seed-fragility.  Default under "
                         "--mlp = gate_mix_warmup (600).")
    ap.add_argument("--front-end", dest="front_end", default=None,
                    choices=["none", "multiscale_conv"],
                    help="input-stage local combiner (card b1926d5d): 'multiscale_conv' enables "
                         "the committed cheap causal multi-scale conv so h_embed carries local "
                         "context for the memory + MLP-workhorse + gate (reasoner reads raw "
                         "tokens).  Default (omitted) = the shipped 'none'.")
    ap.add_argument("--mlp-depth", dest="mlp_depth", type=int, default=None,
                    help="workhorse GeGLU depth (card b1926d5d): 1 = shipped single bare "
                         "GatedMLP; >=2 stacks residual+norm rungs for more per-token capacity. "
                         "Only meaningful with --mlp.  Default (omitted) = 1.")
    ap.add_argument("--front-end-workhorse-only", dest="front_end_workhorse_only",
                    action="store_true",
                    help="range-separation variant (card b1926d5d): route the conv front-end to "
                         "the MLP workhorse ONLY (memory + reasoner + gate keep raw h_embed).  "
                         "Requires --front-end multiscale_conv + --mlp.")
    ap.add_argument("--conv-identity-init", dest="conv_identity_init", action="store_true",
                    help="identity-init the conv front-end (card b1926d5d): exact pass-through "
                         "at step 0, so the optimizer opens the multi-scale mixing where it "
                         "helps.  Requires --front-end multiscale_conv.")
    args = ap.parse_args(argv)
    seeds = tuple(int(s) for s in args.seeds.split(",") if s.strip())

    if args.smoke:
        # Inherits the VERIFIED defaults (delta_layers=1, tie_embeddings=False); only the
        # cheap-run knobs are overridden.
        cfg = TandemConfig(train_steps=120, batch_size=16, seg_len=256, k_train=8,
                           d_model=64, eval_batches=2, eval_batch=64,
                           test_r_depths=(4, 6, 16), log_every=40, locate_warmup=40,
                           gate_mix_warmup=30, gate_commit_anneal=40, lr_warmup=20)
        print(">>> SMOKE", flush=True)
    else:
        cfg = TandemConfig()
        print(">>> FULL", flush=True)
    if args.batch is not None:
        cfg.batch_size = args.batch
    if args.steps is not None:
        cfg.train_steps = args.steps
    if args.seg is not None:
        cfg.seg_len = args.seg
    if args.layers is not None:
        cfg.delta_layers = args.layers
    if args.tie:
        cfg.tie_embeddings = True
    if args.mlp:
        cfg.mlp_enabled = True
        # VERIFIED (card a7948491): the 3-way needs the CURRICULUM (type_warmup), not the flat
        # forced-mix (which orphans M — the memory grabs the easy plain type).  Default it on
        # for --mlp unless --flat-mix opts into the documented failing ablation.
        if not args.flat_mix and cfg.type_warmup == 0:
            cfg.type_warmup = cfg.gate_mix_warmup
    # Explicit type-warmup override (robustness knob) — must precede the --r-ramp block so the
    # ramp-delay default (type_warmup + commit_anneal) picks up the overridden value.
    if args.type_warmup is not None:
        cfg.type_warmup = args.type_warmup
    # PROGRESSIVE R-depth curriculum (card cff8f5ee).  Must run AFTER the --mlp block so the
    # ramp-delay default can read the type_warmup it sets.
    if args.r_depth_max is not None:
        cfg.r_depth_max = args.r_depth_max
    if args.r_ramp:
        cfg.r_depth_ramp = True
        # VERIFIED RECIPE (card cff8f5ee).  Acc is 3-seed robust: acc_M 1.000, acc_R
        # 0.997/0.999/0.992/0.940+/-0.026@D32 (every seed clears in-dist>=0.95 + D32>=0.85; vs
        # the pre-card shipped D32 ~0.37).  ROUTING is the seed-fragile axis: at type_warmup
        # 600 it is 2/3 (one seed lets P drift onto the memory during the ramp), so the robust
        # default is type_warmup 1200 — it FIRMS the P->mlp basin BEFORE the ramp (the memory
        # is kept off plain text long enough that it cannot out-capture the per-token MLP on
        # P), verified to flip the failing seed to a full PASS (route M/P -> 1.0, D32 0.969).
        # --steps/--k-train/--batch/--type-warmup/--ramp-* still override.  The query_forward
        # fast path (reasoner_query_only, wired into tandem_step) makes it cheap (~1.5GB/seed).
        if args.type_warmup is None and cfg.mlp_enabled:
            cfg.type_warmup = max(cfg.type_warmup, 1200)
        if args.steps is None:
            cfg.train_steps = 8500
        if args.k_train is None and cfg.k_train < cfg.r_depth_max + 2:
            cfg.k_train = 30
        # Batch 40 = card a7948491's proven routing-STABLE batch (the 3-way M/P routing is
        # batch-sensitive: at batch 24 the permutation-invariant gate lets P drift onto the
        # memory and orphan M — measured; batch 40 holds).  The fast path removes the VRAM
        # cost that previously forced a smaller batch.
        if args.batch is None:
            cfg.batch_size = 40
        # Train depths now span start..max: report()/eval read train_r_depths[0]=start (the
        # shallow M/P/dissociation depth) and max()=r_depth_max (the in-dist boundary).
        cfg.train_r_depths = (cfg.r_depth_start, cfg.r_depth_max)
        # NOTE: the ramp DELAY (routing-locks-before-ramp: deepen only after the M/R/P basin has
        # committed) is left as the SENTINEL here and resolved at runtime in _train_one via
        # _resolve_ramp_delay — so the invariant holds for EVERY entry point, not just this CLI.
    if args.ramp_delay is not None:
        cfg.r_depth_ramp_delay = args.ramp_delay
    if args.ramp_steps is not None:
        cfg.r_depth_ramp_steps = args.ramp_steps
    if args.k_train is not None:
        cfg.k_train = args.k_train
    # Workhorse strengthening A/B (card b1926d5d): additive, default to the shipped backbone.
    if args.front_end is not None:
        cfg.front_end = args.front_end
    if args.mlp_depth is not None:
        cfg.mlp_depth = args.mlp_depth
    if args.front_end_workhorse_only:
        cfg.front_end_workhorse_only = True
    if args.conv_identity_init:
        cfg.conv_identity_init = True
    return cfg, seeds, args.out


def main() -> None:
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "merge":
        ap = argparse.ArgumentParser()
        ap.add_argument("merge")
        ap.add_argument("shards", nargs="+")
        ap.add_argument("--out", default="")
        args = ap.parse_args()
        merged_seeds: list[dict] = []
        cfg_dict: dict = {}
        for p in args.shards:
            sh = json.load(open(p))
            cfg_dict = sh.get("config", cfg_dict)
            merged_seeds.extend(sh.get("seeds", sh.get("seed_results", [])))
        # report() only needs the depth grids; reconstruct those (typed) and default the rest.
        # (Consequence: a MERGE output's "config" block carries DEFAULT k_train/type_warmup —
        # cite the per-seed run JSONs, not the merge output, for the recipe.)
        cfg = TandemConfig(
            train_r_depths=tuple(int(x) for x in cfg_dict.get("train_r_depths", (4, 5, 6))),
            test_r_depths=tuple(int(x) for x in cfg_dict.get("test_r_depths", (4, 6, 16, 32))),
            mlp_enabled=bool(cfg_dict.get("mlp_enabled", False)),
        )
        out = report(merged_seeds, cfg)
        if args.out:
            json.dump(out, open(args.out, "w"), indent=2, default=str)
            print(f"wrote {args.out}")
        return

    cfg, seeds, out = _build_run_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}"
          + (f" {torch.cuda.get_device_name(0)}" if device.type == "cuda" else ""), flush=True)
    print(f"    delta_layers={cfg.delta_layers} tie_embeddings={cfg.tie_embeddings} "
          f"mlp_enabled={cfg.mlp_enabled} type_warmup={cfg.type_warmup} "
          f"front_end={cfg.front_end} mlp_depth={cfg.mlp_depth} "
          f"wh_only={cfg.front_end_workhorse_only}", flush=True)
    if cfg.r_depth_ramp:
        delay_str = str(cfg.r_depth_ramp_delay) if cfg.r_depth_ramp_delay >= 0 else "auto"
        print(f"    r_depth_ramp: widening [{cfg.r_depth_start}, cur_max] with cur_max "
              f"{cfg.r_depth_start}->{cfg.r_depth_max} over {cfg.r_depth_ramp_steps} steps "
              f"(delay {delay_str}), k_train={cfg.k_train}, steps={cfg.train_steps}", flush=True)

    seed_results: list[dict] = []
    for seed in seeds:
        seed_results.append(_train_one(cfg, seed, device, verbose=True))
        if out:  # write incrementally so a killed run resumes from disk
            json.dump({"config": asdict(cfg), "seeds": seed_results}, open(out, "w"),
                      indent=2, default=str)
            print(f"  wrote {out} ({len(seed_results)} seeds)", flush=True)

    report(seed_results, cfg)
    print("done", flush=True)


if __name__ == "__main__":
    main()
