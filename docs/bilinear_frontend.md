# Phase 2 — Factorized Bilinear (MFB) Windowed Front-End

This document is the design + math contract for the project's **first novel
component** (card **86347418**): a windowed *bilinear* ("outer-product") token
front-end, implemented memory-safely via **factorized low-rank Hadamard pooling
(MFB)**, plus a standalone, **ablatable** language model (`bilinear_lm`) built
around it.

The component answers one falsifiable question: *does explicit second-order
(multiplicative) local token interaction help?* — measured against the Phase 0b
matched-parameter baselines (`docs/baselines.md`) via an interaction-mode switch
and a parameter-matched control. It is deliberately **local** (window `W`): it
tests the multiplicative-interaction *primitive*, not long-range modelling
(that is the GNNs' job in later phases). Frame any comparison accordingly.

| Item | Location |
|------|----------|
| Front-end module | `models/components/bilinear_frontend.py` (`BilinearFrontEnd`) |
| Language model | `models/bilinear_lm.py` (`@register_model("bilinear_lm")`) |
| Config fields | `config.py` → `ModelConfig` (`bilinear_*`, `post_mixer_*`, `materialized_*`, `interaction_mode`) |
| Tests | `tests/test_bilinear.py` |

---

## 1. The problem: the outer-product blow-up

The intended interaction is: each token's embedding `emb_t ∈ R^E` (E = `d_model`,
intentionally tiny — default 128) interacts via an **outer product** with itself
and its `W-1` trailing neighbours (`W` = `bilinear_window`, default 16), and the
result is compressed toward a `64×64 = 4096`-dim per-token feature.

The **naive** realisation forms, for every position `t` and offset `d`, the
matrix `emb_t emb_{t-d}^T ∈ R^{E×E}` and stacks them into a `(T, E, E, W)`
tensor. At `E = 128, W = 16` that is

```
128 · 128 · 16 · 4 bytes ≈ 1.05 MiB  PER TOKEN.
```

At a GPT-1/2 sequence (T≈1024) and a real batch this is multiple GiB for a
*single intermediate*, before activations and gradients — infeasible on the
12 GB RTX 3060.

The prior-art pass (verified) found this full bilinear interaction is
**~98 % redundant**: it is equivalent to a degree-2 polynomial kernel, and a
**low-rank / Hadamard factorisation (MLB / MFB)** recovers the pairwise
multiplicative signal *without ever forming the `E×E` matrix*, while being more
parameter-efficient than sketch-based methods at small scale. That factorisation
is the contribution of this component.

---

## 2. The factorized MFB identity (the math)

For two operands `x, y ∈ R^E` and a single bilinear output `z_i = x^T W_i y`,
factorise the weight matrix `W_i ∈ R^{E×E}` as a sum of `k` rank-1 terms
(`k` = `bilinear_k`, the MFB factor):

```
W_i = Σ_{j=1..k} u_{ij} v_{ij}^T ,    u_{ij}, v_{ij} ∈ R^E .
```

Then, expanding and reassociating (never forming `W_i`):

```
z_i = x^T W_i y
    = x^T ( Σ_{j=1..k} u_{ij} v_{ij}^T ) y
    = Σ_{j=1..k} (x^T u_{ij}) (v_{ij}^T y)          ← scalars, no E×E matrix
    = Σ_{j=1..k} ( Ũ_i^T x  ∘  Ṽ_i^T y )_j           ∘ = Hadamard (elementwise)
```

Stacking all `o` outputs (`o` = `bilinear_o`, default 4096) and concatenating the
factors into projection matrices `Ũ, Ṽ ∈ R^{E × (k·o)}`:

```
z = SumPool( (Ũ^T x)  ∘  (Ṽ^T y),  k )            z ∈ R^o
```

where `SumPool(·, k)` reshapes the `k·o`-vector to `(o, k)` and sums over the `k`
axis. **The only intermediates are `(k·o)`-sized vectors — no `E×E` axis pair
ever exists.** `MLB` (Kim et al., arXiv:1610.04325) is the `k = 1` special case;
`MFB` (Yu et al., arXiv:1708.01471) is the general `k ≥ 1` form.

### 2.1 Windowing (causal, shared weights)

For position `t` the query is `x_t` and the partner at offset `d` is `x_{t-d}`
(`d = 0` is the *self* term `x ∘ x`, a quadratic feature of one token — kept, but
note it carries less *interaction* information than the cross terms `d > 0`). The
partner sequence is built by **left-padding the time axis and slicing**
(`_causal_shift`); positions `t < d` (no valid trailing neighbour) are
zero-padded, keeping next-token training leak-free. The embedding axis is never
touched, so the windowing introduces no `E×E` intermediate either.

The projections `Ũ, Ṽ` are **shared across all `W` offsets**, so the parameter
cost is `O(E·k·o)`, not `O(W·E·k·o)`. The per-offset specialisation lives only in
cheap aggregation weights (`offset_weights`, shape `(W, o)`, initialised to 1 so
the untrained module reduces to a plain sum over offsets); set
`bilinear_offset_weighting="sum"` to drop them entirely.

### 2.2 MFB output normalisation

After aggregating across offsets, MFB applies (in `_mfb_normalize`):

1. **Power normalisation** (signed square-root): `sign(z) · sqrt(|z| + ε)` —
   tames the heavy-tailed magnitude of bilinear features.
2. **L2 normalisation** over the `o` axis.
3. **Dropout** (`front_end_dropout`).

---

## 3. Memory: factorized vs naive

| Path | Dominant intermediate | Bytes (E=128, W=16, k=2, o=4096, fp32) |
|------|-----------------------|-----------------------------------------|
| Naive outer product | `(T, E, E, W)` | `E·E·W·4 ≈ 1.05 MiB / token` |
| **Factorized MFB** | `(T, k·o)` | `k·o·4 ≈ 32 KiB / token` (~33× smaller, and **no `E×E` axis**) |

Asymptotically the factorized path is `O(T·W·k·o)` work with `O(T·k·o)` peak
memory, versus the naive `O(T·W·E²)` memory. The **memory guard** test
(`test_factorized_path_never_materializes_emb_by_emb`) makes this a hard
invariant: it wraps the forward in a `TorchFunctionMode` that inspects the shape
of **every** tensor op output and asserts none has an `(E, E)` trailing axis pair.

**Measured (RTX 3060, bf16, B=4, T=512, ~176 M params, activation
checkpointing on):** peak VRAM ≈ **3.1 GiB** for `factorized_mfb` — well under
the 12 GB budget, where the single naive `(T, E, E, W)` tensor alone would be
≈ 2 GiB before activations/grads.

The **factorization-correctness** test
(`test_factorized_matches_bruteforce_bilinear`) is the proof the math is right:
on a tiny config it reconstructs each `W_i` explicitly (formed *only in the test*)
and checks `z = SumPool(Ũ^T x ∘ Ṽ^T y, k)` equals `x^T W_i y` to tolerance,
including the per-offset weights.

---

## 4. The three interaction modes (the ablation machinery)

Selected by `cfg.model.interaction_mode`. All three emit `(B, T, o)` and share
the causal windowing, so the surrounding LM is mode-agnostic.

| Mode | What it is | Role |
|------|-----------|------|
| `factorized_mfb` *(default)* | The factorized low-rank Hadamard interaction above. | The contribution under test. |
| `control_linear` | A windowed **linear** mixer over the same `W` shifted neighbours (`Linear(W·E → o)`), with the **same post-norm** — but **no multiplicative term**. | The ablation **null**: capacity-matched, so if MFB does not beat it, the second-order interaction is not pulling its weight. |
| `materialized_cnn` | Reduce `E → r` (`materialized_reduce_dim`, default 32), form the **small** `(r, r, W)` interaction explicitly, run a 2-D CNN over it (W offsets as channels). | The honest test of the original *"CNN over the interaction matrix"* idea (an inversion of Bilinear-CNN, arXiv:1504.07889) at **~16× lower memory** than the naive `128×128` map. A variant, *not* the default — it is novel/unproven. |

The `control_linear` parameter count `W·E·o` is the same order as the MFB
projections' `2·E·k·o` for typical `k=2, W=16` (`16 = 2·2·4`), so the comparison
is roughly capacity-matched out of the box; use `match_params` for an exact
budget match against the baselines.

---

## 5. The `bilinear_lm` language model

`@register_model("bilinear_lm")` (`models/bilinear_lm.py`). Architecture:

```
token ids (B, T)
  → nn.Embedding                         (B, T, d_model)   # phonological hook writes here
  → BilinearFrontEnd (windowed MFB)       (B, T, o)
  → front_proj  o → trunk_width           (B, T, trunk_width)
  → PostMixerBlock × N
       (causal depthwise-separable 1-D CNN over the sequence + GeGLU gated MLP,
        both pre-norm + residual)
  → RMSNorm
  → trunk_to_embed  trunk_width → d_model  (B, T, d_model)  # only if trunk_width ≠ d_model
  → tied LM head                          (B, T, vocab)
```

Design choices and contracts:

* **Honours `forward(x, targets) → (loss, logits)`** — the only thing the
  model-agnostic `Trainer` sees. **Zero trainer changes.**
* **`self.embed` is the `nn.Embedding`** the embedding-init hook writes to.
  `_init_weights()` runs **first**, then the optional `embedding_init` hook
  (card e1644700, phonological init) has the **final say** — same ordering as the
  Transformer baseline (regression SF-1). Tested by
  `test_bilinear_lm_respects_embedding_init_hook`.
* **The front-end only mixes within a local window per position**; the causal
  depthwise-separable conv in the post-mixer carries cross-position information
  *along the sequence*, and the gated MLP is the per-position non-linearity.
* **Narrow embedding, wide trunk.** The embedding (`d_model`) is intentionally
  tiny (default 128); the post-mixer trunk runs at `post_mixer_width` (`0` =
  "use `d_model`"). This lets depth × width scale the model to GPT-1/2 budgets
  **without bloating the embedding table** — the bulk of parameters live in the
  front-end and trunk, as intended. `trunk_to_embed` bridges back to `d_model`
  for the tied head (skipped when `trunk_width == d_model`).
* **Sizeable by `match_params`.** `num_parameters(trainable_only=True)` is
  exposed; the model builds for any `(d_model, post_mixer_layers, ...)` the
  search proposes. A Transformer baseline can be size-matched to a `bilinear_lm`
  target (`test_match_params_sizes_transformer_to_bilinear_lm`).
* **Depth-aware init.** Residual output projections (conv pointwise + gated-MLP
  down-projection) are scaled by `1/sqrt(2·post_mixer_layers)` so the
  residual-stream variance stays bounded as depth grows.
* **Activation-checkpointing compatible** (`activation_checkpointing` toggles
  `torch.utils.checkpoint` over the post-mixer blocks), and runs in bf16 on the
  3060.

### 5.1 Reaching GPT-1/2 scale

Example configs (vocab 50257, `d_model=128`, `o=4096`, `k=2`, `W=16`):

| `post_mixer_width` | `post_mixer_layers` | Params |
|--------------------|--------------------|--------|
| 1024 | 12 | ≈ 177 M |
| 1536 | 24 | ≈ 752 M |

The `100–350 M` GPT-1/2 band is reached by tuning width/depth between these
(e.g. width 1024, 12–18 layers). At 177 M, the embedding is only ~6.4 M of the
total; the trunk (~164 M) and front-end (~6 M) carry the budget.

---

## 6. Config fields (all additive to `ModelConfig`)

| Field | Default | Meaning |
|-------|---------|---------|
| `bilinear_window` | 16 | `W` — self + `W-1` trailing neighbours |
| `bilinear_k` | 2 | MFB factor `k` (sum-pool group size; `k=1` = MLB) |
| `bilinear_o` | 4096 | per-token output dim `o` (`= 64×64`) |
| `interaction_mode` | `"factorized_mfb"` | `factorized_mfb` \| `control_linear` \| `materialized_cnn` |
| `front_end_dropout` | 0.1 | MFB dropout (after power + L2 norm) |
| `bilinear_offset_weighting` | `"learned"` | `sum` \| `learned` offset aggregation |
| `post_mixer_layers` | 4 | post-mixer depth |
| `post_mixer_kernel` | 7 | causal depthwise conv kernel width |
| `post_mixer_ff_mult` | 4 | gated-MLP inner expansion (× trunk width) |
| `post_mixer_width` | 0 | trunk width (`0` = `d_model`) |
| `materialized_reduce_dim` | 32 | `materialized_cnn`: reduced dim `r` |
| `materialized_cnn_channels` | 32 | `materialized_cnn`: 2-D CNN hidden channels |

Existing baselines and the eval harness are **not** modified — `bilinear_lm`
only consumes the registry, the model-agnostic `Trainer`, `match_params`
(`sizing.py`), and the embedding-init hook.

---

## 7. Planned ablation protocol

This card delivers the **component + unit validation + ablation machinery**. The
comparative *training* runs are a separate follow-up experiment card (real 3060
time) and are **not** required here. The protocol they will follow:

1. **Size-match.** Pick a target param budget in the GPT-1/2 band. Use
   `match_params` to size the Transformer and Mamba baselines to it, and tune
   `bilinear_lm` (`post_mixer_width` / `post_mixer_layers`) to the same budget.
2. **Primary ablation (is multiplication useful?).** Train `bilinear_lm` with
   `interaction_mode="factorized_mfb"` vs `interaction_mode="control_linear"`
   (the matched-param, no-multiply null), everything else fixed. A BPB/perplexity
   gain for MFB over the control is the evidence that the second-order local
   interaction is a useful primitive.
3. **Reference points.** Compare both against the size-matched Transformer and
   Mamba on the same corpora and the same `eval/` harness (BPB, perplexity).
   Because the front-end is local, frame this as a test of the *interaction
   primitive*, not of long-range modelling.
4. **Variant probe (optional).** Run `materialized_cnn` to see whether the
   original "CNN over the interaction matrix" idea is competitive at small scale,
   acknowledging it is novel/unproven.
5. **Knobs.** Sweep `k` (1 = MLB vs ≥2 = MFB), `o`, `W`, and
   `bilinear_offset_weighting` (sum vs learned) as secondary ablations.

---

## References

- Yu et al., *Multi-modal Factorized Bilinear Pooling with Co-Attention Learning
  for Visual Question Answering*, arXiv:1708.01471 (MFB; the
  `z = SumPool(Ũ^T x ∘ Ṽ^T y, k)` identity, power + L2 norm).
- Kim et al., *Hadamard Product for Low-rank Bilinear Pooling*,
  arXiv:1610.04325 (MLB; the `k = 1` special case).
- Lin et al., *Bilinear CNN Models for Fine-grained Visual Recognition*,
  arXiv:1504.07889 (the vision precedent that `materialized_cnn` inverts).
```
