# Multi-scale Conv Front-End (cheap local combiner)

This document is the design + correctness contract for the **multi-scale causal
conv input stage** (card **ed853f9c**, realising design note **8b5341f0**): an
optional, config-flagged stage that enriches each token embedding with
multi-scale *trailing local context* and condenses it back to ONE `d_model`
vector per token, feeding the committed `delta_memory_lm` memory backbone better
per-position features (richer keys/values).

It is the **cheap, additive** version of the bilinear front-end's instinct.
The bilinear (`bilinear_frontend.md`) was the *expensive, multiplicative* way to
capture local multi-token meaning ("new york" ≠ "new" + "york"); the Phase 2
ablation showed it tied its cheap linear null at ~2× cost. A causal conv over
the embedding sequence is the cheap, additive way to do the same thing — and the
evidence points at the cheap side.

| Item | Location |
|------|----------|
| Front-end module | `models/components/multiscale_conv_embed.py` (`MultiScaleConvEmbedding`) |
| Wired into | `models/delta_memory_lm.py` (after the embedding + phonological hook, before the memory layers) |
| Config fields | `config.py` → `ModelConfig` (`front_end`, `conv_widths`, `conv_condense`, `conv_depthwise`) |
| Tests | `tests/test_multiscale_conv_embed.py` |

This is the clean **one-flag A/B**: `front_end="none"` (the committed backbone,
byte-for-byte) vs `front_end="multiscale_conv"` (the enriched input stage).

---

## 1. The pipeline

Per position `t`, with token embeddings `x ∈ (B, T, d)` (`d = d_model`):

```
token embeddings (B, T, d)
  -> bank of CAUSAL depthwise-separable 1-D convs at dyadic widths {1,2,4,8,16}
       width-1  == pointwise / identity scale (the plain per-token embedding)
       width-k  == a learned k-gram: a combination of the k embeddings in the
                   trailing window ending at t
     -> S = len(conv_widths) scale tensors, each (B, T, d)
  -> stack over the scale axis                                   -> (B, T, S, d)
  -> condense (a 1×1 conv == a per-position projection SHARED across all T)
                                                                 -> (B, T, d)
```

The output has the **same shape** as the input — it is a drop-in enrichment of
the embedding, so the rest of `delta_memory_lm` is unchanged.

### Why a sliding conv, not a lookup table

You cannot table-lookup an n-gram (`vocab**k` entries — at byte level
`256**16`). A *sliding kernel* turns the n-gram into a learned **function** of
the trailing window; that is what makes it buildable. This is the proven
n-gram-CNN-embedding family: **Kim 2014** (n-gram CNN embeddings),
**Charformer / GBST** (multi-block-size embeddings), **ByteNet / TCN** (dilated
causal-conv input stacks).

### Dyadic, not contiguous, widths

Widths `{1,2,4,8,16}` (≈ `log(W)` kernels) capture multi-scale structure for a
fraction of the cost of all contiguous widths `1..16` — widths 15 and 16 see
nearly the same window, so most of the contiguous bank is redundant.

### Depthwise-separable convs

With `conv_depthwise=True` (default) each scale is a **depthwise** conv (mixes
within a channel across the trailing window, `groups=d`) followed by a
**pointwise** 1×1 conv (mixes channels per position) — the cheap factorisation.
`conv_depthwise=False` uses a single full 1-D conv per scale instead (mixes time
and channels at once; more parameters).

---

## 2. The condense step (1×1-conv collapse)

The cross-token mixing already happened inside the sliding-window convs, so
collapsing `(B, T, S, d) → (B, T, d)` is **purely per-position** — trivially
causal (a 1×1 conv / per-position projection cannot see the future). Two modes
(`conv_condense`):

* **`concat_proj`** (default) — reshape `(B, T, S, d) → (B, T, S·d)` then a
  shared `Linear(S·d, d)`. Mixes across both scales and dims; the literal
  1×1-conv collapse. Simplest, most expressive.
* **`soft_select`** (GBST-style) — a shared `Linear(S·d, S) → softmax` over the
  `S` scales → a **convex blend** of the scale-embeddings:
  `enriched = Σ_s w_s · scale_s`, `w = softmax(...)`, `Σ_s w_s = 1`. The output
  stays in embedding space, and the per-position granularity is readable off the
  weights (interpretable).

### Weight sharing — the key correctness point

The condense kernel is **ONE learned function applied identically to every
token's `(S, d)` slice** — the defining property of a 1×1 conv / `nn.Linear`
broadcast over the sequence. It is **not** position-indexed (no `T` / `max_seq_len`
axis in its weight). Sharing is what gives **translation invariance** and
**length generalisation**: a model trained at one length applies the same
condense at any other length.

> Test: `test_condense_is_weight_shared_across_positions` feeds two different
> positions identical `(S, d)` slices and asserts identical condensed output;
> `test_concat_proj_condense_is_a_single_linear_over_scale_dim_axis` checks the
> weight shape is `(d, S·d)` with no length axis.

---

## 3. Causality (the load-bearing property)

For an autoregressive LM the front-end **must not leak the future**: the output
at position `t` may depend only on inputs `≤ t`. A leak here would silently
inflate every long-context perplexity-vs-position result the thesis depends on.

Two mechanisms guarantee it:

1. **Each conv is left-padded only.** A width-`k` conv left-pads by `k-1` and
   never right-pads, so `output[t]` depends on inputs `[t-(k-1) .. t]` — the
   trailing window. (Implementation: `F.pad(h, (width-1, 0))` then `Conv1d` with
   `padding=0`, the repo's established causal-conv idiom, see
   `bilinear_lm.DepthwiseSeparableConv1d` and `mamba._causal_conv`.)
2. **The condense is per-position.** It operates on each `(S, d)` slice
   independently, so it cannot mix across time at all.

> Tests (both `< 1e-6`):
> `test_front_end_module_is_causal_by_perturbation` (perturb token `t+1` in the
> module input → outputs at `≤ t` unchanged, for every condense mode and
> depthwise on/off) and
> `test_delta_memory_lm_with_front_end_is_causal_by_perturbation` (the same probe
> end-to-end through the full model with the front-end ON → logits at `≤ t`
> unchanged). A centred (non-left-padded) conv or a leaky condense fails these.

---

## 4. The `none` no-op guarantee

`front_end="none"` (the default) constructs **no module at all**
(`self.front_end is None`) and `forward` skips it entirely. No extra parameters,
no extra RNG draws, no state-dict keys — so the committed `delta_memory_lm` is
**byte-for-byte unchanged**, and the existing delta tests + the full suite stay
green.

> Tests: `test_front_end_none_state_dict_identical_to_baseline` (same keys + param
> count, no `front_end.*` keys) and
> `test_front_end_none_forward_is_bit_identical_under_fixed_seed` (under a fixed
> seed the logits are bit-identical to the no-field baseline).

---

## 5. Config

```python
# config.py → ModelConfig
front_end: str = "none"                 # "none" | "multiscale_conv"
conv_widths: list[int] = [1, 2, 4, 8, 16]   # dyadic widths; each >= 1; width-1 = identity scale
conv_condense: str = "concat_proj"      # "concat_proj" | "soft_select"
conv_depthwise: bool = True             # depthwise-separable (True) vs full conv (False)
```

All fields are **additive** and default-preserving; the YAML loader is
`hasattr`-guarded, so no loader change was needed. Only `delta_memory_lm` reads
them — the baselines and `bilinear_lm` ignore them.

---

## 6. The ablation protocol

This card builds the component, the flag, and the unit validation. The
**comparative ablation RUN** is a separate follow-up experiment (like the prior
ones), not part of this card:

* **Arm A:** `delta_memory_lm`, `front_end="none"` — plain single-token embedding.
* **Arm B:** `delta_memory_lm`, `front_end="multiscale_conv"` — the multi-scale
  conv front-end + condense.
* **Held fixed:** the same `delta_memory_lm` backbone, **matched parameters**
  (size Arm A up to Arm B with `match_params`, or vice versa), byte-level text8,
  **bits-per-byte** as the metric.
* **Optional 3rd arm:** `conv_depthwise=False` (full conv) or an MLP-kernel / NiN
  conv, to test whether within-window nonlinearity buys anything vs the added cost.

One clean A/B isolating the **input representation** — cheap, and it does not
disturb the memory.

### Honest expectations

This is still **local** front-end machinery. Phase 2 showed elaborate local
front-ends have not earned their keep in this setup, and GBST / Charformer gains
in the literature are *modest* — often more efficiency (shorter sequences via
downsampling) than raw quality. Treat this as a clean isolated ablation; the
expected upside is "a nice cheap representation boost," not a step change.

### Phonological / multilingual tie-in

Contextual embeddings lean on *context* over single-token identity, reducing
reliance on the brittle (English-biased) per-token phonological prior. This
**composes** with the phonological init (card e1644700): keep the init on the
per-token embedding table; the conv bank builds context on top of it. A hedge,
not a multilingual fix.
