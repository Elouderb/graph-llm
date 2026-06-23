# Phase 3 — Persistent Memory Stage: Gated-DeltaNet Delta-Rule Matrix Memory

This document is the design contract for the project's **central thesis
component** (card **e2c6ea95**): a *persistent, bounded, fixed-size matrix
memory* that **replaces the context window**. It is delivered as a registered
standalone LM, `delta_memory_lm`, so it can be validated head-to-head against the
Mamba baseline on long-context probes.

> **The claim under test:** *context windows are obsolete — a fixed-size memory
> updated online can carry the information an attention KV-cache carries, without
> growing with the sequence.* This stage builds the memory and proves its
> mechanics; the comparative **training** experiment is a separate follow-up
> card (§6).

It is **not** a graph and **not** attention with a KV cache. The entire "memory"
is a small matrix `S` *per head* that is updated — edited in place — token by
token. Its size is **independent of the sequence length `T`**; that is the
bounded-memory property the thesis rests on.

---

## 1. The memory and its update

### Files

| Piece | Module | Class |
|-------|--------|-------|
| The memory layer | `models/components/delta_memory.py` | `GatedDeltaMemory` |
| The standalone LM | `models/delta_memory_lm.py` | `DeltaMemoryLM` (`@register_model("delta_memory_lm")`) |
| Config fields | `config.py` | `ModelConfig.delta_*` |

### State

Per head, `S` is a `(d_k, d_v)` matrix — a **key → value associative store**.
With `H` heads per layer, a layer's full state is `(B, H, d_k, d_v)`. **This
shape does not depend on `T`.** Compare an attention KV cache, which is
`(B, H, T, d_head)` and grows linearly with every token — that growth is exactly
what a context window bounds, and exactly what this memory removes.

### The recurrence (Gated-DeltaNet)

With feature map `φ` (here L2-normalisation, §3), a per-(token, head) **forget
gate** `α_t ∈ (0, 1)` and a **write strength** `β_t ∈ (0, 1]`:

```
READ   (causal):   o_t = S_{t-1}ᵀ φ(q_t)                              # (d_v,)
WRITE  (delta):    S_t = α_t · S_{t-1} + β_t · φ(k_t) (v_t − S_{t-1}ᵀ φ(k_t))ᵀ
```

The read uses `S_{t-1}` — the state **before** this token's write — so output
`t` depends only on tokens `≤ t`. The layer is therefore **causal by
construction**, with no mask needed (proven by a perturbation probe; §5).

### Why it overwrites instead of accumulating (the key idea)

The term `u_t = v_t − S_{t-1}ᵀ φ(k_t)` is the **delta / prediction error**: the
difference between the value *currently bound to key `k`* in the memory and the
*new* value `v`. The update writes the **correction** `u_t`, not `v` itself.

* Plain linear attention / fast-weights accumulates: `S += v kᵀ`. Writing
  `(k, v1)` then `(k, v2)` leaves `Sᵀk ≈ v1 + v2` — stale bindings pile up and
  the memory saturates.
* The **delta rule** writes the correction, so writing `(k, v1)` then `(k, v2)`
  leaves `Sᵀk ≈ v2`. The new binding **overwrites / edits** the old one in
  place. This makes the memory **self-evicting** and **bounded**: re-using a key
  replaces its slot rather than consuming new capacity.

This overwrite-vs-accumulate behaviour is the load-bearing test
(`tests/test_delta_memory.py::test_delta_rule_overwrites_not_accumulates`): with
`β=1, α=1` and an orthonormal key, reading after `(k, v1), (k, v2)` returns
exactly `v2`, **not** `v1 + v2`.

---

## 2. "One gradient-descent step" interpretation

The write is **literally one step of online gradient descent** on the per-token
associative recall loss

```
L_t(S) = ½ ‖ Sᵀ φ(k_t) − v_t ‖²
```

Its gradient is `∇_S L_t = φ(k_t) (Sᵀ φ(k_t) − v_t)ᵀ`, so a single GD step with
step size `β_t`, preceded by the forget decay `α_t`, gives

```
S_t = α_t S_{t-1} − β_t ∇_S L_t |_{S = S_{t-1}}
    = α_t S_{t-1} + β_t φ(k_t) (v_t − S_{t-1}ᵀ φ(k_t))ᵀ
```

— exactly the write rule above. So the memory performs in-context **test-time
learning**: each token takes one SGD step to better recall its own value, and
`α_t` slowly forgets stale keys. `β_t` is the (learned, input-dependent) learning
rate; `α_t` is the (learned, input-dependent) weight decay. This connects the
component to the Fast Weight Programmers view (a fast network whose weights are
written by a slow network) and to the modern "linear attention is test-time
regression" framing.

---

## 3. The feature map φ and the gating

* **Feature map** (`delta_feature_map`): default `"l2"` — L2-normalise keys and
  queries. Following Gated-DeltaNet, normalising keys bounds `‖φ(k)‖ = 1`, which
  keeps the delta GD step well-scaled (the effective per-key learning rate does
  not blow up with key magnitude) and keeps the read `Sᵀφ(q)` a bounded
  similarity. `"silu_l2"` applies SiLU then L2-normalises; `"identity"` disables
  the map (used by the math tests, where an exact hand reference is checked).
* **Write strength** `β_t = σ(W_β x_t) ∈ (0, 1)` — a learned, input-dependent
  scalar per `(token, head)` (sigmoid is strictly `(0, 1)`). `β → 0` skips a
  write; `β → 1` approaches a full delta replacement of the key's binding (the
  math tests drive the exact `β = 1` limit through the scan directly).
* **Forget gate** `α_t ∈ (0, 1)` — Gated-DeltaNet's scalar global decay per
  `(token, head)`, parameterised as `α_t = exp(−softplus(W_α x_t))` (stable, and
  strictly `< 1` — a true contraction). At zero input `α_t = exp(−log 2) ≈ 0.5`
  (moderate forgetting at init); the learned bias moves each head's decay from
  there. Setting `delta_use_forget_gate = False` recovers the **ungated
  DeltaNet** (`α_t ≡ 1`).

This is the Gated-DeltaNet form: a *scalar* gate on the whole state (cheap,
chunkable) rather than a per-dimension diagonal gate (RWKV-7 / DPLR territory),
which keeps the recurrence in the efficient delta-rule family.

---

## 4. Capacity: why fixed size is enough

A `(d_k, d_v)` matrix can store at most **`d_k` mutually-orthogonal key→value
bindings** without interference (it is rank-`d_k` at most; non-orthogonal keys
cross-talk). So **per-head capacity is `≤ d_k`**. The design consequences:

* **Budget `d_k` conservatively.** A head is a small fixed store, not an
  unbounded log.
* **Scale via `heads × layers × width`, not by inflating one head's `d_k`.**
  `H` heads give `H` independent stores; depth lets later layers re-bind and
  compose; the residual stream / MLP width carries representational capacity.
  This is how `delta_memory_lm` reaches GPT-1/2 budgets
  (`delta_layers × delta_n_heads × delta_head_{k,v}_dim`).

The forget gate `α_t` is what makes a *finite* store viable over an *infinite*
stream: old, un-refreshed bindings decay, freeing effective capacity, so the
memory degrades gracefully rather than hard-saturating.

---

## 5. Correctness guarantees (the tests)

`tests/test_delta_memory.py`:

* **Causality (perturbation probe)** — perturbing token `t+1` leaves logits at
  every position `≤ t` unchanged (`< 1e-6`), at both the layer and the full-LM
  level. Non-negotiable for an autoregressive LM: a leak here silently inflates
  every long-context result.
* **Delta-rule overwrite-not-accumulate** — write `(k, v1)` then `(k, v2)`
  (`β=1, α=1`); reading `k` returns `≈ v2`, not `v1 + v2`. Plus an exact 3-step
  scan checked against an independent by-hand reference (with non-trivial `α, β`).
* **Bounded / fixed-size state** — the state shape is captured at every step for
  `T = 4` and `T = 64` and asserted constant `= (B, H, d_k, d_v)`; no per-step
  growth.
* Forward/backward finite for every feature map × gate combination; the LM
  builds at a ~GPT-1/2 param count and trains a tiny synthetic step;
  `match_params` sizes both a Mamba and a Transformer baseline to it; the
  registry resolves `delta_memory_lm` and honours the phonological-init hook.

---

## 6. Implementation choice: v1 is a sequential pure-PyTorch scan

v1 implements the recurrence as a **clean sequential scan in plain PyTorch** —
correctness over speed, exactly like the Mamba baseline's `_selective_scan`
(`docs/baselines.md` §1.2). The only Python loop is over the time axis; the
batch/head/matrix work is `einsum`. The scan runs in fp32 for numerical
stability and casts back to the input dtype (bf16/amp compatible; validated on
the RTX 3060 with a bf16 forward+backward).

**We deliberately do not depend on the `fla-hub` / `flash-linear-attention`
Triton/CUDA kernels for v1.** Those carry the same build/toolchain risk on the
3060 that made us hand-roll Mamba, and a silently-unavailable memory stage would
defeat the purpose. The sequential scan is mathematically the same recurrence;
it is just slower because it materialises the per-token state instead of using
the **chunkwise-parallel** form.

**Throughput caveat & follow-up.** The `O(T)`-iteration scan is the right trade
for a research artifact whose first job is to be *correct and validatable*, but
it is not training-throughput-optimal. The documented optimisation is the
**chunkwise-parallel delta-rule kernel** (DeltaNet, arXiv:2406.06484): the
sequence is processed in chunks with the intra-chunk delta interactions solved
by a small triangular (WY-representation) inverse and the inter-chunk state
carried recurrently — recovering near-attention throughput while staying exact.
If/when added, a vectorised/chunked path **must** match this sequential
reference within tolerance (a test, mirroring the bilinear factorization-vs-
bruteforce proof).

---

## 7. Planned validation vs Mamba (separate follow-up card)

This card builds the component + LM + unit validation + GPU smoke. The
comparative **training** experiment is a separate follow-up needing 3060 time
(like the Phase 2 ablation). The protocol — measuring *the thesis*, not
aggregate perplexity:

1. **Matched-parameter setup.** Use `match_params` to size `mamba`,
   `transformer`, and `delta_memory_lm` to the same trainable-parameter budget
   (already unit-tested here for Mamba and the Transformer). Train all three on
   the same corpus with the same trainer (it is model-agnostic; zero changes).
2. **Perplexity-vs-context curve**, not aggregate ppl. Use
   `eval/long_context.py::position_loss_curve` to plot loss as a function of
   token position on sequences **longer than the training window**. The thesis
   prediction: the delta memory holds (or degrades gracefully) past the window
   where a fixed-context Transformer cannot, and is competitive with / beats
   Mamba (a 1.3B DeltaNet beats Mamba & GLA in the literature).
3. **Passkey / recall probe.** Use
   `eval/long_context.py::run_passkey_probe` across depths and context lengths.
   The delta rule's overwrite/eviction should give cleaner recall of a single
   planted key at long range than an accumulating linear-attention memory.
4. **Report curves and per-depth recall**, not a single scalar — a single
   aggregate ppl can hide exactly the long-range behaviour the claim is about.

---

## 8. References

* **DeltaNet** — Yang, Wang, Zhang, Shen, Kim, Pan. *Parallelizing Linear
  Transformers with the Delta Rule over Sequence Length.* arXiv:2406.06484
  (NeurIPS 2024). The chunkwise-parallel delta rule.
* **Gated DeltaNet** — Yang, Kautz, Hatamizadeh. *Gated Delta Networks:
  Improving Mamba2 with Delta Rule.* arXiv:2412.06464 (ICLR 2025). Adds the
  scalar forget gate `α_t` to the delta rule.
* **Fast Weight Programmers** — Schlag, Irie, Schmidhuber. *Linear Transformers
  Are Secretly Fast Weight Programmers.* arXiv:2102.11174 (ICML 2021). The
  fast-weight / delta-rule view of linear attention and its bounded-memory
  capacity.
* **Mamba** — Gu, Dao. arXiv:2312.00752. The recurrent-state baseline this stage
  is measured against (`docs/baselines.md`).
