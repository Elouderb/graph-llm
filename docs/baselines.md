# Phase 0 Baselines & Evaluation Protocol

This document is the apples-to-apples contract for the project's falsifiability
spine (card **424e3a8e**): the matched-parameter **Transformer** and **Mamba**
baselines, the datasets they train on, and the evaluation harness every later
component must beat. The headline claim — *"a memory GNN makes the context
window obsolete"* — is only meaningful **measured** against a recurrent-state
model that already streams over arbitrary length. Mamba is that load-bearing
baseline; the Transformer is the attention reference point.

---

## 1. Baselines

Both baselines are registered with the model registry and obey the same
`forward(x, targets) -> (loss, logits)` contract, so the model-agnostic
`Trainer` runs either with zero changes.

| Name | Module | Class | Family |
|------|--------|-------|--------|
| `transformer` | `models/baselines/transformer.py` | `TransformerBaseline` | attention (decoder-only) |
| `mamba` | `models/baselines/mamba.py` | `MambaBaseline` | recurrent state (selective SSM) |

### 1.1 Transformer

Decoder-only, pre-norm, fully config-driven (`d_model`, `n_layers`, `n_heads`,
`d_ff`, `max_seq_len`). RMSNorm, RoPE positional encoding (with on-demand cache
extension so it runs **past the training window**), causal attention via
`F.scaled_dot_product_attention`, optionally tied input/output embeddings,
GPT-2-style init with depth-aware (`1/sqrt(2·n_layers)`) residual-projection
scaling so deep configs from `match_params` train stably.

### 1.2 Mamba (pure-PyTorch selective SSM)

**Dependency choice (important):** we deliberately do **not** depend on the
`mamba-ssm` package. Its `selective_scan` is a hand-written fused **CUDA**
kernel that must compile against a matching CUDA toolchain. This build
environment (and the 12 GB RTX 3060 target's dev path) is CPU-first, so that
kernel is impractical and would silently make the recurrent baseline
unavailable — and the recurrent baseline is the whole point. Instead we
implement the selective scan as a **readable sequential recurrence in plain
PyTorch**.

It is mathematically equivalent to the reference selective SSM (same block
structure, same zero-order-hold discretisation, same gating):

```
in_proj:  d_model            -> 2 * d_inner          (split into x, z)
conv1d:   depthwise causal conv over x               (width d_conv)
x_proj:   d_inner            -> dt_rank + 2 * d_state (split into dt, B, C)
dt_proj:  dt_rank            -> d_inner               (then softplus -> delta)
scan:     dA = exp(delta·A),  dB = delta·B
          h_t = dA·h_{t-1} + dB·x_t,   y_t = C_t·h_t + D·x_t
gate:     y = y * silu(z)
out_proj: d_inner            -> d_model
```

`A` is stored in log space (`A = -exp(A_log)`) so it stays strictly negative
(stable decay); `dt_proj.bias` is initialised so `softplus(bias) ∈ [dt_min,
dt_max]`, mirroring the reference. There is **no positional embedding** — the
recurrence is inherently sequential, which is exactly the property that lets it
stream past any fixed context window.

> **Throughput caveat.** The pure-PyTorch scan materialises the recurrence as a
> Python `for`-loop over the time axis, so it is **substantially slower** than
> the fused CUDA kernel (and slower than the parallel attention path) — roughly
> linear in sequence length with a large Python-overhead constant. This is fine
> for a research baseline (correctness and architecture parity over speed) and
> for CPU CI, but expect Mamba training wall-clock to dominate at long sequence
> lengths. If/when a CUDA box is available, swapping in `mamba-ssm`'s
> `selective_scan_fn` behind the same `MambaBlock` interface is a localised
> change. Keep batch/sequence sizes modest when running the pure-PyTorch path.

### 1.3 SSM hyperparameters (config)

Added to `ModelConfig` (ignored by the Transformer):

| field | default | meaning |
|-------|---------|---------|
| `d_state` | 16 | SSM state dimension *N* |
| `d_conv` | 4 | depthwise causal conv kernel width |
| `expand` | 2 | inner expansion (`d_inner = expand · d_model`) |
| `dt_rank` | `"auto"` | rank of the `dt` projection (`auto = ceil(d_model/16)`) |
| `dt_min` / `dt_max` | 0.001 / 0.1 | softplus-space `dt` bias init range |

---

## 2. Parameter matching (`models/baselines/sizing.py`)

`count_params(model)` returns the trainable parameter count. `match_params(
target_params, base_cfg, tolerance=0.05)` returns a **new** config (the template
is never mutated) whose model lands within **±5 %** of a target — so the two
baselines (and, later, our own model) are size-matched and any quality
difference reflects architecture, not capacity.

The search is **two-dimensional over width and depth**: for each candidate depth
(`n_layers`) in a small window around the base, the width (`d_model`) is
binary-searched (param count is monotonic in width at fixed depth), keeping
`d_ff` at the template's `d_ff/d_model` ratio and snapping width to a multiple of
`n_heads`; the overall closest `(depth, width)` is returned. Searching depth as
well as width closes the granularity gaps a width-only search leaves when the
`n_heads` grid is coarse relative to the target. Candidates are **built and
counted directly** rather than scored by a hand-derived formula, so the matcher
stays exact even if a baseline's architecture changes.

### Reference matched configs

Measured on this build (vocab = 256, `tie_embeddings=True`, `max_seq_len=1024`).
Mamba is size-matched to the Transformer target via `match_params` (its
`d_model` / `n_layers` are chosen by the search; `d_state=16`, `expand=2`):

| Tier | Transformer config | Transformer params | Matched Mamba config | Mamba params | Δ |
|------|--------------------|---------------------|----------------------|--------------|---|
| small  | `d_model=256, n_layers=6, n_heads=8, d_ff=1024`   | 4,787,456  | `d_model=488, n_layers=3`  | 4,756,048  | 0.7 % |
| medium | `d_model=512, n_layers=8, n_heads=8, d_ff=2048`   | 25,305,600 | `d_model=888, n_layers=5`  | 25,371,936 | 0.3 % |
| large  | `d_model=768, n_layers=12, n_heads=12, d_ff=3072` | 85,150,464 | `d_model=1224, n_layers=9` | 85,831,776 | 0.8 % |

> The Phase 0 byte vocab is 256; param counts shift when the 16k tokenizer
> (card e1644700) lands (the embedding table grows). Re-run `match_params`
> against the new `vocab_size` at that point — the matcher is vocab-aware.

---

## 3. Datasets (`data/loader.py`)

All corpora are lazily fetched via the `datasets` library, **cached** under
`DataConfig.data_dir` (which is `.gitignore`d), and fall back to the
deterministic synthetic dataset when offline so CI stays green with no network.
Select via `DataConfig.source`.

| `source` | corpus | split | encoding | eval metric |
|----------|--------|-------|----------|-------------|
| `enwik8` | enwik8 (100 MB Wikipedia XML) | **90M / 5M / 5M** bytes (canonical, contiguous) | byte-level (vocab 256) | bits-per-byte |
| `text8`  | text8 (cleaned enwik8) | **90M / 5M / 5M** bytes (canonical, contiguous) | byte-level (vocab 256) | bits-per-byte |
| `enwik9` | enwik9 (1 GB Wikipedia XML, `http://mattmahoney.net/dc/enwik9.zip`) | **990M / 5M / 5M** bytes (canonical, contiguous; same 5M/5M held-out tail convention as enwik8/text8) | byte-level (vocab 256), raw bytes — no XML stripping | bits-per-byte |
| `wikitext103` | WikiText-103 (raw) | provided train split, byte-level | byte-level (vocab 256) for now | bits-per-byte (see seam) |
| `tinystories` | roneneldan/TinyStories | random | byte-level | tiny-scale coherence |
| `synthetic` | deterministic random tokens | random | n/a | offline smoke / CI |

The enwik8/text8/enwik9/wikitext103 loaders return a **fixed** `(train, val)` pair
built from the **canonical contiguous split** (NOT a random slice), so reported
numbers are comparable across runs and across models. The split helpers
(`canonical_byte_splits`, `make_canonical_split_datasets`) are unit-tested on
in-memory byte fixtures — never a live download.

**enwik9 (card 69776c3e)** is the next rung above text8/enwik8 (~10^8 bytes) for
the 5M → 15M → 50M parameter mini-ladder (card f82d95dc). Unlike the other
corpora (served pre-decoded via the HF `datasets` hub), enwik9 has no HF dataset
id: it is fetched once via stdlib `urllib` + `zipfile` from Matt Mahoney's
canonical ~330 MB zip and cached as the raw, uncompressed ~1 GB (exactly
1,000,000,000 bytes) dump. Cache location follows the same
`DataConfig.data_dir`-relative convention as every other corpus in this table
(`<data_dir>/enwik9.bin`, `.gitignore`d); on this project's dev machine that
cache is shared across worktrees at `~/.cache/graph_llm/data/enwik9.bin`
(`DataConfig(source="enwik9", data_dir="~/.cache/graph_llm/data")` — set once,
reused by every tree without re-downloading). The cache write is atomic
(temp-file-in-same-dir + `os.replace`) and both a fresh fetch and an existing
cache file are checked against the exact expected size, so a kill-mid-download
or a truncated cache raises loudly instead of silently training on corrupt
bytes.

### Eval metric at Phase 0 = bits-per-byte (byte-level)

BPB is computed byte-level and is therefore **tokenizer-independent**, making it
the cleanest cross-model comparison **before** the real tokenizer lands. That is
why even WikiText-103 (normally a *token-level perplexity* benchmark) is scored
byte-level here.

> **Tokenizer seam (do not remove).** To move WikiText-103 (or any corpus) to
> token-level perplexity once the 16k tokenizer (card e1644700) ships, replace
> the byte encoder in the fetchers with the trained tokenizer's `encode` and
> pass its vocab size through. The chunking and the canonical split are
> tokenizer-agnostic, so **only the encoding line changes** — nothing downstream
> (metrics, trainer, harness) is affected. Phase 0 does **not** depend on the
> tokenizer card.

---

## 4. Evaluation harness

### 4.1 Core LM metrics (`eval/metrics.py`)

* `perplexity(model, loader)` — `exp(mean NLL)`, token-weighted.
* `bits_per_byte(model, loader, bytes_per_token=1.0)` — `mean_NLL / ln2 /
  bytes_per_token`. Token-weighted, so a short final batch does not bias the
  result. For byte-level models `bytes_per_token = 1.0`; for a subword tokenizer
  later, pass the corpus's average bytes-per-token ratio.

### 4.2 Long-context probes (`eval/long_context.py`)

Both probes are built to run on sequences **longer than the training window** —
that is what tests extrapolation / memory rather than mere in-window fit.

* **Passkey / needle-in-a-haystack** (`make_passkey_example`,
  `run_passkey_probe`, `score_passkey_retrieval`). A short secret key is buried
  in long semantically-empty filler at a configurable depth; the model is asked
  to recall it; scoring is **exact-match** on the retrieved key. `greedy_generate`
  reads off the answer using only the `(loss, logits)` contract, so it is
  model-agnostic.
* **Per-token-position loss curve** (`position_loss_curve`). Mean next-token loss
  at each position across a batch of long sequences. A model that genuinely uses
  long context keeps the curve flat/decreasing with position; one that has
  forgotten earlier tokens shows loss rising past its usable window. The eval
  script reports the in-window vs. out-of-window means side by side.

Everything is byte-level by default (tokenizer-independent); an `encode` /
`decode` callable can be injected to use a real tokenizer later without changing
the probe logic.

### 4.3 Eval CLI (`scripts/eval.py`)

Runs the **full suite** on a checkpoint and emits **JSON** (`--json-out`) plus a
pretty-printed table:

```bash
python scripts/eval.py \
    --config configs/<cfg>.yaml \
    --checkpoint checkpoints/<ckpt>.pt \
    --json-out results/eval.json \
    --long-context-len 2048      # default: 2x the training window
```

The JSON records: model name, param count, train window, long-context length,
core metrics (perplexity, BPB), the full position-loss curve plus in/out-of
window means, and the per-depth passkey results with accuracy.

---

## 5. Protocol summary (apples-to-apples checklist)

1. Pick a parameter tier (§2) and `match_params` both baselines to it.
2. Train both on the **same** corpus / encoding / budget (`source`,
   `seq_len`, `max_steps`, optimiser config) on the 12 GB target.
3. Report **byte-level BPB** on the canonical val split as the headline
   cross-model number (tokenizer-independent at Phase 0).
4. Report the **passkey** retrieval accuracy and **position-loss** curve at a
   context length **beyond** the training window — this is where the recurrent
   baseline's streaming behaviour (and, later, our memory-GNN claim) is actually
   tested.
5. Keep the Mamba throughput caveat (§1.2) in mind when budgeting wall-clock.
