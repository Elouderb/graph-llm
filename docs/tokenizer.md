# Tokenizer — Phase 1 (card e1644700)

## Overview

Phase 1 introduces a byte-level BPE tokenizer with vocabulary size **exactly 16,384** and a phonological embedding initialiser that encodes articulatory features before training begins.

---

## Vocabulary rationale

| Design choice | Justification |
|---|---|
| **Byte-level** | No out-of-vocabulary tokens; any UTF-8 text is representable. Special tokens are reserved without conflicting with byte values. |
| **BPE** | Simple, well-understood; deterministic given corpus + random seed; compatible with HuggingFace `tokenizers`. |
| **16,384 (2^14)** | Large enough for common English subword merges; small enough that the embedding matrix stays ~8 MB at d_model=128 (fp32). Fits comfortably in the 12 GB GPU budget. |
| **4 special tokens** | `<pad>`, `<unk>`, `<bos>`, `<eos>` are included in the 16,384 count. The byte alphabet (256 characters) is seeded by `ByteLevel.alphabet()`. |

---

## Files

```
src/graph_llm/tokenizer/
    __init__.py          — package init, exports public API + triggers registration
    bpe.py               — BPETokenizer class (HF tokenizers backend)
    phonological_init.py — phonological initialiser + registry hook

scripts/
    train_tokenizer.py   — train and save the 16k BPE tokenizer
    probe_embedding.py   — silhouette-score probe (phonological vs. random)

tests/
    test_tokenizer.py    — unit tests for all Phase 1 deliverables
```

---

## Training the tokenizer

### Offline (bundled synthetic corpus, always works)

```bash
python scripts/train_tokenizer.py \
    --output tokenizer/bpe_16k.json
```

This uses 5,000 random-word sentences sampled from ``/usr/share/dict/words``
(104k entries, available on all standard Linux/macOS systems).  No network
access is required.  The resulting vocabulary is exactly 16,384 tokens and is
sufficient for all tests and ablations; it is not a realistic language-model vocab.

### With TinyStories (recommended for realistic experiments)

```bash
python scripts/train_tokenizer.py \
    --source tinystories \
    --data-dir data/ \
    --output tokenizer/bpe_16k.json
```

Downloads ~10,000 TinyStories from HuggingFace on first run (cached under
`data/tinystories_cache/`).

### With enwik8

```bash
python scripts/train_tokenizer.py \
    --source enwik8 \
    --data-dir data/ \
    --output tokenizer/bpe_16k.json
```

Uses the first 20 MB of enwik8 (Wikipedia XML).

---

## Using the tokenizer in a config

Set two fields in your YAML config:

```yaml
data:
  encoder: bpe
  bpe_tokenizer_path: tokenizer/bpe_16k.json
```

The default is `encoder: byte` (the existing `ByteLevelEncoder`), which is
unchanged.  `get_encoder(cfg)` in `data/loader.py` dispatches between the two.

---

## Phoneme mapping method

```
BPE token
  └─► strip byte-level prefix (Ġ / ▁ / ##), lowercase
        └─► CMUdict lookup  ──► ARPABET phones (first pronunciation)
              └─► _ARPA_TO_IPA mapping ──► IPA symbol(s) per phone
                    └─► panphon.FeatureTable.word_to_vector_list()
                          └─► 24-d articulatory feature vector per IPA segment
                                └─► average all segments ──► 24-d token vector
```

### ARPABET → IPA

`phonological_init.py` contains a static 39-entry `_ARPA_TO_IPA` dict.
Diphthongs (AW, AY, EY, OW, OY) are split into their component IPA segments
so panphon processes each separately and the vectors are averaged.

### panphon feature dimension

panphon produces 24-d binary/ternary articulatory feature vectors.  The
dimension is constant across panphon versions.

### Projection to d_model

A fixed random projection matrix W of shape `(24, d_model)` is sampled from
`N(0, 1/sqrt(24))` using a seeded NumPy RNG (seed=0 by default).  Each
mapped token's 24-d feature vector is multiplied by W to produce a `d_model`-d
initial embedding.  Unmapped tokens keep the default `nn.Embedding` random init.

### Coverage

Coverage is the fraction of vocabulary tokens that have a CMUdict entry and
a parseable phoneme sequence.  Expected values:

| Corpus | Approx. coverage |
|---|---|
| Synthetic (bundled) | ~5–15% (small word set, many repeated subword fragments) |
| TinyStories | ~15–30% |
| enwik8 | ~20–35% |

Coverage is logged at INFO level and returned by `compute_phonological_vectors`.

**Coverage denominator note:** the denominator is the *total* vocabulary count
(``len(vocab)``), which includes special tokens (``<pad>``, ``<unk>``,
``<bos>``, ``<eos>``), raw byte tokens (256 entries from the byte-level
alphabet), punctuation tokens, and numeric tokens.  None of these typically
appear in CMUdict, so the reported coverage fraction is always lower than the
fraction of *word-like* tokens that were matched.  This is intentional: the
denominator represents the true cost basis (all embedding rows that receive a
phonological value relative to the total weight matrix).

---

## Phonological embedding init recipe

```python
from graph_llm.tokenizer.bpe import BPETokenizer
from graph_llm.tokenizer.phonological_init import phonological_init_fn

# 1. Load trained tokenizer
tok = BPETokenizer.from_pretrained("tokenizer/bpe_16k.json")

# 2. Build model (runs registry hook — no-op for phonological without vocab)
model = build_model(cfg)

# 3. Apply phonological init with vocab
phonological_init_fn(
    model.embed.weight,
    vocab_size=cfg.model.vocab_size,
    d_model=cfg.model.d_model,
    vocab=tok.get_vocab(),
)
```

The registry hook `@register_embedding_init("phonological")` is a no-op shim
called by the model constructor without a tokenizer vocab.  The real init is a
separate call with the tokenizer.

---

## Ablation toggle

The `model.embedding_init` config field selects the init method:

| Config value | Effect |
|---|---|
| `null` / `None` (default) | Standard `nn.Embedding` random init |
| `"phonological"` | `apply_embedding_init()` loads the BPE tokenizer and runs `phonological_init_fn` with the vocab |

### How the toggle is wired

`scripts/train.py` calls `apply_embedding_init(model, cfg)` immediately after
`build_model(cfg)`.  When `cfg.model.embedding_init` is `"phonological"`, this
helper:

1. Validates that `cfg.data.encoder == "bpe"` and `cfg.data.bpe_tokenizer_path`
   is set (raises `ValueError` otherwise — fast fail so misconfiguration is caught
   before training starts).
2. Loads the trained `BPETokenizer` from the configured path.
3. Calls `phonological_init_fn(model.embed.weight, vocab_size, d_model, vocab=tok.get_vocab())`.

When `embedding_init` is `None` (or any other value), `apply_embedding_init` is
a no-op and the embedding weight is whatever `build_model` produced.

To use `phonological` init you must also set the BPE tokenizer path:

```yaml
# configs/base.yaml
model:
  embedding_init: phonological

data:
  encoder: bpe
  bpe_tokenizer_path: tokenizer/bpe_16k.json
```

To run a phonological-vs-random A/B on identical data and model seed:

```bash
# Phonological
python scripts/train.py --config configs/base.yaml

# Random (control) — override at CLI
python scripts/train.py --config configs/base.yaml \
    model.embedding_init=null
```

Both runs use the same `train.seed`, so all other random operations are
identical and the only variable is the embedding init.

---

## Probe: phonological clustering vs. random

The probe measures how well phonological init separates voiced from unvoiced
consonant tokens using silhouette score (cosine distance):

```bash
python scripts/probe_embedding.py \
    --tokenizer tokenizer/bpe_16k.json \
    --d-model 128
```

Output example:

```
============================================================
Phonological Embedding Probe Results
============================================================
  N labelled tokens (voiced/unvoiced): 42 / 38
  Silhouette (phonological):  +0.1823
  Silhouette (random init):   -0.0041
  Delta (phon - random):      +0.1864
============================================================
  PASS: phonological init clusters voiced/unvoiced better than random.
```

A positive delta confirms that the articulatory structure is injected before
any training occurs.  The probe does NOT require any trained model — it tests
only the initialisation.

---

## Dependencies

Added to `pyproject.toml`:

```toml
"cmudict>=1.0"      # pronunciation lexicon (ships as a package asset, offline)
"panphon>=0.20"     # articulatory feature vectors (ships own data, offline)
"scikit-learn>=1.4" # silhouette_score for the probe
```

Both `cmudict` and `panphon` bundle their lexicon/data as package assets.
No network access is needed at init time or inference time.
