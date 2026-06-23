"""Phonological embedding initialiser for card e1644700.

Pipeline
--------
For each token in the vocabulary:

1. Strip the byte-level prefix character (Ġ / Ċ) to get the raw subword text.
2. Normalise to lowercase and look up in CMUdict.  On a miss, try simple
   heuristic stripping of leading punctuation or trailing morpheme markers
   (##, ▁) — common in BPE vocabularies.
3. For each phoneme in the first CMUdict pronunciation, strip the ARPABET
   stress digit, map to an IPA symbol, then call panphon to get the 24-d
   articulatory feature vector.
4. Average all per-phoneme vectors → one 24-d feature vector per token.
5. Linear project (W: 24 → d_model, no bias) to produce the token's initial
   embedding.  W is drawn once from N(0, 1/24) and fixed for the init pass.
6. Tokens with no CMUdict entry receive the default nn.Embedding random init
   (already in the weight tensor before this function is called).

Coverage is logged and returned so downstream scripts can report it.

Registration
------------
This module registers ``phonological_init_fn`` under the name ``"phonological"``
via :func:`graph_llm.models.registry.register_embedding_init`.  Importing the
module (done automatically when ``embedding_init="phonological"`` is resolved)
is sufficient to activate the registration — no other wiring is needed.

Offline operation
-----------------
Both CMUdict and panphon ship their data as package assets; no network access
is required.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from torch import Tensor

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ARPABET → IPA mapping
# ---------------------------------------------------------------------------

# Maps ARPABET phones (uppercase, no stress digit) to IPA symbols that panphon
# can parse.  Diphthongs are split into their component IPA segments so panphon
# can process each symbol independently.
_ARPA_TO_IPA: dict[str, list[str]] = {
    "AA": ["ɑ"],
    "AE": ["æ"],
    "AH": ["ʌ"],
    "AO": ["ɔ"],
    "AW": ["a", "ʊ"],
    "AY": ["a", "ɪ"],
    "B":  ["b"],
    "CH": ["t", "ʃ"],
    "D":  ["d"],
    "DH": ["ð"],
    "EH": ["ɛ"],
    "ER": ["ɝ"],
    "EY": ["e", "ɪ"],
    "F":  ["f"],
    "G":  ["ɡ"],
    "HH": ["h"],
    "IH": ["ɪ"],
    "IY": ["i"],
    "JH": ["d", "ʒ"],
    "K":  ["k"],
    "L":  ["l"],
    "M":  ["m"],
    "N":  ["n"],
    "NG": ["ŋ"],
    "OW": ["o", "ʊ"],
    "OY": ["ɔ", "ɪ"],
    "P":  ["p"],
    "R":  ["ɹ"],
    "S":  ["s"],
    "SH": ["ʃ"],
    "T":  ["t"],
    "TH": ["θ"],
    "UH": ["ʊ"],
    "UW": ["u"],
    "V":  ["v"],
    "W":  ["w"],
    "Y":  ["j"],
    "Z":  ["z"],
    "ZH": ["ʒ"],
}

_N_ARPA_FEATURES: int = 24  # panphon feature dimension


# ---------------------------------------------------------------------------
# Lazy singletons
# ---------------------------------------------------------------------------

_cmudict_cache: dict[str, list[str]] | None = None
_panphon_ft = None  # panphon.FeatureTable singleton


def _get_cmudict() -> dict[str, list[str]]:
    """Return CMUdict as {word_lower: [ARPABET phones, ...]}."""
    global _cmudict_cache
    if _cmudict_cache is None:
        import cmudict as _cmu  # type: ignore[import-untyped]

        raw: dict[str, list[list[str]]] = _cmu.dict()
        # Keep only the first pronunciation for each word.
        _cmudict_cache = {w: phones[0] for w, phones in raw.items()}
    return _cmudict_cache


def _get_ft():  # noqa: ANN201
    """Return the panphon FeatureTable singleton."""
    global _panphon_ft
    if _panphon_ft is None:
        import panphon  # type: ignore[import-untyped]

        _panphon_ft = panphon.FeatureTable()
    return _panphon_ft


# ---------------------------------------------------------------------------
# Per-token phoneme vector
# ---------------------------------------------------------------------------

# Byte-level BPE prefix for a leading space (Ġ = U+0120)
_BPE_SPACE = "Ġ"
# Regex to strip non-alpha characters from subword text for lookup
_STRIP_RE = re.compile(r"[^a-z\-']+")


def _clean_token(token: str) -> str:
    """Strip BPE artefacts and non-alphabetic characters for CMUdict lookup."""
    # Remove byte-level space prefix
    text = token.lstrip(_BPE_SPACE)
    # Remove sub-word markers used by other tokenisers (## or ▁)
    text = text.lstrip("▁").lstrip("#")
    # Lower-case and strip non-alpha (digits, punctuation)
    text = text.lower()
    text = _STRIP_RE.sub("", text)
    return text


def _phones_to_feature_vector(phones: list[str]) -> np.ndarray | None:
    """Convert ARPABET phone list → averaged panphon feature vector (24-d).

    Returns None if no phone resolves to a panphon vector.
    """
    ft = _get_ft()
    all_vecs: list[list[float]] = []
    for arpa in phones:
        bare = arpa.rstrip("012")  # strip stress digits
        ipa_segs = _ARPA_TO_IPA.get(bare.upper())
        if ipa_segs is None:
            continue
        for ipa_sym in ipa_segs:
            vecs = ft.word_to_vector_list(ipa_sym, numeric=True)
            if vecs:
                all_vecs.extend(vecs)
    if not all_vecs:
        return None
    return np.mean(all_vecs, axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# Main init function
# ---------------------------------------------------------------------------


def compute_phonological_vectors(
    vocab: dict[str, int],
) -> tuple[dict[int, np.ndarray], float]:
    """Map every token in *vocab* to a 24-d articulatory feature vector.

    Parameters
    ----------
    vocab:
        ``{token_string: token_id}`` mapping from the trained tokenizer.

    Returns
    -------
    vectors:
        ``{token_id: feature_vec}`` for tokens that have a CMUdict entry.
    coverage:
        Fraction of vocab tokens that received a phonological vector (0–1).

        .. note::

            The denominator is ``len(vocab)`` — the *total* vocabulary count —
            which includes special tokens (``<pad>``, ``<unk>``, ``<bos>``,
            ``<eos>``), raw byte tokens (256 entries), punctuation, and numeric
            tokens.  None of these map to CMUdict entries, so reported coverage
            is always lower than the fraction of *word-like* tokens that were
            matched.  For a 16,384-token BPE vocab trained on English text,
            expect ~5–35% depending on corpus diversity.
    """
    cmudict = _get_cmudict()
    vectors: dict[int, np.ndarray] = {}

    for token, tid in vocab.items():
        key = _clean_token(token)
        if not key:
            continue
        phones = cmudict.get(key)
        if phones is None:
            continue
        vec = _phones_to_feature_vector(phones)
        if vec is not None:
            vectors[tid] = vec

    coverage = len(vectors) / max(len(vocab), 1)
    return vectors, coverage


def phonological_init_fn(
    weight: Tensor,
    vocab_size: int,
    d_model: int,
    *,
    vocab: dict[str, int] | None = None,
    seed: int = 0,
) -> None:
    """Apply phonological initialisation to an embedding weight tensor.

    This function is registered as ``"phonological"`` and called by the model
    constructor **after** ``_init_weights()`` so the phonological values take
    precedence over the default random init for mappable tokens.

    Parameters
    ----------
    weight:
        The ``nn.Embedding.weight`` tensor, shape (vocab_size, d_model).
        Modified in-place.
    vocab_size:
        Number of tokens (must match ``weight.shape[0]``).
    d_model:
        Embedding dimension (must match ``weight.shape[1]``).
    vocab:
        ``{token: id}`` dict from the trained BPETokenizer.  If None, the
        function is a no-op and emits a warning (useful for unit-test stubs
        that only test the registration hook).
    seed:
        RNG seed for the projection matrix W.
    """
    if vocab is None:
        _log.warning(
            "phonological_init_fn called without a vocab dict — "
            "no phonological structure injected, embedding left at random init. "
            "Setting embedding_init='phonological' in the config is NOT sufficient on its own. "
            "You must also call apply_embedding_init(model, cfg) from your training script "
            "after build_model(cfg) returns. "
            "See graph_llm.tokenizer.phonological_init.apply_embedding_init for details."
        )
        return

    phon_vecs, coverage = compute_phonological_vectors(vocab)
    _log.info(
        "Phonological init: %d / %d tokens mapped (coverage %.1f%%)",
        len(phon_vecs),
        vocab_size,
        coverage * 100,
    )

    if not phon_vecs:
        _log.warning("No tokens mapped; embedding left at random init.")
        return

    # Build projection matrix W: (24, d_model).
    # Fixed seed so the projection is reproducible for a given d_model.
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((_N_ARPA_FEATURES, d_model)).astype(np.float32)
    W /= np.sqrt(_N_ARPA_FEATURES)  # scale ~ N(0, 1/sqrt(24))

    # Target standard deviation for all embedding rows (matches nn.Embedding
    # default random init in the surrounding rows so phonological and random
    # tokens start at the same norm scale, ensuring *direction* carries
    # phonological structure rather than magnitude).
    target_std = 0.02

    # Move tensors to the same device (and dtype) as the embedding weight so
    # this works on CUDA as well as CPU.
    dev = weight.device
    dt = weight.dtype
    W_t = torch.from_numpy(W).to(dev, dtype=dt)  # (24, d_model)

    with torch.no_grad():
        for tid, fvec in phon_vecs.items():
            if tid >= weight.shape[0]:
                continue
            fvec_t = torch.from_numpy(fvec).to(dev, dtype=dt).unsqueeze(0)  # (1, 24)
            proj = (fvec_t @ W_t).squeeze(0)                                 # (d_model,)

            # Rescale so the projected row has the same per-component std as
            # the surrounding random-init rows (~0.02).  Without this the
            # projected rows are O(0.5–1.0) per component — 25–50× larger —
            # creating a norm artifact that confounds the phonological-vs-random
            # ablation. We preserve *direction* (phonological content) and set
            # *scale* to match the surrounding rows.
            row_std = proj.std()
            if row_std > 0:
                proj = proj * (target_std / row_std)

            weight[tid] = proj

    _log.info(
        "Phonological init complete: %d tokens initialised at std≈%.4f, "
        "%d left at random (%.1f%% unmapped).",
        len(phon_vecs),
        target_std,
        vocab_size - len(phon_vecs),
        (1 - coverage) * 100,
    )


# ---------------------------------------------------------------------------
# End-to-end wiring helper (called from training script)
# ---------------------------------------------------------------------------


def apply_embedding_init(model: torch.nn.Module, cfg: object) -> None:
    """Apply phonological embedding init when the config requests it.

    This is the *only* place the ablation toggle is wired end-to-end.
    Call it immediately after ``build_model(cfg)`` in training scripts:

    .. code-block:: python

        model = build_model(cfg)
        apply_embedding_init(model, cfg)  # no-op if embedding_init is not "phonological"

    Parameters
    ----------
    model:
        The built model.  Must expose ``model.embed.weight`` (nn.Embedding).
    cfg:
        A :class:`graph_llm.config.Config` instance.

    Raises
    ------
    ValueError
        If ``cfg.model.embedding_init == "phonological"`` but
        ``cfg.data.encoder != "bpe"`` or ``cfg.data.bpe_tokenizer_path`` is None.
    """
    # Resolve cfg attribute access defensively to work with dataclasses and
    # plain namespace objects alike.
    model_cfg = getattr(cfg, "model", cfg)
    data_cfg = getattr(cfg, "data", cfg)

    if getattr(model_cfg, "embedding_init", None) != "phonological":
        return  # Nothing to do — random init or another method.

    encoder = getattr(data_cfg, "encoder", None)
    tok_path = getattr(data_cfg, "bpe_tokenizer_path", None)

    if encoder != "bpe" or not tok_path:
        raise ValueError(
            "cfg.model.embedding_init='phonological' requires "
            "cfg.data.encoder='bpe' and a non-empty cfg.data.bpe_tokenizer_path. "
            f"Got encoder={encoder!r}, bpe_tokenizer_path={tok_path!r}."
        )

    from graph_llm.tokenizer.bpe import BPETokenizer  # local import to avoid cycles

    tok = BPETokenizer.from_pretrained(tok_path)
    vocab = tok.get_vocab()

    weight = model.embed.weight  # type: ignore[attr-defined]
    vocab_size, d_model = weight.shape

    if weight.shape[0] < len(vocab):
        raise ValueError(
            f"model.embed.weight vocab_size={weight.shape[0]} is smaller than "
            f"tokenizer vocab size={len(vocab)}. "
            f"Set cfg.model.vocab_size={len(vocab)} so the embedding table "
            f"is large enough for all BPE token ids."
        )

    _log.info(
        "apply_embedding_init: running phonological init "
        "(vocab_size=%d, d_model=%d, tokenizer=%s)",
        vocab_size,
        d_model,
        tok_path,
    )
    phonological_init_fn(weight, vocab_size, d_model, vocab=vocab)


# ---------------------------------------------------------------------------
# Registry hook — executed on import
# ---------------------------------------------------------------------------

from graph_llm.models.registry import register_embedding_init  # noqa: E402


@register_embedding_init("phonological")
def _registered_phonological_init(
    weight: Tensor, vocab_size: int, d_model: int
) -> None:
    """Registry shim — delegates to :func:`phonological_init_fn`.

    When called from the model constructor (which only passes weight,
    vocab_size, d_model), operates without a vocab dict (no-op phonologically,
    random init remains).  For real phonological init, call
    :func:`phonological_init_fn` directly with ``vocab=tokenizer.get_vocab()``.

    Rationale: the model registry contract is
    ``(weight, vocab_size, d_model) -> None`` and the model constructor does
    not have access to the tokenizer vocab.  The recommended usage pattern is::

        model = build_model(cfg)           # hooks run, embedding gets random init
        phon_init(model.embed.weight,      # then overwrite with phonological values
                  vocab_size, d_model,
                  vocab=tokenizer.get_vocab())
    """
    # Called via registry hook — vocab not available here.
    # The actual phonological values are written by the training script or
    # probe script after the tokenizer vocab is loaded.
    phonological_init_fn(weight, vocab_size, d_model, vocab=None)
