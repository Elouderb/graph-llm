"""Probe phonological embedding cluster quality vs. random init.

Usage
-----
Run after training a BPE tokenizer::

    python scripts/probe_embedding.py \\
        --tokenizer tokenizer/bpe_16k.json \\
        --d-model 128

The script initialises two embedding matrices (same vocab, same shape) and
computes silhouette scores for a binary partition of voiced vs. unvoiced
consonants — a purely articulatory distinction that phonological init
should recover, but random init cannot.

Metrics reported
----------------
- N_voiced / N_unvoiced: number of tokens in each category
- silhouette_phonological: silhouette score for phonological init
- silhouette_random: silhouette score for random init (control)
- delta: difference (higher = better phonological clustering)

Exit code 0 on success; 1 if the probe cannot run (not enough mapped tokens).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# IPA voiced-consonant reference set (single IPA chars used by _ARPA_TO_IPA)
# ---------------------------------------------------------------------------

# Voiced consonants that panphon marks [+voice].
# We probe voiced vs. unvoiced on the set of tokens that CMUdict maps to purely
# consonantal pronunciations (avoids the complexity of vowel voicing).
_VOICED_CONSONANTS_IPA = {"b", "d", "ɡ", "v", "ð", "z", "ʒ", "m", "n", "ŋ", "l", "ɹ", "w", "j"}
_UNVOICED_CONSONANTS_IPA = {"p", "t", "k", "f", "θ", "s", "ʃ", "h"}

# ARPABET phones that are purely voiced or unvoiced consonants
# (excluding stops that are context-dependent; we keep it simple)
_VOICED_ARPA = {"B", "D", "G", "V", "DH", "Z", "ZH", "M", "N", "NG", "L", "R", "W", "Y"}
_UNVOICED_ARPA = {"P", "T", "K", "F", "TH", "S", "SH", "HH"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _classify_token(phones: list[str]) -> str | None:
    """Return 'voiced', 'unvoiced', or None if mixed/ambiguous."""
    bare_phones = {p.rstrip("012").upper() for p in phones}
    has_voiced = bool(bare_phones & _VOICED_ARPA)
    has_unvoiced = bool(bare_phones & _UNVOICED_ARPA)
    # Purely consonantal and single-category
    if has_voiced and not has_unvoiced:
        return "voiced"
    if has_unvoiced and not has_voiced:
        return "unvoiced"
    return None


def _build_phonological_embedding(
    vocab: dict[str, int],
    d_model: int,
    seed: int = 0,
) -> torch.Tensor:
    """Return a (vocab_size, d_model) embedding with phonological init.

    Delegates to :func:`graph_llm.tokenizer.phonological_init.phonological_init_fn`
    so the probe uses *exactly* the same projection and norm-rescaling logic as
    the training path — no reimplementation.
    """
    from graph_llm.tokenizer.phonological_init import phonological_init_fn

    vocab_size = max(vocab.values()) + 1

    # Start from the same random init as the training path so unmapped tokens
    # are not all-zero and the comparison against random-init is fair.
    torch.manual_seed(seed)
    weight = torch.empty(vocab_size, d_model)
    torch.nn.init.normal_(weight, mean=0.0, std=0.02)

    phonological_init_fn(weight, vocab_size, d_model, vocab=vocab, seed=seed)
    return weight


def _build_random_embedding(vocab: dict[str, int], d_model: int, seed: int = 99) -> torch.Tensor:
    vocab_size = max(vocab.values()) + 1
    torch.manual_seed(seed)
    weight = torch.empty(vocab_size, d_model)
    torch.nn.init.normal_(weight, mean=0.0, std=0.02)
    return weight


def _collect_label_vectors(
    vocab: dict[str, int],
    embedding: torch.Tensor,
) -> tuple[list[int], np.ndarray]:
    """Return (labels, vectors) for tokens with a voiced/unvoiced classification."""
    import cmudict as _cmu  # type: ignore[import-untyped]

    from graph_llm.tokenizer.phonological_init import _clean_token

    raw = _cmu.dict()
    cmudict_first: dict[str, list[str]] = {w: v[0] for w, v in raw.items()}

    labels: list[int] = []  # 0=voiced, 1=unvoiced
    vecs: list[np.ndarray] = []

    for token, tid in vocab.items():
        key = _clean_token(token)
        if not key:
            continue
        phones = cmudict_first.get(key)
        if phones is None:
            continue
        label = _classify_token(phones)
        if label is None:
            continue
        labels.append(0 if label == "voiced" else 1)
        vecs.append(embedding[tid].detach().numpy())

    return labels, np.array(vecs) if vecs else np.empty((0, embedding.shape[1]))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Probe phonological vs. random embedding cluster quality."
    )
    parser.add_argument(
        "--tokenizer",
        required=True,
        help="Path to the saved BPETokenizer JSON.",
    )
    parser.add_argument(
        "--d-model",
        type=int,
        default=128,
        help="Embedding dimension (must match training config; default: 128).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed for the projection matrix and random baseline (default: 0).",
    )
    args = parser.parse_args()

    from graph_llm.tokenizer.bpe import BPETokenizer

    _log.info("Loading tokenizer from %s", args.tokenizer)
    tok = BPETokenizer.from_pretrained(args.tokenizer)
    vocab = tok.get_vocab()
    _log.info("Vocab size: %d", len(vocab))

    _log.info("Building phonological embedding (d_model=%d)…", args.d_model)
    phon_emb = _build_phonological_embedding(vocab, args.d_model, seed=args.seed)

    _log.info("Building random embedding (control)…")
    rand_emb = _build_random_embedding(vocab, args.d_model, seed=args.seed + 99)

    _log.info("Collecting voiced/unvoiced labels…")
    labels, phon_vecs = _collect_label_vectors(vocab, phon_emb)
    _, rand_vecs = _collect_label_vectors(vocab, rand_emb)

    if len(labels) < 4:
        _log.error(
            "Only %d labelled tokens found — not enough for a silhouette score. "
            "Try a larger/more realistic corpus.",
            len(labels),
        )
        sys.exit(1)

    from sklearn.metrics import silhouette_score  # type: ignore[import-untyped]

    labels_arr = np.array(labels)
    n_voiced = int((labels_arr == 0).sum())
    n_unvoiced = int((labels_arr == 1).sum())
    _log.info("Labelled tokens: %d voiced, %d unvoiced", n_voiced, n_unvoiced)

    sil_phon = float(silhouette_score(phon_vecs, labels_arr, metric="cosine"))
    sil_rand = float(silhouette_score(rand_vecs, labels_arr, metric="cosine"))
    delta = sil_phon - sil_rand

    print("\n" + "=" * 60)
    print("Phonological Embedding Probe Results")
    print("=" * 60)
    print(f"  N labelled tokens (voiced/unvoiced): {n_voiced} / {n_unvoiced}")
    print(f"  Silhouette (phonological):  {sil_phon:+.4f}")
    print(f"  Silhouette (random init):   {sil_rand:+.4f}")
    print(f"  Delta (phon - random):      {delta:+.4f}")
    print("=" * 60)
    if delta > 0:
        print("  PASS: phonological init clusters voiced/unvoiced better than random.")
    else:
        print("  INFO: phonological init does not outperform random on this corpus.")
    print()

    sys.exit(0)


if __name__ == "__main__":
    main()
