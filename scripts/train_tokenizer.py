"""Train and save the 16,384-vocab byte-level BPE tokenizer.

Usage
-----
Offline (bundled synthetic corpus, always works)::

    python scripts/train_tokenizer.py --output tokenizer/bpe_16k.json

With TinyStories (if available; requires network on first run)::

    python scripts/train_tokenizer.py --source tinystories \\
        --data-dir data/ --output tokenizer/bpe_16k.json

With enwik8::

    python scripts/train_tokenizer.py --source enwik8 \\
        --data-dir data/ --output tokenizer/bpe_16k.json

After training, set ``data.encoder = bpe`` and
``data.bpe_tokenizer_path = <output>`` in your YAML config.

Corpus choices
--------------
``synthetic`` (default, offline):
    A bundled ~1 MB pseudo-English corpus of common English words repeated
    to give BPE enough material to reach 16,384 merges.  Not a realistic
    language distribution but is sufficient to produce a deterministic,
    self-consistent vocab that the tests can verify without network access.

``tinystories``:
    Loads from HuggingFace ``roneneldan/TinyStories``.  First 10,000 stories
    (~10 MB of English text) are used.  Produces a realistic vocabulary.

``enwik8``:
    Loads the first 20 MB of enwik8 (Wikipedia XML).  Realistic but slower.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow running from repo root without editable install
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from graph_llm.tokenizer.bpe import VOCAB_SIZE, BPETokenizer  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Bundled synthetic corpus
# ---------------------------------------------------------------------------

_DICT_FILE = Path("/usr/share/dict/words")

# Fallback seed sentences used when the system dictionary is unavailable.
# Repeated 1000x to give BPE enough text (results in a smaller vocab if
# the corpus is insufficiently diverse).
_FALLBACK_SENTENCES = [
    "the quick brown fox jumps over the lazy dog",
    "in the beginning was the word and the word was good",
    "once upon a time in a land far far away there lived a king",
    "to be or not to be that is the question",
    "it was the best of times it was the worst of times",
    "sphinx of black quartz judge my vow",
    "pack my box with five dozen liquor jugs",
    "how vexingly quick daft zebras jump",
    "the five boxing wizards jump quickly",
    "a quick brown fox jumps over the lazy dog while singing",
]


def _build_synthetic_corpus(n_sentences: int = 5_000, seed: int = 42) -> list[str]:
    """Return a diverse corpus for offline BPE training.

    Primary:  sample random-word sentences from /usr/share/dict/words
              (>100k entries → sufficient diversity for 16,384 merges).
    Fallback: repeat the built-in pangram set (vocab may be < 16k).
    """
    import random as _random

    if _DICT_FILE.exists():
        words = _DICT_FILE.read_text().splitlines()
        rng = _random.Random(seed)
        sentences: list[str] = []
        for _ in range(n_sentences):
            n = rng.randint(5, 15)
            sentence = " ".join(rng.choices(words, k=n))
            sentences.append(sentence.lower())
        return sentences

    _log.warning(
        "/usr/share/dict/words not found; using built-in fallback corpus. "
        "The resulting vocabulary may be smaller than %d.",
        VOCAB_SIZE,
    )
    return _FALLBACK_SENTENCES * 1_000


# ---------------------------------------------------------------------------
# Corpus loaders
# ---------------------------------------------------------------------------


def _load_tinystories(data_dir: str, n_stories: int = 10_000) -> list[str]:
    from datasets import load_dataset  # type: ignore[import-untyped]

    _log.info("Loading TinyStories (first %d stories)…", n_stories)
    cache = str(Path(data_dir) / "tinystories_cache")
    ds = load_dataset("roneneldan/TinyStories", split="train", cache_dir=cache)
    return [row["text"] for row in ds.select(range(min(n_stories, len(ds))))]  # type: ignore[union-attr]


def _load_enwik8(data_dir: str, max_bytes: int = 20_000_000) -> list[str]:
    from datasets import load_dataset  # type: ignore[import-untyped]

    _log.info("Loading enwik8 (first %d bytes)…", max_bytes)
    cache = str(Path(data_dir) / "enwik8_cache")
    ds = load_dataset("enwik8", split="train", cache_dir=cache, trust_remote_code=True)
    text: str = "".join(ds["text"])[:max_bytes]  # type: ignore[index]
    # Split into ~100-char chunks so the iterator has many strings
    return [text[i : i + 200] for i in range(0, len(text), 200)]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and save the 16k BPE tokenizer.")
    parser.add_argument(
        "--source",
        choices=["synthetic", "tinystories", "enwik8"],
        default="synthetic",
        help="Corpus source (default: synthetic — fully offline).",
    )
    parser.add_argument(
        "--data-dir",
        default="data/",
        help="Directory for corpus caches (default: data/).",
    )
    parser.add_argument(
        "--output",
        default="tokenizer/bpe_16k.json",
        help="Output path for the saved tokenizer JSON (default: tokenizer/bpe_16k.json).",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=VOCAB_SIZE,
        help=f"Vocabulary size (default: {VOCAB_SIZE}).",
    )
    parser.add_argument(
        "--min-frequency",
        type=int,
        default=1,
        help="Minimum merge frequency (default: 1). Use 1 to guarantee the target vocab size.",
    )
    args = parser.parse_args()

    _log.info("Corpus source: %s", args.source)

    if args.source == "synthetic":
        texts = _build_synthetic_corpus()
        _log.info(
            "Using bundled synthetic corpus (%d sentences, ~%d chars).",
            len(texts),
            sum(len(t) for t in texts),
        )
    elif args.source == "tinystories":
        texts = _load_tinystories(args.data_dir)
    elif args.source == "enwik8":
        texts = _load_enwik8(args.data_dir)
    else:
        raise ValueError(f"Unknown source: {args.source}")

    _log.info(
        "Training BPE tokenizer: vocab_size=%d, min_frequency=%d, corpus_size=%d strings…",
        args.vocab_size,
        args.min_frequency,
        len(texts),
    )

    tok = BPETokenizer.train_on_corpus(
        texts,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        show_progress=True,
    )

    actual = tok._tok.get_vocab_size()
    _log.info("Training complete. Actual vocab size: %d", actual)
    assert actual == args.vocab_size, (
        f"Expected exactly {args.vocab_size} tokens but got {actual}. "
        "Try --min-frequency=1 to allow all rare pairs to be merged."
    )

    tok.save(args.output)
    _log.info("Tokenizer saved to: %s", args.output)

    # Quick smoke-check
    test_text = "the quick brown fox"
    ids = tok.encode(test_text)
    decoded = tok.decode(ids)
    _log.info("Smoke check encode/decode: %r → %s → %r", test_text, ids, decoded)
    assert decoded.strip() == test_text, f"Round-trip mismatch: {decoded!r} != {test_text!r}"
    _log.info("Round-trip OK.")


if __name__ == "__main__":
    main()
