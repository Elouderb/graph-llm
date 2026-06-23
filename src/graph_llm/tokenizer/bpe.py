"""Byte-level BPE tokenizer backed by HuggingFace ``tokenizers``.

Vocab size is fixed at 16,384 (2^14) to match the Phase 1 research budget.
The tokenizer is deterministic and versioned: the saved JSON captures the full
training outcome so that ``from_pretrained`` always produces identical results.

Offline corpus note
-------------------
``train_on_corpus`` accepts any Python iterator of strings and uses
``train_from_iterator``, so no disk files are required at training time.
The ``scripts/train_tokenizer.py`` script bundles a small synthetic English
corpus that is always available offline.  If TinyStories or enwik8 is present
it will be preferred (better vocabulary coverage).

Interface
---------
The class satisfies the interface promised in the Phase 0 stub::

    tok.vocab_size          # int — always 16_384
    tok.encode(text)        # str -> list[int]
    tok.decode(ids)         # list[int] -> str
    BPETokenizer.from_pretrained(path)  # load from saved JSON

Version
-------
TOKENIZER_VERSION = "1.0.0" — bump when vocab or training recipe changes.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from pathlib import Path

from tokenizers import Tokenizer as _HFTokenizer
from tokenizers import decoders, pre_tokenizers, processors
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

_log = logging.getLogger(__name__)

VOCAB_SIZE: int = 16_384
TOKENIZER_VERSION: str = "1.0.0"

# Special tokens: PAD=0, UNK=1, BOS=2, EOS=3
_SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]


class BPETokenizer:
    """Byte-level BPE tokenizer with vocab = 16,384.

    Parameters
    ----------
    _hf_tokenizer:
        Internal HuggingFace tokenizer instance.  Do not construct directly;
        use :meth:`train_on_corpus` or :meth:`from_pretrained`.
    """

    vocab_size: int = VOCAB_SIZE

    def __init__(self, _hf_tokenizer: _HFTokenizer) -> None:
        self._tok = _hf_tokenizer

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def encode(self, text: str) -> list[int]:
        """Encode a string to a list of token IDs."""
        return self._tok.encode(text).ids

    def decode(self, ids: list[int]) -> str:
        """Decode a list of token IDs back to a string."""
        return self._tok.decode(ids)

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save the tokenizer as a single versioned JSON file.

        The file includes a ``graph_llm_version`` key so callers can detect
        schema changes without parsing the full HF JSON.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        raw = json.loads(self._tok.to_str(pretty=True))
        raw["graph_llm_version"] = TOKENIZER_VERSION
        path.write_text(json.dumps(raw, indent=2, ensure_ascii=False))
        _log.info("Tokenizer saved to %s (version %s)", path, TOKENIZER_VERSION)

    @classmethod
    def from_pretrained(cls, path: str | Path) -> BPETokenizer:
        """Load a previously saved tokenizer from *path*.

        Raises
        ------
        FileNotFoundError
            If *path* does not exist.
        ValueError
            If the file carries an unrecognised ``graph_llm_version``.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Tokenizer file not found: {path}")

        raw = json.loads(path.read_text())
        version = raw.pop("graph_llm_version", None)
        if version is None:
            _log.warning("Tokenizer file missing graph_llm_version; loading anyway.")
        elif version != TOKENIZER_VERSION:
            raise ValueError(
                f"Tokenizer version mismatch: file has '{version}', "
                f"code expects '{TOKENIZER_VERSION}'."
            )

        tok = _HFTokenizer.from_str(json.dumps(raw))
        return cls(tok)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    @classmethod
    def train_on_corpus(
        cls,
        texts: Iterable[str],
        vocab_size: int = VOCAB_SIZE,
        min_frequency: int = 2,
        show_progress: bool = True,
    ) -> BPETokenizer:
        """Train a byte-level BPE tokenizer on *texts*.

        Parameters
        ----------
        texts:
            An iterator of plain-text strings (lines or documents).
        vocab_size:
            Target vocabulary size.  Defaults to 16,384.
        min_frequency:
            Minimum merge frequency.  2 is standard.
        show_progress:
            Whether to show the HF training progress bar.

        Returns
        -------
        BPETokenizer
            Trained and ready-to-use tokenizer instance.
        """
        tok = _HFTokenizer(BPE(unk_token="<unk>"))
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
        tok.decoder = decoders.ByteLevel()
        tok.post_processor = processors.ByteLevel(trim_offsets=True)

        trainer = BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=_SPECIAL_TOKENS,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=show_progress,
        )

        tok.train_from_iterator(texts, trainer=trainer)

        actual = tok.get_vocab_size()
        if actual != vocab_size:
            _log.warning(
                "Requested vocab_size=%d but got %d (corpus may be too small).",
                vocab_size,
                actual,
            )

        return cls(tok)

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def get_vocab(self) -> dict[str, int]:
        """Return the full vocabulary as {token: id}."""
        return self._tok.get_vocab()

    def __repr__(self) -> str:
        return (
            f"BPETokenizer(vocab_size={self._tok.get_vocab_size()}, "
            f"version={TOKENIZER_VERSION!r})"
        )
