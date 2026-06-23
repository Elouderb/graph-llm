"""Dataset loader and trivial encoder for Phase 0.

The smoke path (source="synthetic") is fully offline and deterministic.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split

from graph_llm.config import DataConfig

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Encoder protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class TextEncoder(Protocol):
    """Minimal interface shared by ByteLevelEncoder and BPETokenizer."""

    def encode(self, text: str) -> list[int]: ...


# ---------------------------------------------------------------------------
# Byte-level encoder
# ---------------------------------------------------------------------------


class ByteLevelEncoder:
    """Trivial identity byte encoder: vocab_size = 256, no UNK.

    Suitable only for byte-level models.  The real tokenizer (16k BPE +
    phonological init) is implemented in card e1644700.
    """

    vocab_size: int = 256

    def encode(self, text: str) -> list[int]:
        """Encode a UTF-8 string to a list of byte ints."""
        return list(text.encode("utf-8", errors="replace"))

    def decode(self, ids: list[int]) -> str:
        """Decode a list of byte ints to a string."""
        return bytes(ids).decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------


class SyntheticDataset(Dataset):
    """Deterministic random-token dataset — no disk/network I/O.

    Generates *n_sequences* sequences of length *seq_len* from a uniform
    distribution over [0, vocab_size).  Useful for offline smoke tests and
    CI.
    """

    def __init__(
        self,
        seq_len: int,
        vocab_size: int = 256,
        n_sequences: int = 256,
        seed: int = 0,
    ) -> None:
        rng = np.random.default_rng(seed)
        self._data = torch.from_numpy(
            rng.integers(0, vocab_size, size=(n_sequences, seq_len + 1), dtype=np.int64)
        )

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        seq = self._data[idx]
        return seq[:-1], seq[1:]  # (input, target), each shape (seq_len,)


class _TextChunkDataset(Dataset):
    """Chunk a flat token array into fixed-length sequences."""

    def __init__(self, tokens: np.ndarray, seq_len: int) -> None:
        n_chunks = (len(tokens) - 1) // seq_len
        self._tokens = torch.from_numpy(tokens[: n_chunks * seq_len + 1].copy())
        self._seq_len = seq_len

    def __len__(self) -> int:
        return (len(self._tokens) - 1) // self._seq_len

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = idx * self._seq_len
        chunk = self._tokens[start : start + self._seq_len + 1]
        return chunk[:-1], chunk[1:]


# ---------------------------------------------------------------------------
# Public builder
# ---------------------------------------------------------------------------


def build_dataloaders(
    cfg: DataConfig,
    vocab_size: int = 256,
    seed: int = 0,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """Build (train_loader, val_loader) from DataConfig.

    Falls back to the synthetic dataset if *source* is not "enwik8" or
    "tinystories", or if the download fails (e.g., no network access).

    Args:
        cfg: DataConfig with source, seq_len, batch_size, val_fraction, data_dir.
        vocab_size: Token vocabulary size (used for synthetic mode).
        seed: RNG seed for dataset splits.
        num_workers: DataLoader worker processes.

    Returns:
        ``(train_loader, val_loader)`` as :class:`torch.utils.data.DataLoader`.
    """
    source = cfg.source.lower()

    # Real byte-level corpora with a *canonical contiguous* split (enwik8/text8:
    # 90M/5M/5M; wikitext103: provided train/valid/test).  These return a fixed
    # (train, val) pair built from the standard split rather than a random split,
    # so results are comparable across runs and across models.  See card
    # 424e3a8e and docs/baselines.md.
    if source in ("enwik8", "text8", "wikitext103"):
        loaders = _try_build_canonical_loaders(cfg, num_workers)
        if loaders is not None:
            return loaders
        # Offline / download failure -> deterministic synthetic fallback so the
        # pipeline still runs (and CI stays green without network access).
        _log.warning("'%s' unavailable; falling back to synthetic data.", source)
        dataset = _make_synthetic(cfg, vocab_size, seed)
    elif source == "synthetic":
        dataset = _make_synthetic(cfg, vocab_size, seed)
    elif source == "tinystories":
        dataset = _try_load_tinystories(cfg) or _make_synthetic(cfg, vocab_size, seed)
    else:
        dataset = _make_synthetic(cfg, vocab_size, seed)

    return _split_and_load(dataset, cfg, seed, num_workers)


def _make_synthetic(cfg: DataConfig, vocab_size: int, seed: int) -> SyntheticDataset:
    return SyntheticDataset(
        seq_len=cfg.seq_len,
        vocab_size=vocab_size,
        n_sequences=max(64, cfg.batch_size * 32),
        seed=seed,
    )


def _split_and_load(
    dataset: Dataset,
    cfg: DataConfig,
    seed: int,
    num_workers: int,
) -> tuple[DataLoader, DataLoader]:
    n_val = max(1, int(len(dataset) * cfg.val_fraction))  # type: ignore[arg-type]
    n_train = len(dataset) - n_val  # type: ignore[arg-type]
    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=generator)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
    )
    return train_loader, val_loader


def _try_load_enwik8(cfg: DataConfig) -> _TextChunkDataset | None:
    """Attempt to load a slice of enwik8; return None on failure."""
    try:
        from datasets import load_dataset  # type: ignore[import-untyped]

        data_dir = Path(cfg.data_dir)
        cache = str(data_dir / "enwik8_cache")
        ds = load_dataset("enwik8", split="train", cache_dir=cache, trust_remote_code=True)
        # Use first 5 MB for smoke runs
        text: str = ds[0]["text"][:5_000_000]  # type: ignore[index]
        tokens = np.frombuffer(text.encode("utf-8", errors="replace"), dtype=np.uint8).astype(
            np.int64
        )
        return _TextChunkDataset(tokens, cfg.seq_len)
    except Exception as e:
        _log.warning("enwik8 load failed, falling back to synthetic data: %s", e)
        return None


def _try_load_tinystories(cfg: DataConfig) -> _TextChunkDataset | None:
    """Attempt to load TinyStories; return None on failure."""
    try:
        from datasets import load_dataset  # type: ignore[import-untyped]

        data_dir = Path(cfg.data_dir)
        cache = str(data_dir / "tinystories_cache")
        ds = load_dataset("roneneldan/TinyStories", split="train", cache_dir=cache)
        # Concatenate first 5000 stories
        texts: list[str] = [row["text"] for row in ds.select(range(min(5000, len(ds))))]  # type: ignore[union-attr]
        big_text = "\n".join(texts)
        encoder = get_encoder(cfg)
        tokens = np.array(encoder.encode(big_text), dtype=np.int64)
        return _TextChunkDataset(tokens, cfg.seq_len)
    except Exception as e:
        _log.warning("TinyStories load failed, falling back to synthetic data: %s", e)
        return None


# ---------------------------------------------------------------------------
# Canonical byte-level corpora (enwik8 / text8 / WikiText-103)  — card 424e3a8e
# ---------------------------------------------------------------------------
#
# Eval-metric focus at Phase 0 is **bits-per-byte (BPB)**, computed byte-level.
# BPB is tokenizer-independent, so it is the cleanest cross-model comparison
# before the real 16k tokenizer (card e1644700) lands.  WikiText-103 is *wired*
# here but also scored byte-level (BPB) for now.
#
# TOKENIZER SEAM (do NOT remove): to switch WikiText-103 (or any corpus) to
# token-level perplexity once card e1644700 ships, replace the byte encoder
# below with the trained tokenizer's ``encode`` and pass that vocab size
# through. The chunking (`_TextChunkDataset`) and the canonical split are
# tokenizer-agnostic, so only the *encoding* line changes — nothing downstream.

# Canonical enwik8/text8 split: first 90M bytes train, next 5M val, next 5M test.
_ENWIK8_TRAIN_BYTES = 90_000_000
_ENWIK8_VAL_BYTES = 5_000_000
_ENWIK8_TEST_BYTES = 5_000_000


def _bytes_to_tokens(raw: bytes) -> np.ndarray:
    """View raw bytes as an ``int64`` token array (byte-level, vocab=256)."""
    return np.frombuffer(raw, dtype=np.uint8).astype(np.int64)


def canonical_byte_splits(
    tokens: np.ndarray,
    train_bytes: int = _ENWIK8_TRAIN_BYTES,
    val_bytes: int = _ENWIK8_VAL_BYTES,
    test_bytes: int = _ENWIK8_TEST_BYTES,
) -> dict[str, np.ndarray]:
    """Slice a flat byte-token array into the canonical train/val/test split.

    Contiguous, in order (NOT random) — this is the standard enwik8/text8
    protocol (90M/5M/5M).  When the corpus is shorter than the requested sizes
    (e.g. a small fixture), the splits are scaled down proportionally so the
    function is still usable in unit tests.

    Args:
        tokens: 1-D ``int64`` array of byte tokens.
        train_bytes: Train-split length in bytes.
        val_bytes: Val-split length in bytes.
        test_bytes: Test-split length in bytes.

    Returns:
        ``{"train": ..., "val": ..., "test": ...}`` as ``np.ndarray`` views.
    """
    total_req = train_bytes + val_bytes + test_bytes
    n = len(tokens)
    if n < total_req:
        # Scale the split proportionally for short corpora / fixtures.
        scale = n / total_req
        train_bytes = int(train_bytes * scale)
        val_bytes = int(val_bytes * scale)
        # test gets the remainder so the three splits tile the array exactly
    train = tokens[:train_bytes]
    val = tokens[train_bytes : train_bytes + val_bytes]
    test = tokens[train_bytes + val_bytes :]
    return {"train": train, "val": val, "test": test}


def _cache_path(cfg: DataConfig, name: str) -> Path:
    """Return the lazy on-disk cache path for a raw corpus dump (gitignored)."""
    return Path(cfg.data_dir) / f"{name}.bin"


def _load_or_fetch_bytes(cfg: DataConfig, name: str, fetch) -> bytes | None:
    """Return the raw corpus bytes, using a lazy on-disk cache.

    On the first call ``fetch()`` downloads the corpus (via ``datasets``); the
    raw bytes are cached under ``cfg.data_dir`` (which is ``.gitignore``d) so
    subsequent runs are offline.  Returns ``None`` if the fetch fails (no
    network), letting callers fall back to synthetic data.
    """
    path = _cache_path(cfg, name)
    if path.exists():
        return path.read_bytes()
    try:
        raw = fetch()
    except Exception as e:  # noqa: BLE001 — any fetch failure -> offline fallback
        _log.warning("%s fetch failed (offline?): %s", name, e)
        return None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    except OSError as e:
        _log.warning("Could not cache %s to %s: %s", name, path, e)
    return raw


def _fetch_enwik8_bytes() -> bytes:
    from datasets import load_dataset  # type: ignore[import-untyped]

    # enwik8 is the standard first-100MB Wikipedia byte-stream benchmark; HF
    # returns it pre-decoded as text rows. Re-encode WITHOUT silent replacement
    # so any decode/encode surprise fails loudly instead of silently corrupting
    # the byte stream that bits-per-byte is defined on (a Latin-1-vs-UTF-8
    # round-trip would otherwise inflate non-ASCII bytes unnoticed). NOTE: HF
    # decoding is assumed UTF-8 (lossless round-trip); validate len(raw) against
    # the canonical ~10^8 bytes before trusting BPB on a real enwik8 run.
    ds = load_dataset("enwik8", split="train", trust_remote_code=True)
    text = "".join(ds["text"])  # type: ignore[index]
    return text.encode("utf-8")


def _fetch_text8_bytes() -> bytes:
    from datasets import load_dataset  # type: ignore[import-untyped]

    ds = load_dataset("afmck/text8", split="train")
    text = "".join(ds["text"])  # type: ignore[index]
    return text.encode("utf-8", errors="replace")


def _fetch_wikitext103_bytes() -> bytes:
    from datasets import load_dataset  # type: ignore[import-untyped]

    # NOTE (tokenizer seam): scored byte-level (BPB) for now; swap the encoder
    # for the 16k tokenizer (card e1644700) to move to token-level perplexity.
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
    text = "".join(ds["text"])  # type: ignore[index]
    return text.encode("utf-8", errors="replace")


_CANONICAL_FETCHERS = {
    "enwik8": _fetch_enwik8_bytes,
    "text8": _fetch_text8_bytes,
    "wikitext103": _fetch_wikitext103_bytes,
}


def make_canonical_split_datasets(
    raw: bytes,
    seq_len: int,
    encoder: TextEncoder | None = None,
) -> dict[str, _TextChunkDataset]:
    """Build {train,val,test} chunk datasets from raw corpus bytes.

    By default uses byte-level encoding (vocab=256).  Pass an ``encoder``
    (e.g. a :class:`~graph_llm.tokenizer.BPETokenizer`) to tokenize at the
    BPE level instead.  The canonical 90M/5M/5M byte-offset split is applied
    first, then each split is encoded independently so there is no cross-split
    context leakage.

    Exposed for unit tests so the split/chunk logic can be exercised on a small
    in-memory fixture with no network or disk I/O.
    """
    if encoder is None or isinstance(encoder, ByteLevelEncoder):
        # Fast path: byte-level, no decode round-trip needed.
        tokens = _bytes_to_tokens(raw)
        splits = canonical_byte_splits(tokens)
        return {name: _TextChunkDataset(arr, seq_len) for name, arr in splits.items()}

    # BPE (or other text encoder): split bytes by offset first, then encode each
    # split as text so no cross-split context leaks through the tokenizer.
    total = len(raw)
    total_req = _ENWIK8_TRAIN_BYTES + _ENWIK8_VAL_BYTES + _ENWIK8_TEST_BYTES
    if total < total_req:
        scale = total / total_req
        train_end = int(_ENWIK8_TRAIN_BYTES * scale)
        val_end = train_end + int(_ENWIK8_VAL_BYTES * scale)
    else:
        train_end = _ENWIK8_TRAIN_BYTES
        val_end = _ENWIK8_TRAIN_BYTES + _ENWIK8_VAL_BYTES

    byte_splits = {
        "train": raw[:train_end],
        "val": raw[train_end:val_end],
        "test": raw[val_end:],
    }
    result: dict[str, _TextChunkDataset] = {}
    for name, chunk in byte_splits.items():
        text = chunk.decode("utf-8", errors="replace")
        ids = np.array(encoder.encode(text), dtype=np.int64)
        result[name] = _TextChunkDataset(ids, seq_len)
    return result


# ---------------------------------------------------------------------------
# Encoder factory (card e1644700) — additive, does NOT modify above functions
# ---------------------------------------------------------------------------


def get_encoder(cfg: DataConfig):  # noqa: ANN201
    """Return the encoder specified by ``cfg.encoder``.

    Returns either a :class:`ByteLevelEncoder` (``encoder="byte"``, default) or
    a :class:`~graph_llm.tokenizer.BPETokenizer` (``encoder="bpe"``).

    For ``"bpe"``, ``cfg.bpe_tokenizer_path`` must point to a saved tokenizer
    JSON.  If the path is missing, raises ``FileNotFoundError``.

    The returned object exposes ``.encode(text) -> list[int]``,
    ``.decode(ids) -> str``, and ``.vocab_size: int`` — the same interface as
    :class:`ByteLevelEncoder`.
    """
    enc = getattr(cfg, "encoder", "byte")
    if enc == "byte":
        return ByteLevelEncoder()
    if enc == "bpe":
        from graph_llm.tokenizer.bpe import BPETokenizer  # lazy import

        path = getattr(cfg, "bpe_tokenizer_path", None)
        if path is None:
            raise ValueError(
                "cfg.data.bpe_tokenizer_path must be set when encoder='bpe'."
            )
        return BPETokenizer.from_pretrained(path)
    raise ValueError(f"Unknown encoder '{enc}'. Choose 'byte' or 'bpe'.")


def _try_build_canonical_loaders(
    cfg: DataConfig, num_workers: int
) -> tuple[DataLoader, DataLoader] | None:
    """Build (train_loader, val_loader) for a canonical byte-level corpus.

    The val split is the canonical validation set (NOT a random slice of train),
    so reported BPB is the standard held-out number.  Returns ``None`` on
    download failure so the caller can fall back to synthetic data.
    """
    source = cfg.source.lower()
    fetch = _CANONICAL_FETCHERS.get(source)
    if fetch is None:
        return None
    raw = _load_or_fetch_bytes(cfg, source, fetch)
    if raw is None:
        return None
    encoder = get_encoder(cfg)
    datasets_by_split = make_canonical_split_datasets(raw, cfg.seq_len, encoder=encoder)
    train_ds = datasets_by_split["train"]
    val_ds = datasets_by_split["val"]
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
    )
    return train_loader, val_loader
