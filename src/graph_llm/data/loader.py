"""Dataset loader and trivial encoder for Phase 0.

The smoke path (source="synthetic") is fully offline and deterministic.
"""

from __future__ import annotations

import contextlib
import logging
import os
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
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
# Ordered-segment stream (cross-segment persistent-memory training — card
# 61f900ca, piece 3)
# ---------------------------------------------------------------------------
#
# The standard ``_TextChunkDataset`` + shuffled ``DataLoader`` above deliberately
# SHUFFLES chunks, which destroys cross-chunk continuity — fine for the committed
# within-sequence training, useless for teaching the delta-memory state to carry
# across boundaries.  The ordered stream below is the contrast: it yields ORDERED
# CONTIGUOUS segments so consecutive segments tile a single continuous stream, and
# the SegmentedTrainer carries the per-layer memory state across them.  It is
# additive — the shuffled path is untouched.


@dataclass
class OrderedSegment:
    """One ordered contiguous segment of the byte stream.

    Attributes:
        inputs: ``(B, segment_len)`` input token ids.
        targets: ``(B, segment_len)`` next-token targets (the stream shifted by 1).
        stream_reset: ``True`` iff this segment begins a fresh stream window and the
            carried per-layer memory state must be DROPPED (reset) before it — a
            simulated document boundary, or the very first segment.  ``False`` means
            the carried state from the previous segment continues into this one.
    """

    inputs: torch.Tensor
    targets: torch.Tensor
    stream_reset: bool


class OrderedSegmentStream:
    """Iterator over ORDERED CONTIGUOUS segments of a flat token stream.

    The flat ``tokens`` array is split into ``batch_size`` equal-length parallel
    sub-streams (strided so each row is a contiguous slice of the original stream),
    then walked left-to-right in ``segment_len``-token segments.  Consecutive
    segments tile each sub-stream continuously, so a model that carries its state
    across segment boundaries sees the whole prior sub-stream through its bounded
    memory — exactly what the cross-segment-memory training needs.  This is the
    deliberate contrast with the shuffled ``_TextChunkDataset`` loader (kept intact).

    A ``stream_reset`` flag is raised on the first segment and every
    ``stream_reset_interval`` segments thereafter (``0`` == only the first), telling
    the trainer to drop the carried state there (a simulated document boundary;
    text8 is one long stream, so the default never resets within it).

    Args:
        tokens: 1-D ``int64`` token array (the whole stream).
        segment_len: Tokens per segment ``T`` (``>= 1``).
        batch_size: Number ``B`` of parallel sub-streams.
        stream_reset_interval: Reset the carried state every this many segments
            (``0`` == never reset within the stream, only the first segment resets).

    Yields:
        :class:`OrderedSegment` instances, in stream order, until each sub-stream is
        exhausted.  The number of segments per epoch is
        ``(len(tokens) // batch_size - 1) // segment_len``.
    """

    def __init__(
        self,
        tokens: np.ndarray,
        segment_len: int,
        batch_size: int,
        stream_reset_interval: int = 0,
    ) -> None:
        if segment_len < 1:
            raise ValueError(f"segment_len must be >= 1, got {segment_len}")
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        if stream_reset_interval < 0:
            raise ValueError(
                f"stream_reset_interval must be >= 0, got {stream_reset_interval}"
            )
        tokens = np.asarray(tokens)
        if tokens.ndim != 1:
            raise ValueError(f"tokens must be 1-D, got shape {tokens.shape}")

        # Carve the stream into B contiguous equal-length sub-streams.  We need one
        # extra token per row so the last input has a target.
        per_stream = len(tokens) // batch_size
        if per_stream < segment_len + 1:
            raise ValueError(
                f"stream too short: {len(tokens)} tokens / {batch_size} streams = "
                f"{per_stream} per stream < segment_len+1 = {segment_len + 1}"
            )
        usable = per_stream * batch_size
        grid = tokens[:usable].reshape(batch_size, per_stream)
        self._grid = torch.from_numpy(np.ascontiguousarray(grid)).long()
        self._segment_len = segment_len
        self._batch_size = batch_size
        self._stream_reset_interval = stream_reset_interval
        # We can emit a segment whenever inputs [s, s+T) AND target [s+1, s+T+1) fit.
        self._n_segments = (per_stream - 1) // segment_len

    def __len__(self) -> int:
        return self._n_segments

    def __iter__(self):  # noqa: ANN204 — Iterator[OrderedSegment]
        T = self._segment_len
        for i in range(self._n_segments):
            start = i * T
            inputs = self._grid[:, start : start + T]
            targets = self._grid[:, start + 1 : start + T + 1]
            if self._stream_reset_interval == 0:
                reset = i == 0
            else:
                reset = i % self._stream_reset_interval == 0
            yield OrderedSegment(inputs=inputs, targets=targets, stream_reset=reset)


def iter_ordered_segments(
    tokens: np.ndarray,
    segment_len: int,
    batch_size: int,
    stream_reset_interval: int = 0,
):  # noqa: ANN201 — Iterator[OrderedSegment]
    """Convenience: build and iterate an :class:`OrderedSegmentStream` in one call.

    Equivalent to ``iter(OrderedSegmentStream(...))``; see that class for argument
    semantics.
    """
    return iter(
        OrderedSegmentStream(
            tokens,
            segment_len=segment_len,
            batch_size=batch_size,
            stream_reset_interval=stream_reset_interval,
        )
    )


def load_text8_bytes(cfg: DataConfig) -> np.ndarray | None:
    """Load the cached text8 byte stream as an ``int64`` token array.

    Reads the lazy on-disk cache written by :func:`_load_or_fetch_bytes` (the same
    ``data/text8.bin`` path the canonical loaders use).  Returns ``None`` when the
    cache is absent and the corpus cannot be fetched (offline), so callers can fall
    back to synthetic data without a hard failure.  Used by the SegmentedTrainer to
    build an ordered-segment stream over the one long text8 stream.
    """
    raw = _load_or_fetch_bytes(cfg, "text8", _fetch_text8_bytes)
    if raw is None:
        return None
    return _bytes_to_tokens(raw)


def load_enwik9_bytes(cfg: DataConfig) -> np.ndarray | None:
    """Load the cached enwik9 byte stream as an ``int64`` token array.

    Mirrors :func:`load_text8_bytes` (card 69776c3e): same cache-first contract, same
    return type, same ``None``-on-offline fallback -- so a trainer (e.g.
    :class:`~graph_llm.train.segmented.SegmentedTrainer`) can switch between the two
    corpora by flag/arg alone.  enwik9 (Matt Mahoney's 10^9-byte Wikipedia XML dump,
    http://mattmahoney.net/dc/enwik9.zip) is the natural next rung above text8
    (~10^8 bytes) for the 5M -> 15M -> 50M parameter mini-ladder.

    Deliberately RAW bytes -- no XML stripping.  This is a byte-level LM corpus
    (vocab_size=256, tokenizer-independent BPB), matching the existing enwik8/text8
    convention in this module; a future cleaned-text variant is a separate concern.

    Reads the lazy on-disk cache (``<cfg.data_dir>/enwik9.bin``, gitignored);
    downloads once via :func:`_fetch_enwik9_bytes` if absent.  Returns ``None`` when
    the cache is absent and the corpus cannot be fetched (offline), so callers can
    fall back to a smaller corpus / synthetic data instead of a hard failure.
    """
    raw = _load_or_fetch_bytes(
        cfg, "enwik9", _fetch_enwik9_bytes, expected_size=_EXPECTED_CORPUS_SIZES.get("enwik9")
    )
    if raw is None:
        return None
    return _bytes_to_tokens(raw)


_CORPUS_BYTE_LOADERS: dict[str, Callable[[DataConfig], np.ndarray | None]] = {
    "text8": load_text8_bytes,
    "enwik9": load_enwik9_bytes,
}


def load_corpus_split(
    source: str,
    cfg: DataConfig | None = None,
    train_frac: float = 0.98,
) -> tuple[np.ndarray, np.ndarray]:
    """Load a cached byte corpus and split it into disjoint contiguous (train, eval) slices.

    Generalises :func:`~graph_llm.train.tandem.load_text8_split` to any of this
    module's flat byte-stream loaders (``"text8"`` or ``"enwik9"``) behind one
    ``source`` flag, so a trainer can move to enwik9 without touching its split
    logic (card 69776c3e).  The split is a contiguous cut (NOT random), so eval rows
    are never memorised training windows.

    Args:
        source: ``"text8"`` or ``"enwik9"``.
        cfg: Optional :class:`DataConfig` (only ``data_dir`` is consulted); defaults
            to ``DataConfig(source=source)`` when omitted.
        train_frac: Fraction of the stream kept for training (default 0.98 -- enwik9
            is 10x text8, so the same 5-50M-byte eval slice is a much smaller
            fraction of the whole; pass 0.8 to match ``load_text8_split``'s ratio).

    Returns:
        ``(train_tokens, eval_tokens)`` as ``int64`` arrays.

    Raises:
        ValueError: If ``source`` is not a known corpus name.
        RuntimeError: If the corpus is not cached and cannot be downloaded
            (offline) -- mirrors ``load_text8_split``'s hard failure; callers that
            want a soft fallback should catch this.
    """
    load_fn = _CORPUS_BYTE_LOADERS.get(source)
    if load_fn is None:
        raise ValueError(
            f"load_corpus_split: unknown source {source!r}; "
            f"choose one of {sorted(_CORPUS_BYTE_LOADERS)}"
        )
    cfg = cfg or DataConfig(source=source)
    arr = load_fn(cfg)
    if arr is None:  # pragma: no cover - requires the cached corpus
        raise RuntimeError(
            f"{source} cache required (data_dir={cfg.data_dir!r}); no cache present "
            "and the fetch failed (offline?)."
        )
    cut = int(len(arr) * train_frac)
    return arr[:cut], arr[cut:]


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
    # 90M/5M/5M; enwik9: 900M/50M/50M -- card 69776c3e, the next rung above
    # text8/enwik8 for the parameter mini-ladder; wikitext103: provided
    # train/valid/test).  These return a fixed (train, val) pair built from the
    # standard split rather than a random split, so results are comparable across
    # runs and across models.  See card 424e3a8e and docs/baselines.md.
    if source in ("enwik8", "text8", "wikitext103", "enwik9"):
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

        data_dir = Path(cfg.data_dir).expanduser()
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

        data_dir = Path(cfg.data_dir).expanduser()
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

# Canonical enwik9 split (card 69776c3e): same 5M/5M val/test tail sizes as
# enwik8/text8 (comparable held-out set sizes across corpora), train gets the
# remaining 990M of the 10^9-byte total.
_ENWIK9_TRAIN_BYTES = 990_000_000
_ENWIK9_VAL_BYTES = 5_000_000
_ENWIK9_TEST_BYTES = 5_000_000

# Per-source canonical split sizes consulted by :func:`_try_build_canonical_loaders`.
# Falls back to the enwik8 sizes for any source not listed here.
_SOURCE_SPLIT_BYTES: dict[str, tuple[int, int, int]] = {
    "enwik9": (_ENWIK9_TRAIN_BYTES, _ENWIK9_VAL_BYTES, _ENWIK9_TEST_BYTES),
}


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
    return Path(cfg.data_dir).expanduser() / f"{name}.bin"


# Known-exact corpus byte sizes (card 69776c3e review, comment 264): checked
# against BOTH a freshly-fetched corpus and an EXISTING on-disk cache, so a
# truncated download or a kill-mid-write is a loud failure instead of a
# silently-trusted corrupt cache on the next run. Only enwik9 has a single,
# well-known exact size (Matt Mahoney's canonical 10^9-byte dump); enwik8/text8/
# wikitext103 vary slightly by HF snapshot/decoding, so they are intentionally
# left out of this dict (unchanged behavior -- no check for those sources).
_EXPECTED_CORPUS_SIZES: dict[str, int] = {"enwik9": 1_000_000_000}


def _load_or_fetch_bytes(
    cfg: DataConfig, name: str, fetch, expected_size: int | None = None
) -> bytes | None:
    """Return the raw corpus bytes, using a lazy on-disk cache.

    On the first call ``fetch()`` downloads the corpus (via ``datasets``); the
    raw bytes are cached under ``cfg.data_dir`` (which is ``.gitignore``d) so
    subsequent runs are offline.  Returns ``None`` if the fetch fails (no
    network), letting callers fall back to synthetic data.

    The cache write is atomic: the bytes are written to a temp file in the
    SAME directory as the final cache path, then moved into place with
    ``os.replace`` (an atomic rename on a given filesystem). Without this, a
    process killed mid-``write_bytes`` (e.g. an interrupted multi-GB enwik9
    download) leaves a truncated file at the final path that the next run's
    ``path.exists()`` check would trust with zero validation.

    Args:
        expected_size: exact byte length the corpus MUST have (e.g. enwik9's
            well-known 1_000_000_000 bytes). When given, this is checked
            against BOTH a freshly-fetched corpus (before caching it -- a
            mismatch is NOT cached) and an existing on-disk cache (before
            trusting it); either mismatch raises ``ValueError`` rather than
            silently proceeding with truncated/corrupt data. ``None`` (the
            default) skips the check, preserving prior behavior exactly.
    """
    path = _cache_path(cfg, name)
    if path.exists():
        raw = path.read_bytes()
        if expected_size is not None and len(raw) != expected_size:
            raise ValueError(
                f"{name} cache at {path} is {len(raw):,} bytes, expected exactly "
                f"{expected_size:,} -- this looks like a truncated/corrupted cache "
                "(e.g. an interrupted download or a process killed mid-write). "
                "Delete the file and re-run to refetch."
            )
        return raw
    try:
        raw = fetch()
    except Exception as e:  # noqa: BLE001 — any fetch failure -> offline fallback
        _log.warning("%s fetch failed (offline?): %s", name, e)
        return None
    if expected_size is not None and len(raw) != expected_size:
        raise ValueError(
            f"{name} fetch returned {len(raw):,} bytes, expected exactly "
            f"{expected_size:,} -- treating this as a corrupted/incomplete "
            "download; NOT caching it. Re-run to retry."
        )
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{name}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(raw)
            os.replace(tmp_name, path)
        except OSError:
            with contextlib.suppress(OSError):
                os.remove(tmp_name)
            raise
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


_ENWIK9_URL = "http://mattmahoney.net/dc/enwik9.zip"


def _fetch_enwik9_bytes() -> bytes:
    """Download + extract enwik9 (card 69776c3e): raw Wikipedia XML bytes, no cleaning.

    Unlike enwik8/text8/wikitext103 (served pre-decoded via the HF ``datasets`` hub),
    enwik9 has no HF dataset id: Matt Mahoney publishes it directly as a ~330 MB zip
    (http://mattmahoney.net/dc/enwik9.zip) containing one member, canonically named
    "enwik9".  Fetched with stdlib ``urllib`` + ``zipfile`` (no new dependency); the
    lazy on-disk cache in :func:`_load_or_fetch_bytes` means this network call (and
    the ~1 GB extraction) only happens once per machine -- see that function's
    ``cfg.data_dir`` cache path.  Raises on any failure so the caller's existing
    offline fallback applies (this fetcher is never invoked once cached).
    """
    import io
    import urllib.request
    import zipfile

    with urllib.request.urlopen(_ENWIK9_URL, timeout=300) as resp:  # noqa: S310
        zip_bytes = resp.read()
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        names = zf.namelist()
        member = "enwik9" if "enwik9" in names else names[0]
        return zf.read(member)


_CANONICAL_FETCHERS = {
    "enwik8": _fetch_enwik8_bytes,
    "text8": _fetch_text8_bytes,
    "wikitext103": _fetch_wikitext103_bytes,
    "enwik9": _fetch_enwik9_bytes,
}


def make_canonical_split_datasets(
    raw: bytes,
    seq_len: int,
    encoder: TextEncoder | None = None,
    train_bytes: int = _ENWIK8_TRAIN_BYTES,
    val_bytes: int = _ENWIK8_VAL_BYTES,
    test_bytes: int = _ENWIK8_TEST_BYTES,
) -> dict[str, _TextChunkDataset]:
    """Build {train,val,test} chunk datasets from raw corpus bytes.

    By default uses byte-level encoding (vocab=256).  Pass an ``encoder``
    (e.g. a :class:`~graph_llm.tokenizer.BPETokenizer`) to tokenize at the
    BPE level instead.  The canonical byte-offset split (90M/5M/5M by default;
    pass ``train_bytes``/``val_bytes``/``test_bytes`` for a different corpus, e.g.
    enwik9's 990M/5M/5M -- card 69776c3e) is applied first, then each split is
    encoded independently so there is no cross-split context leakage.

    Exposed for unit tests so the split/chunk logic can be exercised on a small
    in-memory fixture with no network or disk I/O.
    """
    if encoder is None or isinstance(encoder, ByteLevelEncoder):
        # Fast path: byte-level, no decode round-trip needed.
        tokens = _bytes_to_tokens(raw)
        splits = canonical_byte_splits(tokens, train_bytes, val_bytes, test_bytes)
        return {name: _TextChunkDataset(arr, seq_len) for name, arr in splits.items()}

    # BPE (or other text encoder): split bytes by offset first, then encode each
    # split as text so no cross-split context leaks through the tokenizer.
    total = len(raw)
    total_req = train_bytes + val_bytes + test_bytes
    if total < total_req:
        scale = total / total_req
        train_end = int(train_bytes * scale)
        val_end = train_end + int(val_bytes * scale)
    else:
        train_end = train_bytes
        val_end = train_bytes + val_bytes

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
    raw = _load_or_fetch_bytes(
        cfg, source, fetch, expected_size=_EXPECTED_CORPUS_SIZES.get(source)
    )
    if raw is None:
        return None
    encoder = get_encoder(cfg)
    train_bytes, val_bytes, test_bytes = _SOURCE_SPLIT_BYTES.get(
        source, (_ENWIK8_TRAIN_BYTES, _ENWIK8_VAL_BYTES, _ENWIK8_TEST_BYTES)
    )
    datasets_by_split = make_canonical_split_datasets(
        raw,
        cfg.seq_len,
        encoder=encoder,
        train_bytes=train_bytes,
        val_bytes=val_bytes,
        test_bytes=test_bytes,
    )
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
