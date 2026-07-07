"""enwik9 byte-level data loader tests (card 69776c3e, piece 1).

These tests exercise the loader's CACHE + SPLIT logic on a small synthetic
in-memory fixture -- they never download the real ~330 MB enwik9.zip.  The
real download/cache path is exercised manually once per machine and logged to
the card (see card 69776c3e comments), not exercised here.
"""

from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from graph_llm.config import DataConfig
from graph_llm.data.loader import (
    _CANONICAL_FETCHERS,
    _EXPECTED_CORPUS_SIZES,
    _cache_path,
    _fetch_enwik9_bytes,
    _load_or_fetch_bytes,
    build_dataloaders,
    canonical_byte_splits,
    load_corpus_split,
    load_enwik9_bytes,
    make_canonical_split_datasets,
)


def _fake_enwik9_bytes(n: int = 5000, seed: int = 0) -> bytes:
    """Small deterministic byte blob standing in for real enwik9 content."""
    rng = np.random.default_rng(seed)
    return bytes(rng.integers(0, 256, size=n, dtype=np.uint8).tobytes())


def test_enwik9_registered_as_canonical_fetcher() -> None:
    assert "enwik9" in _CANONICAL_FETCHERS
    assert _CANONICAL_FETCHERS["enwik9"] is _fetch_enwik9_bytes


def test_canonical_byte_splits_scales_down_for_enwik9_sizes() -> None:
    """A small fixture split with the enwik9-sized (990M/5M/5M) request scales down
    proportionally, exactly like the existing enwik8 sizing (same code path)."""
    tokens = np.frombuffer(_fake_enwik9_bytes(2000), dtype=np.uint8).astype(np.int64)
    splits = canonical_byte_splits(
        tokens, train_bytes=990_000_000, val_bytes=5_000_000, test_bytes=5_000_000
    )
    assert set(splits) == {"train", "val", "test"}
    total = sum(len(v) for v in splits.values())
    assert total == len(tokens)
    # Train should dominate (990/1000 of the request), val/test small but present.
    assert len(splits["train"]) > len(splits["val"])
    assert len(splits["train"]) > len(splits["test"])


def test_make_canonical_split_datasets_enwik9_sizes() -> None:
    raw = _fake_enwik9_bytes(4000)
    datasets = make_canonical_split_datasets(
        raw, seq_len=32, train_bytes=990_000_000, val_bytes=5_000_000, test_bytes=5_000_000
    )
    assert set(datasets) == {"train", "val", "test"}
    for ds in datasets.values():
        assert len(ds) >= 0
    # Non-trivial chunks exist in at least the (dominant) train split.
    assert len(datasets["train"]) > 0


def test_load_enwik9_bytes_uses_disk_cache_not_network(tmp_path, monkeypatch) -> None:
    """Seed the on-disk cache directly; confirm the loader reads it WITHOUT ever
    calling the network fetcher (proves the offline/cached path, matching the
    ``load_text8_bytes`` contract this mirrors)."""
    raw = _fake_enwik9_bytes(1234, seed=1)
    cache_path = tmp_path / "enwik9.bin"
    cache_path.write_bytes(raw)

    def _boom() -> bytes:  # pragma: no cover - must never be called
        raise AssertionError("network fetch should not be called when cache exists")

    monkeypatch.setattr("graph_llm.data.loader._fetch_enwik9_bytes", _boom)
    # This test exercises the cache-hit/no-network contract on a small synthetic
    # fixture, not the real-corpus size check (covered separately below) -- disable
    # the exact-length guard so a deliberately tiny fixture is still accepted here.
    monkeypatch.setattr("graph_llm.data.loader._EXPECTED_CORPUS_SIZES", {})

    cfg = DataConfig(source="enwik9", data_dir=str(tmp_path))
    tokens = load_enwik9_bytes(cfg)
    assert tokens is not None
    assert tokens.dtype == np.int64
    expected = np.frombuffer(raw, dtype=np.uint8).astype(np.int64)
    assert np.array_equal(tokens, expected)


def test_load_enwik9_bytes_returns_none_when_offline_and_uncached(tmp_path, monkeypatch) -> None:
    def _fail() -> bytes:
        raise OSError("no network in test")

    monkeypatch.setattr("graph_llm.data.loader._fetch_enwik9_bytes", _fail)
    cfg = DataConfig(source="enwik9", data_dir=str(tmp_path))
    assert load_enwik9_bytes(cfg) is None


def test_build_dataloaders_enwik9_source_from_cache(tmp_path, monkeypatch) -> None:
    """``build_dataloaders(DataConfig(source="enwik9"))`` builds real (train, val)
    loaders from a pre-seeded cache -- no network, no synthetic fallback."""
    raw = _fake_enwik9_bytes(6000, seed=2)
    (tmp_path / "enwik9.bin").write_bytes(raw)

    def _boom() -> bytes:  # pragma: no cover
        raise AssertionError("network fetch should not be called when cache exists")

    monkeypatch.setattr("graph_llm.data.loader._fetch_enwik9_bytes", _boom)
    monkeypatch.setattr("graph_llm.data.loader._EXPECTED_CORPUS_SIZES", {})

    cfg = DataConfig(source="enwik9", data_dir=str(tmp_path), seq_len=16, batch_size=4)
    train_loader, val_loader = build_dataloaders(cfg, vocab_size=256, seed=0)
    assert len(train_loader.dataset) > 0  # type: ignore[arg-type]
    x, y = next(iter(train_loader))
    assert x.shape[1] == 16
    assert y.shape == x.shape


def test_load_corpus_split_enwik9(tmp_path, monkeypatch) -> None:
    raw = _fake_enwik9_bytes(3000, seed=3)
    (tmp_path / "enwik9.bin").write_bytes(raw)
    monkeypatch.setattr("graph_llm.data.loader._EXPECTED_CORPUS_SIZES", {})
    cfg = DataConfig(source="enwik9", data_dir=str(tmp_path))
    train, ev = load_corpus_split("enwik9", cfg=cfg, train_frac=0.9)
    assert len(train) + len(ev) == len(raw)
    assert len(train) == int(len(raw) * 0.9)
    # Disjoint contiguous slices of the same stream (train first, eval last).
    expected = np.frombuffer(raw, dtype=np.uint8).astype(np.int64)
    assert np.array_equal(np.concatenate([train, ev]), expected)


def test_load_corpus_split_unknown_source_raises() -> None:
    with pytest.raises(ValueError):
        load_corpus_split("not-a-real-corpus")


def test_load_corpus_split_raises_without_cache_or_network(tmp_path, monkeypatch) -> None:
    def _fail() -> bytes:
        raise OSError("no network in test")

    monkeypatch.setattr("graph_llm.data.loader._fetch_enwik9_bytes", _fail)
    cfg = DataConfig(source="enwik9", data_dir=str(tmp_path))
    with pytest.raises(RuntimeError):
        load_corpus_split("enwik9", cfg=cfg)


def test_enwik9_registered_in_expected_corpus_sizes() -> None:
    """The review's Major finding #1 (card 69776c3e comment 264): enwik9's
    well-known exact decompressed size must be registered so both a fresh fetch
    and an existing cache are validated against it."""
    assert _EXPECTED_CORPUS_SIZES["enwik9"] == 1_000_000_000


def test_load_or_fetch_bytes_raises_on_existing_cache_size_mismatch(tmp_path) -> None:
    """A cache file that does NOT match the expected exact size (e.g. a truncated
    write left by a killed process) must be a loud failure, not silently trusted."""
    cfg = DataConfig(source="enwik9", data_dir=str(tmp_path))
    (tmp_path / "enwik9.bin").write_bytes(b"\x00" * 999)  # deliberately wrong size

    def _boom() -> bytes:  # pragma: no cover - cache exists, must not be called
        raise AssertionError("fetch should not be called when a cache file exists")

    with pytest.raises(ValueError, match="expected exactly 1,000,000,000"):
        _load_or_fetch_bytes(cfg, "enwik9", _boom, expected_size=1_000_000_000)


def test_load_or_fetch_bytes_raises_on_fresh_fetch_size_mismatch_and_does_not_cache(
    tmp_path,
) -> None:
    """A fresh fetch that returns the wrong number of bytes (corrupted/incomplete
    download) must raise AND must NOT be written to the cache path."""
    cfg = DataConfig(source="enwik9", data_dir=str(tmp_path))

    def _short_fetch() -> bytes:
        return b"\x00" * 999  # wrong size -- simulates a truncated download

    with pytest.raises(ValueError, match="expected exactly 1,000,000,000"):
        _load_or_fetch_bytes(cfg, "enwik9", _short_fetch, expected_size=1_000_000_000)

    assert not (tmp_path / "enwik9.bin").exists()


def test_load_enwik9_bytes_raises_on_real_cache_size_mismatch(tmp_path) -> None:
    """End-to-end through the public ``load_enwik9_bytes`` entry point (not just
    the private helper): a wrong-sized real cache file must raise, not silently
    return truncated/corrupt tokens."""
    (tmp_path / "enwik9.bin").write_bytes(b"\x00" * 1234)
    cfg = DataConfig(source="enwik9", data_dir=str(tmp_path))
    with pytest.raises(ValueError, match="expected exactly 1,000,000,000"):
        load_enwik9_bytes(cfg)


def test_load_or_fetch_bytes_writes_cache_atomically_via_temp_and_replace(tmp_path) -> None:
    """The cache write must go through a temp file in the SAME directory + an
    atomic ``os.replace`` -- no stray temp files left behind on success, and no
    partially-written file ever visible at the final cache path."""
    cfg = DataConfig(source="text8", data_dir=str(tmp_path))
    raw = _fake_enwik9_bytes(4096, seed=7)

    def _fetch() -> bytes:
        return raw

    result = _load_or_fetch_bytes(cfg, "text8", _fetch)
    assert result == raw

    cache_path = _cache_path(cfg, "text8")
    assert cache_path.exists()
    assert cache_path.read_bytes() == raw
    # No leftover temp files in the cache directory after a successful write.
    leftover_tmp = [p for p in tmp_path.iterdir() if p.name.startswith(".text8.")]
    assert leftover_tmp == []


def test_load_or_fetch_bytes_atomic_write_survives_replace_failure(tmp_path, monkeypatch) -> None:
    """If the final ``os.replace`` step fails, the temp file is cleaned up (no
    debris left in the cache dir) and the fetched bytes are still returned to the
    caller for this run (soft-fail on caching, matching the pre-existing
    OSError-tolerant contract for a failed cache write)."""
    import graph_llm.data.loader as loader_mod

    cfg = DataConfig(source="text8", data_dir=str(tmp_path))
    raw = _fake_enwik9_bytes(2048, seed=8)

    def _fetch() -> bytes:
        return raw

    def _boom_replace(src, dst):  # noqa: ANN001, ARG001
        raise OSError("simulated replace failure")

    monkeypatch.setattr(loader_mod.os, "replace", _boom_replace)

    result = _load_or_fetch_bytes(cfg, "text8", _fetch)
    assert result == raw  # still usable this run even though caching failed
    assert not _cache_path(cfg, "text8").exists()
    leftover_tmp = [p for p in tmp_path.iterdir() if p.name.startswith(".text8.")]
    assert leftover_tmp == []  # temp file cleaned up, not left as debris


def test_fetch_enwik9_bytes_extracts_single_member_zip(monkeypatch) -> None:
    """Exercise ``_fetch_enwik9_bytes``'s zip-extraction logic against a small FAKE
    zip served by a stubbed ``urlopen`` -- no real network call."""
    payload = b"hello enwik9 fixture"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("enwik9", payload)
    zip_bytes = buf.getvalue()

    class _FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self) -> bytes:
            return zip_bytes

    def _fake_urlopen(url, timeout=None):  # noqa: ANN001, ARG001
        assert url == "http://mattmahoney.net/dc/enwik9.zip"
        return _FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen)
    result = _fetch_enwik9_bytes()
    assert result == payload
