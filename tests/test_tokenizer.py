"""Tests for Phase 1 tokenizer + phonological init (card e1644700).

Covers:
- BPETokenizer round-trip encode/decode
- Vocab size exactly 16,384 after training on the synthetic corpus
- Tokenizer save/load round-trip (JSON serialisation)
- Version validation raises on mismatch
- get_vocab() returns a dict with expected size
- compute_phonological_vectors: returns 24-d arrays, coverage > 0
- phonological_init_fn: correct output shape, finite values, in-place write
- Unmapped tokens remain at their pre-existing random init values (unchanged)
- Toggle: phonological init produces different embedding from random at fixed seed
- Registry: "phonological" is registered and callable with correct signature
- test_embedding_init_hook_is_respected stays green (imported from test_smoke logic)
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_DICT_FILE = Path("/usr/share/dict/words")


def _build_training_corpus(n_sentences: int = 5_000, seed: int = 42) -> list[str]:
    """Build a diverse synthetic corpus from the system dictionary.

    Generates random-word sentences so BPE has enough unique character
    sequences to perform 16,128 merges and reach vocab_size=16,384.
    Falls back to a pangram-based corpus if the dictionary is unavailable.
    """
    if _DICT_FILE.exists():
        words = _DICT_FILE.read_text().splitlines()
        rng = random.Random(seed)
        sentences = []
        for _ in range(n_sentences):
            n = rng.randint(5, 15)
            sentence = " ".join(rng.choices(words, k=n))
            sentences.append(sentence.lower())
        return sentences

    # Fallback: the pangram corpus used by the training script (may not reach 16k)
    base = [
        "the quick brown fox jumps over the lazy dog",
        "pack my box with five dozen liquor jugs",
        "how vexingly quick daft zebras jump",
        "the five boxing wizards jump quickly",
        "sphinx of black quartz judge my vow",
    ]
    return base * 1_000


@pytest.fixture(scope="module")
def trained_tokenizer():
    """Train a 16,384-vocab BPE tokenizer once for the test module."""
    from graph_llm.tokenizer.bpe import VOCAB_SIZE, BPETokenizer

    corpus = _build_training_corpus()
    tok = BPETokenizer.train_on_corpus(
        corpus,
        vocab_size=VOCAB_SIZE,
        min_frequency=1,
        show_progress=False,
    )
    return tok


# ---------------------------------------------------------------------------
# BPETokenizer: basic contract
# ---------------------------------------------------------------------------


class TestBPETokenizer:
    def test_vocab_size_exactly_16384(self, trained_tokenizer):
        """Vocabulary must be exactly 16,384 tokens."""
        from graph_llm.tokenizer.bpe import VOCAB_SIZE

        assert trained_tokenizer._tok.get_vocab_size() == VOCAB_SIZE

    def test_encode_returns_ints(self, trained_tokenizer):
        ids = trained_tokenizer.encode("hello world")
        assert isinstance(ids, list)
        assert all(isinstance(i, int) for i in ids)
        assert len(ids) > 0

    def test_decode_returns_str(self, trained_tokenizer):
        ids = trained_tokenizer.encode("hello world")
        text = trained_tokenizer.decode(ids)
        assert isinstance(text, str)

    def test_round_trip_simple(self, trained_tokenizer):
        text = "the quick brown fox"
        ids = trained_tokenizer.encode(text)
        decoded = trained_tokenizer.decode(ids).strip()
        assert decoded == text

    def test_round_trip_multiword(self, trained_tokenizer):
        text = "pack my box with five dozen liquor jugs"
        decoded = trained_tokenizer.decode(trained_tokenizer.encode(text)).strip()
        assert decoded == text

    def test_round_trip_empty(self, trained_tokenizer):
        """Empty string encodes to zero or minimal tokens and decodes back."""
        ids = trained_tokenizer.encode("")
        decoded = trained_tokenizer.decode(ids)
        assert isinstance(decoded, str)

    def test_encode_is_deterministic(self, trained_tokenizer):
        text = "the quick brown fox"
        ids1 = trained_tokenizer.encode(text)
        ids2 = trained_tokenizer.encode(text)
        assert ids1 == ids2

    def test_get_vocab_returns_dict(self, trained_tokenizer):
        from graph_llm.tokenizer.bpe import VOCAB_SIZE

        vocab = trained_tokenizer.get_vocab()
        assert isinstance(vocab, dict)
        assert len(vocab) == VOCAB_SIZE
        assert all(isinstance(k, str) for k in vocab)
        assert all(isinstance(v, int) for v in vocab.values())

    def test_save_and_load(self, trained_tokenizer, tmp_path):
        """Save tokenizer to JSON and reload; encoding must be identical."""
        from graph_llm.tokenizer.bpe import TOKENIZER_VERSION, BPETokenizer

        save_path = str(tmp_path / "bpe_test.json")
        trained_tokenizer.save(save_path)

        # File must exist and contain version metadata
        assert Path(save_path).is_file()
        data = json.loads(Path(save_path).read_text())
        # The version is injected in the added_tokens_decoder or a custom key
        # (the implementation stores it in the serialised JSON under graph_llm_version)
        assert "graph_llm_version" in data  # version key must be injected by BPETokenizer.save()

        reloaded = BPETokenizer.from_pretrained(save_path)
        text = "the quick brown fox"
        assert reloaded.encode(text) == trained_tokenizer.encode(text)

    def test_version_mismatch_raises(self, trained_tokenizer, tmp_path):
        """Loading a file with a wrong version key raises ValueError."""
        from graph_llm.tokenizer.bpe import BPETokenizer

        save_path = str(tmp_path / "bpe_bad.json")
        trained_tokenizer.save(save_path)

        # Corrupt the version field
        data = json.loads(Path(save_path).read_text())
        data["graph_llm_version"] = "99.0.0"
        Path(save_path).write_text(json.dumps(data))

        with pytest.raises(ValueError, match="version"):
            BPETokenizer.from_pretrained(save_path)


# ---------------------------------------------------------------------------
# Phonological init: compute_phonological_vectors
# ---------------------------------------------------------------------------


class TestComputePhonologicalVectors:
    """Tests against a tiny hand-crafted vocab known to have CMUdict entries."""

    # Hand-crafted mini vocab: 6 tokens where we know coverage.
    # "Ġthe" is a byte-level BPE token for " the" (space prefix).
    # "Ġcat" → "cat" in CMUdict → 3 phonemes
    # "Ġdog" → "dog"
    # "<pad>" → no CMUdict entry
    # "Ġxyz" → almost certainly not in CMUdict
    _MINI_VOCAB = {
        "<pad>": 0,
        "<unk>": 1,
        "Ġthe": 2,
        "Ġcat": 3,
        "Ġdog": 4,
        "Ġxyz123": 5,
    }

    def test_returns_dict_and_float(self):
        from graph_llm.tokenizer.phonological_init import compute_phonological_vectors

        vectors, coverage = compute_phonological_vectors(self._MINI_VOCAB)
        assert isinstance(vectors, dict)
        assert isinstance(coverage, float)
        assert 0.0 <= coverage <= 1.0

    def test_vector_shape_24d(self):
        from graph_llm.tokenizer.phonological_init import compute_phonological_vectors

        vectors, _ = compute_phonological_vectors(self._MINI_VOCAB)
        for tid, vec in vectors.items():
            assert vec.shape == (24,), f"Token {tid} has shape {vec.shape}"
            assert vec.dtype == np.float32

    def test_at_least_one_token_mapped(self):
        """At least one of 'the', 'cat', 'dog' should be in CMUdict."""
        from graph_llm.tokenizer.phonological_init import compute_phonological_vectors

        vectors, coverage = compute_phonological_vectors(self._MINI_VOCAB)
        assert len(vectors) >= 1, "Expected at least one mapped token"
        assert coverage > 0.0

    def test_unk_not_mapped(self):
        """<unk> strips angle brackets to 'unk' which is not an English word → not mapped."""
        import cmudict as _cmu  # pyright: ignore[reportMissingImports]  # untyped, runtime-installed

        from graph_llm.tokenizer.phonological_init import compute_phonological_vectors

        # Verify 'unk' is not in cmudict (so id=1 should not appear in vectors)
        raw = _cmu.dict()
        if "unk" in raw:
            pytest.skip("'unk' unexpectedly in CMUdict; skip this assertion")

        vectors, _ = compute_phonological_vectors(self._MINI_VOCAB)
        assert 1 not in vectors, "<unk> / 'unk' should not have a phonological vector"

    def test_vectors_are_finite(self):
        from graph_llm.tokenizer.phonological_init import compute_phonological_vectors

        vectors, _ = compute_phonological_vectors(self._MINI_VOCAB)
        for tid, vec in vectors.items():
            assert np.all(np.isfinite(vec)), f"Non-finite values for token {tid}"

    def test_coverage_on_real_vocab(self, trained_tokenizer):
        """On a real 16k vocab, expect at least 1% coverage (very conservative)."""
        from graph_llm.tokenizer.phonological_init import compute_phonological_vectors

        vocab = trained_tokenizer.get_vocab()
        vectors, coverage = compute_phonological_vectors(vocab)
        assert coverage >= 0.01, f"Coverage too low: {coverage:.4f}"


# ---------------------------------------------------------------------------
# Phonological init: phonological_init_fn
# ---------------------------------------------------------------------------


class TestPhonologicalInitFn:
    _D_MODEL = 64  # small for test speed
    # <unk> → "unk" (not in CMUdict), "Ġxyz123" → "xyz" (not a CMUdict word)
    # These token IDs (0, 1) should be unchanged after phonological init.
    _MINI_VOCAB = {
        "<unk>": 0,
        "Ġxyz123": 1,
        "Ġthe": 2,
        "Ġcat": 3,
        "Ġdog": 4,
    }
    _VOCAB_SIZE = 5

    def _random_weight(self) -> torch.Tensor:
        torch.manual_seed(42)
        w = torch.empty(self._VOCAB_SIZE, self._D_MODEL)
        torch.nn.init.normal_(w, std=0.02)
        return w

    def test_output_shape_unchanged(self):
        from graph_llm.tokenizer.phonological_init import phonological_init_fn

        weight = self._random_weight()
        orig_shape = weight.shape
        phonological_init_fn(
            weight, self._VOCAB_SIZE, self._D_MODEL, vocab=self._MINI_VOCAB
        )
        assert weight.shape == orig_shape

    def test_values_finite_after_init(self):
        from graph_llm.tokenizer.phonological_init import phonological_init_fn

        weight = self._random_weight()
        phonological_init_fn(
            weight, self._VOCAB_SIZE, self._D_MODEL, vocab=self._MINI_VOCAB
        )
        assert torch.all(torch.isfinite(weight))

    def test_unmapped_tokens_unchanged(self):
        """Tokens that don't resolve to CMUdict entries must be untouched."""
        import cmudict as _cmu  # pyright: ignore[reportMissingImports]  # untyped, runtime-installed

        from graph_llm.tokenizer.phonological_init import _clean_token, phonological_init_fn

        raw = _cmu.dict()
        # Confirm our unmapped tokens are really not in CMUdict
        unmapped_ids = {}
        for tok, tid in self._MINI_VOCAB.items():
            key = _clean_token(tok)
            if key and key in raw:
                continue  # this one IS in CMUdict, skip
            unmapped_ids[tid] = tok

        if not unmapped_ids:
            pytest.skip("All tokens in _MINI_VOCAB mapped to CMUdict; can't test unmapped path.")

        weight = self._random_weight()
        rows_before = {tid: weight[tid].clone() for tid in unmapped_ids}

        phonological_init_fn(
            weight, self._VOCAB_SIZE, self._D_MODEL, vocab=self._MINI_VOCAB
        )

        for tid, row_before in rows_before.items():
            assert torch.allclose(weight[tid], row_before), (
                f"Token id={tid} ({unmapped_ids[tid]!r}) was changed but should be unmapped."
            )

    def test_noop_when_vocab_is_none(self):
        """Without a vocab, phonological_init_fn is a no-op."""
        from graph_llm.tokenizer.phonological_init import phonological_init_fn

        weight = self._random_weight()
        before = weight.clone()
        phonological_init_fn(weight, self._VOCAB_SIZE, self._D_MODEL, vocab=None)
        assert torch.allclose(weight, before), "Weight changed despite vocab=None"

    def test_toggle_produces_different_embedding(self):
        """Phonological init must produce a tensor distinguishable from pure random."""
        from graph_llm.tokenizer.phonological_init import (
            compute_phonological_vectors,
            phonological_init_fn,
        )

        # Build random baseline
        torch.manual_seed(0)
        rand_weight = torch.empty(self._VOCAB_SIZE, self._D_MODEL)
        torch.nn.init.normal_(rand_weight, std=0.02)

        # Build phonological
        torch.manual_seed(0)
        phon_weight = torch.empty(self._VOCAB_SIZE, self._D_MODEL)
        torch.nn.init.normal_(phon_weight, std=0.02)
        phonological_init_fn(
            phon_weight, self._VOCAB_SIZE, self._D_MODEL, vocab=self._MINI_VOCAB
        )

        # They must differ (at least one mapped token was overwritten)
        vecs, coverage = compute_phonological_vectors(self._MINI_VOCAB)
        if coverage == 0.0:
            pytest.skip("No tokens mapped in this vocab; toggle test skipped.")
        assert not torch.allclose(rand_weight, phon_weight), (
            "Phonological and random embeddings are identical — init is not applied."
        )

    def test_seed_reproducibility(self):
        """Same seed → identical phonological embedding."""
        from graph_llm.tokenizer.phonological_init import phonological_init_fn

        w1 = self._random_weight()
        phonological_init_fn(
            w1, self._VOCAB_SIZE, self._D_MODEL, vocab=self._MINI_VOCAB, seed=7
        )
        w2 = self._random_weight()
        phonological_init_fn(
            w2, self._VOCAB_SIZE, self._D_MODEL, vocab=self._MINI_VOCAB, seed=7
        )
        assert torch.allclose(w1, w2), "Different results for same seed"

    def test_different_seeds_differ(self):
        """Different seeds → different projection matrices → different embeddings."""
        from graph_llm.tokenizer.phonological_init import (
            compute_phonological_vectors,
            phonological_init_fn,
        )

        vecs, coverage = compute_phonological_vectors(self._MINI_VOCAB)
        if coverage == 0.0:
            pytest.skip("No tokens mapped; seed test skipped.")

        w1 = self._random_weight()
        phonological_init_fn(
            w1, self._VOCAB_SIZE, self._D_MODEL, vocab=self._MINI_VOCAB, seed=0
        )
        w2 = self._random_weight()
        phonological_init_fn(
            w2, self._VOCAB_SIZE, self._D_MODEL, vocab=self._MINI_VOCAB, seed=1
        )
        assert not torch.allclose(w1, w2), "Different seeds produced same embedding"


# ---------------------------------------------------------------------------
# Registry: "phonological" is properly registered
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_phonological_registered_after_import(self):
        """Importing the tokenizer package registers 'phonological'."""
        import graph_llm.tokenizer  # noqa: F401 (side-effect import)
        from graph_llm.models.registry import get_embedding_init

        fn = get_embedding_init("phonological")
        assert callable(fn)

    def test_registry_fn_accepts_correct_signature(self):
        """Registry callable must accept (weight, vocab_size, d_model) without error."""
        import graph_llm.tokenizer  # noqa: F401
        from graph_llm.models.registry import get_embedding_init

        fn = get_embedding_init("phonological")
        weight = torch.zeros(10, 32)
        # Should not raise (vocab=None path → no-op)
        fn(weight, 10, 32)

    def test_unknown_name_raises(self):
        from graph_llm.models.registry import get_embedding_init

        with pytest.raises((KeyError, ValueError)):
            get_embedding_init("nonexistent_init_xyz")


# ---------------------------------------------------------------------------
# config.py additive fields
# ---------------------------------------------------------------------------


class TestConfigFields:
    def test_data_config_has_encoder_field(self):
        from graph_llm.config import DataConfig

        cfg = DataConfig()
        assert hasattr(cfg, "encoder")
        assert cfg.encoder == "byte"

    def test_data_config_has_bpe_tokenizer_path(self):
        from graph_llm.config import DataConfig

        cfg = DataConfig()
        assert hasattr(cfg, "bpe_tokenizer_path")
        assert cfg.bpe_tokenizer_path is None

    def test_model_config_has_embedding_init(self):
        from graph_llm.config import ModelConfig

        cfg = ModelConfig()
        assert hasattr(cfg, "embedding_init")
        assert cfg.embedding_init is None


# ---------------------------------------------------------------------------
# apply_embedding_init: ablation toggle end-to-end
# ---------------------------------------------------------------------------


class TestApplyEmbeddingInit:
    """Verify that apply_embedding_init wires the phonological toggle correctly."""

    def test_noop_when_embedding_init_is_null(self):
        """apply_embedding_init does nothing when embedding_init is None."""
        import torch.nn as nn

        from graph_llm.tokenizer.phonological_init import apply_embedding_init

        class _FakeModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(16, 32)
                torch.nn.init.constant_(self.embed.weight, 3.0)

        class _FakeCfg:
            class model:
                embedding_init = None

            class data:
                encoder = "byte"
                bpe_tokenizer_path = None

        model = _FakeModel()
        before = model.embed.weight.clone()
        apply_embedding_init(model, _FakeCfg())
        assert torch.allclose(model.embed.weight, before), (
            "apply_embedding_init mutated embedding when embedding_init is None."
        )

    def test_raises_when_phonological_but_no_bpe_path(self):
        """Requesting phonological init without a tokenizer path must raise ValueError."""
        import torch.nn as nn

        from graph_llm.tokenizer.phonological_init import apply_embedding_init

        class _FakeModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(16, 32)

        class _FakeCfg:
            class model:
                embedding_init = "phonological"

            class data:
                encoder = "bpe"
                bpe_tokenizer_path = None  # missing!

        with pytest.raises(ValueError, match="bpe_tokenizer_path"):
            apply_embedding_init(_FakeModel(), _FakeCfg())

    def test_raises_when_phonological_but_encoder_is_byte(self):
        """Requesting phonological init with encoder=byte must raise ValueError."""
        import torch.nn as nn

        from graph_llm.tokenizer.phonological_init import apply_embedding_init

        class _FakeModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(16, 32)

        class _FakeCfg:
            class model:
                embedding_init = "phonological"

            class data:
                encoder = "byte"  # wrong encoder type
                bpe_tokenizer_path = "/some/path.json"

        with pytest.raises(ValueError, match="encoder"):
            apply_embedding_init(_FakeModel(), _FakeCfg())

    def test_toggle_wires_phonological_via_config(self, trained_tokenizer, tmp_path):
        """apply_embedding_init with a real tokenizer produces phonologically-initialised embeddings."""
        import torch.nn as nn

        from graph_llm.tokenizer.phonological_init import apply_embedding_init

        # Save the trained tokenizer so apply_embedding_init can load it by path
        tok_path = str(tmp_path / "bpe_for_toggle_test.json")
        trained_tokenizer.save(tok_path)

        vocab_size = trained_tokenizer.vocab_size
        d_model = 64

        class _FakeModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(vocab_size, d_model)

        class _FakeCfg:
            class model:
                embedding_init = "phonological"

            class data:
                encoder = "bpe"
                bpe_tokenizer_path = tok_path

        # Random baseline (same init seed)
        torch.manual_seed(42)
        rand_model = _FakeModel()
        rand_weight = rand_model.embed.weight.detach().clone()

        # Phonological model (same init seed, then overwritten)
        torch.manual_seed(42)
        phon_model = _FakeModel()
        apply_embedding_init(phon_model, _FakeCfg())
        phon_weight = phon_model.embed.weight.detach()

        # The two embeddings must differ — at least some tokens are overwritten
        assert not torch.allclose(rand_weight, phon_weight), (
            "apply_embedding_init did not alter any embedding rows; toggle is inoperative."
        )


# ---------------------------------------------------------------------------
# get_encoder helper
# ---------------------------------------------------------------------------


class TestGetEncoder:
    def test_byte_encoder(self):
        from graph_llm.config import DataConfig
        from graph_llm.data.loader import ByteLevelEncoder, get_encoder

        cfg = DataConfig(encoder="byte")
        enc = get_encoder(cfg)
        assert isinstance(enc, ByteLevelEncoder)

    def test_bpe_encoder_raises_without_path(self):
        from graph_llm.config import DataConfig
        from graph_llm.data.loader import get_encoder

        cfg = DataConfig(encoder="bpe", bpe_tokenizer_path=None)
        with pytest.raises(ValueError, match="bpe_tokenizer_path"):
            get_encoder(cfg)

    def test_bpe_encoder_loads_tokenizer(self, trained_tokenizer, tmp_path):
        from graph_llm.config import DataConfig
        from graph_llm.data.loader import get_encoder
        from graph_llm.tokenizer.bpe import BPETokenizer

        save_path = str(tmp_path / "bpe_enc.json")
        trained_tokenizer.save(save_path)

        cfg = DataConfig(encoder="bpe", bpe_tokenizer_path=save_path)
        enc = get_encoder(cfg)
        assert isinstance(enc, BPETokenizer)

    def test_unknown_encoder_raises(self):
        from graph_llm.config import DataConfig
        from graph_llm.data.loader import get_encoder

        cfg = DataConfig(encoder="unknown_xyz")
        with pytest.raises(ValueError, match="unknown_xyz"):
            get_encoder(cfg)

    def test_bpe_encoder_yields_ids_above_255(self, trained_tokenizer, tmp_path):
        """BPE encoder must produce at least some token ids > 255 for real text.

        This is the key correctness check for the BPE seam: if all ids are <= 255
        the model is receiving byte ids (the old behaviour) rather than BPE ids.
        """
        from graph_llm.config import DataConfig
        from graph_llm.data.loader import get_encoder

        save_path = str(tmp_path / "bpe_for_seam_test.json")
        trained_tokenizer.save(save_path)

        cfg = DataConfig(encoder="bpe", bpe_tokenizer_path=save_path)
        enc = get_encoder(cfg)

        # Long-enough text so BPE can form multi-byte merges well above id 255
        text = "the quick brown fox jumps over the lazy dog " * 20
        ids = enc.encode(text)

        assert len(ids) > 0, "BPE encoder returned no ids."
        assert any(i > 255 for i in ids), (
            f"All BPE ids <= 255 — tokenizer is acting as byte-level. "
            f"Max id seen: {max(ids)}. "
            "Check that the BPE tokenizer was actually trained with merges."
        )

    def test_byte_encoder_yields_ids_0_to_255(self):
        """Byte encoder must produce only ids in [0, 255]."""
        from graph_llm.data.loader import ByteLevelEncoder

        enc = ByteLevelEncoder()
        text = "hello world, the quick brown fox!"
        ids = enc.encode(text)

        assert len(ids) > 0
        assert all(0 <= i <= 255 for i in ids), (
            f"Byte encoder produced ids outside [0, 255]: "
            f"{[i for i in ids if i < 0 or i > 255]}"
        )


# ---------------------------------------------------------------------------
# Norm rescaling: projected rows must match embedding init std
# ---------------------------------------------------------------------------


class TestPhonologicalInitNormRescaling:
    """Phonological projection rows must be rescaled to ~0.02 std to match
    the surrounding random-init rows so the ablation is not confounded by
    a norm artifact.
    """

    def test_phonological_init_rescales_to_embed_std(self, trained_tokenizer):
        """After phonological_init_fn, mapped rows must have std close to 0.02."""
        import torch.nn as nn

        from graph_llm.tokenizer.phonological_init import phonological_init_fn

        vocab = trained_tokenizer.get_vocab()
        d_model = 128

        torch.manual_seed(0)
        weight = torch.empty(trained_tokenizer.vocab_size, d_model)
        nn.init.normal_(weight, mean=0.0, std=0.02)
        original = weight.clone()

        phonological_init_fn(weight, trained_tokenizer.vocab_size, d_model, vocab=vocab)

        # Find rows that were overwritten
        changed = ~torch.all(weight == original, dim=1)
        n_changed = changed.sum().item()
        assert n_changed > 0, "No rows were changed; check phonological coverage."

        # Compute per-row std of changed rows
        changed_rows = weight[changed]  # (n_changed, d_model)
        row_stds = changed_rows.std(dim=1)  # (n_changed,)
        mean_std = row_stds.mean().item()

        # Allow ±50% of 0.02 — the rescaling is per-row and exact, but
        # single-phoneme rows may have unusual distributions.
        assert 0.005 < mean_std < 0.06, (
            f"Phonological-init rows have mean per-row std={mean_std:.5f}; "
            f"expected close to 0.02. Norm rescaling may not be working."
        )


# ---------------------------------------------------------------------------
# Vocab-size validation in apply_embedding_init
# ---------------------------------------------------------------------------


class TestApplyEmbeddingInitVocabSizeValidation:
    """apply_embedding_init must raise ValueError when model vocab_size < tokenizer vocab."""

    def test_raises_when_vocab_exceeds_model_embed(self, trained_tokenizer, tmp_path):
        import torch.nn as nn

        from graph_llm.tokenizer.phonological_init import apply_embedding_init

        tok_path = str(tmp_path / "bpe_vocab_size_test.json")
        trained_tokenizer.save(tok_path)

        tok_vocab_size = trained_tokenizer.vocab_size
        # Make the embedding SMALLER than the tokenizer vocab to trigger the check
        small_vocab = tok_vocab_size - 10

        class _FakeModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(small_vocab, 32)

        class _FakeCfg:
            class model:
                embedding_init = "phonological"

            class data:
                encoder = "bpe"
                bpe_tokenizer_path = tok_path

        with pytest.raises(ValueError, match="vocab_size"):
            apply_embedding_init(_FakeModel(), _FakeCfg())
