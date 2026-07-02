"""Tests for the synthetic reasoning corpus (card 255a4424).

Covers:
- Oracle solvability (~100%) for all three generator types.
- Determinism: same seed -> same corpus.
- Difficulty monotonicity: harder difficulty -> weaker n-gram baseline.
- Byte-level contract: tokens are int64, values in 0..255.
- Loader produces correct shapes compatible with _TextChunkDataset.
- answer_accuracy metric.
- extract_answer_from_tokens helper.
"""

from __future__ import annotations

import collections
import random
from typing import TYPE_CHECKING

import numpy as np
import pytest

from graph_llm.config import DataConfig
from graph_llm.data.reasoning_corpus import (
    ArithmeticGenerator,
    CodePredictionGenerator,
    ReasoningExample,
    TransitiveLogicGenerator,
    answer_accuracy,
    build_reasoning_corpus,
    build_reasoning_dataloaders,
    extract_answer_from_tokens,
    oracle_solve,
)

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_rng(seed: int = 42) -> random.Random:
    return random.Random(seed)


def _generate_n(gen: object, n: int, seed: int = 42) -> list[ReasoningExample]:
    """Generate n examples from any generator."""
    rng = _make_rng(seed)
    results: list[ReasoningExample] = []
    for _ in range(n):
        ex = gen.generate(rng)  # type: ignore[attr-defined]
        results.append(ex)
    return results


# ---------------------------------------------------------------------------
# Token contract tests
# ---------------------------------------------------------------------------


class TestByteContract:
    """All token arrays must be int64 with values in 0..255 (vocab=256)."""

    def test_arithmetic_tokens_dtype_and_range(self) -> None:
        ex = ArithmeticGenerator(n_ops=2).generate(_make_rng())
        assert ex.tokens.dtype == np.int64, "tokens must be int64"
        assert int(ex.tokens.min()) >= 0
        assert int(ex.tokens.max()) <= 255

    def test_transitive_tokens_dtype_and_range(self) -> None:
        ex = TransitiveLogicGenerator(chain_len=3).generate(_make_rng())
        assert ex.tokens.dtype == np.int64
        assert int(ex.tokens.min()) >= 0
        assert int(ex.tokens.max()) <= 255

    def test_code_tokens_dtype_and_range(self) -> None:
        ex = CodePredictionGenerator(n_vars=2).generate(_make_rng())
        assert ex.tokens.dtype == np.int64
        assert int(ex.tokens.min()) >= 0
        assert int(ex.tokens.max()) <= 255

    def test_corpus_train_tokens_dtype(self) -> None:
        corpus = build_reasoning_corpus(n_train=20, n_val=5, seed=0)
        assert corpus.train_tokens.dtype == np.int64
        assert corpus.val_tokens.dtype == np.int64
        assert int(corpus.train_tokens.min()) >= 0
        assert int(corpus.train_tokens.max()) <= 255

    def test_prompt_len_consistency(self) -> None:
        """prompt_len must equal byte-length of example.prompt."""
        for gen in [
            ArithmeticGenerator(n_ops=2),
            TransitiveLogicGenerator(chain_len=3),
            CodePredictionGenerator(n_vars=2),
        ]:
            for ex in _generate_n(gen, 10):
                expected_prompt_len = len(ex.prompt.encode("utf-8"))
                assert ex.prompt_len == expected_prompt_len, (
                    f"{ex.kind}: prompt_len={ex.prompt_len} but "
                    f"len(prompt.encode())={expected_prompt_len}"
                )

    def test_tokens_length_equals_text_bytes(self) -> None:
        for gen in [
            ArithmeticGenerator(n_ops=2),
            TransitiveLogicGenerator(chain_len=3),
            CodePredictionGenerator(n_vars=2),
        ]:
            for ex in _generate_n(gen, 5):
                expected = len(ex.text.encode("utf-8"))
                assert len(ex.tokens) == expected, (
                    f"{ex.kind}: len(tokens)={len(ex.tokens)} but "
                    f"len(text.encode())={expected}"
                )


# ---------------------------------------------------------------------------
# Oracle solvability
# ---------------------------------------------------------------------------


class TestOracleSolvability:
    """Oracle must solve >= 95% (target: ~100%) of each generator's examples."""

    N_EXAMPLES = 200
    THRESHOLD = 0.95

    def _oracle_accuracy(self, gen: object, n: int = N_EXAMPLES) -> float:
        examples = _generate_n(gen, n)
        correct = 0
        for ex in examples:
            try:
                prediction = oracle_solve(ex)
                if prediction == ex.answer:
                    correct += 1
            except Exception:  # pragma: no cover
                pass
        return correct / n

    def test_arithmetic_oracle(self) -> None:
        acc = self._oracle_accuracy(ArithmeticGenerator(n_ops=3))
        assert acc >= self.THRESHOLD, f"Arithmetic oracle accuracy {acc:.3f} < {self.THRESHOLD}"

    def test_transitive_oracle(self) -> None:
        acc = self._oracle_accuracy(TransitiveLogicGenerator(chain_len=4))
        assert acc >= self.THRESHOLD, f"Transitive oracle accuracy {acc:.3f} < {self.THRESHOLD}"

    def test_code_oracle(self) -> None:
        acc = self._oracle_accuracy(CodePredictionGenerator(n_vars=3))
        assert acc >= self.THRESHOLD, f"Code oracle accuracy {acc:.3f} < {self.THRESHOLD}"

    def test_oracle_matches_stored_answer(self) -> None:
        """oracle_solve must return the same string that is stored in example.answer."""
        for gen in [
            ArithmeticGenerator(n_ops=2),
            TransitiveLogicGenerator(chain_len=3),
            CodePredictionGenerator(n_vars=2),
        ]:
            for ex in _generate_n(gen, 50):
                oracle = oracle_solve(ex)
                assert oracle == ex.answer, (
                    f"{ex.kind}: oracle={oracle!r} but example.answer={ex.answer!r}\n"
                    f"  text: {ex.text!r}"
                )


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    """Same seed must produce identical examples in identical order."""

    def test_corpus_deterministic_by_seed(self) -> None:
        c1 = build_reasoning_corpus(n_train=50, n_val=10, seed=99)
        c2 = build_reasoning_corpus(n_train=50, n_val=10, seed=99)
        assert len(c1.train) == len(c2.train)
        for e1, e2 in zip(c1.train, c2.train):
            assert e1.text == e2.text, "Train examples differ across identical seeds"
        for e1, e2 in zip(c1.val, c2.val):
            assert e1.text == e2.text, "Val examples differ across identical seeds"

    def test_different_seeds_produce_different_corpora(self) -> None:
        c1 = build_reasoning_corpus(n_train=50, n_val=10, seed=1)
        c2 = build_reasoning_corpus(n_train=50, n_val=10, seed=2)
        texts1 = {e.text for e in c1.train}
        texts2 = {e.text for e in c2.train}
        # With 50 examples and randomised parameters the overlap should be small.
        overlap = len(texts1 & texts2)
        assert overlap < len(texts1), "Different seeds produced identical corpora"

    def test_val_set_independent_of_n_train(self) -> None:
        """Changing n_train must not change the val set."""
        c1 = build_reasoning_corpus(n_train=100, n_val=20, seed=7)
        c2 = build_reasoning_corpus(n_train=200, n_val=20, seed=7)
        # val_seed is derived from the top-level seed independently of n_train usage.
        assert [e.text for e in c1.val] == [e.text for e in c2.val], (
            "Val set changed when n_train changed (val_seed not independent)"
        )

    def test_token_arrays_deterministic(self) -> None:
        c1 = build_reasoning_corpus(n_train=30, n_val=5, seed=42)
        c2 = build_reasoning_corpus(n_train=30, n_val=5, seed=42)
        np.testing.assert_array_equal(c1.train_tokens, c2.train_tokens)
        np.testing.assert_array_equal(c1.val_tokens, c2.val_tokens)


# ---------------------------------------------------------------------------
# Difficulty monotonicity
# ---------------------------------------------------------------------------
# A weak baseline = "repeat the most common answer from training", i.e. a
# frequency-based lookup.  Harder difficulty means more unique answers, so
# the majority-vote baseline accuracy falls.


def _majority_vote_accuracy(examples: list[ReasoningExample]) -> float:
    """Accuracy of predicting the most-common answer for every example."""
    if not examples:
        return 0.0
    counter: collections.Counter[str] = collections.Counter(
        ex.answer for ex in examples
    )
    majority = counter.most_common(1)[0][0]
    return sum(1 for ex in examples if ex.answer == majority) / len(examples)


class TestDifficultyMonotonicity:
    """Harder difficulty -> weaker majority-vote baseline (lower accuracy)."""

    N = 300

    def test_arithmetic_monotone(self) -> None:
        easy = _generate_n(ArithmeticGenerator(n_ops=1), self.N, seed=11)
        hard = _generate_n(ArithmeticGenerator(n_ops=6), self.N, seed=11)
        acc_easy = _majority_vote_accuracy(easy)
        acc_hard = _majority_vote_accuracy(hard)
        assert acc_easy >= acc_hard, (
            f"Arithmetic monotonicity failed: easy={acc_easy:.3f} hard={acc_hard:.3f}"
        )

    def test_transitive_monotone(self) -> None:
        # For transitive logic the majority-vote over answer letters is uniform
        # (same 20-letter pool at all chain lengths).  Instead, use a
        # "first-entity" baseline: guess the first word of the text.
        # With chain_len=2, shuffled clauses give each of the 2 entities a 50%
        # chance of appearing first → the winner (answer) appears first ~50%
        # of the time.  With chain_len=8, the winner is one of 7 clause-leading
        # entities → appears first ~1/7 ≈ 14% of the time.
        import re as _re

        def first_entity_accuracy(examples: list[ReasoningExample]) -> float:
            correct = 0
            for ex in examples:
                m = _re.match(r"(\w+)", ex.text)
                pred = m.group(1) if m else ""
                if pred == ex.answer:
                    correct += 1
            return correct / len(examples)

        easy = _generate_n(TransitiveLogicGenerator(chain_len=2), self.N, seed=12)
        hard = _generate_n(TransitiveLogicGenerator(chain_len=8), self.N, seed=12)
        acc_easy = first_entity_accuracy(easy)
        acc_hard = first_entity_accuracy(hard)
        assert acc_easy > acc_hard, (
            f"Transitive monotonicity failed: easy={acc_easy:.3f} hard={acc_hard:.3f}"
        )

    def test_code_monotone(self) -> None:
        easy = _generate_n(CodePredictionGenerator(n_vars=1), self.N, seed=13)
        hard = _generate_n(CodePredictionGenerator(n_vars=5), self.N, seed=13)
        acc_easy = _majority_vote_accuracy(easy)
        acc_hard = _majority_vote_accuracy(hard)
        assert acc_easy >= acc_hard, (
            f"Code monotonicity failed: easy={acc_easy:.3f} hard={acc_hard:.3f}"
        )


# ---------------------------------------------------------------------------
# answer_accuracy metric
# ---------------------------------------------------------------------------


class TestAnswerAccuracy:
    def test_all_correct(self) -> None:
        examples = _generate_n(ArithmeticGenerator(n_ops=2), 20)
        predictions = [ex.answer for ex in examples]
        assert answer_accuracy(examples, predictions) == pytest.approx(1.0)

    def test_all_wrong(self) -> None:
        examples = _generate_n(ArithmeticGenerator(n_ops=2), 10)
        predictions = ["WRONG"] * len(examples)
        assert answer_accuracy(examples, predictions) == pytest.approx(0.0)

    def test_half_correct(self) -> None:
        examples = _generate_n(ArithmeticGenerator(n_ops=2), 10)
        predictions = [ex.answer if i % 2 == 0 else "X" for i, ex in enumerate(examples)]
        acc = answer_accuracy(examples, predictions)
        assert 0.4 <= acc <= 0.6

    def test_empty_returns_zero(self) -> None:
        assert answer_accuracy([], []) == pytest.approx(0.0)

    def test_strips_whitespace(self) -> None:
        examples = _generate_n(ArithmeticGenerator(n_ops=2), 5)
        # Add spurious whitespace to the prediction.
        predictions = [" " + ex.answer + " " for ex in examples]
        assert answer_accuracy(examples, predictions) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# extract_answer_from_tokens
# ---------------------------------------------------------------------------


class TestExtractAnswerFromTokens:
    def test_roundtrip_arithmetic(self) -> None:
        ex = ArithmeticGenerator(n_ops=2).generate(_make_rng(0))
        extracted = extract_answer_from_tokens(ex.tokens, ex.prompt_len)
        assert extracted == ex.answer

    def test_roundtrip_transitive(self) -> None:
        ex = TransitiveLogicGenerator(chain_len=3).generate(_make_rng(1))
        extracted = extract_answer_from_tokens(ex.tokens, ex.prompt_len)
        assert extracted == ex.answer

    def test_roundtrip_code(self) -> None:
        ex = CodePredictionGenerator(n_vars=2).generate(_make_rng(2))
        extracted = extract_answer_from_tokens(ex.tokens, ex.prompt_len)
        assert extracted == ex.answer

    def test_empty_answer_when_prompt_exceeds_tokens(self) -> None:
        ex = ArithmeticGenerator(n_ops=1).generate(_make_rng(3))
        extracted = extract_answer_from_tokens(ex.tokens, len(ex.tokens) + 10)
        assert extracted == ""


# ---------------------------------------------------------------------------
# Corpus and loader shapes
# ---------------------------------------------------------------------------


class TestCorpusAndLoader:
    def test_corpus_sizes(self) -> None:
        corpus = build_reasoning_corpus(n_train=80, n_val=20, seed=5)
        assert len(corpus.train) == 80
        assert len(corpus.val) == 20

    def test_flat_token_arrays_non_empty(self) -> None:
        corpus = build_reasoning_corpus(n_train=10, n_val=5, seed=6)
        assert len(corpus.train_tokens) > 0
        assert len(corpus.val_tokens) > 0

    def test_flat_token_newline_separators(self) -> None:
        """Flat token array must contain newline (10) separators between examples."""
        corpus = build_reasoning_corpus(n_train=5, n_val=2, seed=7)
        # At least n_train newlines should appear.
        newline_count = int((corpus.train_tokens == ord("\n")).sum())
        assert newline_count >= len(corpus.train)

    def test_dataloaders_batch_shape(self) -> None:
        cfg = DataConfig(source="reasoning", seq_len=64, batch_size=4)
        train_loader, val_loader = build_reasoning_dataloaders(
            cfg,
            seed=0,
            n_train=500,
            n_val=100,
            arithmetic_n_ops=2,
            transitive_chain_len=3,
            code_n_vars=1,
        )
        # Check one batch from each loader.
        inputs, targets = next(iter(train_loader))
        assert inputs.shape == (4, 64), f"Expected (4,64), got {inputs.shape}"
        assert targets.shape == (4, 64), f"Expected (4,64), got {targets.shape}"
        # targets should be inputs shifted by 1 (next-token prediction)
        # We can't check this exactly because _TextChunkDataset chunks overlap,
        # but dtypes should match.
        assert inputs.dtype == targets.dtype

    def test_dataloader_val_shape(self) -> None:
        cfg = DataConfig(source="reasoning", seq_len=32, batch_size=2)
        _, val_loader = build_reasoning_dataloaders(
            cfg, seed=1, n_train=200, n_val=50
        )
        inputs, targets = next(iter(val_loader))
        assert inputs.shape[1] == 32
        assert targets.shape[1] == 32

    def test_mix_proportions(self) -> None:
        """Check that kind proportions roughly match the defaults (arith=0.5, trans=0.4, code=0.1)."""
        corpus = build_reasoning_corpus(n_train=1000, n_val=100, seed=42)
        counts: dict[str, int] = collections.Counter(e.kind for e in corpus.train)  # type: ignore[assignment]
        total = len(corpus.train)
        assert counts.get("arithmetic", 0) / total == pytest.approx(0.5, abs=0.05)
        assert counts.get("transitive", 0) / total == pytest.approx(0.4, abs=0.05)
        assert counts.get("code", 0) / total == pytest.approx(0.1, abs=0.05)


# ---------------------------------------------------------------------------
# Generator edge cases
# ---------------------------------------------------------------------------


class TestGeneratorEdgeCases:
    def test_arithmetic_n_ops_1(self) -> None:
        ex = ArithmeticGenerator(n_ops=1).generate(_make_rng())
        assert oracle_solve(ex) == ex.answer

    def test_transitive_chain_len_2(self) -> None:
        ex = TransitiveLogicGenerator(chain_len=2).generate(_make_rng())
        assert oracle_solve(ex) == ex.answer

    def test_code_n_vars_1(self) -> None:
        ex = CodePredictionGenerator(n_vars=1).generate(_make_rng())
        assert oracle_solve(ex) == ex.answer

    def test_arithmetic_invalid_n_ops(self) -> None:
        with pytest.raises(ValueError):
            ArithmeticGenerator(n_ops=0)

    def test_transitive_invalid_chain_len(self) -> None:
        with pytest.raises(ValueError):
            TransitiveLogicGenerator(chain_len=1)

    def test_code_invalid_n_vars(self) -> None:
        with pytest.raises(ValueError):
            CodePredictionGenerator(n_vars=0)

    def test_answer_is_last_token_span(self) -> None:
        """The answer must be at the tail of tokens (prompt_len .. end)."""
        for gen in [
            ArithmeticGenerator(n_ops=2),
            TransitiveLogicGenerator(chain_len=3),
            CodePredictionGenerator(n_vars=2),
        ]:
            for ex in _generate_n(gen, 20):
                tail_bytes = bytes(ex.tokens[ex.prompt_len :].astype(np.uint8))
                tail_str = tail_bytes.decode("utf-8", errors="replace")
                assert tail_str == ex.answer, (
                    f"{ex.kind}: tail={tail_str!r} != answer={ex.answer!r}\n"
                    f"  text: {ex.text!r}"
                )
