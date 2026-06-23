"""Long-context harness tests (card 424e3a8e).

Passkey probe construction + exact-match scoring on synthetic examples, and the
per-token-position loss curve.  All offline / CPU.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import graph_llm.models.baselines  # noqa: F401 — registers baselines
from graph_llm.config import Config, ModelConfig
from graph_llm.eval.long_context import (
    greedy_generate,
    make_passkey_example,
    position_loss_curve,
    run_passkey_probe,
    score_passkey_retrieval,
)
from graph_llm.models import build_model

# ---------------------------------------------------------------------------
# Passkey prompt construction
# ---------------------------------------------------------------------------


def test_passkey_example_contains_key_and_question() -> None:
    ex = make_passkey_example("12345", context_tokens=200, depth_fraction=0.5)
    assert "12345" in ex.prompt
    assert ex.prompt.strip().endswith("The pass key is")
    assert ex.answer == "12345"
    assert ex.depth_fraction == 0.5


def test_passkey_prompt_exceeds_training_window() -> None:
    """The whole point: the prompt must be longer than a small training window."""
    train_window = 64
    ex = make_passkey_example("424242", context_tokens=4 * train_window, depth_fraction=0.5)
    assert ex.prompt_len_tokens > train_window, (
        "passkey prompt must exceed the training window to test extrapolation"
    )


@pytest.mark.parametrize("depth", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_passkey_depth_positions_the_key(depth: float) -> None:
    """The key lands earlier in the prompt for smaller depth fractions."""
    ex = make_passkey_example("777", context_tokens=400, depth_fraction=depth)
    key_pos = ex.prompt.index("The pass key is 777")
    frac = key_pos / len(ex.prompt)
    # Loose monotonicity: depth 0 places near start, depth 1 near end.
    if depth == 0.0:
        assert frac < 0.5
    if depth == 1.0:
        assert frac > 0.5


def test_passkey_rejects_bad_depth() -> None:
    with pytest.raises(ValueError):
        make_passkey_example("1", context_tokens=10, depth_fraction=1.5)


# ---------------------------------------------------------------------------
# Exact-match scoring
# ---------------------------------------------------------------------------


def test_score_exact_match_true_and_false() -> None:
    # Key at the start of the (short) continuation — the natural answer to a
    # prompt ending "The pass key is " — counts as retrieved.
    assert score_passkey_retrieval("12345", "12345") is True
    assert score_passkey_retrieval("  12345 is the key", "12345") is True
    # Conservative: a key buried AFTER other text is NOT credited, so the probe
    # cannot over-credit a non-answer (protects the long-context claim's
    # integrity — see score_passkey_retrieval docstring).
    assert score_passkey_retrieval("the answer is 12345.", "12345") is False
    assert score_passkey_retrieval("99999", "12345") is False
    assert score_passkey_retrieval("", "12345") is False


def test_passkey_probe_scores_a_planted_model() -> None:
    """A model that always emits the planted key's bytes scores 100% accuracy.

    This validates the probe's prompt -> generate -> score wiring end-to-end on a
    synthetic example, without training anything.
    """
    key = "55555"
    key_bytes = list(key.encode("utf-8"))

    class _PlantedModel(torch.nn.Module):
        """Ignores context; cycles through the key bytes one per step.

        ``greedy_generate`` calls ``forward`` once per generated token, growing
        the sequence by one each time, so keying the emitted byte off the
        current length ``T`` cycles ``key[0], key[1], ...`` across calls.  The
        key therefore appears as a contiguous substring of the continuation.
        """

        def __init__(self, vocab_size: int, prompt_len: int) -> None:
            super().__init__()
            self.vocab_size = vocab_size
            self._emit = key_bytes
            self._prompt_len = prompt_len

        def forward(self, x, targets=None):
            B, T = x.shape
            logits = torch.zeros(B, T, self.vocab_size)
            step = max(0, T - self._prompt_len)  # 0 on the first generate call
            nxt = self._emit[step % len(self._emit)]
            logits[0, -1, nxt] = 10.0
            return torch.zeros(1), logits

    # Build the example ourselves so the planted model knows the prompt length;
    # this exercises the same construct -> encode -> generate -> score path the
    # probe uses, on a fully synthetic example.
    ex = make_passkey_example(key, context_tokens=120, depth_fraction=0.5)
    prompt_ids = torch.tensor(list(ex.prompt.encode("utf-8")), dtype=torch.long)
    model = _PlantedModel(vocab_size=256, prompt_len=len(prompt_ids))

    gen_ids = greedy_generate(model, prompt_ids, max_new_tokens=len(key) + 2)
    generated = bytes(i % 256 for i in gen_ids).decode("utf-8", errors="replace")
    assert key in generated, f"planted model should emit key; got {generated!r}"
    assert score_passkey_retrieval(generated, ex.answer) is True


def test_run_passkey_probe_end_to_end_smoke() -> None:
    """run_passkey_probe wires construct->generate->score across depths.

    Uses a real (untrained) baseline: we don't assert it *retrieves* the key
    (it can't, untrained) — only that the harness produces a well-formed result
    with one entry per depth and a valid accuracy.
    """
    cfg = Config(model=ModelConfig(name="transformer", vocab_size=256, d_model=32,
                                   n_heads=4, n_layers=1, d_ff=64, max_seq_len=16, dropout=0.0))
    model = build_model(cfg)
    res = run_passkey_probe(model, context_tokens=64, depths=[0.0, 0.5, 1.0], vocab_size=256)
    assert 0.0 <= res["accuracy"] <= 1.0
    assert len(res["per_depth"]) == 3
    for entry in res["per_depth"]:
        assert entry["prompt_len_tokens"] > 16  # exceeds training window
        assert isinstance(entry["retrieved"], bool)


def test_greedy_generate_is_deterministic_and_right_length() -> None:
    cfg = Config(model=ModelConfig(name="transformer", vocab_size=256, d_model=32,
                                   n_heads=4, n_layers=1, d_ff=64, max_seq_len=32, dropout=0.0))
    model = build_model(cfg)
    model.eval()
    prompt = torch.randint(0, 256, (8,))
    g1 = greedy_generate(model, prompt, max_new_tokens=5)
    g2 = greedy_generate(model, prompt, max_new_tokens=5)
    assert len(g1) == 5
    assert g1 == g2, "greedy decoding must be deterministic"


# ---------------------------------------------------------------------------
# Per-token-position loss curve
# ---------------------------------------------------------------------------


def test_position_loss_curve_shape_and_finiteness() -> None:
    cfg = Config(model=ModelConfig(name="transformer", vocab_size=256, d_model=32,
                                   n_heads=4, n_layers=1, d_ff=64, max_seq_len=32, dropout=0.0))
    model = build_model(cfg)
    # Sequences LONGER than the training window (32): use length 80.
    seqs = torch.randint(0, 256, (4, 80))
    curve = position_loss_curve(model, seqs)
    assert curve.shape == (79,), f"curve must be (T-1,), got {curve.shape}"
    assert bool((curve == curve).all()), "curve must be finite (no NaNs)"
    assert (curve >= 0).all(), "cross-entropy losses are non-negative"


def test_position_loss_curve_runs_for_mamba_beyond_window() -> None:
    """Mamba must also produce a position-loss curve past its training window."""
    cfg = Config(model=ModelConfig(name="mamba", vocab_size=256, d_model=32,
                                   n_heads=4, n_layers=1, d_ff=64, max_seq_len=16, dropout=0.0))
    model = build_model(cfg)
    seqs = torch.randint(0, 256, (3, 48))  # 48 > 16 training window
    curve = position_loss_curve(model, seqs)
    assert curve.shape == (47,)
    assert (curve >= 0).all()


def test_position_loss_curve_rejects_bad_shapes() -> None:
    cfg = Config(model=ModelConfig(name="transformer", vocab_size=256, d_model=32,
                                   n_heads=4, n_layers=1, d_ff=64, max_seq_len=32, dropout=0.0))
    model = build_model(cfg)
    with pytest.raises(ValueError):
        position_loss_curve(model, torch.randint(0, 256, (10,)))  # 1-D
    with pytest.raises(ValueError):
        position_loss_curve(model, torch.randint(0, 256, (2, 1)))  # T < 2
