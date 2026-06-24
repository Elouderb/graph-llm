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

import graph_llm.models  # noqa: F401 — registers "delta_memory_lm"
import graph_llm.models.baselines  # noqa: F401 — registers baselines
from graph_llm.config import Config, ModelConfig
from graph_llm.eval.long_context import (
    carried_stream_bpb,
    greedy_generate,
    make_cross_segment_passkey,
    make_passkey_example,
    position_loss_curve,
    run_passkey_probe,
    run_segments,
    score_cross_segment_passkey,
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


# ---------------------------------------------------------------------------
# Cross-segment persistent-memory harness (card 61f900ca)
# ---------------------------------------------------------------------------
#
# These tests verify the harness MECHANICS run end-to-end on a small delta_memory_lm
# and that carried vs reset modes feed different state.  Actual passkey RETRIEVAL
# SUCCESS requires a trained model (piece 4 / the experiment), which is out of
# scope here; an untrained model is expected to retrieve nothing.


def _delta_cfg(max_seq_len: int = 32, scan: str = "chunkwise") -> Config:
    """A small delta_memory_lm config for the cross-segment harness tests."""
    return Config(
        model=ModelConfig(
            name="delta_memory_lm", vocab_size=256, d_model=32, delta_layers=2,
            delta_n_heads=4, delta_head_k_dim=16, delta_head_v_dim=16,
            delta_feature_map="l2", delta_use_forget_gate=True, delta_ff_mult=2,
            delta_dropout=0.0, dropout=0.0, max_seq_len=max_seq_len,
            delta_scan=scan, delta_chunk_size=8,
        )
    )


def test_run_segments_carry_vs_reset_differ_after_first_segment() -> None:
    """Carried state changes a later segment's logits; reset leaves it independent.

    The first segment has no prior context, so carry and reset must agree on it.
    A later segment under carry sees the earlier segments through the threaded
    memory, so its logits MUST differ from the reset run — the mechanical proof
    that the two modes feed different state.
    """
    torch.manual_seed(0)
    model = build_model(_delta_cfg())
    model.eval()
    segs = [torch.randint(0, 256, (20,)), torch.randint(0, 256, (20,)),
            torch.randint(0, 256, (15,))]

    carried = run_segments(model, segs, carry=True)
    reset = run_segments(model, segs, carry=False)

    assert carried.carried is True and reset.carried is False
    assert carried.final_states is not None and len(carried.final_states) == 2
    assert reset.final_states is None
    assert len(carried.per_segment_logits) == len(reset.per_segment_logits) == 3

    # First segment: no prior context -> identical.
    assert torch.equal(carried.per_segment_logits[0], reset.per_segment_logits[0])
    # Later segments: carry sees the prior stream -> must differ.
    d1 = (carried.per_segment_logits[1] - reset.per_segment_logits[1]).abs().max().item()
    d2 = (carried.per_segment_logits[2] - reset.per_segment_logits[2]).abs().max().item()
    assert d1 > 1e-5, "carried 2nd-segment logits should differ from reset"
    assert d2 > 1e-5, "carried 3rd-segment logits should differ from reset"


def test_run_segments_falls_back_to_reset_for_stateless_model() -> None:
    """A model without the carry API runs per-segment regardless of ``carry``."""
    cfg = Config(model=ModelConfig(name="transformer", vocab_size=256, d_model=32,
                                   n_heads=4, n_layers=1, d_ff=64, max_seq_len=32, dropout=0.0))
    model = build_model(cfg)
    segs = [torch.randint(0, 256, (10,)), torch.randint(0, 256, (10,))]
    run = run_segments(model, segs, carry=True)
    assert run.carried is False  # transformer has no states_in/return_states
    assert run.final_states is None
    assert len(run.per_segment_logits) == 2


def test_run_segments_rejects_empty() -> None:
    model = build_model(_delta_cfg())
    with pytest.raises(ValueError):
        run_segments(model, [])


def test_make_cross_segment_passkey_structure() -> None:
    """Key lives in segment 0, query in the last segment; answer is outside it."""
    ex = make_cross_segment_passkey("12345", n_segments=4, segment_tokens=40)
    assert len(ex.segment_ids) == 4
    assert ex.passkey_segment_index == 0
    assert ex.query_segment_index == 3
    assert ex.answer == "12345"
    # Segment 0 decodes to text containing the key; the final segment is the query.
    seg0 = bytes(int(i) % 256 for i in ex.segment_ids[0].tolist()).decode("utf-8", "replace")
    last = bytes(int(i) % 256 for i in ex.segment_ids[-1].tolist()).decode("utf-8", "replace")
    assert "12345" in seg0
    assert "pass key" in last
    assert "12345" not in last, "answer must be OUTSIDE the query segment's window"
    assert ex.total_tokens == sum(int(s.numel()) for s in ex.segment_ids)


@pytest.mark.parametrize("bad", [{"n_segments": 1}, {"segment_tokens": 0}])
def test_make_cross_segment_passkey_rejects_bad_args(bad: dict) -> None:
    kwargs = {"n_segments": 3, "segment_tokens": 20}
    kwargs.update(bad)
    with pytest.raises(ValueError):
        make_cross_segment_passkey("1", **kwargs)  # type: ignore[arg-type]


def test_cross_segment_passkey_harness_runs_carry_and_reset() -> None:
    """The cross-segment passkey harness runs end-to-end in both modes.

    Verifies MECHANICS only: a well-formed result with the right flags in each
    mode.  An untrained model retrieves nothing — actual retrieval is piece 4.
    The discriminating property here is that ``carried`` is True only in carry
    mode (so reset cannot, even in principle, see the key segment).
    """
    torch.manual_seed(0)
    model = build_model(_delta_cfg())
    model.eval()
    ex = make_cross_segment_passkey("12345", n_segments=4, segment_tokens=40)

    res_carry = score_cross_segment_passkey(model, ex, carry=True)
    res_reset = score_cross_segment_passkey(model, ex, carry=False)

    assert res_carry["carried"] is True
    assert res_reset["carried"] is False
    assert res_carry["answer_outside_query_window"] is True
    for res in (res_carry, res_reset):
        assert isinstance(res["retrieved"], bool)
        assert isinstance(res["generated"], str)
        assert res["query_segment_tokens"] == int(ex.segment_ids[-1].numel())


def test_carried_stream_bpb_runs_both_modes() -> None:
    """Long-stream carried bpb runs in carry and reset modes and is finite > 0."""
    torch.manual_seed(0)
    model = build_model(_delta_cfg())
    model.eval()
    stream = torch.randint(0, 256, (200,))

    bpb_carry = carried_stream_bpb(model, stream, segment_len=32, carry=True)
    bpb_reset = carried_stream_bpb(model, stream, segment_len=32, carry=False)

    for bpb in (bpb_carry, bpb_reset):
        assert isinstance(bpb, float)
        assert bpb == bpb  # not NaN
        assert bpb > 0.0


def test_carried_stream_bpb_accepts_2d_and_rejects_bad_shapes() -> None:
    model = build_model(_delta_cfg())
    model.eval()
    # (1, L) is accepted (squeezed to (L,)).
    bpb = carried_stream_bpb(model, torch.randint(0, 256, (1, 64)), segment_len=16)
    assert bpb > 0.0
    with pytest.raises(ValueError):
        carried_stream_bpb(model, torch.randint(0, 256, (64,)), segment_len=0)
    with pytest.raises(ValueError):
        carried_stream_bpb(model, torch.randint(0, 256, (3, 64)), segment_len=16)  # 2-D, N>1
    with pytest.raises(ValueError):
        carried_stream_bpb(model, torch.randint(0, 256, (1,)), segment_len=16)  # L < 2
