"""Segmented stateful training tests (card 61f900ca, piece 3).

The cross-segment-memory TRAINING subsystem: ordered-segment streaming,
truncated-BPTT with detach at the window boundary, the state-distribution-exposure
augmentation, and the synthetic cross-segment retrieval task generator.  All
CPU / offline.

The LOAD-BEARING test is :func:`test_truncated_bptt_detach_severs_gradient` — the
defining correctness invariant of truncated BPTT: a carried state is differentiable
while the autograd graph is connected, but receives ZERO gradient once detached at
the window boundary, so gradients never flow across more than ``K`` segments.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import graph_llm.models  # noqa: F401 — registers "delta_memory_lm"
from graph_llm.config import Config, DataConfig, ModelConfig, TrainConfig
from graph_llm.data.loader import OrderedSegmentStream, iter_ordered_segments
from graph_llm.data.synthetic_tasks import (
    CrossSegmentTaskSampler,
    make_cross_segment_task,
    masked_token_loss,
)
from graph_llm.models import build_model
from graph_llm.models.components.delta_memory import DeltaMemoryState
from graph_llm.train.segmented import (
    SegmentedTrainer,
    detach_states,
    perturb_states,
)


def _state_leaves(state: DeltaMemoryState) -> list[torch.Tensor]:
    """The differentiable tensors inside one :class:`DeltaMemoryState`.

    The carried per-layer state is the delta-memory matrix plus (when the causal
    conv is enabled, card 571d50ec) the conv tail; both must receive gradient when
    the truncated-BPTT graph is connected and zero once detached.
    """
    leaves = [state.memory]
    if state.conv_tail is not None:
        leaves.append(state.conv_tail)
    return leaves


def _mem_cfg(**overrides: Any) -> ModelConfig:
    """A small delta_memory_lm config for the segmented trainer on CPU."""
    base: dict[str, Any] = {
        "name": "delta_memory_lm",
        "vocab_size": 64,
        "d_model": 32,
        "delta_layers": 2,
        "delta_n_heads": 4,
        "delta_head_k_dim": 16,
        "delta_head_v_dim": 16,
        "delta_dropout": 0.0,
        "dropout": 0.0,
        "max_seq_len": 64,
        "front_end": "none",
        "segment_len": 8,
        "bptt_window": 2,
    }
    base.update(overrides)
    return ModelConfig(**base)


def _full_cfg(model: ModelConfig, **train_overrides: Any) -> Config:
    train_kwargs: dict[str, Any] = {
        "max_steps": 30,
        "warmup_steps": 2,
        "lr": 2e-3,
        "mixed_precision": "no",
        "log_every": 1000,
    }
    train_kwargs.update(train_overrides)
    return Config(
        model=model,
        data=DataConfig(source="synthetic", seq_len=model.segment_len, batch_size=4),
        train=TrainConfig(**train_kwargs),
    )


def _learnable_stream(period: int = 16, vocab: int = 64, repeats: int = 3000) -> np.ndarray:
    """A repeating byte stream whose period the model CAN learn (so loss can drop)."""
    return np.tile(np.arange(period, dtype=np.int64) % vocab, repeats)


# ---------------------------------------------------------------------------
# THE GATE: truncated-BPTT detach correctness
# ---------------------------------------------------------------------------


def test_detach_states_severs_graph() -> None:
    """``detach_states`` returns states with no grad_fn and requires_grad False.

    Covers the :class:`DeltaMemoryState` carry (card 571d50ec): both the memory
    matrix and the conv tail must be detached.
    """
    live = [
        DeltaMemoryState(
            memory=(torch.randn(2, 4, 8, 8, requires_grad=True) * 2.0),
            conv_tail=(torch.randn(2, 3, 16, requires_grad=True) * 2.0),
        )
        for _ in range(2)
    ]
    # Both constituents are graph-connected (have a grad_fn).
    assert all(
        s.memory.grad_fn is not None and s.conv_tail is not None
        and s.conv_tail.grad_fn is not None
        for s in live
    )
    det = detach_states(live)
    assert det is not None
    for d in det:
        assert isinstance(d, DeltaMemoryState)
        for t in _state_leaves(d):
            assert t.grad_fn is None
            assert not t.requires_grad
    assert detach_states(None) is None

    # A bare-tensor state (the pre-conv path / disabled conv) is still accepted.
    bare = [torch.randn(2, 4, 8, 8, requires_grad=True) * 2.0]
    det_bare = detach_states(bare)
    assert det_bare is not None
    assert det_bare[0].grad_fn is None and not det_bare[0].requires_grad  # type: ignore[union-attr]


def test_truncated_bptt_detach_severs_gradient() -> None:
    """LOAD-BEARING: gradients do NOT flow past the detached window boundary.

    Isolate graph CONNECTIVITY from the numerical dependence of the state values:
    feed the SAME carried-state values into a window twice — once as a graph-leaf
    that requires grad (CONNECTED), once detached (the truncated-BPTT boundary).
    When connected the carried-state leaf is differentiable (non-zero grad); once
    detached the very same leaf receives EXACTLY zero gradient — gradients cannot
    cross more than the K-segment window.
    """
    torch.manual_seed(0)
    cfg = _full_cfg(_mem_cfg())
    model = build_model(cfg)
    model.train()

    seg_a = torch.randint(0, cfg.model.vocab_size, (2, 8))
    seg_b = torch.randint(0, cfg.model.vocab_size, (2, 8))
    tgt_b = torch.randint(0, cfg.model.vocab_size, (2, 8))

    # Produce a realistic carried (final) state from an earlier segment.
    _, _, state_a = model(seg_a, None, None, True)

    # CONNECTED: same VALUES, but differentiable leaves (matrix + conv tail).
    # Gradients must reach them.
    leaf = [
        DeltaMemoryState(
            memory=s.memory.detach().clone().requires_grad_(True),
            conv_tail=(
                None
                if s.conv_tail is None
                else s.conv_tail.detach().clone().requires_grad_(True)
            ),
        )
        for s in state_a
    ]
    leaf_tensors = [t for s in leaf for t in _state_leaves(s)]
    model.zero_grad(set_to_none=True)
    loss_conn, _, _ = model(seg_b, tgt_b, leaf, True)
    loss_conn.backward()
    connected_grad = sum(
        (lf.grad.norm().item() if lf.grad is not None else 0.0) for lf in leaf_tensors
    )
    assert connected_grad > 1e-8, (
        "the carried state must be differentiable when the graph is connected — "
        "otherwise the within-window gradient never reaches it"
    )

    # DETACHED (what the trainer does at the window boundary): same values, detached.
    detached = detach_states(leaf)
    assert detached is not None
    for d in detached:
        assert isinstance(d, DeltaMemoryState)
        for t in _state_leaves(d):
            assert t.grad_fn is None and not t.requires_grad
    for lf in leaf_tensors:
        lf.grad = None
    model.zero_grad(set_to_none=True)
    loss_det, _, _ = model(seg_b, tgt_b, detached, True)
    loss_det.backward()
    detached_grad = sum(
        (lf.grad.norm().item() if lf.grad is not None else 0.0) for lf in leaf_tensors
    )
    assert detached_grad == 0.0, (
        "no gradient may flow into a state that was detached at the window boundary"
    )


def test_perturb_states_changes_values_and_detaches() -> None:
    """State-noise augmentation perturbs each state (matrix + conv tail) and detaches.

    Both constituents of a :class:`DeltaMemoryState` are noised (card 571d50ec):
    the conv tail is part of the carried context the model must learn to read from.
    """
    states = [
        DeltaMemoryState(
            memory=torch.randn(2, 4, 8, 8, requires_grad=True),
            conv_tail=torch.randn(2, 3, 16, requires_grad=True),
        )
        for _ in range(2)
    ]
    out = perturb_states(states, noise_std=0.3, generator=torch.Generator().manual_seed(0))
    assert out is not None
    for orig, pert in zip(states, out, strict=True):
        assert isinstance(pert, DeltaMemoryState)
        assert (pert.memory - orig.memory.detach()).abs().max().item() > 0.0
        assert pert.memory.grad_fn is None and not pert.memory.requires_grad
        assert pert.conv_tail is not None and orig.conv_tail is not None
        assert (pert.conv_tail - orig.conv_tail.detach()).abs().max().item() > 0.0
        assert pert.conv_tail.grad_fn is None and not pert.conv_tail.requires_grad
    assert perturb_states(None, 0.1) is None


# ---------------------------------------------------------------------------
# Ordered-segment stream
# ---------------------------------------------------------------------------


def test_ordered_stream_is_contiguous_and_strided() -> None:
    """Consecutive segments tile each sub-stream; rows are strided sub-streams."""
    tokens = np.arange(40, dtype=np.int64)
    stream = OrderedSegmentStream(tokens, segment_len=4, batch_size=2)
    segs = list(stream)
    assert len(stream) == len(segs) == 4
    # Row 0 is the first contiguous sub-stream (tokens 0..19); row 1 the next.
    assert segs[0].inputs[0].tolist() == [0, 1, 2, 3]
    assert segs[0].targets[0].tolist() == [1, 2, 3, 4]
    assert segs[1].inputs[0].tolist() == [4, 5, 6, 7]  # contiguous continuation
    assert segs[0].inputs[1].tolist() == [20, 21, 22, 23]  # strided second stream


def test_ordered_stream_reset_flags() -> None:
    """Only the first segment resets when interval=0; every interval otherwise."""
    tokens = np.arange(40, dtype=np.int64)
    no_reset = [s.stream_reset for s in OrderedSegmentStream(tokens, 4, 2, 0)]
    assert no_reset == [True, False, False, False]
    every_two = [s.stream_reset for s in OrderedSegmentStream(tokens, 4, 2, 2)]
    assert every_two == [True, False, True, False]


def test_ordered_stream_rejects_too_short() -> None:
    with pytest.raises(ValueError):
        OrderedSegmentStream(np.arange(4, dtype=np.int64), segment_len=8, batch_size=2)


def test_iter_ordered_segments_matches_class() -> None:
    tokens = np.arange(40, dtype=np.int64)
    a = [s.inputs.tolist() for s in iter_ordered_segments(tokens, 4, 2)]
    b = [s.inputs.tolist() for s in OrderedSegmentStream(tokens, 4, 2)]
    assert a == b


# ---------------------------------------------------------------------------
# Synthetic cross-segment retrieval task generator
# ---------------------------------------------------------------------------


def test_cross_segment_task_structure() -> None:
    """Key in an early segment, query in the last, answer masked + outside window."""
    task = make_cross_segment_task("12345", n_segments=3, segment_tokens=20)
    assert len(task.segment_inputs) == 3
    assert task.answer == "12345"
    assert task.passkey_segment_index == 0
    assert task.query_segment_index == 2
    assert task.answer_outside_query_window is True
    # Leading segments carry NO scored positions (mask all False) — they only build
    # the memory.
    assert all(not bool(m.any()) for m in task.segment_masks[:-1])
    # The final segment scores exactly the answer tokens, and those targets decode
    # back to the key.
    final_tgt = task.segment_targets[-1][0]
    final_mask = task.segment_masks[-1][0]
    assert int(final_mask.sum()) == len("12345")
    answer_ids = final_tgt[final_mask].tolist()
    assert bytes(answer_ids).decode() == "12345"


def test_cross_segment_task_rejects_bad_args() -> None:
    with pytest.raises(ValueError):
        make_cross_segment_task("1", n_segments=1, segment_tokens=10, key_digits=1)
    with pytest.raises(ValueError):
        make_cross_segment_task("1", n_segments=2, segment_tokens=0, key_digits=1)
    # Mismatched key_digits: passkey "abc" has 3 chars but key_digits=5 → ValueError.
    with pytest.raises(ValueError, match="key_digits"):
        make_cross_segment_task("abc", n_segments=2, segment_tokens=10, key_digits=5)


def test_task_sampler_draws_varied_valid_tasks() -> None:
    sampler = CrossSegmentTaskSampler(segment_tokens=16, min_segments=2, max_segments=4, seed=0)
    keys = set()
    for _ in range(8):
        task = sampler.sample()
        assert task.answer_outside_query_window is True
        assert int(task.segment_masks[-1].sum()) == len(task.answer)
        keys.add(task.answer)
    assert len(keys) > 1, "sampler should draw distinct random passkeys"


def test_cross_segment_task_key_digits_1_produces_single_answer_token() -> None:
    """key_digits=1: answer is exactly 1 token, mask scores exactly 1 position."""
    # 1-digit keys run from 1–9 (single decimal digit).
    task = make_cross_segment_task("7", n_segments=3, segment_tokens=20, key_digits=1)
    assert task.answer == "7"
    # Leading segments have no scored positions.
    assert all(not bool(m.any()) for m in task.segment_masks[:-1])
    # Final segment mask scores exactly 1 position (1 byte = 1 token).
    final_mask = task.segment_masks[-1][0]
    assert int(final_mask.sum()) == 1, "key_digits=1 must score exactly 1 answer token"
    # The masked target token decodes back to the key character.
    final_tgt = task.segment_targets[-1][0]
    answer_ids = final_tgt[final_mask].tolist()
    assert bytes(answer_ids).decode() == "7"


def test_task_sampler_key_digits_1_easy_probe() -> None:
    """CrossSegmentTaskSampler with key_digits=1 generates 1-character passkeys."""
    sampler = CrossSegmentTaskSampler(
        segment_tokens=16, min_segments=2, max_segments=3, key_digits=1, seed=42
    )
    for _ in range(10):
        task = sampler.sample()
        assert len(task.answer) == 1, "1-digit key must be a single character"
        assert task.answer.isdigit(), "passkey must be a decimal digit"
        assert task.answer_outside_query_window is True
        # Mask scores exactly 1 position.
        assert int(task.segment_masks[-1].sum()) == 1
        # The masked target decodes to the key.
        final_tgt = task.segment_targets[-1][0]
        final_mask = task.segment_masks[-1][0]
        assert bytes(final_tgt[final_mask].tolist()).decode() == task.answer


def test_masked_token_loss_scores_only_masked_positions() -> None:
    torch.manual_seed(0)
    logits = torch.randn(1, 5, 64)
    targets = torch.randint(0, 64, (1, 5))
    # All-False mask -> zero loss (no contribution).
    zero = masked_token_loss(logits, targets, torch.zeros(1, 5, dtype=torch.bool))
    assert float(zero) == 0.0
    # A mask selecting one position equals the CE at that position.
    mask = torch.zeros(1, 5, dtype=torch.bool)
    mask[0, 3] = True
    got = masked_token_loss(logits, targets, mask)
    expected = torch.nn.functional.cross_entropy(logits[0, 3:4], targets[0, 3:4])
    assert torch.allclose(got, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# SegmentedTrainer end-to-end
# ---------------------------------------------------------------------------


def test_segmented_trainer_reduces_loss_and_carries_state() -> None:
    """End-to-end: loss decreases over windows AND state is carried across segments."""
    torch.manual_seed(0)
    np.random.seed(0)
    cfg = _full_cfg(_mem_cfg(stream_reset_interval=0), max_steps=40, lr=3e-3)
    model = build_model(cfg)
    stream = _learnable_stream()
    trainer = SegmentedTrainer(cfg, model, stream, device=torch.device("cpu"))

    # Confirm a non-None states_in is genuinely threaded across segments.
    seen = {"total": 0, "carried": 0}
    orig = model.forward

    def wrapped(x, targets=None, states_in=None, return_states=False):  # noqa: ANN001
        seen["total"] += 1
        if states_in is not None:
            seen["carried"] += 1
        return orig(x, targets, states_in, return_states)

    model.forward = wrapped  # type: ignore[method-assign]
    history = trainer.train()

    assert len(history) == 40
    first = float(np.mean(history[:5]))
    last = float(np.mean(history[-5:]))
    assert last < first, f"loss must decrease: first5={first:.4f} last5={last:.4f}"
    assert seen["carried"] > 0, "state must be carried across segment boundaries"
    assert trainer.state.detach_count > 0, "each window must detach the carried state"
    assert trainer.state.carried_segment_count > 0


def test_segmented_trainer_state_noise_activates_when_enabled() -> None:
    """The state-distribution-exposure augmentation fires iff enabled."""
    torch.manual_seed(0)
    np.random.seed(0)
    stream = _learnable_stream()

    on = _full_cfg(_mem_cfg(state_noise_prob=1.0, state_noise_std=0.2), max_steps=20)
    trainer_on = SegmentedTrainer(on, build_model(on), stream, device=torch.device("cpu"))
    trainer_on.train()
    assert trainer_on.state.state_noise_count > 0

    off = _full_cfg(_mem_cfg(state_noise_prob=0.0), max_steps=20)
    trainer_off = SegmentedTrainer(off, build_model(off), stream, device=torch.device("cpu"))
    trainer_off.train()
    assert trainer_off.state.state_noise_count == 0


def test_segmented_trainer_interleaves_synthetic_tasks() -> None:
    """Synthetic cross-segment tasks are drawn at the configured fraction."""
    torch.manual_seed(0)
    np.random.seed(0)
    stream = _learnable_stream()
    cfg = _full_cfg(_mem_cfg(synthetic_task_fraction=0.5), max_steps=30)
    trainer = SegmentedTrainer(cfg, build_model(cfg), stream, device=torch.device("cpu"))
    trainer.train()
    assert trainer.state.synthetic_task_count > 0
    # And the LM stream still ran some windows.
    assert trainer.state.synthetic_task_count < 30


def test_segmented_trainer_bptt_window_one_detaches_every_segment() -> None:
    """K=1 detaches every segment (pure exposure, no through-segment gradient)."""
    torch.manual_seed(0)
    np.random.seed(0)
    stream = _learnable_stream()
    cfg = _full_cfg(_mem_cfg(bptt_window=1), max_steps=20)
    trainer = SegmentedTrainer(cfg, build_model(cfg), stream, device=torch.device("cpu"))
    history = trainer.train()
    assert len(history) == 20
    # Every one of the 20 windows is a single segment -> 20 detaches.
    assert trainer.state.detach_count == 20
