"""Checkpoint/resume equivalence tests for SegmentedTrainer (card 53e55fd2).

Acceptance bar (mirrors ``tests/test_tandem_checkpoint.py``'s idiom, card 69776c3e):
train N steps straight vs train k steps -> checkpoint -> FRESH trainer resume ->
train N-k steps must reproduce IDENTICAL per-step losses, final model parameters,
AND the carried DeltaMemoryState -- CPU, tiny config, covering BOTH a single-fusion
tandem model (``tandem_blocks=1``, the default) and a STACKED model
(``tandem_blocks=2``, card d05da2db / since commit cda73d0 the carried-state list
length is ``len(model.stacked_blocks)`` rather than ``delta_layers`` -- the
checkpoint machinery must handle both shapes generically, per
``detach_states`` / ``states_to_device``).

The tiny config deliberately turns on ``state_noise_prob`` and
``synthetic_task_fraction`` (both > 0) so the resume proof exercises every RNG
stream SegmentedTrainer draws from: the schedule generator (``_sched_rng``, used
for BOTH the use-synthetic draw and the state-noise draw), the noise generator
(``_noise_rng``, used by ``perturb_states``), and the synthetic-task sampler's own
numpy generator -- not just the trivial all-off path.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from torch import Tensor

import graph_llm.models  # noqa: F401 — registers "delta_memory_lm"
from graph_llm.config import Config, DataConfig, ModelConfig, TrainConfig
from graph_llm.models import build_model
from graph_llm.models.components.delta_memory import DeltaMemoryState
from graph_llm.train.checkpoint import load_training_checkpoint
from graph_llm.train.segmented import SegmentedTrainer


def _mem_cfg(stacked: bool, **overrides: Any) -> ModelConfig:
    """A tiny CPU-fast tandem delta_memory_lm config exercising the state-noise +
    synthetic-task RNG paths, in either single-fusion or stacked (2-block) mode."""
    base: dict[str, Any] = dict(
        name="delta_memory_lm",
        vocab_size=64,
        d_model=16,
        delta_layers=2,
        delta_n_heads=1,
        delta_head_k_dim=8,
        delta_head_v_dim=8,
        delta_dropout=0.0,
        dropout=0.0,
        max_seq_len=32,
        front_end="none",
        segment_len=8,
        bptt_window=2,
        stream_reset_interval=0,
        state_noise_prob=0.5,
        state_noise_std=0.1,
        synthetic_task_fraction=0.3,
        synthetic_key_digits=2,
        tandem_enabled=True,
        reasoning_segment_len=8,
        causal_reasoner_steps=2,
        causal_reasoner_key_dim=8,
        causal_reasoner_query_window=4,
        causal_reasoner_conv_kernel=3,
        causal_reasoner_gru_layers=1,
    )
    if stacked:
        base["tandem_blocks"] = 2
    base.update(overrides)
    return ModelConfig(**base)


def _full_cfg(model: ModelConfig, **train_overrides: Any) -> Config:
    train_kwargs: dict[str, Any] = dict(
        max_steps=8, warmup_steps=2, lr=2e-3, mixed_precision="no", log_every=1000, seed=123,
    )
    train_kwargs.update(train_overrides)
    return Config(
        model=model,
        data=DataConfig(source="synthetic", seq_len=model.segment_len, batch_size=2),
        train=TrainConfig(**train_kwargs),
    )


def _stream(period: int = 24, vocab: int = 64, repeats: int = 400) -> np.ndarray:
    return np.tile(np.arange(period, dtype=np.int64) % vocab, repeats)


def _build_trainer(cfg: Config, tokens: np.ndarray, seed: int) -> SegmentedTrainer:
    """Build a FRESH model (its own random init, matching the tandem-checkpoint test's
    convention) then wrap it in a fresh SegmentedTrainer -- mirrors how a real
    interrupted-and-resumed run starts from a brand-new process."""
    torch.manual_seed(seed)
    model = build_model(cfg)
    return SegmentedTrainer(cfg, model, tokens, device=torch.device("cpu"))


def _state_dicts_equal(a: dict, b: dict) -> bool:
    if a.keys() != b.keys():
        return False
    return all(torch.equal(a[k].cpu(), b[k].cpu()) for k in a)


def _states_equal(
    a: list[DeltaMemoryState | Tensor] | None, b: list[DeltaMemoryState | Tensor] | None
) -> bool:
    if a is None or b is None:
        return a is None and b is None
    if len(a) != len(b):
        return False
    for sa, sb in zip(a, b, strict=True):
        assert isinstance(sa, DeltaMemoryState) and isinstance(sb, DeltaMemoryState)
        if not torch.equal(sa.memory.cpu(), sb.memory.cpu()):
            return False
        if (sa.conv_tail is None) != (sb.conv_tail is None):
            return False
        if sa.conv_tail is not None:
            assert sb.conv_tail is not None
            if not torch.equal(sa.conv_tail.cpu(), sb.conv_tail.cpu()):
                return False
    return True


@pytest.mark.parametrize("stacked", [False, True], ids=["single_fusion", "stacked_blocks2"])
def test_resume_equivalence_losses_params_and_carried_state(
    tmp_path: Path, stacked: bool
) -> None:
    """THE acceptance test (card 53e55fd2): straight N-step run vs checkpoint-at-k +
    fresh-trainer resume + (N-k)-step run must match EXACTLY -- losses, final model
    parameters, and the carried DeltaMemoryState (any shape).

    Both the checkpointed run and the resumed run use the SAME ``max_steps`` as the
    straight run (the real-world invariant: an interrupted job's total step budget
    does not change just because it got interrupted) -- unlike
    ``tests/test_tandem_checkpoint.py``'s ``b1_cfg`` convenience of a SMALLER
    ``train_steps``, which is only safe there because the tandem trainer's own LR
    schedule (warmup-to-constant) does not depend on the total step count.
    SegmentedTrainer's default schedule is COSINE, whose shape *does* depend on
    ``max_steps`` (see ``train/optim.py::build_scheduler``), so giving the
    checkpointed run a smaller ``max_steps`` than the reference run would make their
    LR trajectories genuinely different SCHEDULES, not a checkpoint bug -- fixed here
    by keeping ``max_steps`` identical across all three runs and instead choosing
    ``k > n_steps / 2`` so ``checkpoint_every=k`` still produces exactly one on-disk
    checkpoint within the run.
    """
    m = _mem_cfg(stacked)
    tokens = _stream()
    seed = 7
    n_steps = 8
    k = 5  # > n_steps/2 so checkpoint_every=k fires exactly once within n_steps

    # (a) straight N-step run (the reference trajectory).
    straight_cfg = _full_cfg(m, max_steps=n_steps)
    straight = _build_trainer(straight_cfg, tokens, seed)
    straight_losses = straight.train()
    assert len(straight_losses) == n_steps

    # (b1) an IDENTICALLY-configured N-step run (same max_steps -- same schedule
    # shape) that ALSO checkpoints at step k.  Checkpointing is read-only w.r.t. the
    # trainer's own state (captures via .get_state() / .state_dict(), never mutates
    # an RNG stream), so this run's OWN loss trajectory must equal the reference
    # run's exactly -- asserted as a sanity check that checkpointing itself is
    # trajectory-neutral, mirroring the eval-hook's equivalent contract.
    ckpt_dir = tmp_path / "ckpts"
    b1_cfg = _full_cfg(
        m, max_steps=n_steps, checkpoint_dir=str(ckpt_dir), checkpoint_every=k
    )
    b1 = _build_trainer(b1_cfg, tokens, seed)
    b1_losses = b1.train()
    assert b1_losses == straight_losses

    ckpt_files = sorted(ckpt_dir.glob("segmented_ckpt_step*.pt"))
    assert len(ckpt_files) == 1, f"expected exactly one checkpoint, found {ckpt_files}"
    ckpt_path = ckpt_files[0]
    ckpt = load_training_checkpoint(ckpt_path)
    assert ckpt["step"] == k
    assert ckpt["rng_state"] is not None

    # (b2) resume from the checkpoint, in a FRESH trainer wrapping a freshly (randomly)
    # initialised model -- load_state_dict must fully overwrite it -- and train the
    # remaining N-k steps under the SAME max_steps (the real resume invariant above).
    b2_cfg = _full_cfg(m, max_steps=n_steps, resume_from=str(ckpt_path))
    b2 = _build_trainer(b2_cfg, tokens, seed)
    b2_losses = b2.train()

    assert len(b2_losses) == n_steps - k
    assert b2_losses == straight_losses[k:]
    assert _state_dicts_equal(b2.model.state_dict(), straight.model.state_dict())
    assert _states_equal(b2._last_carried_state, straight._last_carried_state)  # noqa: SLF001


def test_resume_without_checkpoint_is_a_pure_no_op_default() -> None:
    """checkpoint_every=0 / resume_from=None (the shipped defaults) must not create
    any checkpoint files or change train()'s behaviour."""
    m = _mem_cfg(stacked=False)
    cfg = _full_cfg(m)
    assert cfg.train.checkpoint_every == 0
    assert cfg.train.resume_from is None
    trainer = _build_trainer(cfg, _stream(), seed=1)
    losses = trainer.train()
    assert len(losses) == cfg.train.max_steps


def test_checkpoint_payload_captures_full_rng_state_and_segmented_extras(
    tmp_path: Path,
) -> None:
    """The checkpoint payload must include the trainer's own named torch.Generator
    states (``_sched_rng`` / ``_noise_rng`` -- NOT covered by the global
    ``torch_cpu`` RNG capture alone) and every segmented-specific extra needed for
    exact resume."""
    m = _mem_cfg(stacked=False)
    ckpt_dir = tmp_path / "ckpts"
    cfg = _full_cfg(m, max_steps=2, checkpoint_dir=str(ckpt_dir), checkpoint_every=2)
    trainer = _build_trainer(cfg, _stream(), seed=1)
    trainer.train()
    ckpt_files = list(ckpt_dir.glob("*.pt"))
    assert len(ckpt_files) == 1
    ckpt = load_training_checkpoint(ckpt_files[0])
    for key in ("model_state", "optimizer_state", "scheduler_state", "rng_state", "step", "extra"):
        assert key in ckpt
    rng = ckpt["rng_state"]
    assert isinstance(rng["torch_cpu"], torch.Tensor)
    assert set(rng["torch_generators"].keys()) == {"sched", "noise"}
    assert isinstance(rng["torch_generators"]["sched"], torch.Tensor)
    assert "synthetic_task" in rng["numpy"]
    extra = ckpt["extra"]
    for key in (
        "segments_consumed",
        "carried_state",
        "carried_segment_count",
        "detach_count",
        "state_noise_count",
        "synthetic_task_count",
    ):
        assert key in extra


def test_map_location_cpu_default_keeps_rng_tensors_cpu(tmp_path: Path) -> None:
    """The card-69776c3e trap, re-verified here: load_training_checkpoint's default
    (map_location="cpu") must leave the RNG-state ByteTensors on CPU so
    torch.set_rng_state / Generator.set_state accept them without extra coercion."""
    m = _mem_cfg(stacked=False)
    ckpt_dir = tmp_path / "ckpts"
    cfg = _full_cfg(m, max_steps=1, checkpoint_dir=str(ckpt_dir), checkpoint_every=1)
    trainer = _build_trainer(cfg, _stream(), seed=1)
    trainer.train()
    ckpt_path = next(iter(ckpt_dir.glob("*.pt")))
    ckpt = load_training_checkpoint(ckpt_path)  # default map_location="cpu"
    assert ckpt["rng_state"]["torch_cpu"].device.type == "cpu"
    for t in ckpt["rng_state"]["torch_generators"].values():
        assert t.device.type == "cpu"
