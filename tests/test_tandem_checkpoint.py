"""Checkpoint/resume equivalence tests for train/tandem.py's `_train_one` (card 69776c3e).

Acceptance bar: train N steps straight vs train k steps -> checkpoint -> resume ->
train N-k steps must reproduce IDENTICAL per-step losses and final model parameters
(CPU, tiny config).  The design argument (see train/checkpoint.py's module docstring):
the depth-ramp / type-warmup / forced-mix / commit-anneal curriculum inside
`_train_one` is a PURE FUNCTION of the loop's `step` counter (or of the model's own
`_tandem_step` buffer, already inside `model.state_dict()`), so a checkpoint that
captures {step, model, optimizer, scheduler, every RNG stream drawn from} is
sufficient for exact resume -- no extra schedule counters are threaded through.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import graph_llm.models  # noqa: F401 -- registers "delta_memory_lm"
from graph_llm.train.checkpoint import load_training_checkpoint
from graph_llm.train.tandem import TandemConfig, _train_one


def _tiny_cfg(**overrides: Any) -> TandemConfig:
    """A tiny CPU-fast TandemConfig exercising the 2-way M+R curriculum (type_warmup +
    gate_mix_warmup/commit_anneal + gate noise, so the resume test actually exercises
    torch's global RNG + the python `rng` stream + the model's own step buffer)."""
    base: dict[str, Any] = dict(
        train_r_depths=(4,),
        test_r_depths=(4,),
        k_train=4,
        train_steps=8,
        batch_size=4,
        seg_len=48,
        n_segments=2,
        n_chains=1,
        d_model=16,
        delta_layers=1,
        delta_n_heads=1,
        delta_head_dim=8,
        eval_batches=1,
        eval_batch=4,
        lr=1e-3,
        lr_warmup=2,
        log_every=1_000_000,  # suppress progress printing
        type_warmup=3,
        gate_mix_warmup=3,
        gate_commit_anneal=3,
        gate_noise_std=0.5,  # non-zero: exercises torch's global RNG every step
    )
    base.update(overrides)
    return TandemConfig(**base)


def _state_dicts_equal(a: dict, b: dict) -> bool:
    if a.keys() != b.keys():
        return False
    return all(torch.equal(a[k].cpu(), b[k].cpu()) for k in a)


def test_resume_equivalence_losses_and_params_match_exactly(tmp_path) -> None:
    device = torch.device("cpu")
    seed = 123
    n_steps = 8
    k = 3

    # (a) straight N-step run.
    straight_cfg = _tiny_cfg(train_steps=n_steps)
    straight = _train_one(straight_cfg, seed, device, verbose=False, capture_model=True)

    # (b1) first k steps, checkpointing at the end.
    ckpt_dir = tmp_path / "ckpts"
    b1_cfg = _tiny_cfg(
        train_steps=k,
        checkpoint_dir=str(ckpt_dir),
        checkpoint_every=k,  # exactly one checkpoint, at the very last of the k steps
    )
    b1 = _train_one(b1_cfg, seed, device, verbose=False, capture_model=True)
    assert len(b1["loss_history"]) == k
    # Losses for the shared first k steps must match the straight run's first k.
    assert b1["loss_history"] == straight["loss_history"][:k]

    ckpt_files = sorted(ckpt_dir.glob("tandem_ckpt_seed*_step*.pt"))
    assert len(ckpt_files) == 1, f"expected exactly one checkpoint, found {ckpt_files}"
    ckpt_path = ckpt_files[0]
    ckpt = load_training_checkpoint(ckpt_path)
    assert ckpt["step"] == k
    assert ckpt["rng_state"] is not None

    # (b2) resume from the checkpoint and train the remaining N-k steps.
    b2_cfg = _tiny_cfg(train_steps=n_steps, resume_from=str(ckpt_path))
    b2 = _train_one(b2_cfg, seed, device, verbose=False, capture_model=True)

    assert len(b2["loss_history"]) == n_steps - k
    assert b2["loss_history"] == straight["loss_history"][k:]
    assert _state_dicts_equal(b2["final_model_state"], straight["final_model_state"])


def test_resume_without_checkpoint_is_a_pure_no_op_default() -> None:
    """checkpoint_dir=None / resume_from=None (the shipped defaults) must not create
    any files or change the return-dict's existing keys' values."""
    device = torch.device("cpu")
    cfg = _tiny_cfg()
    assert cfg.checkpoint_dir is None
    assert cfg.resume_from is None
    result = _train_one(cfg, seed=7, device=device, verbose=False)
    assert "acc_M" in result
    assert "loss_history" in result
    assert "final_model_state" not in result  # capture_model defaults False


def test_checkpoint_payload_captures_full_rng_state(tmp_path) -> None:
    device = torch.device("cpu")
    ckpt_dir = tmp_path / "ckpts"
    cfg = _tiny_cfg(train_steps=2, checkpoint_dir=str(ckpt_dir), checkpoint_every=2)
    _train_one(cfg, seed=1, device=device, verbose=False)
    ckpt_files = list(ckpt_dir.glob("*.pt"))
    assert len(ckpt_files) == 1
    ckpt = load_training_checkpoint(ckpt_files[0])
    for key in ("model_state", "optimizer_state", "scheduler_state", "rng_state", "step"):
        assert key in ckpt
    rng = ckpt["rng_state"]
    assert rng["python_random"] is not None
    assert isinstance(rng["torch_cpu"], torch.Tensor)
