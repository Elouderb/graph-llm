"""Reusable checkpoint/resume helpers with full RNG capture (card 69776c3e).

``model.state_dict()`` / ``optimizer.state_dict()`` / ``scheduler.state_dict()``
only capture what PyTorch itself tracks.  The *schedule* a trainer layers on top
(a step counter driving a warmup, an annealed loss weight, a curriculum ramp) is
typically a PURE FUNCTION of that step counter plus whatever RNG streams the loop
draws from -- so a checkpoint that captures ``{step, model/optimizer/scheduler
state, every RNG stream in play}`` reproduces the WHOLE schedule on resume with no
separate bookkeeping, *provided* the training loop resumes its counting variable
at the saved step (not 0) and nothing between save and resume consumes those RNG
streams out of order.

This is exactly the shape of ``train/tandem.py``'s ``_train_one``: the depth-ramp
window, the type-warmup / forced-mix curriculum, and the gate commit-anneal weight
are all computed directly from ``step`` (the loop variable) or from the model's own
``_tandem_step`` buffer (already captured by ``model.state_dict()``) -- so no extra
counters need to be threaded through the checkpoint payload beyond what is captured
here.  Verified by inspection (card 69776c3e comments) and by the resume-equivalence
test in ``tests/test_tandem_checkpoint.py``.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def capture_rng_state(
    py_rng: random.Random | None = None,
    np_rngs: dict[str, np.random.Generator] | None = None,
) -> dict[str, Any]:
    """Snapshot every RNG stream a training loop might draw from.

    Args:
        py_rng: The trainer's own ``random.Random`` instance (e.g. draws example
            kinds / curriculum depths), or ``None`` to skip.
        np_rngs: Named ``np.random.Generator`` instances (e.g. plain-text
            train/eval samplers), or ``None``/empty to skip.

    Returns:
        A plain dict safe to ``torch.save``: ``"python_random"`` (the
        ``random.Random`` state tuple, or ``None``), ``"numpy"`` (name ->
        bit-generator state dict), ``"torch_cpu"`` (a ``torch.ByteTensor``), and
        ``"torch_cuda"`` (list of per-device ``torch.ByteTensor``, or ``None`` when
        CUDA is unavailable).  Torch's CPU (and CUDA) global RNG matters whenever
        model code draws from it during training -- e.g. the tandem gate's
        exploration noise (``gate_noise_std``, ``torch.randn_like``), which fires
        every training step by default.
    """
    return {
        "python_random": py_rng.getstate() if py_rng is not None else None,
        "numpy": (
            {name: g.bit_generator.state for name, g in np_rngs.items()} if np_rngs else {}
        ),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
    }


def restore_rng_state(
    state: dict[str, Any],
    py_rng: random.Random | None = None,
    np_rngs: dict[str, np.random.Generator] | None = None,
) -> None:
    """Restore RNG streams captured by :func:`capture_rng_state`, in place.

    Args:
        state: A dict produced by :func:`capture_rng_state`.
        py_rng: The ``random.Random`` instance to restore into (must be the SAME
            object the trainer continues to draw from), or ``None`` to skip.
        np_rngs: Named ``np.random.Generator`` instances to restore into (keys must
            match those passed to :func:`capture_rng_state`), or ``None``/empty.
    """
    if py_rng is not None and state.get("python_random") is not None:
        py_rng.setstate(state["python_random"])
    saved_numpy = state.get("numpy") or {}
    for name, g in (np_rngs or {}).items():
        if name in saved_numpy:
            g.bit_generator.state = saved_numpy[name]
    torch.set_rng_state(state["torch_cpu"])
    cuda_state = state.get("torch_cuda")
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_state)


def save_training_checkpoint(
    path: str | Path,
    *,
    step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any = None,
    rng_state: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Save a full resumable checkpoint: model + optimizer + scheduler + RNG + extra.

    Args:
        path: Destination file path; parent directories are created as needed.
        step: The trainer's own step counter at save time (the resume loop restarts
            here, NOT at 0).
        model: The model being trained (``model.state_dict()`` is saved).
        optimizer: The optimizer (``optimizer.state_dict()`` is saved).
        scheduler: Optional LR scheduler (``scheduler.state_dict()`` is saved when
            given; ``None`` otherwise).
        rng_state: A dict from :func:`capture_rng_state`, or ``None`` to skip RNG
            capture (NOT recommended for exact resume-equivalence).
        extra: Any additional plain-old-data to round-trip (e.g. a config dict).

    Returns:
        The resolved :class:`~pathlib.Path` written to.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "step": step,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "rng_state": rng_state,
        "extra": extra or {},
    }
    torch.save(payload, path)
    return path


def load_training_checkpoint(path: str | Path, map_location: Any = "cpu") -> dict[str, Any]:
    """Load a checkpoint saved by :func:`save_training_checkpoint`.

    ``weights_only=False`` is explicit: the payload's ``rng_state``/``extra``
    entries are plain Python objects (tuples, dicts) outside the tensor-only safe
    path, matching the existing convention in ``train/trainer.py``. Only load
    checkpoints your own trainer wrote.
    """
    return torch.load(Path(path), map_location=map_location, weights_only=False)


__all__ = [
    "capture_rng_state",
    "restore_rng_state",
    "save_training_checkpoint",
    "load_training_checkpoint",
]
