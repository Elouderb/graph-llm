"""Segmented stateful trainer — truncated BPTT for cross-segment memory.

Card 61f900ca, piece 3.  This is the training subsystem that teaches
``delta_memory_lm`` to actually USE the cross-segment state the piece-1 API
carries.  The standard :class:`~graph_llm.train.trainer.Trainer` (shuffled chunks,
state reset per sequence) is left UNTOUCHED; this is a separate, opt-in driver.

The mechanics, in order of importance:

* **Ordered-segment streaming.**  Consecutive segments of the byte stream are
  processed in order (via :class:`~graph_llm.data.loader.OrderedSegmentStream`),
  and the per-layer delta-memory state is carried across segment boundaries with
  ``model(x, targets, states_in, return_states=True)`` (the piece-1 API).  A later
  segment therefore sees the whole prior stream through the bounded memory.

* **Truncated BPTT, window K (Transformer-XL style).**  Loss is accumulated over
  ``K`` consecutive segments while the autograd graph stays connected through the
  carried state; then ``backward()`` runs, the optimiser steps, and the carried
  state is DETACHED before the next window.  The graph never grows past ``K``
  segments — gradients do NOT flow across a window boundary.  This is THE
  correctness invariant of truncated BPTT (tested directly).  Per arXiv 2507.02782
  a SHORT ``K`` suffices: the dominant lever for long-range USE is EXPOSURE to
  realistic carried-state distributions (which the carry itself supplies), not deep
  through-time gradient.

* **State-distribution exposure augmentation.**  With probability
  ``state_noise_prob`` a window's INITIAL carried state is replaced by a
  noise-perturbed version of itself (Gaussian, scaled to the state's RMS), widening
  the set of carried states the model is trained to read from (arXiv 2507.02782).

* **Synthetic cross-segment retrieval interleaving.**  With probability
  ``synthetic_task_fraction`` a step trains on a synthetic task
  (:class:`~graph_llm.data.synthetic_tasks.CrossSegmentTask` — key in an early
  segment, query later, answer outside the query window) instead of an ordered
  text-stream window.  Plain LM perplexity does not force far-back use; these tasks
  do, scored at the answer positions only.

Reuses :func:`~graph_llm.train.optim.build_optimizer` /
:func:`~graph_llm.train.optim.build_scheduler` and the AMP / grad-clip toolkit
pattern from the standard Trainer (torch 2.5 API: ``GradScaler("cuda")``,
``torch.autocast(device_type=...)``).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader

from graph_llm.data.loader import OrderedSegmentStream, _TextChunkDataset
from graph_llm.data.synthetic_tasks import CrossSegmentTaskSampler, masked_token_loss
from graph_llm.eval.report import build_eval_report, write_eval_report
from graph_llm.models.components.delta_memory import DeltaMemoryState
from graph_llm.train.optim import build_optimizer, build_scheduler
from graph_llm.utils.logging import get_logger, log_metrics, setup_logging
from graph_llm.utils.seed import seed_everything

if TYPE_CHECKING:
    from graph_llm.config import Config

_log = get_logger("segmented_trainer")


# A per-layer carried state is a DeltaMemoryState (the delta-memory matrix + the
# causal-conv tail, card 571d50ec).  A bare Tensor is still accepted for the
# legacy pre-conv path (and tests that hand-build raw states) — both are handled
# uniformly by mapping the detach/perturb over their constituent tensors.


def _detach_one(s: DeltaMemoryState | Tensor) -> DeltaMemoryState | Tensor:
    """Detach one per-layer state (a :class:`DeltaMemoryState` or a bare Tensor)."""
    if isinstance(s, DeltaMemoryState):
        return DeltaMemoryState(
            memory=s.memory.detach(),
            conv_tail=None if s.conv_tail is None else s.conv_tail.detach(),
        )
    return s.detach()


def detach_states(
    states: Sequence[DeltaMemoryState | Tensor] | None,
) -> list[DeltaMemoryState | Tensor] | None:
    """Detach every per-layer carried state, severing the autograd graph.

    Returns a NEW list whose tensors are detached (``requires_grad`` False,
    ``grad_fn`` ``None``), or ``None`` if ``states`` is ``None``.  Called at every
    truncated-BPTT window boundary so the next window's gradients cannot flow back
    through the carried state — the defining property of truncated BPTT.  Each
    per-layer state is a :class:`DeltaMemoryState` (both the delta-memory matrix
    and the conv tail are detached); a bare ``Tensor`` is also accepted for the
    pre-conv path.
    """
    if states is None:
        return None
    return [_detach_one(s) for s in states]


def _perturb_tensor(
    s: Tensor, noise_std: float, generator: torch.Generator | None
) -> Tensor:
    """RMS-scaled Gaussian perturbation of one tensor, detached."""
    s = s.detach()
    rms = s.pow(2).mean().clamp_min(1e-12).sqrt()
    noise = torch.empty_like(s).normal_(generator=generator)
    return s + noise_std * rms * noise


def perturb_states(
    states: Sequence[DeltaMemoryState | Tensor] | None,
    noise_std: float,
    generator: torch.Generator | None = None,
) -> list[DeltaMemoryState | Tensor] | None:
    """Add RMS-scaled Gaussian noise to each carried state (exposure augmentation).

    The perturbation magnitude is ``noise_std`` times each tensor's own RMS, so the
    noise is meaningful regardless of scale.  Both the delta-memory matrix and the
    causal-conv tail of each :class:`DeltaMemoryState` are perturbed (the conv tail
    is part of the carried context the model must learn to read from).  The result
    is detached (it seeds a fresh window).  Returns ``None`` if ``states`` is
    ``None``.

    Args:
        states: Per-layer carried states (each a :class:`DeltaMemoryState` or a
            bare ``(B, H, d_k, d_v)`` tensor) or ``None``.
        noise_std: Std of the Gaussian noise relative to each tensor's RMS.
        generator: Optional RNG for reproducible noise.

    Returns:
        A new list of perturbed, detached states (or ``None``).
    """
    if states is None:
        return None
    out: list[DeltaMemoryState | Tensor] = []
    for s in states:
        if isinstance(s, DeltaMemoryState):
            out.append(
                DeltaMemoryState(
                    memory=_perturb_tensor(s.memory, noise_std, generator),
                    conv_tail=(
                        None
                        if s.conv_tail is None
                        else _perturb_tensor(s.conv_tail, noise_std, generator)
                    ),
                )
            )
        else:
            out.append(_perturb_tensor(s, noise_std, generator))
    return out


@dataclass
class SegmentedTrainState:
    """Mutable training state for the segmented trainer."""

    step: int = 0
    loss_history: list[float] = field(default_factory=list)
    # Diagnostics proving the subsystem is actually engaged:
    carried_segment_count: int = 0  # segments fed a non-None states_in (state carried)
    detach_count: int = 0           # truncated-BPTT window detaches performed
    state_noise_count: int = 0      # windows whose initial state was noise-perturbed
    synthetic_task_count: int = 0   # steps that trained a synthetic cross-segment task


class SegmentedTrainer:
    """Truncated-BPTT trainer that carries cross-segment delta-memory state.

    Args:
        cfg: Full :class:`~graph_llm.config.Config`.  Reads ``cfg.model.segment_len``,
            ``cfg.model.bptt_window`` (K), ``cfg.model.stream_reset_interval``,
            ``cfg.model.state_noise_prob`` / ``state_noise_std``, and
            ``cfg.model.synthetic_task_fraction`` for the segmented schedule, plus the
            usual ``cfg.train`` optimiser / AMP / clip settings.
        model: A ``delta_memory_lm``-style model exposing
            ``forward(x, targets, states_in, return_states)`` (the piece-1 API).
        tokens: 1-D ``int64`` token array — the ordered stream to train on
            (e.g. text8 from :func:`~graph_llm.data.loader.load_text8_bytes`).
        device: Override device (defaults to CUDA when available, else CPU).
        val_tokens: Optional 1-D ``int64`` held-out token array (e.g. a disjoint
            slice from :func:`~graph_llm.data.loader.load_corpus_split`) used ONLY
            by the periodic unified eval report (``cfg.train.eval_every``, card
            69776c3e); ``None`` (default) skips the val-bpb component of that
            report and does not otherwise change training.
    """

    def __init__(
        self,
        cfg: Config,
        model: nn.Module,
        tokens: np.ndarray,
        device: torch.device | None = None,
        val_tokens: np.ndarray | None = None,
    ) -> None:
        self.cfg = cfg
        self.model = model

        setup_logging()
        seed_everything(cfg.train.seed)

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        self.optimizer = build_optimizer(self.model, cfg.train)
        self.scheduler = build_scheduler(self.optimizer, cfg.train)
        self.state = SegmentedTrainState()

        # Mixed precision (mirrors Trainer's torch 2.5 toolkit).
        mp = cfg.train.mixed_precision.lower()
        self._use_amp = mp in ("fp16", "bf16") and self.device.type == "cuda"
        self._amp_dtype = torch.bfloat16 if mp == "bf16" else torch.float16
        self._scaler: GradScaler | None = (
            GradScaler("cuda") if (self._use_amp and mp == "fp16") else None
        )

        # Segmented schedule.
        m = cfg.model
        self._segment_len = m.segment_len
        self._bptt_window = max(1, m.bptt_window)
        self._stream_reset_interval = m.stream_reset_interval
        self._state_noise_prob = m.state_noise_prob
        self._state_noise_std = m.state_noise_std
        self._synthetic_fraction = m.synthetic_task_fraction

        self._stream = OrderedSegmentStream(
            tokens,
            segment_len=self._segment_len,
            batch_size=cfg.data.batch_size,
            stream_reset_interval=self._stream_reset_interval,
        )
        # Synthetic-task sampler (used only when synthetic_task_fraction > 0).
        self._task_sampler = CrossSegmentTaskSampler(
            segment_tokens=self._segment_len,
            vocab_size=m.vocab_size,
            key_digits=m.synthetic_key_digits,
            seed=cfg.train.seed,
        )
        # Dedicated CPU RNG for the schedule decisions (synthetic draw, state noise)
        # so they are reproducible and independent of the model/data RNG stream.
        self._sched_rng = torch.Generator().manual_seed(cfg.train.seed + 1)
        self._noise_rng = torch.Generator(device=self.device).manual_seed(cfg.train.seed + 2)

        # Unified eval-report hook (card 69776c3e): off by default (eval_every=0).
        self._eval_every = cfg.train.eval_every
        self._eval_run_dir = cfg.train.eval_run_dir
        self._val_tokens = val_tokens
        self._val_loader: DataLoader | None = None
        if val_tokens is not None and self._eval_every > 0:
            val_ds = _TextChunkDataset(np.asarray(val_tokens), self._segment_len)
            self._val_loader = DataLoader(
                val_ds, batch_size=cfg.data.batch_size, shuffle=False, drop_last=False
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self) -> list[float]:
        """Run segmented training for ``cfg.train.max_steps`` optimiser steps.

        Each *step* is one truncated-BPTT window of ``K`` consecutive segments (LM
        stream) OR one synthetic cross-segment task, chosen per step with probability
        ``synthetic_task_fraction``.  Returns the per-step loss history.
        """
        self.model.train()
        max_steps = self.cfg.train.max_steps

        segment_iter = iter(self._stream)
        # ``carried`` is the live (graph-connected) state threaded between segments
        # WITHIN a window; it is detached at each window boundary.
        carried: list[DeltaMemoryState | Tensor] | None = None

        while self.state.step < max_steps:
            use_synthetic = (
                self._synthetic_fraction > 0.0
                and float(torch.rand((), generator=self._sched_rng)) < self._synthetic_fraction
            )

            if use_synthetic:
                loss_val = self._train_synthetic_step()
                self.state.synthetic_task_count += 1
                # A synthetic task is self-contained (its own fresh state); the
                # ordered LM stream's carried state is dropped so the next LM window
                # starts clean rather than mixing a synthetic state into the stream.
                carried = None
            else:
                loss_val, carried, segment_iter = self._train_lm_window(
                    segment_iter, carried
                )

            self.optimizer.zero_grad(set_to_none=True)
            self.state.step += 1
            self.state.loss_history.append(loss_val)

            if self.state.step % self.cfg.train.log_every == 0:
                lr = self.scheduler.get_last_lr()[0]
                log_metrics(self.state.step, loss=round(loss_val, 4), lr=round(lr, 6))

            if self._eval_every > 0 and self.state.step % self._eval_every == 0:
                self._run_periodic_eval()

        return self.state.loss_history

    # ------------------------------------------------------------------
    # Unified eval-report hook (card 69776c3e)
    # ------------------------------------------------------------------

    def _run_periodic_eval(self) -> None:
        """Build + write the unified eval report at the current step.

        Runs under ``build_eval_report``'s own no-perturbation guarantee (eval
        under ``no_grad`` + model-mode / RNG-state restore), so this never changes
        the trajectory of the training loop that calls it.
        """
        report = build_eval_report(
            self.model, self.cfg, self._val_loader, self.device, step=self.state.step
        )
        path = write_eval_report(report, self._eval_run_dir, step=self.state.step)
        _log.info("Wrote eval report: %s", path)

    # ------------------------------------------------------------------
    # LM ordered-stream window (truncated BPTT)
    # ------------------------------------------------------------------

    def _train_lm_window(
        self,
        segment_iter,  # noqa: ANN001 — Iterator[OrderedSegment]
        carried: list[DeltaMemoryState | Tensor] | None,
    ) -> tuple[float, list[DeltaMemoryState | Tensor] | None, object]:
        """Run one truncated-BPTT window of up to ``K`` consecutive LM segments.

        Accumulates loss over the window with the carried state graph-connected, runs
        a single ``backward()`` + optimiser step, then DETACHES the carried state for
        the next window.  A ``stream_reset`` segment drops the carried state before
        it (document/stream boundary).

        Returns:
            ``(mean_window_loss, detached_carried_state, segment_iter)`` — the
            iterator is returned because it is re-created on epoch exhaustion.
        """
        self.optimizer.zero_grad(set_to_none=True)

        # Optionally seed this window's initial state with the exposure-augmentation
        # noise (broadens the attainable-state distribution; arXiv 2507.02782).
        window_state = carried
        if (
            window_state is not None
            and self._state_noise_prob > 0.0
            and float(torch.rand((), generator=self._sched_rng)) < self._state_noise_prob
        ):
            window_state = perturb_states(
                window_state, self._state_noise_std, generator=self._noise_rng
            )
            self.state.state_noise_count += 1

        window_loss: Tensor | None = None
        n_scored = 0
        scaler = self._scaler

        for _ in range(self._bptt_window):
            try:
                seg = next(segment_iter)
            except StopIteration:
                segment_iter = iter(self._stream)
                seg = next(segment_iter)

            if seg.stream_reset:
                # Stream/document boundary: drop the carried state (no gradient and no
                # state crosses it).  A reset detaches the window so far too.
                window_state = None

            inputs = seg.inputs.to(self.device)
            targets = seg.targets.to(self.device)

            if window_state is not None:
                self.state.carried_segment_count += 1

            with self._autocast():
                loss, _logits, states_out = cast(
                    "tuple[Tensor, Tensor, list[DeltaMemoryState]]",
                    self.model(inputs, targets, window_state, True),
                )
            window_loss = loss if window_loss is None else window_loss + loss
            n_scored += 1
            # Thread the (still graph-connected) state into the next segment of THIS
            # window.
            window_state = states_out

        assert window_loss is not None  # n_scored >= 1 always
        mean_loss = window_loss / max(1, n_scored)

        if scaler is not None:
            scaler.scale(mean_loss).backward()
            scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.train.grad_clip)
            scaler.step(self.optimizer)
            scaler.update()
        else:
            mean_loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.train.grad_clip)
            self.optimizer.step()
        self.scheduler.step()

        # DETACH the carried state for the next window — the truncated-BPTT boundary.
        next_carried = detach_states(window_state)
        if next_carried is not None:
            self.state.detach_count += 1

        return float(mean_loss.detach()), next_carried, segment_iter

    # ------------------------------------------------------------------
    # Synthetic cross-segment retrieval task
    # ------------------------------------------------------------------

    def _train_synthetic_step(self) -> float:
        """Train one synthetic cross-segment retrieval task (answer-masked loss).

        Feeds the task's segments in order carrying the state across boundaries
        (single connected window — the task is short), and scores cross-entropy ONLY
        at the answer-token positions of the final segment, so the gradient rewards
        retrieving the early-segment key from the carried memory.
        """
        self.optimizer.zero_grad(set_to_none=True)
        task = self._task_sampler.sample()

        # The synthetic step never detaches/perturbs (a self-contained task), so its
        # carried state is always the model's own DeltaMemoryState list.
        states: list[DeltaMemoryState] | None = None
        total_loss: Tensor | None = None
        with self._autocast():
            for inp, tgt, mask in zip(
                task.segment_inputs,
                task.segment_targets,
                task.segment_masks,
                strict=True,
            ):
                inp = inp.to(self.device)
                tgt = tgt.to(self.device)
                mask = mask.to(self.device)
                _loss, logits, states = cast(
                    "tuple[Tensor, Tensor, list[DeltaMemoryState]]",
                    self.model(inp, None, states, True),
                )
                if states is not None:
                    self.state.carried_segment_count += 1
                seg_loss = masked_token_loss(logits, tgt, mask)
                total_loss = seg_loss if total_loss is None else total_loss + seg_loss

        assert total_loss is not None
        scaler = self._scaler
        if scaler is not None:
            scaler.scale(total_loss).backward()
            scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.train.grad_clip)
            scaler.step(self.optimizer)
            scaler.update()
        else:
            total_loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.train.grad_clip)
            self.optimizer.step()
        self.scheduler.step()
        return float(total_loss.detach())

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _autocast(self):
        if self._use_amp:
            return torch.autocast(device_type="cuda", dtype=self._amp_dtype)
        return torch.autocast(device_type="cpu", enabled=False)
