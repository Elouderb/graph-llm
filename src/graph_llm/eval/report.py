"""Unified eval harness (card 69776c3e): val bpb + specialty evals, one JSON report.

Perplexity/bpb alone under-measures this architecture BY DESIGN: the delta-memory
carries cross-segment state and the (optional) tandem reasoner does multi-hop
lookups, neither of which moves ordinary next-byte perplexity much even when
broken (most bytes are predictable from local context alone).  This module
combines the specialty evals built across earlier cards into ONE entry point,
callable periodically during training (see
:class:`~graph_llm.train.segmented.SegmentedTrainer`) and standalone on a saved
checkpoint (``scripts/eval_report.py``), always reporting BOTH axes:

* ``val_bpb`` -- bits-per-byte on a held-out split (:func:`~graph_llm.eval.metrics.bits_per_byte`).
* ``cross_segment_retrieval`` -- carry vs reset answer-token NLL by retrieval
  distance, the graded metric from card 61f900ca piece 4
  (:func:`~graph_llm.eval.long_context.cross_segment_retrieval_nll_by_distance`).
  Any retrieval-style probe here predicts the answer at ``answer_pos - 1`` (the
  answer-copy leak guard); this is inherited unchanged from the reused metric.
* ``reasoning_depth_accuracy`` / ``routing_health`` -- in-model reasoning-chain
  accuracy at several depths + per-type gate routing fractions, ONLY when the
  model's tandem pathway is enabled (``cfg.model.tandem_enabled``).  Reuses
  ``train.tandem``'s own eval helpers (``_eval_M`` / ``_eval_R_depth``) rather than
  reimplementing the reasoning-example construction + answer-position bookkeeping.

Eval must never perturb training: everything below runs under an explicit
``model.eval()`` + ``torch.no_grad()``, and the model's training-mode flag plus the
torch (+ CUDA) global RNG state are snapshotted before and restored after building
the report -- so a periodic call mid-training cannot change the RNG trajectory of
the run that calls it (acceptance criterion 4, card 69776c3e).
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

import torch
import torch.nn as nn

from graph_llm.eval.long_context import cross_segment_retrieval_nll_by_distance
from graph_llm.eval.metrics import bits_per_byte

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from graph_llm.config import Config

_DEFAULT_REASONING_DEPTHS: tuple[int, ...] = (4, 6, 16, 32)
_DEFAULT_RETRIEVAL_N_SEGMENTS: tuple[int, ...] = (2, 3, 4)


def _reasoning_and_routing(
    model: nn.Module,
    cfg: Config,
    *,
    depths: Sequence[int],
    eval_batch: int,
    eval_batches: int,
    seed: int,
    device: torch.device,
) -> tuple[dict[str, float] | None, dict[str, Any] | None]:
    """In-model reasoning-depth accuracy + per-type gate routing (tandem-only).

    Returns ``(None, None)`` when the model's tandem pathway is off -- these
    metrics have no meaning without a reasoner + gate to evaluate.  A FRESH
    ``random.Random(seed)`` drives the synthetic reasoning examples so this never
    touches the caller's own training RNG stream.
    """
    if not getattr(cfg.model, "tandem_enabled", False):
        return None, None

    # Deferred import: avoids a module-level dependency from eval/ (imported early,
    # e.g. by scripts/eval.py) onto the heavier train/tandem.py (which pulls in
    # graph_llm.models + the data pipelines) for callers that never need this path.
    from graph_llm.train.tandem import TandemConfig, _eval_M, _eval_R_depth  # noqa: PLC0415

    m = cfg.model
    tcfg = TandemConfig(
        seg_len=m.reasoning_segment_len,
        n_segments=2,
        n_chains=2,
        k_train=m.causal_reasoner_steps,
        d_model=m.d_model,
        eval_batch=eval_batch,
        eval_batches=eval_batches,
        mlp_enabled=bool(getattr(m, "tandem_mlp_enabled", False)),
    )
    rng = random.Random(seed)

    acc_m, gate_m = _eval_M(model, tcfg, rng, device)
    reasoning: dict[str, float] = {"M": acc_m}
    routing: dict[str, Any] = {
        "M": gate_m.tolist() if hasattr(gate_m, "tolist") else gate_m
    }
    for depth in depths:
        acc_d, gate_d = _eval_R_depth(model, tcfg, depth, rng, device)
        reasoning[f"D{depth}"] = acc_d
        routing[f"D{depth}"] = gate_d.tolist() if hasattr(gate_d, "tolist") else gate_d
    return reasoning, routing


def build_eval_report(
    model: nn.Module,
    cfg: Config,
    val_loader: DataLoader | None,
    device: torch.device,
    *,
    step: int | None = None,
    reasoning_depths: Sequence[int] = _DEFAULT_REASONING_DEPTHS,
    retrieval_n_segments: Sequence[int] = _DEFAULT_RETRIEVAL_N_SEGMENTS,
    retrieval_repeats: int = 8,
    reasoning_eval_batch: int = 32,
    reasoning_eval_batches: int = 2,
    seed: int = 0,
) -> dict[str, Any]:
    """Build the unified eval report: val bpb + specialty evals, one JSON-able dict.

    Args:
        model: A ``delta_memory_lm``-family model (the piece-1 state-carry API is
            required for the cross-segment retrieval metric; the tandem-only
            metrics additionally require ``cfg.model.tandem_enabled``).
        cfg: The model's :class:`~graph_llm.config.Config` (read for
            ``cfg.model.segment_len`` / ``vocab_size`` / ``synthetic_key_digits`` /
            the tandem knobs).
        val_loader: Held-out :class:`~torch.utils.data.DataLoader` for val bpb, or
            ``None`` to skip that metric (``"val_bpb"`` is then ``None`` in the
            report -- e.g. when no val split is available).
        device: Inference device.
        step: Optional training step to stamp on the report (``None`` for a
            standalone/checkpoint-only report).
        reasoning_depths: Chain depths to probe for in-model reasoning accuracy.
        retrieval_n_segments: Retrieval distances (in segments) to probe.
        retrieval_repeats: Sampled tasks per retrieval distance.
        reasoning_eval_batch: Examples per reasoning-accuracy eval batch.
        reasoning_eval_batches: Number of eval batches averaged per depth.
        seed: Seed for the harness's OWN (fresh) RNGs -- never the caller's
            training RNG (see the module docstring's no-perturbation guarantee).

    Returns:
        A JSON-serialisable dict: ``step``, ``tandem_enabled``, ``val_bpb``,
        ``cross_segment_retrieval`` (``{n_segments: {"nll_carry", "nll_reset"}}``),
        ``reasoning_depth_accuracy`` (``None`` unless tandem-enabled), and
        ``routing_health`` (``None`` unless tandem-enabled).
    """
    was_training = model.training
    torch_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

    try:
        model.eval()
        with torch.no_grad():
            val_bpb = (
                bits_per_byte(model, val_loader, device) if val_loader is not None else None
            )
            retrieval = cross_segment_retrieval_nll_by_distance(
                model,
                n_segments_list=list(retrieval_n_segments),
                repeats=retrieval_repeats,
                segment_tokens=cfg.model.segment_len,
                vocab_size=cfg.model.vocab_size,
                key_digits=cfg.model.synthetic_key_digits,
                seed=seed,
                device=device,
            )
            reasoning, routing = _reasoning_and_routing(
                model,
                cfg,
                depths=reasoning_depths,
                eval_batch=reasoning_eval_batch,
                eval_batches=reasoning_eval_batches,
                seed=seed,
                device=device,
            )
    finally:
        torch.set_rng_state(torch_rng)
        if cuda_rng is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(cuda_rng)
        model.train(was_training)

    return {
        "step": step,
        "tandem_enabled": bool(getattr(cfg.model, "tandem_enabled", False)),
        "val_bpb": val_bpb,
        "cross_segment_retrieval": {str(k): v for k, v in retrieval.items()},
        "reasoning_depth_accuracy": reasoning,
        "routing_health": routing,
    }


def write_eval_report(
    report: dict[str, Any],
    run_dir: str | Path,
    step: int | None = None,
) -> Path:
    """Write an eval report to ``run_dir`` and return the written path.

    Args:
        report: A dict from :func:`build_eval_report` (or any JSON-able dict).
        run_dir: Directory to write into (created if missing).
        step: When given, names the file ``eval_step{step:06d}.json`` (periodic
            call); ``None`` writes ``eval_report.json`` (standalone CLI call).
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    name = f"eval_step{step:06d}.json" if step is not None else "eval_report.json"
    path = run_dir / name
    path.write_text(json.dumps(report, indent=2, default=str))
    return path


__all__ = ["build_eval_report", "write_eval_report"]
