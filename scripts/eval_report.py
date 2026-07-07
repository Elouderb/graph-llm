"""Unified eval-report CLI (card 69776c3e) -- run the eval harness on a checkpoint.

Standalone counterpart to periodic in-training calls (see
:class:`~graph_llm.train.segmented.SegmentedTrainer`): loads a config + a saved
checkpoint and writes ONE JSON report covering both axes -- val bpb AND the
specialty evals (cross-segment retrieval carry/reset + graded metric; in-model
reasoning-depth accuracy + routing health when the checkpoint's model has the
tandem pathway enabled).

Usage::

    python scripts/eval_report.py --config configs/smoke.yaml \
        --checkpoint checkpoints/ckpt_step000010_final.pt \
        --run-dir eval_reports/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

import graph_llm.models.baselines  # noqa: F401 — registers "transformer" and "mamba"
from graph_llm.config import load_config
from graph_llm.data import build_dataloaders
from graph_llm.eval.report import build_eval_report, write_eval_report
from graph_llm.models import build_model  # importing the package also registers "bilinear_lm"
from graph_llm.utils.logging import get_logger, setup_logging

_log = get_logger("scripts.eval_report")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Unified eval report (val bpb + specialty evals) on a graph_llm checkpoint"
    )
    p.add_argument("--config", required=True, help="Path to YAML config file")
    p.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    p.add_argument("--run-dir", default="eval_reports/", help="Directory to write the report JSON")
    p.add_argument("--split", default="val", choices=["train", "val"], help="Which split to eval")
    p.add_argument(
        "--reasoning-depths", type=int, nargs="+", default=None,
        help="Override reasoning-chain depths to probe (default: 4 6 16 32)",
    )
    p.add_argument(
        "--retrieval-n-segments", type=int, nargs="+", default=None,
        help="Override cross-segment retrieval distances in segments (default: 2 3 4)",
    )
    p.add_argument("--retrieval-repeats", type=int, default=8, help="Sampled tasks per distance")
    p.add_argument("--seed", type=int, default=0, help="Seed for the harness's own eval RNGs")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = build_dataloaders(
        cfg.data, vocab_size=cfg.model.vocab_size, seed=cfg.train.seed
    )
    loader = val_loader if args.split == "val" else train_loader

    model = build_model(cfg)
    # weights_only=False: checkpoint may embed non-tensor state (RNG, config).
    # Only load checkpoints you produced yourself (see trainer.py TODO).
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)

    overrides: dict = {"retrieval_repeats": args.retrieval_repeats, "seed": args.seed}
    if args.reasoning_depths is not None:
        overrides["reasoning_depths"] = args.reasoning_depths
    if args.retrieval_n_segments is not None:
        overrides["retrieval_n_segments"] = args.retrieval_n_segments

    report = build_eval_report(model, cfg, loader, device, step=ckpt.get("step"), **overrides)
    report["checkpoint"] = args.checkpoint
    report["config"] = args.config
    report["split"] = args.split

    path = write_eval_report(report, args.run_dir)
    _log.info("Wrote eval report: %s", path)
    _log.info("\n" + json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
