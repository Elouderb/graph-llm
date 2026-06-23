"""Eval CLI — run the full Phase 0 eval suite on a checkpoint (card 424e3a8e).

Computes, on a trained checkpoint:
  * bits-per-byte (BPB, byte-level — tokenizer-independent) and perplexity,
  * a per-token-position loss curve (loss vs. position, including positions
    beyond the training window), and
  * a passkey / needle-in-a-haystack retrieval probe at several depths.

Results are emitted as JSON (``--json-out``) and pretty-printed to the log.

Usage::

    python scripts/eval.py --config configs/smoke.yaml \
        --checkpoint checkpoints/ckpt_step000010_final.pt \
        --json-out results/eval.json
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
from graph_llm.eval import bits_per_byte, perplexity
from graph_llm.eval.long_context import position_loss_curve, run_passkey_probe
from graph_llm.models import build_model  # importing the package also registers "bilinear_lm"
from graph_llm.models.baselines import count_params
from graph_llm.utils.logging import get_logger, setup_logging

_log = get_logger("scripts.eval")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a graph_llm checkpoint (full suite)")
    p.add_argument("--config", required=True, help="Path to YAML config file")
    p.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    p.add_argument("--split", default="val", choices=["train", "val"], help="Which split to eval")
    p.add_argument("--json-out", default=None, help="Optional path to write results JSON")
    p.add_argument(
        "--long-context-len",
        type=int,
        default=None,
        help="Sequence length for the position-loss / passkey probes "
        "(default: 2x training window, to test extrapolation).",
    )
    p.add_argument(
        "--n-long-seqs",
        type=int,
        default=8,
        help="Number of long sequences for the position-loss curve.",
    )
    return p.parse_args()


def _build_position_loss(model, vocab_size, seq_len, n_seqs, device):
    """Construct random long sequences and compute the position-loss curve."""
    rng = torch.Generator().manual_seed(0)
    seqs = torch.randint(0, vocab_size, (n_seqs, seq_len), generator=rng)
    curve = position_loss_curve(model, seqs, device=device)
    return curve


def main() -> None:
    args = parse_args()
    setup_logging()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = build_dataloaders(
        cfg.data,
        vocab_size=cfg.model.vocab_size,
        seed=cfg.train.seed,
    )
    loader = val_loader if args.split == "val" else train_loader

    model = build_model(cfg)
    # weights_only=False: checkpoint may contain the Config dataclass object.
    # Only load checkpoints you produced yourself (see trainer.py TODO).
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)

    train_window = cfg.model.max_seq_len
    long_len = args.long_context_len or (2 * train_window)

    # --- Core LM metrics (byte-level BPB is the headline cross-model number) ---
    ppl = perplexity(model, loader, device)
    bpb = bits_per_byte(model, loader, device)

    # --- Per-token-position loss curve (probes context-length behaviour) ---
    pos_curve = _build_position_loss(
        model, cfg.model.vocab_size, long_len, args.n_long_seqs, device
    )

    # --- Passkey / needle retrieval (probes long-range retrieval) ---
    passkey = run_passkey_probe(
        model,
        context_tokens=long_len,
        vocab_size=cfg.model.vocab_size,
        device=device,
    )

    results = {
        "checkpoint": args.checkpoint,
        "config": args.config,
        "model_name": cfg.model.name,
        "params": count_params(model),
        "train_window": train_window,
        "long_context_len": long_len,
        "split": args.split,
        "metrics": {
            "perplexity": ppl,
            "bits_per_byte": bpb,
        },
        "position_loss": {
            "positions": list(range(len(pos_curve))),
            "mean_loss": [float(v) for v in pos_curve],
            "loss_in_window_mean": float(pos_curve[:train_window].mean()),
            "loss_out_of_window_mean": (
                float(pos_curve[train_window:].mean())
                if len(pos_curve) > train_window
                else None
            ),
        },
        "passkey": {
            "accuracy": passkey["accuracy"],
            "per_depth": passkey["per_depth"],
        },
    }

    _pretty_print(results)

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2))
        _log.info("Wrote results JSON: %s", out_path)


def _pretty_print(results: dict) -> None:
    """Pretty-print the results table to the log."""
    m = results["metrics"]
    pl = results["position_loss"]
    pk = results["passkey"]
    lines = [
        "",
        "=" * 60,
        f"  EVAL SUITE — {results['model_name']}  ({results['params']:,} params)",
        "=" * 60,
        f"  checkpoint        : {results['checkpoint']}",
        f"  split             : {results['split']}",
        f"  train window      : {results['train_window']}",
        f"  long-context len  : {results['long_context_len']}",
        "-" * 60,
        f"  perplexity        : {m['perplexity']:.4f}",
        f"  bits-per-byte     : {m['bits_per_byte']:.4f}",
        "-" * 60,
        f"  pos-loss in-window  (mean) : {pl['loss_in_window_mean']:.4f}",
        (
            f"  pos-loss out-window (mean) : {pl['loss_out_of_window_mean']:.4f}"
            if pl["loss_out_of_window_mean"] is not None
            else "  pos-loss out-window (mean) : n/a (long_len <= window)"
        ),
        "-" * 60,
        f"  passkey accuracy  : {pk['accuracy']:.2%}  ({len(pk['per_depth'])} depths)",
        "=" * 60,
        "",
    ]
    _log.info("\n".join(lines))


if __name__ == "__main__":
    main()
