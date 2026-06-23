"""Evaluation metrics: perplexity and bits-per-byte (BPB).

Both metrics are computed from average negative log-likelihood (nats or bits)
over a held-out token sequence.  See ``scripts/eval.py`` for the CLI driver.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

if TYPE_CHECKING:
    pass


def _average_nll(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """Compute mean negative log-likelihood (nats) over *loader*.

    Args:
        model: Model with ``forward(x, targets) -> (loss, logits)``.
        loader: DataLoader yielding ``(x, targets)`` pairs.
        device: Target device.

    Returns:
        Mean NLL in nats (natural log base), weighted by token count so a
        short final batch does not bias the result (standard for ppl/BPB).
    """
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    with torch.no_grad():
        for x, targets in loader:
            x = x.to(device)
            targets = targets.to(device)
            loss, _ = model(x, targets)
            # ``loss`` is the mean NLL over this batch's tokens; reweight by the
            # token count to recover a token-weighted (not batch-weighted) mean.
            n_tokens = int(targets.numel())
            total_nll += loss.item() * n_tokens
            total_tokens += n_tokens
    return total_nll / max(total_tokens, 1)


def perplexity(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device | None = None,
) -> float:
    """Token-level perplexity: ``exp(mean NLL)``.

    Args:
        model: Registered model (forward returns ``(loss, logits)``).
        loader: DataLoader yielding ``(input_ids, target_ids)`` pairs.
        device: Inference device.  Defaults to CPU.

    Returns:
        Scalar perplexity (lower is better).
    """
    if device is None:
        device = torch.device("cpu")
    nll = _average_nll(model, loader, device)
    return math.exp(nll)


def bits_per_byte(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device | None = None,
    bytes_per_token: float = 1.0,
) -> float:
    """Bits-per-byte (BPB): ``mean_NLL / log(2) / bytes_per_token``.

    For byte-level models *bytes_per_token* = 1.0.  For subword models pass
    the average bytes-per-token ratio of the tokenizer (computed from the
    corpus; typically 4–5 for English BPE).

    Args:
        model: Registered model.
        loader: DataLoader.
        device: Inference device.
        bytes_per_token: Ratio of bytes to tokens in the corpus.

    Returns:
        Scalar BPB (lower is better; < 1.0 is state-of-the-art territory).
    """
    if device is None:
        device = torch.device("cpu")
    nll = _average_nll(model, loader, device)
    return nll / (math.log(2) * bytes_per_token)
