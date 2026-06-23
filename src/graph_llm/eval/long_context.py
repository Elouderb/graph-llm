"""Long-context evaluation harness (card 424e3a8e).

Two probes that expose *context-length* behaviour — the thing perplexity alone
hides, and the thing the project's headline claim ("context windows obsolete")
must be measured against, not asserted:

1. **Passkey / needle-in-a-haystack retrieval.**  A short secret (the "passkey")
   is buried inside a long block of filler text; the model is then asked to
   recall it.  Scoring is exact-match on the recalled passkey.  This is the
   canonical long-range *retrieval* test (Mohtashami & Jaggi, 2023).

2. **Per-token-position loss curve.**  Mean next-token loss as a function of
   position in the sequence.  A model that genuinely uses long context keeps
   loss flat or decreasing with position; one that has effectively forgotten
   earlier tokens shows loss rising past its usable window.

Both probes are built to run on sequences **longer than the training window**
on purpose — that is what tests extrapolation / memory rather than mere
in-window fit.  Everything here is byte-level by default (vocab=256) so it is
tokenizer-independent and works with the Phase 0 byte models; an ``encode``
callable can be injected to use a real tokenizer later (card e1644700) without
changing the probe logic.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TypedDict

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

# Default byte-level encode: UTF-8 bytes.  Swap for a tokenizer's ``encode`` to
# move off byte-level (the probe logic is identical either way).
EncodeFn = Callable[[str], Sequence[int]]


def _byte_encode(text: str) -> list[int]:
    return list(text.encode("utf-8", errors="replace"))


# ---------------------------------------------------------------------------
# Passkey / needle-in-a-haystack retrieval probe
# ---------------------------------------------------------------------------

# Filler is repeated to pad the context.  The garden-path sentence is the
# standard passkey-probe filler — semantically empty, so the only retrievable
# information in the whole context is the passkey itself.
_FILLER = (
    "The grass is green. The sky is blue. The sun is yellow. "
    "Here we go. There and back again. "
)
_PREAMBLE = "There is an important piece of information hidden inside a lot of irrelevant text. "
_PASSKEY_TEMPLATE = "The pass key is {key}. Remember it. {key} is the pass key. "
_QUESTION = "What is the pass key? The pass key is "


@dataclass
class PasskeyExample:
    """A constructed passkey prompt and its ground-truth answer.

    Attributes:
        prompt: The full context+question string the model is conditioned on.
        answer: The ground-truth passkey string the model must reproduce.
        prompt_len_tokens: Length of the encoded prompt (so callers can assert
            it exceeds the training window).
        depth_fraction: Where the passkey was inserted (0.0 = start, 1.0 = end).
    """

    prompt: str
    answer: str
    prompt_len_tokens: int
    depth_fraction: float


def make_passkey_example(
    passkey: str,
    context_tokens: int,
    depth_fraction: float = 0.5,
    encode: EncodeFn = _byte_encode,
) -> PasskeyExample:
    """Construct a passkey prompt with the key buried at *depth_fraction*.

    The filler is repeated until the assembled context reaches roughly
    *context_tokens* encoded tokens, then the passkey sentence is spliced in at
    the requested depth.  Set *context_tokens* greater than the model's training
    window to probe extrapolation.

    Args:
        passkey: The secret string to hide and later score against.
        context_tokens: Target encoded length of the filler context.
        depth_fraction: Insertion depth in ``[0, 1]`` (0 = start, 1 = end).
        encode: String -> token-id encoder (default byte-level).

    Returns:
        A :class:`PasskeyExample`.

    Raises:
        ValueError: If *depth_fraction* is outside ``[0, 1]``.
    """
    if not 0.0 <= depth_fraction <= 1.0:
        raise ValueError(f"depth_fraction must be in [0, 1], got {depth_fraction}")

    passkey_sentence = _PASSKEY_TEMPLATE.format(key=passkey)
    # How many filler tokens we need on either side of the passkey.
    filler_unit_len = max(1, len(encode(_FILLER)))
    n_filler_units = max(1, context_tokens // filler_unit_len)
    n_before = int(round(n_filler_units * depth_fraction))
    n_after = n_filler_units - n_before

    context = (
        _PREAMBLE
        + _FILLER * n_before
        + passkey_sentence
        + _FILLER * n_after
    )
    prompt = context + _QUESTION
    return PasskeyExample(
        prompt=prompt,
        answer=passkey,
        prompt_len_tokens=len(encode(prompt)),
        depth_fraction=depth_fraction,
    )


def score_passkey_retrieval(generated: str, answer: str) -> bool:
    """Exact-match scoring: does the model's continuation contain the passkey?

    The model is expected to continue the prompt with the passkey digits; we
    accept the answer if the (stripped) ground-truth key appears at the start of
    the generated continuation, tolerating leading whitespace and trailing
    punctuation/period the model may emit.

    Args:
        generated: The model's generated continuation of the passkey prompt.
        answer: The ground-truth passkey.

    Returns:
        ``True`` iff the passkey was retrieved exactly.
    """
    gen = generated.strip()
    ans = answer.strip()
    # Require the key at the *start* of the continuation (the natural answer).
    # Containment anywhere is deliberately NOT accepted: a model that merely
    # echoes filler happening to contain the key must not score as a retrieval,
    # or the probe would flatter the very long-range claim it exists to falsify.
    return gen.startswith(ans)


@torch.no_grad()
def greedy_generate(
    model: nn.Module,
    prompt_ids: Tensor,
    max_new_tokens: int,
    device: torch.device | None = None,
) -> list[int]:
    """Greedy-decode *max_new_tokens* continuation ids from *prompt_ids*.

    Model-agnostic: relies only on the ``forward(x) -> (loss, logits)`` contract.
    Used by the passkey probe to read off the retrieved key.

    Args:
        model: Registered model.
        prompt_ids: ``(T,)`` or ``(1, T)`` prompt token ids.
        max_new_tokens: Number of tokens to greedily generate.
        device: Inference device (defaults to CPU).

    Returns:
        The list of generated token ids (length ``max_new_tokens``).
    """
    if device is None:
        device = torch.device("cpu")
    model.eval()
    if prompt_ids.dim() == 1:
        prompt_ids = prompt_ids.unsqueeze(0)
    ids = prompt_ids.to(device)
    generated: list[int] = []
    for _ in range(max_new_tokens):
        _, logits = model(ids)
        next_id = int(logits[0, -1].argmax().item())
        generated.append(next_id)
        ids = torch.cat([ids, torch.tensor([[next_id]], device=device)], dim=1)
    return generated


class PasskeyDepthResult(TypedDict):
    """Per-depth outcome of the passkey probe."""

    depth: float
    prompt_len_tokens: int
    retrieved: bool
    generated: str


class PasskeyProbeResult(TypedDict):
    """Aggregate passkey-probe result: accuracy plus one entry per depth."""

    accuracy: float
    per_depth: list[PasskeyDepthResult]


def run_passkey_probe(
    model: nn.Module,
    context_tokens: int,
    depths: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
    passkeys: Sequence[str] | None = None,
    vocab_size: int = 256,
    encode: EncodeFn = _byte_encode,
    decode: Callable[[Sequence[int]], str] | None = None,
    device: torch.device | None = None,
) -> PasskeyProbeResult:
    """Run the passkey probe across several insertion depths.

    Constructs one passkey example per depth, greedily decodes the model's
    answer, and scores exact-match retrieval.  Designed for ``context_tokens``
    beyond the training window.

    Args:
        model: Registered model (byte-level by default).
        context_tokens: Filler length (set > training window for extrapolation).
        depths: Passkey insertion depths to sweep.
        passkeys: Keys to hide (default: a distinct 5-digit key per depth).
        vocab_size: Token vocab (passkey ids must be < vocab_size for byte mode).
        encode: String encoder.
        decode: Token-id -> string decoder (default: byte decode).
        device: Inference device.

    Returns:
        ``{"accuracy": float, "per_depth": [ {depth, prompt_len, retrieved,
        generated} ... ]}``.
    """
    if device is None:
        device = torch.device("cpu")

    def _default_byte_decode(ids: Sequence[int]) -> str:
        return bytes(int(i) % 256 for i in ids).decode("utf-8", errors="replace")

    decode_fn: Callable[[Sequence[int]], str] = decode or _default_byte_decode
    if passkeys is None:
        rng = np.random.default_rng(0)
        passkeys = [str(int(rng.integers(10_000, 99_999))) for _ in depths]
    elif len(passkeys) != len(depths):
        raise ValueError(
            f"passkeys length ({len(passkeys)}) must match depths length "
            f"({len(depths)}); otherwise the probe would silently cover only some depths."
        )

    per_depth: list[PasskeyDepthResult] = []
    n_correct = 0
    for depth, key in zip(depths, passkeys, strict=True):
        ex = make_passkey_example(key, context_tokens, depth, encode=encode)
        prompt_ids = torch.tensor(
            [t % vocab_size for t in encode(ex.prompt)], dtype=torch.long
        )
        gen_ids = greedy_generate(model, prompt_ids, max_new_tokens=len(key) + 2, device=device)
        generated = decode_fn(gen_ids)
        retrieved = score_passkey_retrieval(generated, ex.answer)
        n_correct += int(retrieved)
        per_depth.append(
            {
                "depth": depth,
                "prompt_len_tokens": ex.prompt_len_tokens,
                "retrieved": retrieved,
                "generated": generated,
            }
        )
    accuracy = n_correct / max(1, len(per_depth))
    return {"accuracy": accuracy, "per_depth": per_depth}


# ---------------------------------------------------------------------------
# Per-token-position loss curve
# ---------------------------------------------------------------------------


@torch.no_grad()
def position_loss_curve(
    model: nn.Module,
    sequences: Tensor,
    device: torch.device | None = None,
    chunk_batch: int = 8,
) -> np.ndarray:
    """Mean next-token loss at each position across a batch of sequences.

    For each position ``t`` in ``[0, T-2]`` we average the cross-entropy of
    predicting token ``t+1`` from the prefix, over all sequences.  A model that
    uses long context keeps this curve flat/decreasing; one that forgets shows
    it rising past its usable window.  Pass sequences *longer than the training
    window* to expose extrapolation behaviour.

    Args:
        model: Registered model (``forward(x) -> (loss, logits)``).
        sequences: ``(N, T)`` token ids, ``T`` may exceed the training window.
        device: Inference device.
        chunk_batch: Sequences processed per forward pass (memory control).

    Returns:
        ``np.ndarray`` of shape ``(T - 1,)`` — mean loss at each target position.
    """
    if device is None:
        device = torch.device("cpu")
    model.eval()
    if sequences.dim() != 2:
        raise ValueError(f"sequences must be 2-D (N, T), got shape {tuple(sequences.shape)}")
    N, T = sequences.shape
    if T < 2:
        raise ValueError(f"sequences must have length >= 2, got T={T}")

    loss_sum = torch.zeros(T - 1, dtype=torch.float64)
    count = 0
    for start in range(0, N, chunk_batch):
        batch = sequences[start : start + chunk_batch].to(device)
        inp = batch[:, :-1]
        tgt = batch[:, 1:]
        _, logits = model(inp)  # (b, T-1, vocab)
        # Per-position cross-entropy, no reduction over positions.
        vocab = logits.shape[-1]
        per_tok = nn.functional.cross_entropy(
            logits.reshape(-1, vocab),
            tgt.reshape(-1),
            reduction="none",
        ).view(tgt.shape)  # (b, T-1)
        loss_sum += per_tok.sum(dim=0).double().cpu()
        count += per_tok.shape[0]
    return (loss_sum / max(1, count)).numpy()
