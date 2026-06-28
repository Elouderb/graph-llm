"""Synthetic cross-segment retrieval TRAINING tasks (card 61f900ca, piece 3).

Plain language-model perplexity does not, on its own, force a model to USE
information from far back in the stream — most next-token predictions are
answerable from local context, so the carried cross-segment memory can be safely
ignored and the loss barely moves.  These synthetic tasks remove that escape:
they place a key/value (a passkey) in an EARLY segment, fill the middle with
semantically-empty filler, and put the QUERY in a LATER segment whose own window
does NOT contain the answer.  The only way to predict the answer tokens is to
read the carried bounded memory.  Interleaving them into segmented training (at a
configurable fraction) supplies the gradient signal that teaches
``delta_memory_lm`` to use its persistent state.

This module reuses the eval-side cross-segment passkey constructor
(:func:`graph_llm.eval.long_context.make_cross_segment_passkey`) so the TRAINING
distribution matches the EVAL probe by construction — the same key-in-early /
query-in-late / answer-outside-window structure.  The only addition here is
turning each example into supervised ``(inputs, targets)`` segment pairs with an
answer-position MASK so the trainer scores ONLY the answer tokens (the retrieval
signal), not the filler.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from graph_llm.eval.long_context import (
    EncodeFn,
    _byte_encode,
    make_cross_segment_passkey,
)


@dataclass
class CrossSegmentTask:
    """A supervised cross-segment retrieval task split into ordered segments.

    The task is run by feeding the segments in order while carrying the per-layer
    memory state across boundaries (exactly as the eval harness does), then scoring
    the cross-entropy of the answer tokens in the final segment — but ONLY at the
    masked answer positions, so the gradient rewards retrieving the early-segment
    key, not modelling filler.

    Attributes:
        segment_inputs: One ``(1, T_i)`` input tensor per segment (ordered).
        segment_targets: One ``(1, T_i)`` next-token target tensor per segment,
            aligned with ``segment_inputs`` (target = input shifted by 1 within the
            assembled stream).
        segment_masks: One ``(1, T_i)`` bool mask per segment; ``True`` only at the
            positions whose target is an ANSWER token (all in the final segment).
            The trainer multiplies the per-position loss by this mask so only the
            retrieval is scored.
        answer: The ground-truth passkey string.
        query_segment_index: Index of the segment holding the query (the last).
        passkey_segment_index: Index of the segment holding the key (0).
        answer_outside_query_window: Always ``True`` by construction (key in an
            earlier segment than the query) — the property that makes the carried
            memory necessary.
    """

    segment_inputs: list[Tensor]
    segment_targets: list[Tensor]
    segment_masks: list[Tensor]
    answer: str
    query_segment_index: int
    passkey_segment_index: int
    answer_outside_query_window: bool


def make_cross_segment_task(
    passkey: str,
    n_segments: int,
    segment_tokens: int,
    vocab_size: int = 256,
    encode: EncodeFn = _byte_encode,
) -> CrossSegmentTask:
    """Build one supervised cross-segment retrieval task.

    Wraps :func:`~graph_llm.eval.long_context.make_cross_segment_passkey` (key in
    segment 0, filler in the middle, query in the last segment) and appends the
    ANSWER tokens to the final query segment, producing per-segment
    ``(inputs, targets, mask)`` where the mask selects only the answer-token target
    positions.

    The assembled stream is::

        [seg0: preamble + KEY + filler] [seg1..k-2: filler] [segk-1: QUESTION + ANSWER]

    and the supervised signal is the next-token prediction of the ANSWER digits at
    the tail of the final segment — answerable only from the carried memory, since
    the KEY is several segments back, outside the final segment's own window.

    Args:
        passkey: The secret to hide in segment 0 and require as the answer.
        n_segments: Number of segments (``>= 2``).
        segment_tokens: Approximate token length of each non-final segment.
        vocab_size: Token vocab (ids are taken mod ``vocab_size`` for byte mode).
        encode: String -> token-id encoder (default byte-level).

    Returns:
        A :class:`CrossSegmentTask`.

    Raises:
        ValueError: If ``n_segments < 2`` or ``segment_tokens < 1`` (propagated from
            :func:`make_cross_segment_passkey`).
    """
    base = make_cross_segment_passkey(
        passkey,
        n_segments=n_segments,
        segment_tokens=segment_tokens,
        vocab_size=vocab_size,
        encode=encode,
    )
    # Encode the answer string to ids (the digits the model must emit after the
    # query).  These become the supervised targets at the tail of the final segment.
    answer_ids = [t % vocab_size for t in encode(passkey)]
    if not answer_ids:
        raise ValueError("passkey encodes to zero tokens; cannot supervise an answer")

    segments = list(base.segment_ids)
    query = segments[-1]  # (Tq,) the QUESTION ids
    answer_tensor = torch.tensor(answer_ids, dtype=torch.long)
    # Final supervised segment = QUESTION followed by the ANSWER digits.  Predicting
    # token i+1 from i: the positions whose TARGET is an answer token are exactly the
    # last len(answer_ids) target positions of this segment.
    final_stream = torch.cat([query, answer_tensor], dim=0)  # (Tq + A,)

    segment_inputs: list[Tensor] = []
    segment_targets: list[Tensor] = []
    segment_masks: list[Tensor] = []

    # Leading segments (key + filler): standard next-token supervision is NOT scored
    # for the retrieval signal (mask all-False), but the inputs still build up the
    # carried memory.  Targets are the within-segment shift; with an all-False mask
    # they contribute nothing to the masked retrieval loss but keep the shapes
    # uniform for the trainer.
    for seg in segments[:-1]:
        inp = seg[:-1] if seg.numel() >= 2 else seg
        tgt = seg[1:] if seg.numel() >= 2 else seg
        n = int(inp.numel())
        segment_inputs.append(inp.unsqueeze(0))
        segment_targets.append(tgt.unsqueeze(0))
        segment_masks.append(torch.zeros(1, n, dtype=torch.bool))

    # Final segment: inputs = stream[:-1], targets = stream[1:], mask True only where
    # the target is an answer token (the last len(answer_ids) positions).
    f_inp = final_stream[:-1]
    f_tgt = final_stream[1:]
    n_final = int(f_inp.numel())
    mask = torch.zeros(n_final, dtype=torch.bool)
    n_answer = len(answer_ids)
    if n_answer > n_final:  # pragma: no cover - n_final >= Tq-1 + A >> A
        raise ValueError("answer longer than the final supervised segment")
    mask[n_final - n_answer :] = True
    segment_inputs.append(f_inp.unsqueeze(0))
    segment_targets.append(f_tgt.unsqueeze(0))
    segment_masks.append(mask.unsqueeze(0))

    return CrossSegmentTask(
        segment_inputs=segment_inputs,
        segment_targets=segment_targets,
        segment_masks=segment_masks,
        answer=passkey,
        query_segment_index=base.query_segment_index,
        passkey_segment_index=base.passkey_segment_index,
        answer_outside_query_window=base.query_segment_index
        != base.passkey_segment_index,
    )


class CrossSegmentTaskSampler:
    """Sample random cross-segment retrieval tasks for interleaving into training.

    Draws a fresh random passkey per task (so the model cannot memorise a fixed
    answer) and a (optionally random) number of segments / segment length, producing
    a :class:`CrossSegmentTask`.  The trainer pulls one of these with probability
    ``synthetic_task_fraction`` per step.

    Args:
        segment_tokens: Token length of each non-final segment (matches the LM
            stream's ``segment_len`` so the synthetic and real segments are the same
            size for the carried state).
        min_segments: Minimum number of segments per task (``>= 2``).
        max_segments: Maximum number of segments per task (``>= min_segments``).
        key_digits: Number of decimal digits in each random passkey.
        vocab_size: Token vocab.
        encode: String -> token-id encoder (default byte-level).
        seed: RNG seed for reproducibility.
    """

    def __init__(
        self,
        segment_tokens: int,
        min_segments: int = 2,
        max_segments: int = 4,
        key_digits: int = 5,
        vocab_size: int = 256,
        encode: EncodeFn = _byte_encode,
        seed: int = 0,
    ) -> None:
        if min_segments < 2:
            raise ValueError(f"min_segments must be >= 2, got {min_segments}")
        if max_segments < min_segments:
            raise ValueError(
                f"max_segments ({max_segments}) must be >= min_segments ({min_segments})"
            )
        if key_digits < 1:
            raise ValueError(f"key_digits must be >= 1, got {key_digits}")
        self._segment_tokens = segment_tokens
        self._min_segments = min_segments
        self._max_segments = max_segments
        self._key_digits = key_digits
        self._vocab_size = vocab_size
        self._encode = encode
        self._rng = np.random.default_rng(seed)

    def sample(self) -> CrossSegmentTask:
        """Draw one random :class:`CrossSegmentTask`."""
        low = 10 ** (self._key_digits - 1)
        high = 10**self._key_digits
        key = str(int(self._rng.integers(low, high)))
        n_segments = int(
            self._rng.integers(self._min_segments, self._max_segments + 1)
        )
        return make_cross_segment_task(
            key,
            n_segments=n_segments,
            segment_tokens=self._segment_tokens,
            vocab_size=self._vocab_size,
            encode=self._encode,
        )


def masked_token_loss(
    logits: Tensor,
    targets: Tensor,
    mask: Tensor,
) -> Tensor:
    """Mean cross-entropy over the ``mask``-selected target positions.

    Used to score ONLY the answer tokens of a cross-segment task (mask ``True`` at
    answer positions), so the gradient rewards retrieving the early-segment key and
    ignores the filler.  Returns a scalar tensor; if the mask selects no positions
    the loss is ``0`` (no contribution).

    Args:
        logits: ``(B, T, vocab)`` next-token logits.
        targets: ``(B, T)`` target ids.
        mask: ``(B, T)`` bool mask selecting scored positions.

    Returns:
        Scalar masked-mean cross-entropy.
    """
    vocab = logits.shape[-1]
    flat_logits = logits.reshape(-1, vocab)
    flat_targets = targets.reshape(-1)
    flat_mask = mask.reshape(-1)
    if not bool(flat_mask.any()):
        return logits.new_zeros(())
    per_tok = F.cross_entropy(flat_logits, flat_targets, reduction="none")
    selected = per_tok[flat_mask]
    return selected.mean()


__all__ = [
    "CrossSegmentTask",
    "CrossSegmentTaskSampler",
    "make_cross_segment_task",
    "masked_token_loss",
]
