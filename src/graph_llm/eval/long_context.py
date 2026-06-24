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

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TypedDict, cast

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


# ---------------------------------------------------------------------------
# Cross-segment persistent-memory harness (card 61f900ca)
# ---------------------------------------------------------------------------
#
# These probes drive ``delta_memory_lm`` over ORDERED consecutive segments,
# carrying the per-layer delta-memory state across segment boundaries via the
# piece-1 API ``forward(x, targets, states_in, return_states)``.  They are the
# eval side of the "memory replaces the context window" thesis: with the state
# CARRIED, information from an EARLIER segment is retrievable in a LATER segment
# even though it lies outside that later segment's own window; with the state
# RESET per segment, it cannot be.  Carry is defined over the delta-memory layers
# only and assumes ``front_end="none"`` (the committed backbone) — see
# ``DeltaMemoryLM.forward``.


@dataclass
class SegmentRun:
    """Outcome of running a model over an ordered list of segments.

    Attributes:
        carried: Whether the per-layer state was carried across boundaries
            (``True``) or reset per segment (``False``).
        per_segment_logits: One ``(B, T_i, vocab)`` logits tensor per segment.
        final_states: The per-block final states after the last segment when
            ``carried`` (each ``(B, H, d_k, d_v)``), else ``None``.
    """

    carried: bool
    per_segment_logits: list[Tensor]
    final_states: list[Tensor] | None


def _supports_state_carry(model: nn.Module) -> bool:
    """Heuristic: does ``model.forward`` accept ``states_in`` / ``return_states``?

    ``delta_memory_lm`` does (card 61f900ca); the baselines do not.  The harness
    falls back to plain per-segment forwards for models without the API, so the
    reset-mode comparison still runs for any model.
    """
    import inspect

    try:
        params = inspect.signature(model.forward).parameters
    except (TypeError, ValueError):  # pragma: no cover - exotic forwards
        return False
    return "states_in" in params and "return_states" in params


@torch.no_grad()
def run_segments(
    model: nn.Module,
    segments: Sequence[Tensor],
    carry: bool = True,
    device: torch.device | None = None,
) -> SegmentRun:
    """Run *model* over ordered *segments*, optionally carrying per-layer state.

    Each segment is a ``(B, T_i)`` (or ``(T_i,)``) block of token ids; the
    segments are consecutive pieces of one stream.  When ``carry`` is ``True``
    and the model exposes the state-carry API (``delta_memory_lm``), the per-block
    final delta-memory state from each segment seeds the next — so a later
    segment sees the whole prior stream through the bounded memory, not just its
    own ``T_i`` tokens.  When ``carry`` is ``False`` (or the model lacks the API)
    each segment is run independently (state reset) — the control condition.

    Args:
        model: Registered model.  ``delta_memory_lm`` carries state; others are
            run per segment (reset) regardless of ``carry``.
        segments: Ordered consecutive token-id blocks (``(B, T_i)`` or ``(T_i,)``).
        carry: Carry the per-layer state across boundaries (default ``True``).
        device: Inference device (defaults to CPU).

    Returns:
        A :class:`SegmentRun` with one logits tensor per segment and the final
        per-block states (when carried).
    """
    if device is None:
        device = torch.device("cpu")
    model.eval()
    if not segments:
        raise ValueError("segments must be non-empty")

    use_carry = carry and _supports_state_carry(model)
    states: list[Tensor] | None = None
    per_segment_logits: list[Tensor] = []
    for seg in segments:
        ids = seg.to(device)
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        if use_carry:
            _, logits, states = cast(
                "tuple[Tensor, Tensor, list[Tensor]]",
                model(ids, None, states, True),
            )
        else:
            _, logits = model(ids)
        per_segment_logits.append(logits)

    return SegmentRun(
        carried=use_carry,
        per_segment_logits=per_segment_logits,
        final_states=states if use_carry else None,
    )


@dataclass
class CrossSegmentPasskey:
    """A multi-segment passkey example whose answer is OUTSIDE the query segment.

    The passkey lives in segment 0; segments ``1..k-2`` are semantically-empty
    filler; the query ("What is the pass key? ...") is in the final segment ``k-1``.
    A model that processes the segments INDEPENDENTLY (reset state) cannot answer
    — the key is not in the final segment's own window.  Only a CARRIED bounded
    memory can retrieve it, which is exactly the property under test.

    Attributes:
        segment_ids: One ``(T_i,)`` LongTensor of token ids per segment.
        answer: The ground-truth passkey string.
        query_segment_index: Index of the segment holding the query (the last).
        passkey_segment_index: Index of the segment holding the key (0).
        total_tokens: Total token count across all segments.
    """

    segment_ids: list[Tensor]
    answer: str
    query_segment_index: int
    passkey_segment_index: int
    total_tokens: int


def make_cross_segment_passkey(
    passkey: str,
    n_segments: int,
    segment_tokens: int,
    vocab_size: int = 256,
    encode: EncodeFn = _byte_encode,
) -> CrossSegmentPasskey:
    """Build a multi-segment passkey example (key in seg 0, query in the last seg).

    Segment 0 holds the preamble + passkey sentence padded to ``segment_tokens``;
    segments ``1..n-2`` are pure filler of ``segment_tokens`` each; the final
    segment holds the question.  Each segment is independently tokenised so the
    pieces tile a single stream when run in order.

    Args:
        passkey: The secret to hide in segment 0 and score against.
        n_segments: Number of segments (``>= 2``: at least key-segment + query).
        segment_tokens: Approximate token length of each non-final segment.
        vocab_size: Token vocab (ids are taken mod ``vocab_size`` for byte mode).
        encode: String -> token-id encoder (default byte-level).

    Returns:
        A :class:`CrossSegmentPasskey`.

    Raises:
        ValueError: If ``n_segments < 2`` or ``segment_tokens < 1``.
    """
    if n_segments < 2:
        raise ValueError(f"n_segments must be >= 2, got {n_segments}")
    if segment_tokens < 1:
        raise ValueError(f"segment_tokens must be >= 1, got {segment_tokens}")

    def _to_ids(text: str) -> Tensor:
        return torch.tensor([t % vocab_size for t in encode(text)], dtype=torch.long)

    def _pad_text_to(base: str, target: int) -> str:
        """Repeat filler after *base* until it reaches ~*target* encoded tokens."""
        text = base
        while len(encode(text)) < target:
            text += _FILLER
        return text

    passkey_sentence = _PASSKEY_TEMPLATE.format(key=passkey)
    seg0_text = _pad_text_to(_PREAMBLE + passkey_sentence, segment_tokens)

    segment_texts = [seg0_text]
    for _ in range(n_segments - 2):
        segment_texts.append(_pad_text_to(_FILLER, segment_tokens))
    segment_texts.append(_QUESTION)  # final query segment

    segment_ids = [_to_ids(t) for t in segment_texts]
    total = int(sum(int(s.numel()) for s in segment_ids))
    return CrossSegmentPasskey(
        segment_ids=segment_ids,
        answer=passkey,
        query_segment_index=n_segments - 1,
        passkey_segment_index=0,
        total_tokens=total,
    )


class CrossSegmentPasskeyResult(TypedDict):
    """Outcome of a cross-segment passkey probe in one mode (carry or reset)."""

    carried: bool
    retrieved: bool
    generated: str
    query_segment_tokens: int
    answer_outside_query_window: bool


@torch.no_grad()
def score_cross_segment_passkey(
    model: nn.Module,
    example: CrossSegmentPasskey,
    carry: bool = True,
    vocab_size: int = 256,
    decode: Callable[[Sequence[int]], str] | None = None,
    device: torch.device | None = None,
) -> CrossSegmentPasskeyResult:
    """Score cross-segment passkey retrieval in carry or reset mode.

    Feeds segments ``0..k-2`` to build up (carry) or discard (reset) the memory
    state, then greedily decodes the answer from the FINAL (query) segment seeded
    with the carried (or fresh) state, and scores exact-match retrieval.  In reset
    mode the query segment sees none of the earlier segments, so retrieval is
    impossible by construction — the discriminating control.

    Args:
        model: Registered model (carry only takes effect for ``delta_memory_lm``).
        example: A :class:`CrossSegmentPasskey`.
        carry: Carry the state across the priming segments (default ``True``).
        vocab_size: Token vocab.
        decode: Token-id -> string decoder (default: byte decode).
        device: Inference device.

    Returns:
        A :class:`CrossSegmentPasskeyResult`.
    """
    if device is None:
        device = torch.device("cpu")
    model.eval()

    def _default_byte_decode(ids: Sequence[int]) -> str:
        return bytes(int(i) % 256 for i in ids).decode("utf-8", errors="replace")

    decode_fn: Callable[[Sequence[int]], str] = decode or _default_byte_decode

    segments = example.segment_ids
    prime, query = segments[:-1], segments[-1]
    use_carry = carry and _supports_state_carry(model)

    # Prime the memory over the leading segments (key + filler).  In reset mode we
    # simply do not carry the resulting state into the query segment.
    states: list[Tensor] | None = None
    if use_carry:
        primed = run_segments(model, prime, carry=True, device=device)
        states = primed.final_states

    # Greedy-decode the answer from the query segment, seeded with the carried
    # state (carry) or a fresh state (reset).
    query_ids = query.to(device).unsqueeze(0)
    generated_ids = _greedy_generate_with_state(
        model,
        query_ids,
        max_new_tokens=len(example.answer) + 2,
        states_in=states,
        device=device,
    )
    generated = decode_fn(generated_ids)
    retrieved = score_passkey_retrieval(generated, example.answer)
    return {
        "carried": use_carry,
        "retrieved": retrieved,
        "generated": generated,
        "query_segment_tokens": int(query.numel()),
        # The key is in segment 0; the query segment is the last one, so the
        # answer is always outside the query segment's own window.
        "answer_outside_query_window": example.query_segment_index
        != example.passkey_segment_index,
    }


@torch.no_grad()
def _greedy_generate_with_state(
    model: nn.Module,
    prompt_ids: Tensor,
    max_new_tokens: int,
    states_in: list[Tensor] | None,
    device: torch.device,
) -> list[int]:
    """Greedy-decode from *prompt_ids* seeded with carried per-layer *states_in*.

    Re-runs the growing query segment from ``states_in`` each step (the carried
    state represents the PRIOR segments, which never change), so the decode is
    causal and the carried memory is visible at every generated position.  Falls
    back to the stateless :func:`greedy_generate` when the model lacks the API.

    Cost: O(T^2) in the query-segment length (the full query is re-run per
    generated token), like :func:`greedy_generate`.  Fine for the eval harness
    (short answers); do NOT reuse on a latency-sensitive path without an
    incremental decode.
    """
    if not _supports_state_carry(model) or states_in is None:
        return greedy_generate(model, prompt_ids, max_new_tokens, device=device)

    ids = prompt_ids.to(device)
    if ids.dim() == 1:
        ids = ids.unsqueeze(0)
    generated: list[int] = []
    for _ in range(max_new_tokens):
        _, logits, _ = cast(
            "tuple[Tensor, Tensor, list[Tensor]]",
            model(ids, None, states_in, True),
        )
        next_id = int(logits[0, -1].argmax().item())
        generated.append(next_id)
        ids = torch.cat([ids, torch.tensor([[next_id]], device=device)], dim=1)
    return generated


@torch.no_grad()
def carried_stream_bpb(
    model: nn.Module,
    stream: Tensor,
    segment_len: int,
    carry: bool = True,
    device: torch.device | None = None,
) -> float:
    """Bits-per-byte of a long byte *stream* run in ordered segments.

    Splits the ``(L,)`` (or ``(1, L)``) byte stream into consecutive
    ``segment_len`` chunks and runs them in order.  With ``carry`` the per-layer
    delta-memory state threads across segment boundaries (so each segment is
    scored *with* the prior stream in memory); with ``carry=False`` each segment
    is scored independently (reset) — the control.  Each segment contributes its
    ``T_i - 1`` within-segment next-token positions (predict ``ids[1:]`` from
    ``ids[:-1]``); these positions are where the carried memory shows up.  The
    cross-boundary prediction of each segment's first token (the discarded last
    logit of the previous forward) is NOT explicitly scored, so carry and reset
    are scored over the same positions and differ only in the state those
    positions see.  The summed loss is converted to bits per byte.

    Args:
        model: Registered model (carry only takes effect for ``delta_memory_lm``).
        stream: ``(L,)`` or ``(1, L)`` byte-id stream, ``L >= 2``.
        segment_len: Tokens per segment (``>= 1``).
        carry: Carry state across segments (default ``True``).
        device: Inference device.

    Returns:
        Mean bits-per-byte over all scored next-token positions.
    """
    if device is None:
        device = torch.device("cpu")
    model.eval()
    if segment_len < 1:
        raise ValueError(f"segment_len must be >= 1, got {segment_len}")
    if stream.dim() == 2 and stream.shape[0] == 1:
        stream = stream[0]
    if stream.dim() != 1:
        raise ValueError(f"stream must be 1-D (L,) or (1, L), got {tuple(stream.shape)}")
    L = int(stream.numel())
    if L < 2:
        raise ValueError(f"stream must have length >= 2, got L={L}")

    stream = stream.to(device)
    use_carry = carry and _supports_state_carry(model)
    states: list[Tensor] | None = None
    nats_sum = 0.0
    tokens = 0

    for start in range(0, L, segment_len):
        seg = stream[start : start + segment_len]
        if seg.numel() < 2:
            # A length-1 tail has no in-segment next-token target; still feed it
            # so the carried state advances, but it contributes no scored tokens.
            if use_carry and seg.numel() == 1:
                _, _, states = cast(
                    "tuple[Tensor, Tensor, list[Tensor]]",
                    model(seg.unsqueeze(0), None, states, True),
                )
            continue
        ids = seg.unsqueeze(0)
        if use_carry:
            _, logits, states = cast(
                "tuple[Tensor, Tensor, list[Tensor]]",
                model(ids, None, states, True),
            )
        else:
            _, logits = model(ids)

        # In-segment next-token positions: predict ids[1:] from ids[:-1].  With
        # carry these positions see the prior stream through the threaded state;
        # under reset they see only the current segment — that is the comparison.
        in_logits = logits[0, :-1]              # (T-1, vocab)
        in_tgt = seg[1:]                        # (T-1,)
        nats = nn.functional.cross_entropy(in_logits, in_tgt, reduction="sum")
        nats_sum += float(nats)
        tokens += int(in_tgt.numel())

    mean_nats = nats_sum / max(1, tokens)
    return mean_nats / math.log(2.0)
