"""In-segment multi-chain REASONING synthetic tasks (card 2dd3400f).

The tandem's reasoner half is validated (in the scratchpad) on an in-segment
multi-chain walk; this module ports that task into the repo as the R-SYNTHETIC
half of the mixed training stream, alongside the existing cross-segment
memory-retrieval synthetic (:mod:`graph_llm.data.synthetic_tasks`, the M half).

Two example kinds, both single-byte-answer and processed as ``n_segments``
segments of length ``L`` (the answer is the final content byte of the LAST
segment), engineered so each single pathway PROVABLY fails its off-specialty:

Type M (memory) — cross-SEGMENT retrieval
    Segment 0 holds a binding ``K=V;`` (K a letter, V a digit); the LAST segment
    asks ``get(K)?<V>``.  The binding is >= one segment back, so only the carried
    delta-memory state can bridge it — the segment-bounded reasoner cannot.

Type R (reasoning) — within-segment multi-chain walk
    Early segments are inert filler; the LAST segment holds ``n_chains`` interleaved
    shuffled assignment chains + ``print(<head of one chain>) -> <root value>``.
    Walking the queried chain (multi-hop) is what a single-shot memory recall
    cannot do; the memory caps at the ``1/n_chains`` guess ceiling.

Aux labels for the CAUSAL reasoner (card e4e8a4dc — the WIN binds at clause
COMPLETION, not the LHS byte, so a strictly-causal window has seen the whole
``lhs=rhs``):

* ``locate_offset`` — the clause-END byte of the queried head's clause (the locate
  target / teacher-forced walk seed).
* ``walk_traj`` — the ordered clause-END offsets ALONG the queried chain
  (``[head_end, hop1_end, ..., root_end, ...]``, length ``steps + 1``) — the
  per-hop walk-aux target trajectory.

These supervise the reasoner ONLY on the R rows (locate-CE + per-hop walk-aux);
the memory (M) rows and any real-text rows get none.  Offsets are byte offsets
within the last (answer) segment (all clause chars are single-byte ASCII).
"""

from __future__ import annotations

import random
import re
import string
from dataclasses import dataclass

import numpy as np

# M key pool (distinct letters) and answer digits.
_M_KEY_POOL: list[str] = list(string.ascii_uppercase)
ANSWER_DIGITS: str = string.digits  # M answers: one of 0-9

# Filler excludes '=' '(' ')' '?' and the letters of 'print'/'get' structure only via
# the structural chars, so no spurious binding / query / clause can form.
_FILLER_CHARS: str = string.ascii_lowercase + string.digits + " .,"

# R name pool: 26 letters + 260 letter+digit pairs = 286 distinct content-addressable
# names (enough for several deep chains).  R root-value pool: 36 distinct single bytes
# (digits + uppercase) so a non-chaining "guess one of the n_chains roots" caps at
# 1/n_chains while pure chance is 1/36.
_R_NAME_POOL: list[str] = list(string.ascii_lowercase) + [
    a + d for a in string.ascii_lowercase for d in "0123456789"
]
R_VALUE_POOL: tuple[str, ...] = tuple(string.digits + string.ascii_uppercase)

PAD_VALUE = 0

_CLAUSE_RE = re.compile(r"(?:^|[;\s])(\w+)=(\w+)")
_PRINT_RE = re.compile(r"print\((\w+)\)")


@dataclass
class ReasoningExample:
    """One labelled mixed example (segmented).

    Attributes:
        seg_tokens: ``(n_segments, L)`` int64 per-segment byte view (right-padded).
        seg_lens: ``(n_segments,)`` real (unpadded) length of each segment.
        answer: single answer byte.
        kind: ``"M"`` (memory) or ``"R"`` (reasoning).
        answer_seg: index of the segment holding the answer (the last).
        answer_pos: index WITHIN ``answer_seg`` of the answer byte.
        locate_offset: R — clause-END byte of the queried head clause (within the
            answer segment); ``-1`` for M.
        walk_traj: R — ordered clause-END offsets along the queried chain (length
            ``walk_steps + 1``); all ``-1`` for M.
        text_segments: raw per-segment strings (oracle / debugging).
        depth: R chain depth (0 for M).
    """

    seg_tokens: np.ndarray
    seg_lens: np.ndarray
    answer: str
    kind: str
    answer_seg: int
    answer_pos: int
    locate_offset: int
    walk_traj: np.ndarray
    text_segments: list[str]
    depth: int


def _tok(s: str) -> np.ndarray:
    return np.frombuffer(s.encode("utf-8"), dtype=np.uint8).astype(np.int64)


_FILLER_ARR = np.frombuffer(_FILLER_CHARS.encode("ascii"), dtype=np.uint8)


def _filler(rng: random.Random, n: int) -> str:
    """Inert filler of length ``n`` (no structural chars).  Vectorised: draws the
    bytes with numpy (seeded off ``rng`` for reproducibility) instead of a per-char
    Python loop — the batch-generation hot path."""
    if n <= 0:
        return ""
    idx = np.random.default_rng(rng.getrandbits(63)).integers(0, len(_FILLER_ARR), size=n)
    return _FILLER_ARR[idx].tobytes().decode("ascii")


def _pad_segment(s: str, seg_len: int) -> tuple[np.ndarray, int]:
    toks = _tok(s)
    if len(toks) > seg_len:
        toks = toks[:seg_len]
    real = len(toks)
    if real < seg_len:
        toks = np.concatenate([toks, np.full(seg_len - real, PAD_VALUE, dtype=np.int64)])
    return toks, real


def _clause_end_offset(text: str, query_head: str) -> int:
    """Clause-END byte of the queried head clause ``<head>=<rhs>`` in ``text``.

    The queried head appears as an LHS exactly once; its clause is terminated by the
    next ``;`` (present for every clause, incl. the last, since ``text`` continues
    with ``"; print(...)"``).  The clause-end is the byte just before that ``;``.
    """
    m = re.search(rf"(?:^|[;\s]){re.escape(query_head)}=", text)
    if m is None:  # pragma: no cover - generator invariant
        raise ValueError(f"queried head {query_head!r} LHS not found in {text!r}")
    lhs = m.start()
    while lhs < len(text) and text[lhs] in "; \t":
        lhs += 1
    semi = text.index(";", lhs)
    return semi - 1


def _chain_end_positions(code: str, query_head: str, steps: int) -> list[int]:
    """Ordered clause-END offsets along the queried chain, saturating at the root.

    Returns ``[head_end, hop1_end, ..., root_end, root_end, ...]`` of length
    ``steps + 1`` (``code`` is the pre-``->`` part; offsets are byte offsets in it).
    """
    env: dict[str, str] = {}
    end_off: dict[str, int] = {}
    for m in _CLAUSE_RE.finditer(code):
        env[m.group(1)] = m.group(2)
        end_off[m.group(1)] = m.end(2) - 1  # last byte of the RHS
    seq: list[int] = []
    cur = query_head
    for _ in range(len(env) + 1):
        if cur not in env:  # pragma: no cover - generator invariant
            raise ValueError(f"dangling name {cur!r} in {code!r}")
        seq.append(end_off[cur])
        nxt = env[cur]
        if nxt not in env:  # cur is the root clause (RHS is a literal)
            break
        cur = nxt
    if len(seq) < steps + 1:
        seq = seq + [seq[-1]] * (steps + 1 - len(seq))
    return seq[: steps + 1]


def _make_multichain_r(
    rng: random.Random, depth: int, n_chains: int, walk_steps: int, n_decoys: int = 0
) -> tuple[str, str, int, list[int]]:
    """Build the R last-segment string + (answer, clause_end_offset, walk_traj).

    ``n_decoys`` inert ``dk=<value>`` dead-end literal clauses (never on any queried
    path) add distraction — the ff756b87 Run-3 shortcut control: extra root-value
    literals so a net that does not follow the queried chain cannot latch onto the
    small root set.
    """
    need = n_chains * (depth + 1) + n_decoys
    if need > len(_R_NAME_POOL):
        raise ValueError(
            f"n_chains={n_chains} depth={depth} (+{n_decoys} decoys) needs {need} "
            f"names but pool has {len(_R_NAME_POOL)}."
        )
    if n_chains > len(R_VALUE_POOL):
        raise ValueError(f"n_chains={n_chains} > R value pool {len(R_VALUE_POOL)}.")
    names = rng.sample(_R_NAME_POOL, need)
    root_values = rng.sample(list(R_VALUE_POOL), n_chains)
    clauses: list[str] = []
    chain_heads: list[str] = []
    idx = 0
    for c in range(n_chains):
        cn = names[idx : idx + depth + 1]
        idx += depth + 1
        chain_heads.append(cn[0])
        for i in range(depth):
            clauses.append(f"{cn[i]}={cn[i + 1]}")
        clauses.append(f"{cn[-1]}={root_values[c]}")
    for dn in names[idx : idx + n_decoys]:  # inert dead-end literals (distraction)
        clauses.append(f"{dn}={rng.choice(R_VALUE_POOL)}")
    rng.shuffle(clauses)
    qc = rng.randrange(n_chains)
    query_head = chain_heads[qc]
    answer = root_values[qc]
    code = "; ".join(clauses)
    text = code + f"; print({query_head}) -> " + answer
    locate = _clause_end_offset(text, query_head)
    traj = _chain_end_positions(code, query_head, walk_steps)
    return text, answer, locate, traj


def make_reasoning_example(
    rng: random.Random,
    kind: str,
    depth: int,
    seg_len: int,
    n_segments: int,
    n_chains: int,
    walk_steps: int,
    n_decoys: int = 0,
) -> ReasoningExample:
    """Generate one mixed example (``kind`` ``"M"`` or ``"R"``)."""
    if n_segments < 2:
        raise ValueError("n_segments must be >= 2 (M must span a boundary).")
    last = n_segments - 1
    seg_texts: list[str] = []
    walk_traj = np.full(walk_steps + 1, -1, dtype=np.int64)

    if kind == "M":
        key = rng.choice(_M_KEY_POOL)
        val = rng.choice(ANSWER_DIGITS)
        bind = f"{key}={val};"
        seg0 = bind + " " + _filler(rng, max(0, seg_len - len(bind) - 1))
        seg_texts.append(seg0[:seg_len])
        for _ in range(1, last):
            seg_texts.append(_filler(rng, seg_len))
        query = f"get({key})?"
        prefix = _filler(rng, max(0, seg_len - len(query) - 1))
        seg_texts.append((prefix + query + val)[:seg_len])
        answer = val
        ex_depth = 0
        locate_offset = -1
    elif kind == "R":
        for _ in range(last):
            seg_texts.append(_filler(rng, seg_len))
        chain_text, answer, locate_offset, traj = _make_multichain_r(
            rng, depth, n_chains, walk_steps, n_decoys
        )
        if len(_tok(chain_text)) > seg_len:
            raise ValueError(
                f"R depth {depth} x {n_chains} chains is {len(_tok(chain_text))} bytes "
                f"> seg_len {seg_len}; increase seg_len."
            )
        seg_texts.append(chain_text)
        ex_depth = depth
        walk_traj = np.array(traj, dtype=np.int64)
    else:
        raise ValueError(f"kind must be 'M' or 'R', got {kind!r}")

    seg_tokens = np.zeros((n_segments, seg_len), dtype=np.int64)
    seg_lens = np.zeros(n_segments, dtype=np.int64)
    for i, txt in enumerate(seg_texts):
        toks, real = _pad_segment(txt, seg_len)
        seg_tokens[i] = toks
        seg_lens[i] = real
    answer_pos = int(seg_lens[last] - 1)
    return ReasoningExample(
        seg_tokens=seg_tokens,
        seg_lens=seg_lens,
        answer=answer,
        kind=kind,
        answer_seg=last,
        answer_pos=answer_pos,
        locate_offset=locate_offset,
        walk_traj=walk_traj,
        text_segments=seg_texts,
        depth=ex_depth,
    )


def oracle_solve(ex: ReasoningExample) -> str:
    """Independent solver for both types (the acceptance gate)."""
    if ex.kind == "M":
        last_text = ex.text_segments[ex.answer_seg]
        qm = re.search(r"get\((\w)\)\?", last_text)
        if qm is None:
            raise ValueError(f"no get(K)? in last segment: {last_text!r}")
        qkey = qm.group(1)
        for txt in ex.text_segments:
            bm = re.search(rf"(?:^|[^\w])({re.escape(qkey)})=(\d);", txt)
            if bm is not None:
                return bm.group(2)
        raise ValueError(f"no binding for key {qkey!r}")
    if ex.kind == "R":
        code_part = ex.text_segments[ex.answer_seg].split(" -> ")[0]
        pm = _PRINT_RE.search(code_part)
        if pm is None:
            raise ValueError(f"no print() in R last segment: {code_part!r}")
        target = pm.group(1)
        env: dict[str, str] = {}
        for stmt in code_part.split(";"):
            stmt = stmt.strip()
            if stmt.startswith("print(") or stmt == "":
                continue
            am = re.match(r"(\w+)=(\w+)$", stmt)
            if am is None:
                raise ValueError(f"cannot parse R clause: {stmt!r}")
            env[am.group(1)] = am.group(2)
        cur = target
        for _ in range(len(env) + 1):
            if cur not in env:
                raise ValueError(f"dangling {cur!r} in {env}")
            rhs = env[cur]
            if rhs not in env:
                return rhs
            cur = rhs
        raise ValueError(f"R chain did not terminate from {target!r}: {env}")
    raise ValueError(f"unknown kind {ex.kind!r}")


@dataclass
class ReasoningBatch:
    """A tensorisable mixed M+R batch (numpy; the trainer moves it to a device)."""

    seg_tokens: np.ndarray   # (B, n_segments, L) int64
    answer_seg: int
    answer_pos: np.ndarray   # (B,) int64
    answer: np.ndarray       # (B,) int64 target byte
    kind_is_r: np.ndarray    # (B,) bool
    locate_offset: np.ndarray  # (B,) int64 (-1 for M)
    walk_traj: np.ndarray    # (B, walk_steps + 1) int64 (-1 for M)
    examples: list[ReasoningExample]


def make_batch(
    rng: random.Random,
    kinds: list[str],
    depth: int,
    seg_len: int,
    n_segments: int,
    n_chains: int,
    walk_steps: int,
    n_decoys: int = 0,
) -> ReasoningBatch:
    """Build a mixed M+R batch from an explicit list of per-example kinds."""
    exs = [
        make_reasoning_example(rng, k, depth, seg_len, n_segments, n_chains, walk_steps, n_decoys)
        for k in kinds
    ]
    b = len(exs)
    seg_tokens = np.zeros((b, n_segments, seg_len), dtype=np.int64)
    answer_pos = np.zeros(b, dtype=np.int64)
    answer = np.zeros(b, dtype=np.int64)
    kind_is_r = np.zeros(b, dtype=bool)
    locate_offset = np.full(b, -1, dtype=np.int64)
    walk_traj = np.full((b, walk_steps + 1), -1, dtype=np.int64)
    for i, ex in enumerate(exs):
        seg_tokens[i] = ex.seg_tokens
        answer_pos[i] = ex.answer_pos
        answer[i] = ord(ex.answer)
        kind_is_r[i] = ex.kind == "R"
        locate_offset[i] = ex.locate_offset
        walk_traj[i] = ex.walk_traj
    return ReasoningBatch(
        seg_tokens=seg_tokens,
        answer_seg=int(exs[0].answer_seg),
        answer_pos=answer_pos,
        answer=answer,
        kind_is_r=kind_is_r,
        locate_offset=locate_offset,
        walk_traj=walk_traj,
        examples=exs,
    )


def oracle_smoke(
    depths: tuple[int, ...] = (4, 5, 6, 16, 32),
    n_per: int = 300,
    seg_len: int = 512,
    n_segments: int = 2,
    n_chains: int = 2,
    walk_steps: int = 12,
    seed: int = 0,
) -> bool:
    """Assert the oracle solves both types at 100% AND the aux labels are correct."""
    rng = random.Random(seed)
    ok = True
    # M is depth-independent.
    m_ok = all(
        oracle_solve(ex := make_reasoning_example(rng, "M", 0, seg_len, n_segments, n_chains, walk_steps))
        == ex.answer
        for _ in range(n_per)
    )
    ok = ok and m_ok
    for d in depths:
        r_ok = True
        for _ in range(n_per):
            ex = make_reasoning_example(rng, "R", d, seg_len, n_segments, n_chains, walk_steps)
            if oracle_solve(ex) != ex.answer:
                r_ok = False
                break
            # Aux-label correctness: locate offset lands on the queried head's clause
            # end (its byte, then a ';'); the walk trajectory starts at the locate
            # site and — when the walk budget covers the depth (walk_steps >= depth) —
            # saturates at the root clause's last byte (the answer literal).
            seg = ex.text_segments[ex.answer_seg]
            lo = ex.locate_offset
            if not (0 <= lo < ex.answer_pos and seg[lo + 1] == ";"):
                r_ok = False
                break
            code = seg.split(" -> ")[0]
            if int(ex.walk_traj[0]) != lo:
                r_ok = False
                break
            if d <= walk_steps and code[int(ex.walk_traj[-1])] != ex.answer:
                r_ok = False
                break
        ok = ok and r_ok
    return ok


if __name__ == "__main__":
    if not oracle_smoke():
        raise SystemExit("REASONING TASK ORACLE SMOKE FAILED")
    print("reasoning_tasks oracle smoke OK")


__all__ = [
    "ReasoningExample",
    "ReasoningBatch",
    "make_reasoning_example",
    "make_batch",
    "oracle_solve",
    "oracle_smoke",
    "R_VALUE_POOL",
    "ANSWER_DIGITS",
]
