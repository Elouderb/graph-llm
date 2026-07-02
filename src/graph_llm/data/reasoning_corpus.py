"""Synthetic byte-level reasoning corpus for comparative eval (card 255a4424).

Three deterministic, seeded, difficulty-parameterised generators — each produces
examples with a clearly delimited PROMPT and a trailing ANSWER span:

1. **Arithmetic** (``ArithmeticGenerator``): multi-step word problems.
   e.g. "Sam had 7 marbles. He found 8 more. He lost 3. How many now? 12"
   Sweepable ``n_ops`` (number of operations).

2. **Transitive logic** (``TransitiveLogicGenerator``): chain of pairwise
   comparisons followed by an extremum query.
   e.g. "Ann is taller than Bob. Bob is taller than Cal. Who is tallest? Ann"
   Sweepable ``chain_len`` (number of entities).

3. **Code prediction** (``CodePredictionGenerator``): tiny variable-assignment
   programs with a ``print`` statement.
   e.g. "x=3; y=x+4; print(y) -> 7"
   Fixed short form (the "small share" generator per the card).

Public surface
--------------
- ``ReasoningExample``          — single labelled example.
- ``ReasoningCorpus``           — fixed train/val snapshot with byte-token sequences.
- ``build_reasoning_corpus``    — canonical entry point, seeded, deterministic.
- ``build_reasoning_dataloaders`` — (train_loader, val_loader) compatible with
  the text8 / ``_TextChunkDataset`` contract.
- ``answer_accuracy``           — THE key metric: fraction of examples whose
  answer span is predicted correctly.
- Generators: ``ArithmeticGenerator``, ``TransitiveLogicGenerator``,
  ``CodePredictionGenerator``.
"""

from __future__ import annotations

import logging
import random
import string
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader

from graph_llm.config import DataConfig
from graph_llm.data.loader import _TextChunkDataset

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Example dataclass
# ---------------------------------------------------------------------------


@dataclass
class ReasoningExample:
    """One labelled reasoning example.

    Attributes:
        text: The full text string ``PROMPT + ANSWER`` (e.g. "... How many now? 12").
        prompt: The question prefix (without the answer).
        answer: The trailing answer string.
        kind: Generator type: ``"arithmetic"``, ``"transitive"``, or ``"code"``.
        difficulty: Integer difficulty parameter for this example (n_ops / chain_len
            / number of variable assignments).
        tokens: ``np.ndarray`` of ``int64`` byte tokens (0..255) for the full text.
        prompt_len: Number of tokens in the prompt prefix.
    """

    text: str
    prompt: str
    answer: str
    kind: str
    difficulty: int
    tokens: np.ndarray  # int64, shape (len(text.encode()),)
    prompt_len: int     # byte-token length of the prompt (not including answer)


def _text_to_tokens(text: str) -> np.ndarray:
    """Encode a UTF-8 string to an ``int64`` byte-token array (vocab=256)."""
    return np.frombuffer(text.encode("utf-8", errors="replace"), dtype=np.uint8).astype(
        np.int64
    )


# ---------------------------------------------------------------------------
# Generator 1 — Multi-step arithmetic word problems
# ---------------------------------------------------------------------------

_NAMES = [
    "Sam", "Alex", "Jo", "Dana", "Kim", "Lee", "Pat", "Chris",
    "Morgan", "Taylor", "Casey", "Jordan",
]
_OBJECTS = [
    "marbles", "coins", "stickers", "books", "cards", "apples",
    "balls", "pens", "stamps", "chips",
]


class ArithmeticGenerator:
    """Generate multi-step arithmetic word problems.

    Each example starts with a character who *has* some initial count, then
    performs ``n_ops`` single-digit add or subtract operations.  The final
    answer is the accumulated total.

    Format::

        "<Name> had <N> <objects>. He found <A> more. He lost <B>. How many now? <answer>"

    The ANSWER span is everything after the last ``"? "`` (just the number).
    The carry/accumulation is computable token-by-token (for the per-position
    reasoner hypothesis).

    Args:
        n_ops: Number of add/subtract operations after the initial count
            (difficulty parameter; must be >= 1).  Harder = more operations.
    """

    kind: str = "arithmetic"

    def __init__(self, n_ops: int = 2) -> None:
        if n_ops < 1:
            raise ValueError(f"n_ops must be >= 1, got {n_ops}")
        self.n_ops = n_ops

    def generate(self, rng: random.Random) -> ReasoningExample:
        name = rng.choice(_NAMES)
        obj = rng.choice(_OBJECTS)
        start = rng.randint(1, 9)
        running = start

        # Build prompt piece by piece
        parts: list[str] = [f"{name} had {start} {obj}."]
        for _ in range(self.n_ops):
            delta = rng.randint(1, 9)
            op = rng.choice(["add", "sub"])
            if op == "add":
                running += delta
                parts.append(f" He found {delta} more.")
            else:
                # Keep total positive: if we'd go negative, add instead.
                if running - delta <= 0:
                    running += delta
                    parts.append(f" He found {delta} more.")
                else:
                    running -= delta
                    parts.append(f" He lost {delta}.")

        question = " How many now?"
        prompt = "".join(parts) + question + " "
        answer = str(running)
        text = prompt + answer

        tokens = _text_to_tokens(text)
        prompt_len = len(_text_to_tokens(prompt))
        return ReasoningExample(
            text=text,
            prompt=prompt,
            answer=answer,
            kind=self.kind,
            difficulty=self.n_ops,
            tokens=tokens,
            prompt_len=prompt_len,
        )


# ---------------------------------------------------------------------------
# Generator 2 — Transitive / relational logic
# ---------------------------------------------------------------------------

# Each entry: (comparative form, superlative form)
_RELATIONS: list[tuple[str, str]] = [
    ("taller", "tallest"),
    ("heavier", "heaviest"),
    ("faster", "fastest"),
    ("older", "oldest"),
    ("richer", "richest"),
    ("stronger", "strongest"),
    ("smarter", "smartest"),
]
_PEOPLE_POOL = list(string.ascii_uppercase)  # A–Z single-letter names


class TransitiveLogicGenerator:
    """Generate transitive / relational logic chain puzzles.

    Produces a strict linear chain of ``chain_len`` entities with ``chain_len - 1``
    pairwise comparisons presented in **shuffled order**, then asks for the extremum
    (tallest / fastest / etc.).  The answer is always the entity that never appears
    on the right-hand side of a comparison (the global maximum).

    Format (clauses shuffled)::

        "Bob is taller than Cal. Ann is taller than Bob. Who is tallest? Ann"

    The ANSWER span is the entity name after the last ``"? "``.
    Answering requires multi-hop / cross-token reasoning: the chain must be
    traversed to find the root.  Shuffled clause order ensures that position-based
    n-gram baselines (e.g. "guess the first entity") degrade monotonically with
    chain length.

    Args:
        chain_len: Number of entities in the chain (difficulty parameter;
            must be >= 2).  Harder = longer chain = more reasoning hops.
    """

    kind: str = "transitive"

    def __init__(self, chain_len: int = 3) -> None:
        if chain_len < 2:
            raise ValueError(f"chain_len must be >= 2, got {chain_len}")
        self.chain_len = chain_len

    def generate(self, rng: random.Random) -> ReasoningExample:
        comparative, superlative = rng.choice(_RELATIONS)
        # Pick chain_len distinct single-letter names in random order.
        # names[0] is the global maximum (the answer); the chain is
        # names[0] > names[1] > ... > names[-1].
        names = rng.sample(_PEOPLE_POOL[:20], self.chain_len)
        # Build the (chain_len-1) pairwise clauses, then SHUFFLE their order.
        # Shuffling is the key to making position-based n-gram baselines fail:
        # a "guess the first entity" heuristic succeeds for chain_len=2 (50%
        # chance the winner appears first) but degrades to ~1/(chain_len-1) for
        # longer chains.  The oracle still finds the unique entity that never
        # appears on the right-hand side — unaffected by clause order.
        pairs: list[tuple[str, str]] = [
            (names[i], names[i + 1]) for i in range(self.chain_len - 1)
        ]
        rng.shuffle(pairs)
        parts: list[str] = [f"{a} is {comparative} than {b}." for a, b in pairs]
        prompt = " ".join(parts) + f" Who is {superlative}? "
        answer = names[0]
        text = prompt + answer

        tokens = _text_to_tokens(text)
        prompt_len = len(_text_to_tokens(prompt))
        return ReasoningExample(
            text=text,
            prompt=prompt,
            answer=answer,
            kind=self.kind,
            difficulty=self.chain_len,
            tokens=tokens,
            prompt_len=prompt_len,
        )


# ---------------------------------------------------------------------------
# Generator 3 — Small code / output-prediction
# ---------------------------------------------------------------------------


class CodePredictionGenerator:
    """Generate tiny variable-assignment programs with a print statement.

    Produces a sequence of integer assignments of the form ``x=<N>`` or
    ``y=x+<N>`` / ``y=x-<N>``, followed by ``print(<var>) -> <value>``.
    The ANSWER span is the integer value after ``"-> "``.

    Format::

        "x=3; y=x+4; print(y) -> 7"

    The number of assignment statements is ``n_vars`` (the difficulty parameter).
    All intermediate values are kept positive (>= 1).

    Args:
        n_vars: Number of variable assignments (difficulty; must be >= 1).
    """

    kind: str = "code"

    def __init__(self, n_vars: int = 2) -> None:
        if n_vars < 1:
            raise ValueError(f"n_vars must be >= 1, got {n_vars}")
        self.n_vars = n_vars

    def generate(self, rng: random.Random) -> ReasoningExample:
        # Variable names: v0, v1, v2, ...
        var_names = [f"v{i}" for i in range(self.n_vars)]
        env: dict[str, int] = {}
        stmts: list[str] = []

        for idx, var in enumerate(var_names):
            if idx == 0:
                val = rng.randint(1, 9)
                env[var] = val
                stmts.append(f"{var}={val}")
            else:
                src = var_names[idx - 1]
                delta = rng.randint(1, 9)
                op = rng.choice(["+", "-"])
                if op == "+" or env[src] - delta <= 0:
                    env[var] = env[src] + delta
                    stmts.append(f"{var}={src}+{delta}")
                else:
                    env[var] = env[src] - delta
                    stmts.append(f"{var}={src}-{delta}")

        last_var = var_names[-1]
        last_val = env[last_var]
        code_str = "; ".join(stmts)
        prompt = f"{code_str}; print({last_var}) -> "
        answer = str(last_val)
        text = prompt + answer

        tokens = _text_to_tokens(text)
        prompt_len = len(_text_to_tokens(prompt))
        return ReasoningExample(
            text=text,
            prompt=prompt,
            answer=answer,
            kind=self.kind,
            difficulty=self.n_vars,
            tokens=tokens,
            prompt_len=prompt_len,
        )


# ---------------------------------------------------------------------------
# Oracle solvers (prove examples are answerable)
# ---------------------------------------------------------------------------


def oracle_solve(example: ReasoningExample) -> str:
    """Return the oracle answer for a reasoning example.

    For arithmetic and code, re-evaluates the expression embedded in the text.
    For transitive logic, parses the chain and returns the head.  This is
    independent of the stored ``example.answer`` so it validates correctness.

    Raises ``ValueError`` if the example cannot be parsed (indicates a generator
    bug — the oracle is the acceptance test).
    """
    if example.kind == "arithmetic":
        return _oracle_arithmetic(example)
    if example.kind == "transitive":
        return _oracle_transitive(example)
    if example.kind == "code":
        return _oracle_code(example)
    raise ValueError(f"Unknown example kind: {example.kind!r}")


def _oracle_arithmetic(ex: ReasoningExample) -> str:
    """Parse and re-compute an arithmetic word-problem answer."""
    import re

    text = ex.text
    # Extract initial count: "<Name> had <N> <objects>."
    m = re.match(r"\w+ had (\d+) \w+\.", text)
    if m is None:
        raise ValueError(f"Cannot parse arithmetic initial count from: {text!r}")
    running = int(m.group(1))

    # Find all "found N more" (add) and "lost N" (sub) operations.
    for found_m in re.finditer(r"He found (\d+) more\.", text):
        running += int(found_m.group(1))
    for lost_m in re.finditer(r"He lost (\d+)\.", text):
        running -= int(lost_m.group(1))

    return str(running)


def _oracle_transitive(ex: ReasoningExample) -> str:
    """Parse a transitive chain and return the extremum entity."""
    import re

    # Extract all "A is <rel>er than B" pairs.
    pairs = re.findall(r"(\w+) is \w+ than (\w+)\.", ex.text)
    if not pairs:
        raise ValueError(f"Cannot parse transitive pairs from: {ex.text!r}")
    # Build partial order: find the node that never appears on the right side.
    all_left = {p[0] for p in pairs}
    all_right = {p[1] for p in pairs}
    roots = all_left - all_right
    if len(roots) != 1:
        raise ValueError(f"Expected 1 root, got {roots} in: {ex.text!r}")
    return roots.pop()


def _oracle_code(ex: ReasoningExample) -> str:
    """Evaluate a tiny variable-assignment program and return the printed value."""
    import re

    # Parse from the text (before " -> ").
    code_part = ex.text.split(" -> ")[0]
    # Extract "print(var)" to find the target variable.
    pm = re.search(r"print\((\w+)\)", code_part)
    if pm is None:
        raise ValueError(f"Cannot find print() in: {code_part!r}")
    target = pm.group(1)
    # Evaluate all "var=expr" statements (safe: only digits, +, -, variable names).
    env: dict[str, int] = {}
    for stmt in code_part.split(";"):
        stmt = stmt.strip()
        am = re.match(r"(\w+)=(.+)$", stmt)
        if am is None:
            continue
        lhs = am.group(1)
        rhs = am.group(2).strip()
        # rhs is either a digit literal or "var+N" / "var-N"
        try:
            env[lhs] = int(rhs)
        except ValueError:
            expr_m = re.match(r"(\w+)([+-])(\d+)$", rhs)
            if expr_m is None:
                raise ValueError(f"Cannot parse rhs: {rhs!r}")
            base_var = expr_m.group(1)
            op = expr_m.group(2)
            delta = int(expr_m.group(3))
            base_val = env[base_var]
            env[lhs] = base_val + delta if op == "+" else base_val - delta
    if target not in env:
        raise ValueError(f"Target variable {target!r} not found in env={env}")
    return str(env[target])


# ---------------------------------------------------------------------------
# Answer-accuracy metric
# ---------------------------------------------------------------------------


def answer_accuracy(
    examples: list[ReasoningExample],
    predictions: list[str],
) -> float:
    """Compute the fraction of examples whose predicted answer matches exactly.

    This is THE key eval metric (bpb may not move; answer-accuracy is where
    reasoning shows).  Both ``predictions[i]`` and ``examples[i].answer`` are
    stripped of leading/trailing whitespace before comparison.

    Args:
        examples: Ground-truth examples (with ``example.answer`` set).
        predictions: Predicted answer strings, one per example.

    Returns:
        Float in ``[0.0, 1.0]``.  Returns ``0.0`` when ``examples`` is empty.
    """
    if not examples:
        return 0.0
    correct = sum(
        p.strip() == e.answer.strip() for e, p in zip(examples, predictions)
    )
    return correct / len(examples)


def extract_answer_from_tokens(
    tokens: np.ndarray,
    prompt_len: int,
) -> str:
    """Decode the answer span from a token array given the prompt length.

    The answer span is ``tokens[prompt_len:]`` decoded as UTF-8 bytes.

    Args:
        tokens: ``int64`` byte-token array (the full example).
        prompt_len: Number of tokens belonging to the prompt prefix.

    Returns:
        Decoded answer string (may be empty if ``prompt_len >= len(tokens)``).
    """
    answer_tokens = tokens[prompt_len:]
    return bytes(answer_tokens.astype(np.uint8)).decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Corpus builder
# ---------------------------------------------------------------------------


@dataclass
class ReasoningCorpus:
    """Fixed train/val snapshot of reasoning examples, byte-level.

    Attributes:
        train: List of training examples.
        val: List of validation examples.
        train_tokens: Concatenated ``int64`` byte-token array for training.
        val_tokens: Concatenated ``int64`` byte-token array for validation.

    Each example is separated from the next by a newline (``\\n``), so the flat
    token stream is a simple concatenation and can be chunked with
    ``_TextChunkDataset`` exactly like text8.
    """

    train: list[ReasoningExample]
    val: list[ReasoningExample]
    train_tokens: np.ndarray   # int64, concatenated byte tokens
    val_tokens: np.ndarray     # int64, concatenated byte tokens


_NEWLINE_TOKENS = np.array([ord("\n")], dtype=np.int64)


def _concat_examples(examples: list[ReasoningExample]) -> np.ndarray:
    """Concatenate example token arrays with a newline separator."""
    if not examples:
        return np.empty(0, dtype=np.int64)
    parts: list[np.ndarray] = []
    for ex in examples:
        parts.append(ex.tokens)
        parts.append(_NEWLINE_TOKENS)
    return np.concatenate(parts)


def build_reasoning_corpus(
    n_train: int = 4000,
    n_val: int = 500,
    seed: int = 42,
    arithmetic_n_ops: int = 3,
    transitive_chain_len: int = 4,
    code_n_vars: int = 2,
    arith_frac: float = 0.5,
    transitive_frac: float = 0.4,
    # code_frac is implicitly 1.0 - arith_frac - transitive_frac
) -> ReasoningCorpus:
    """Build a fixed, seeded, byte-level reasoning corpus.

    Produces a ``ReasoningCorpus`` with ``n_train`` training examples and
    ``n_val`` validation examples.  The corpus is deterministic: the same
    ``seed`` always produces the same examples in the same order.

    The mix is controlled by ``arith_frac`` and ``transitive_frac``; code gets
    the remainder (must be >= 0).  Default: 50% arithmetic, 40% transitive,
    10% code — code is the "small share" per the card.

    Args:
        n_train: Number of training examples.
        n_val: Number of validation examples.
        seed: RNG seed (same seed → same corpus).
        arithmetic_n_ops: Difficulty for arithmetic (number of operations).
        transitive_chain_len: Difficulty for transitive logic (chain length).
        code_n_vars: Difficulty for code prediction (number of variable assignments).
        arith_frac: Fraction of examples that are arithmetic.
        transitive_frac: Fraction of examples that are transitive logic.

    Returns:
        A :class:`ReasoningCorpus` with flat byte-token arrays ready for
        ``_TextChunkDataset``.
    """
    code_frac = 1.0 - arith_frac - transitive_frac
    if code_frac < 0.0:
        raise ValueError(
            f"arith_frac + transitive_frac = {arith_frac + transitive_frac:.3f} > 1.0"
        )

    arith_gen = ArithmeticGenerator(n_ops=arithmetic_n_ops)
    trans_gen = TransitiveLogicGenerator(chain_len=transitive_chain_len)
    code_gen = CodePredictionGenerator(n_vars=code_n_vars)

    rng = random.Random(seed)

    def _generate_split(n: int, split_seed: int) -> list[ReasoningExample]:
        split_rng = random.Random(split_seed)
        examples: list[ReasoningExample] = []
        n_arith = round(n * arith_frac)
        n_trans = round(n * transitive_frac)
        n_code = n - n_arith - n_trans
        for _ in range(n_arith):
            examples.append(arith_gen.generate(split_rng))
        for _ in range(n_trans):
            examples.append(trans_gen.generate(split_rng))
        for _ in range(max(0, n_code)):
            examples.append(code_gen.generate(split_rng))
        # Shuffle so the kinds are interleaved in the flat token stream.
        split_rng.shuffle(examples)
        return examples

    # Use deterministic derived seeds for each split so that changing n_train
    # does NOT change the validation set (and vice versa).
    train_seed = rng.randint(0, 2**31 - 1)
    val_seed = rng.randint(0, 2**31 - 1)

    train_examples = _generate_split(n_train, train_seed)
    val_examples = _generate_split(n_val, val_seed)

    return ReasoningCorpus(
        train=train_examples,
        val=val_examples,
        train_tokens=_concat_examples(train_examples),
        val_tokens=_concat_examples(val_examples),
    )


# ---------------------------------------------------------------------------
# DataLoader builder (text8-compatible contract)
# ---------------------------------------------------------------------------


def build_reasoning_dataloaders(
    cfg: DataConfig,
    seed: int = 42,
    num_workers: int = 0,
    n_train: int = 4000,
    n_val: int = 500,
    arithmetic_n_ops: int = 3,
    transitive_chain_len: int = 4,
    code_n_vars: int = 2,
) -> tuple[DataLoader, DataLoader]:
    """Build ``(train_loader, val_loader)`` for the reasoning corpus.

    Matches the contract of ``build_dataloaders`` in ``loader.py``:
    - Examples are byte-level (uint8 tokens cast to int64, vocab=256).
    - Each ``DataLoader`` batch is ``(inputs, targets)`` of shape ``(B, seq_len)``.
    - Deterministic: same ``seed`` → same split.

    Args:
        cfg: ``DataConfig`` (uses ``cfg.seq_len``, ``cfg.batch_size``).
        seed: RNG seed.
        num_workers: DataLoader worker processes.
        n_train: Training examples.
        n_val: Validation examples.
        arithmetic_n_ops: Arithmetic difficulty.
        transitive_chain_len: Transitive logic difficulty.
        code_n_vars: Code prediction difficulty.

    Returns:
        ``(train_loader, val_loader)`` as :class:`torch.utils.data.DataLoader`.
    """
    corpus = build_reasoning_corpus(
        n_train=n_train,
        n_val=n_val,
        seed=seed,
        arithmetic_n_ops=arithmetic_n_ops,
        transitive_chain_len=transitive_chain_len,
        code_n_vars=code_n_vars,
    )
    train_ds = _TextChunkDataset(corpus.train_tokens, cfg.seq_len)
    val_ds = _TextChunkDataset(corpus.val_tokens, cfg.seq_len)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=True,
        generator=torch.Generator().manual_seed(seed),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
    )
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Smoke demo (called when module is run directly)
# ---------------------------------------------------------------------------


def _smoke() -> None:
    """Print 2–3 examples of each type and run a quick oracle check."""
    rng = random.Random(0)
    generators: list[tuple[str, object]] = [
        ("arithmetic (n_ops=3)", ArithmeticGenerator(n_ops=3)),
        ("transitive (chain=4)", TransitiveLogicGenerator(chain_len=4)),
        ("code (n_vars=2)", CodePredictionGenerator(n_vars=2)),
    ]
    print("\n=== Reasoning Corpus Smoke ===\n")
    for label, gen in generators:
        print(f"--- {label} ---")
        for i in range(3):
            ex = gen.generate(rng)  # type: ignore[attr-defined]
            oracle = oracle_solve(ex)
            ok = oracle == ex.answer
            print(f"  [{i}] {ex.text!r}")
            print(f"       prompt_len={ex.prompt_len}  answer={ex.answer!r}  oracle={oracle!r}  ok={ok}")
        print()

    # Build a tiny corpus and check sizes
    corpus = build_reasoning_corpus(n_train=20, n_val=5, seed=0)
    print(f"corpus.train_tokens shape: {corpus.train_tokens.shape}, dtype={corpus.train_tokens.dtype}")
    print(f"corpus.val_tokens   shape: {corpus.val_tokens.shape},  dtype={corpus.val_tokens.dtype}")
    assert corpus.train_tokens.dtype == np.int64
    assert corpus.val_tokens.dtype == np.int64
    print("\nSmoke passed.")


if __name__ == "__main__":
    _smoke()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "ReasoningExample",
    "ReasoningCorpus",
    "ArithmeticGenerator",
    "TransitiveLogicGenerator",
    "CodePredictionGenerator",
    "oracle_solve",
    "answer_accuracy",
    "extract_answer_from_tokens",
    "build_reasoning_corpus",
    "build_reasoning_dataloaders",
]
