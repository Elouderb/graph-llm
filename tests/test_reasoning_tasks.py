"""Tests for the in-segment R-synthetic reasoning tasks (card 2dd3400f).

The R-synthetic half of the mixed tandem stream must be ORACLE-SOLVABLE and carry
CORRECT aux labels (the causal WIN binds at the clause END): the locate offset lands
on the queried head's clause-end, and the walk trajectory follows the queried chain
to its root literal (the answer).  These labels supervise the reasoner's locate-CE +
per-hop walk-aux ONLY on the reasoning rows.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from graph_llm.data.reasoning_tasks import (
    make_batch,
    make_reasoning_example,
    oracle_smoke,
    oracle_solve,
)


def test_oracle_and_aux_labels_train_depths() -> None:
    """Oracle solves M + R (train depths) at 100% and the aux labels are correct."""
    assert oracle_smoke(depths=(4, 5, 6), n_per=300, walk_steps=12)


def test_oracle_and_aux_labels_extrapolation_depths() -> None:
    """With a walk budget covering the depth, R extrapolation depths also verify."""
    assert oracle_smoke(depths=(4, 6, 16, 32), n_per=150, walk_steps=32)


def test_m_is_cross_segment() -> None:
    """The M binding lives in an EARLIER segment than the query (memory is required)."""
    rng = random.Random(0)
    for _ in range(50):
        ex = make_reasoning_example(rng, "M", 0, 256, 2, 2, 8)
        assert ex.kind == "M"
        assert ex.locate_offset == -1  # M has no chain to locate
        # The binding is NOT in the answer (last) segment.
        assert "=" not in ex.text_segments[ex.answer_seg].split("get(")[0]
        assert oracle_solve(ex) == ex.answer


def test_r_locate_and_walk_labels_consistent() -> None:
    """R locate offset = queried head clause-end; walk trajectory ends at the root."""
    rng = random.Random(3)
    for _ in range(80):
        ex = make_reasoning_example(rng, "R", 5, 256, 2, 2, 8)
        seg = ex.text_segments[ex.answer_seg]
        lo = ex.locate_offset
        assert 0 <= lo < ex.answer_pos
        assert seg[lo + 1] == ";"           # clause end is followed by a ';'
        assert int(ex.walk_traj[0]) == lo   # the walk seeds at the locate site
        code = seg.split(" -> ")[0]
        assert code[int(ex.walk_traj[-1])] == ex.answer  # trajectory reaches the root


def test_make_batch_shapes_and_kinds() -> None:
    """The mixed batch tensorises to the expected shapes and per-row labels."""
    b = make_batch(random.Random(1), ["M", "R", "M", "R"], 4, 256, 2, 2, 8)
    assert b.seg_tokens.shape == (4, 2, 256)
    assert b.answer_pos.shape == (4,)
    assert b.walk_traj.shape == (4, 9)  # walk_steps + 1
    assert b.kind_is_r.tolist() == [False, True, False, True]
    # M rows have no locate/walk labels; R rows do.
    assert b.locate_offset[0] == -1 and b.locate_offset[1] >= 0
    assert (b.walk_traj[0] == -1).all() and (b.walk_traj[1] >= 0).all()
