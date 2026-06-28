"""MQAR associative-recall PROOF test for the GatedDeltaMemory conv (card 571d50ec).

This is the test that proves the short causal conv is the missing key->value
*binding* mechanism.  It ports the ``scratchpad/mqar_testbed.py`` diagnosis into a
deterministic, capped-step repo test: ``embed -> GatedDeltaMemory -> readout`` is
trained with DIRECT supervision on a Multi-Query Associative Recall task (no LM,
no front-end — the delta matrix in isolation), and we measure recall.

The diagnosis (card 61f900ca): the delta write is TOKEN-LOCAL (``k_t`` and ``v_t``
both come from the same token ``x_t``), so without local mixing a value position
never sees its key and the memory cannot bind ``k_i -> v_i``.  The MQAR testbed
measured recall ~0.23 (capped) at conv width 1 vs ~1.0 at width >= 2.  Here we
assert the SAME qualitative result on a small, fast, seeded configuration:

* WITH the conv (``delta_conv_width=4``) recall reaches >= 0.9 at N=8 key->value
  pairs (Kv=16 distinct values) within a bounded Adam budget;
* WITHOUT it (``delta_conv_width=1``) recall is far worse on the identical recipe
  — proving the conv, not extra capacity/steps, is what enables binding.

Deterministic (seeded) and capped-step.  The ~1800-step recipe is too slow for
CPU, so the test is GPU-gated (skipped when CUDA is unavailable); set
``GRAPH_LLM_TEST_DEVICE=cpu`` to force the slow CPU run for a one-off local check.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from graph_llm.config import ModelConfig
from graph_llm.models.components.delta_memory import GatedDeltaMemory

# ---------------------------------------------------------------------------
# MQAR task: interleaved (key, value) pairs, then queries.
# ---------------------------------------------------------------------------
#
# Vocabulary layout (all disjoint id ranges so the model can tell roles apart):
#   keys   : [0, n_keys)
#   values : [n_keys, n_keys + n_values)
#   query marker : a single dedicated id
# A sequence is:  k0 v0 k1 v1 ... k_{N-1} v_{N-1}  Q ki  Q kj ...
# At each position immediately AFTER a "Q ki" query, the target is the value v
# that was bound to k_i in the prefix.  Loss/recall are scored at those positions
# ONLY (the rest of the sequence is unsupervised), exactly the MQAR protocol.


def _make_mqar_batch(
    batch: int,
    n_pairs: int,
    n_keys: int,
    n_values: int,
    n_queries: int,
    *,
    generator: torch.Generator,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build one MQAR batch (fully vectorised — no per-row Python loop).

    Returns ``(tokens, targets, score_mask)`` where ``tokens`` is ``(B, T)`` input
    ids, ``targets`` is ``(B, T)`` (the bound value at each scored position, 0
    elsewhere), and ``score_mask`` is ``(B, T)`` bool marking the query-answer
    positions.  Layout per row: ``[k v]*N`` then ``[Q k]*n_queries``; the answer is
    scored AT each query-key position (the model must predict the bound value).
    """
    val_lo = n_keys
    query_marker = n_keys + n_values

    # Distinct keys per row: argsort of random scores == a per-row permutation.
    perm = torch.rand(batch, n_keys, generator=generator, device=device).argsort(dim=1)
    keys = perm[:, :n_pairs]                                   # (B, N) distinct key ids
    vals = torch.randint(
        val_lo, val_lo + n_values, (batch, n_pairs), generator=generator, device=device
    )                                                          # (B, N) value ids

    # Binding table: bind[b, key_id] = value bound to that key (0 where unbound).
    bind = torch.zeros(batch, n_keys, dtype=torch.long, device=device)
    bind.scatter_(1, keys, vals)

    # Queries: pick n_queries of the N bound keys per row (with replacement).
    q_sel = torch.randint(0, n_pairs, (batch, n_queries), generator=generator, device=device)
    q_keys = torch.gather(keys, 1, q_sel)                      # (B, n_queries) query key ids
    q_answers = torch.gather(bind, 1, q_keys)                  # (B, n_queries) bound values

    # Assemble the sequence: [k0 v0 k1 v1 ...] then [Q qk0  Q qk1 ...].
    prefix = torch.stack([keys, vals], dim=2).reshape(batch, 2 * n_pairs)  # (B, 2N)
    marker = torch.full_like(q_keys, query_marker)
    query_block = torch.stack([marker, q_keys], dim=2).reshape(batch, 2 * n_queries)
    tokens = torch.cat([prefix, query_block], dim=1)          # (B, T)

    seq_len = 2 * n_pairs + 2 * n_queries
    targets = torch.zeros(batch, seq_len, dtype=torch.long, device=device)
    score_mask = torch.zeros(batch, seq_len, dtype=torch.bool, device=device)
    # Answer positions: the query-key slots (odd indices in the query block).
    ans_pos = 2 * n_pairs + 1 + 2 * torch.arange(n_queries, device=device)  # (n_queries,)
    targets[:, ans_pos] = q_answers
    score_mask[:, ans_pos] = True
    return tokens, targets, score_mask


class _MQARProbe(nn.Module):
    """embed -> GatedDeltaMemory -> linear readout (the delta matrix in isolation)."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.mem = GatedDeltaMemory(cfg)
        self.readout = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(x)
        h = self.mem(h)
        return self.readout(h)


def _mqar_cfg(conv_width: int, **overrides: Any) -> ModelConfig:
    """Small single-layer GatedDeltaMemory config sized for the recall task.

    d_k=32 comfortably exceeds N=8 bindings (capacity bound is d_k per head); a few
    heads give the optimiser slack.  silu_l2 + forget gate match the validated
    MQAR recipe; only ``delta_conv_width`` differs between the two arms.
    """
    base: dict[str, Any] = {
        "name": "delta_memory_lm",
        "vocab_size": 64,
        "d_model": 128,
        "delta_n_heads": 4,
        "delta_head_k_dim": 32,
        "delta_head_v_dim": 32,
        "delta_feature_map": "silu_l2",
        "delta_use_forget_gate": True,
        "delta_dropout": 0.0,
        "delta_conv_width": conv_width,
        "delta_scan": "chunkwise",
        "delta_chunk_size": 32,
    }
    base.update(overrides)
    return ModelConfig(**base)


def _train_and_eval_recall(
    conv_width: int,
    *,
    device: torch.device,
    steps: int,
    seed: int = 0,
) -> float:
    """Train the probe on MQAR and return final-batch recall at the query positions."""
    torch.manual_seed(seed)
    gen = torch.Generator(device=device).manual_seed(seed)

    n_pairs, n_keys, n_values, n_queries = 8, 16, 16, 8
    cfg = _mqar_cfg(conv_width, vocab_size=n_keys + n_values + 1)
    model = _MQARProbe(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)

    model.train()
    for _ in range(steps):
        tokens, targets, mask = _make_mqar_batch(
            batch=64, n_pairs=n_pairs, n_keys=n_keys, n_values=n_values,
            n_queries=n_queries, generator=gen, device=device,
        )
        logits = model(tokens)
        scored = mask.reshape(-1)
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, cfg.vocab_size)[scored], targets.reshape(-1)[scored]
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    # Recall on a held-out batch (fresh draws from the same generator).
    model.eval()
    with torch.no_grad():
        tokens, targets, mask = _make_mqar_batch(
            batch=256, n_pairs=n_pairs, n_keys=n_keys, n_values=n_values,
            n_queries=n_queries, generator=gen, device=device,
        )
        logits = model(tokens)
        pred = logits.argmax(dim=-1)
        scored = mask
        correct = ((pred == targets) & scored).sum().item()
        total = int(scored.sum().item())
    return correct / max(1, total)


# The recall recipe needs ~1800 Adam steps over a 128-d / 4-head memory; on CPU
# that is minutes (far too slow for the unit suite), so the proof is GPU-gated.
# Set GRAPH_LLM_TEST_DEVICE=cpu to force it on CPU (e.g. for a one-off local check).
_FORCE = os.environ.get("GRAPH_LLM_TEST_DEVICE", "").lower()
_RUN_ON_CPU = _FORCE == "cpu"
_HAS_CUDA = torch.cuda.is_available()


def _device() -> torch.device:
    if _RUN_ON_CPU:
        return torch.device("cpu")
    return torch.device("cuda")


@pytest.mark.skipif(
    not _HAS_CUDA and not _RUN_ON_CPU,
    reason="MQAR recall proof needs ~1800 Adam steps; GPU-gated (set "
    "GRAPH_LLM_TEST_DEVICE=cpu to force the slow CPU run).",
)
def test_mqar_conv_binds_key_to_value() -> None:
    """PROOF: with the conv, the delta matrix binds key->value (recall >= 0.9 at N=8).

    The conv arm (``delta_conv_width=4``) must clear 0.9 recall on 8 bindings; the
    no-conv arm (``delta_conv_width=1``) must be far worse on the IDENTICAL recipe,
    isolating the conv as the binding mechanism.  Seeded + capped steps so it is a
    deterministic check, not a training run.
    """
    device = _device()
    steps = 1800

    recall_conv = _train_and_eval_recall(conv_width=4, device=device, steps=steps)
    recall_noconv = _train_and_eval_recall(conv_width=1, device=device, steps=steps)

    assert recall_conv >= 0.9, (
        f"WITH conv (width=4) recall {recall_conv:.3f} < 0.9 — the conv should let "
        f"the delta matrix bind >= 8 key->value pairs at ~100% (no-conv arm "
        f"{recall_noconv:.3f})."
    )
    # The conv must be MARKEDLY better than no conv (the whole point of the card).
    assert recall_conv >= recall_noconv + 0.3, (
        f"conv recall {recall_conv:.3f} not markedly above no-conv {recall_noconv:.3f}; "
        "the conv is supposed to be the binding mechanism, so the gap must be large."
    )
