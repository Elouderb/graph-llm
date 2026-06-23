"""Baseline + metrics + dataset-logic tests (card 424e3a8e).

All offline / CPU.  Loader logic is exercised on in-memory byte fixtures — never
a live download.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import graph_llm.models.baselines  # noqa: F401 — registers "transformer" and "mamba"
from graph_llm.config import Config, ModelConfig
from graph_llm.data.loader import (
    canonical_byte_splits,
    make_canonical_split_datasets,
)
from graph_llm.eval.metrics import bits_per_byte, perplexity
from graph_llm.models import build_model
from graph_llm.models.baselines import count_params, match_params
from graph_llm.models.baselines.mamba import MambaBlock

BASELINES = ["transformer", "mamba"]


def _base_cfg(name: str) -> Config:
    return Config(
        model=ModelConfig(
            name=name,
            vocab_size=256,
            d_model=64,
            n_heads=4,
            n_layers=2,
            d_ff=256,
            max_seq_len=32,
            dropout=0.0,
        )
    )


# ---------------------------------------------------------------------------
# Registration + forward/backward contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", BASELINES)
def test_baseline_registered_and_builds(name: str) -> None:
    model = build_model(_base_cfg(name))
    assert isinstance(model, torch.nn.Module)
    assert count_params(model) > 0


@pytest.mark.parametrize("name", BASELINES)
def test_baseline_forward_backward_cpu(name: str) -> None:
    """forward(x, targets) -> (scalar loss, (B,T,V) logits); loss.backward() works."""
    cfg = _base_cfg(name)
    model = build_model(cfg)
    B, T = 2, cfg.model.max_seq_len
    x = torch.randint(0, cfg.model.vocab_size, (B, T))
    targets = torch.randint(0, cfg.model.vocab_size, (B, T))

    loss, logits = model(x, targets)
    assert loss.ndim == 0, f"loss must be scalar, got {loss.shape}"
    assert math.isfinite(loss.item())
    assert logits.shape == (B, T, cfg.model.vocab_size)

    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads), (
        "no finite non-zero gradients flowed"
    )


@pytest.mark.parametrize("name", BASELINES)
def test_baseline_eval_mode_zero_loss_without_targets(name: str) -> None:
    cfg = _base_cfg(name)
    model = build_model(cfg)
    model.eval()
    x = torch.randint(0, cfg.model.vocab_size, (1, cfg.model.max_seq_len))
    with torch.no_grad():
        loss, logits = model(x)
    assert float(loss.sum()) == 0.0
    assert logits.shape[-1] == cfg.model.vocab_size


# ---------------------------------------------------------------------------
# Mamba chunkwise scan == sequential oracle (card 18b14615, LOAD-BEARING)
# ---------------------------------------------------------------------------
#
# mamba_scan="chunkwise" is now the DEFAULT, so the fast chunked selective scan
# must reproduce the original sequential _selective_scan numerically — including
# sequence lengths that are NOT a multiple of the chunk size.  A mismatch here is
# a real bug in the chunked scan, not a tolerance to relax.


def _mamba_block(scan: str, *, chunk_size: int = 32, d_model: int = 64) -> MambaBlock:
    torch.manual_seed(0)
    return MambaBlock(
        d_model=d_model, d_state=16, d_conv=4, expand=2, dt_rank="auto",
        dt_min=0.001, dt_max=0.1, scan=scan, chunk_size=chunk_size,
    )


def _mamba_scan_inputs(block: MambaBlock, B: int, T: int, seed: int) -> tuple[torch.Tensor, ...]:
    """Realistic (u, delta, B, C) for a direct selective-scan call, in fp32.

    delta is softplus'd (strictly positive, like the model); B, C random.
    """
    torch.manual_seed(seed)
    d_inner, n = block.d_inner, block.d_state
    u = torch.randn(B, T, d_inner)
    delta = torch.nn.functional.softplus(torch.randn(B, T, d_inner) * 0.5 - 3.0)  # ~ model range
    Bm = torch.randn(B, T, n)
    Cm = torch.randn(B, T, n)
    return u.float(), delta.float(), Bm.float(), Cm.float()


@pytest.mark.parametrize("T", [1, 31, 32, 33, 64, 65, 100, 128, 257])
def test_mamba_chunkwise_scan_matches_sequential(T: int) -> None:
    """Chunked selective scan == sequential oracle within 1e-4 (fp32), incl. T % C != 0."""
    seq_block = _mamba_block("sequential")
    chunk_block = _mamba_block("chunkwise", chunk_size=32)
    chunk_block.load_state_dict(seq_block.state_dict())  # identical A_log / D

    u, delta, Bm, Cm = _mamba_scan_inputs(seq_block, B=2, T=T, seed=T)
    ref = seq_block._selective_scan(u, delta, Bm, Cm)
    fast = chunk_block._selective_scan_chunkwise(u, delta, Bm, Cm)

    assert fast.shape == ref.shape == (2, T, seq_block.d_inner)
    assert torch.isfinite(fast).all(), "chunked selective scan produced non-finite values"
    max_diff = (fast - ref).abs().max().item()
    assert max_diff <= 1e-4, (
        f"mamba chunkwise != sequential oracle (T={T}): max abs diff {max_diff:.3e} > 1e-4"
    )


@pytest.mark.parametrize("chunk_size", [1, 8, 16, 64])
def test_mamba_chunkwise_scan_matches_across_chunk_sizes(chunk_size: int) -> None:
    seq_block = _mamba_block("sequential")
    chunk_block = _mamba_block("chunkwise", chunk_size=chunk_size)
    chunk_block.load_state_dict(seq_block.state_dict())
    u, delta, Bm, Cm = _mamba_scan_inputs(seq_block, B=2, T=70, seed=7)  # 70 % chunk != 0
    ref = seq_block._selective_scan(u, delta, Bm, Cm)
    fast = chunk_block._selective_scan_chunkwise(u, delta, Bm, Cm)
    max_diff = (fast - ref).abs().max().item()
    assert max_diff <= 1e-4, f"C={chunk_size}: max diff {max_diff:.3e} > 1e-4"


def test_mamba_block_forward_chunkwise_matches_sequential() -> None:
    """Full MambaBlock forward agrees between the two scans (shared weights, T % C != 0)."""
    seq_block = _mamba_block("sequential", d_model=64)
    chunk_block = _mamba_block("chunkwise", chunk_size=16, d_model=64)
    chunk_block.load_state_dict(seq_block.state_dict())
    seq_block.eval()
    chunk_block.eval()
    x = torch.randn(2, 50, 64)  # 50 not a multiple of 16
    with torch.no_grad():
        out_seq = seq_block(x)
        out_fast = chunk_block(x)
    max_diff = (out_seq - out_fast).abs().max().item()
    assert max_diff <= 1e-4, f"mamba forward seq vs chunkwise max diff {max_diff:.3e} > 1e-4"


def test_mamba_chunkwise_block_is_causal_by_perturbation() -> None:
    """Causality probe on the FAST Mamba path: perturb token t+1; outputs <= t unchanged."""
    torch.manual_seed(0)
    block = _mamba_block("chunkwise", chunk_size=4, d_model=32)
    block.eval()
    assert block.scan == "chunkwise"
    B, T = 2, 13  # not a multiple of the chunk size (4)
    x = torch.randn(B, T, 32)
    with torch.no_grad():
        out = block(x)
    t = 5
    x_pert = x.clone()
    x_pert[:, t + 1] += torch.randn_like(x_pert[:, t + 1])  # perturb a FUTURE token
    with torch.no_grad():
        out_pert = block(x_pert)
    max_diff = (out[:, : t + 1] - out_pert[:, : t + 1]).abs().max().item()
    assert max_diff < 1e-5, (
        f"chunkwise Mamba leaked future info: perturbing token {t + 1} changed "
        f"outputs at positions <= {t} by {max_diff:.2e}"
    )


def test_mamba_invalid_scan_raises() -> None:
    with pytest.raises(ValueError):
        _mamba_block("nonsense")


# ---------------------------------------------------------------------------
# Parameter matching within ±5 %
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", BASELINES)
@pytest.mark.parametrize("target", [500_000, 2_000_000, 5_000_000])
def test_match_params_within_tolerance(name: str, target: int) -> None:
    base = Config(
        model=ModelConfig(
            name=name,
            vocab_size=256,
            n_heads=8,
            n_layers=4,
            d_ff=2048,
            max_seq_len=128,
            dropout=0.0,
        )
    )
    matched = match_params(target, base, tolerance=0.05)
    achieved = count_params(build_model(matched))
    rel_err = abs(achieved - target) / target
    assert rel_err <= 0.05, (
        f"{name}: target={target:,} achieved={achieved:,} err={rel_err:.3f} > 5%"
    )
    # match_params must not mutate the template config.
    assert base.model.d_model == 512


def test_match_params_does_not_mutate_base() -> None:
    base = _base_cfg("transformer")
    d_model_before = base.model.d_model
    match_params(1_000_000, base)
    assert base.model.d_model == d_model_before


def test_match_params_rejects_nonpositive_target() -> None:
    with pytest.raises(ValueError):
        match_params(0, _base_cfg("transformer"))


# ---------------------------------------------------------------------------
# Metrics: BPB / perplexity vs. hand-computed values
# ---------------------------------------------------------------------------


class _FixedLogitsModel(torch.nn.Module):
    """A model that ignores its input and emits pre-set logits.

    Lets us drive perplexity/BPB from a fixture with a known closed-form NLL.
    """

    def __init__(self, logits: torch.Tensor) -> None:
        super().__init__()
        # logits: (B, T, V)
        self.register_buffer("_logits", logits)

    def forward(self, x: torch.Tensor, targets: torch.Tensor | None = None):
        logits = self._logits
        if targets is not None:
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), targets.reshape(-1)
            )
        else:
            loss = torch.zeros(1)
        return loss, logits


def test_perplexity_and_bpb_match_hand_computed() -> None:
    """Uniform logits over V classes => NLL = ln(V), ppl = V, BPB = log2(V)."""
    V, B, T = 8, 1, 4
    logits = torch.zeros(B, T, V)  # uniform softmax over V
    targets = torch.zeros(B, T, dtype=torch.long)
    ds = TensorDataset(torch.zeros(B, T, dtype=torch.long), targets)
    loader = DataLoader(ds, batch_size=B)

    model = _FixedLogitsModel(logits)
    ppl = perplexity(model, loader, torch.device("cpu"))
    bpb = bits_per_byte(model, loader, torch.device("cpu"))

    assert ppl == pytest.approx(float(V), rel=1e-5)
    assert bpb == pytest.approx(math.log2(V), rel=1e-5)


def test_bpb_is_token_weighted_across_ragged_batches() -> None:
    """Two batches with different token counts: NLL must weight by token count.

    Batch A: 1 token, NLL = ln(2).  Batch B: 3 tokens, NLL = ln(4).
    Token-weighted mean NLL = (1*ln2 + 3*ln4) / 4 = (ln2 + 3*2ln2)/4 = (7/4) ln2.
    => BPB = mean_nll / ln2 = 7/4 = 1.75.
    """

    class _RaggedModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self._step = 0

        def forward(self, x, targets=None):
            # First call: V=2 (NLL=ln2, 1 token); second: V=4 (NLL=ln4, 3 tokens)
            V = 2 if self._step == 0 else 4
            n = 1 if self._step == 0 else 3
            self._step += 1
            logits = torch.zeros(1, n, V)
            t = targets if targets is not None else torch.zeros(1, n, dtype=torch.long)
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, V), t.reshape(-1))
            return loss, logits

    # Two "batches": shapes only matter for token counts; the model ignores x.
    b1 = (torch.zeros(1, 1, dtype=torch.long), torch.zeros(1, 1, dtype=torch.long))
    b2 = (torch.zeros(1, 3, dtype=torch.long), torch.zeros(1, 3, dtype=torch.long))
    loader = [b1, b2]  # plain iterable of (x, targets) — metrics only iterate

    model = _RaggedModel()
    bpb = bits_per_byte(model, loader, torch.device("cpu"))  # type: ignore[arg-type]
    assert bpb == pytest.approx(1.75, rel=1e-5)


# ---------------------------------------------------------------------------
# Dataset logic: canonical byte split + chunking (no network)
# ---------------------------------------------------------------------------


def test_canonical_byte_splits_scale_for_short_corpora() -> None:
    """A short fixture is split proportionally and tiles the array exactly."""
    tokens = np.arange(1000, dtype=np.int64)
    splits = canonical_byte_splits(tokens)
    total = len(splits["train"]) + len(splits["val"]) + len(splits["test"])
    assert total == len(tokens), "splits must tile the corpus with no loss/overlap"
    # Ordering is contiguous (train precedes val precedes test).
    assert splits["train"][0] == 0
    assert splits["train"][-1] + 1 == splits["val"][0]
    assert splits["val"][-1] + 1 == splits["test"][0]
    # 90/5/5 proportions (approximately) preserved.
    assert len(splits["train"]) == pytest.approx(900, abs=2)


def test_canonical_split_full_size_proportions() -> None:
    """At full 100M length the split is exactly 90M/5M/5M."""
    tokens = np.zeros(100_000_000, dtype=np.int64)
    splits = canonical_byte_splits(tokens)
    assert len(splits["train"]) == 90_000_000
    assert len(splits["val"]) == 5_000_000
    assert len(splits["test"]) == 5_000_000


def test_make_canonical_split_datasets_chunks_byte_level() -> None:
    """make_canonical_split_datasets yields chunkable byte datasets per split."""
    raw = bytes(range(256)) * 8  # 2048 bytes
    datasets_by_split = make_canonical_split_datasets(raw, seq_len=16)
    assert set(datasets_by_split) == {"train", "val", "test"}
    train = datasets_by_split["train"]
    x, y = train[0]
    assert x.shape == (16,)
    assert y.shape == (16,)
    # Targets are inputs shifted by one (next-token).
    assert torch.equal(x[1:], y[:-1])
    # Byte-level: all ids in [0, 256).
    assert int(x.max()) < 256 and int(x.min()) >= 0
