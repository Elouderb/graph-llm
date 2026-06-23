"""Throughput benchmark for the recurrent scans (card 18b14615).

Measures training-step throughput (forward + backward + opt.step) and peak VRAM
for the delta-memory LM and the Mamba baseline, each with the **sequential**
(O(T) Python-loop oracle) vs **chunkwise** (T/C-step parallel) scan, plus an
optional torch.compile pass on the chunkwise delta scan.  Reports REAL numbers;
nothing here is faked.

Pin to GPU 1 (RTX 2070, 8 GB) — keep the model small enough to fit 8 GB::

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
        python -u scripts/bench_recurrent_scan.py

The selective-scan / delta-scan implementation is selected by config
(``mamba_scan`` / ``delta_scan``); torch.compile on the raw sequential T-loop is
impractical (it unrolls the data-dependent loop -> pathological compile time),
which is why chunking — not compile — is the throughput fix for both models.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import graph_llm.models  # noqa: F401  registers delta_memory_lm
import graph_llm.models.baselines  # noqa: F401  registers transformer + mamba
from graph_llm.config import Config, ModelConfig
from graph_llm.models import build_model
from graph_llm.models.baselines import count_params


def _delta_cfg(seq_len: int, scan: str, *, d_model: int, layers: int) -> Config:
    """delta_memory_lm sized by (d_model, layers) so it fits the target GPU."""
    m = ModelConfig(
        name="delta_memory_lm",
        vocab_size=8192,
        d_model=d_model,
        delta_layers=layers,
        delta_n_heads=8,
        delta_head_k_dim=64,
        delta_head_v_dim=64,
        delta_ff_mult=4,
        delta_feature_map="l2",
        delta_use_forget_gate=True,
        delta_dropout=0.0,
        dropout=0.0,
        max_seq_len=seq_len,
        delta_scan=scan,
        activation_checkpointing=True,
        tie_embeddings=True,
    )
    return Config(model=m)


def _mamba_cfg(seq_len: int, scan: str, *, d_model: int, layers: int) -> Config:
    """Mamba baseline sized by (d_model, layers) so it fits the target GPU."""
    m = ModelConfig(
        name="mamba",
        vocab_size=8192,
        d_model=d_model,
        n_layers=layers,
        d_state=16,
        d_conv=4,
        expand=2,
        dropout=0.0,
        max_seq_len=seq_len,
        mamba_scan=scan,
        activation_checkpointing=True,
        tie_embeddings=True,
    )
    return Config(model=m)


def bench(
    name: str,
    cfg: Config,
    seq_len: int,
    *,
    batch_size: int,
    compile_model: bool,
    warmup: int,
    iters: int,
    device: torch.device,
) -> None:
    torch.manual_seed(0)
    model = build_model(cfg).to(device)
    model.train()
    n_params = count_params(model)

    # Build the optimizer from the EAGER module's params BEFORE compiling — a
    # compiled module proxies the same parameters, but constructing the optimizer
    # first keeps the param references unambiguous.
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    # ``torch.compile`` returns a callable wrapper; type it as ``nn.Module`` for
    # the type checker (the wrapper proxies ``__call__``).
    run_model: nn.Module = torch.compile(model) if compile_model else model  # type: ignore[assignment]

    vocab = cfg.model.vocab_size
    x = torch.randint(0, vocab, (batch_size, seq_len), device=device)
    y = torch.randint(0, vocab, (batch_size, seq_len), device=device)

    use_cuda = device.type == "cuda"
    if use_cuda:
        torch.cuda.reset_peak_memory_stats(device)

    def step() -> float:
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_cuda):
            loss, _ = run_model(x, y)
        loss.backward()
        opt.step()
        return float(loss.detach())

    # Warmup (also triggers compilation on the first call).
    for _ in range(warmup):
        step()
    if use_cuda:
        torch.cuda.synchronize(device)

    t0 = time.perf_counter()
    last_loss = 0.0
    for _ in range(iters):
        last_loss = step()
    if use_cuda:
        torch.cuda.synchronize(device)
    dt = time.perf_counter() - t0

    it_s = iters / dt
    peak_gb = (torch.cuda.max_memory_allocated(device) / 1e9) if use_cuda else float("nan")
    print(
        f"{name:42s} seq={seq_len:5d} bs={batch_size} params={n_params / 1e6:5.1f}M  "
        f"{it_s:7.3f} it/s  ({dt / iters * 1e3:8.1f} ms/it)  peakVRAM={peak_gb:5.2f} GB  "
        f"loss~{last_loss:.3f}",
        flush=True,
    )
    if use_cuda:
        torch.cuda.empty_cache()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=12)
    ap.add_argument("--seqs", type=int, nargs="+", default=[512, 1024])
    ap.add_argument("--d-model", type=int, default=384)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--seq-oracle-iters", type=int, default=2,
                    help="iters for the slow sequential oracle (the thing we are fixing)")
    ap.add_argument("--with-compile", action="store_true",
                    help="also benchmark torch.compile on the chunkwise delta scan")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"device: {torch.cuda.get_device_name(device)}  torch {torch.__version__}", flush=True)
    else:
        print(f"device: CPU  torch {torch.__version__} (GPU unavailable)", flush=True)
    print("=" * 110, flush=True)

    dm, ly = args.d_model, args.layers
    for seq in args.seqs:
        print(f"\n--- seq_len = {seq} (bf16 autocast + activation checkpointing, "
              f"d_model={dm}, layers={ly}) ---", flush=True)
        oi = args.seq_oracle_iters  # sequential is slow; very few iters
        bench("delta  sequential (oracle)", _delta_cfg(seq, "sequential", d_model=dm, layers=ly), seq,
              batch_size=args.batch_size, compile_model=False,
              warmup=1, iters=oi, device=device)
        bench("delta  chunkwise", _delta_cfg(seq, "chunkwise", d_model=dm, layers=ly), seq,
              batch_size=args.batch_size, compile_model=False,
              warmup=args.warmup, iters=args.iters, device=device)
        if args.with_compile:
            bench("delta  chunkwise + torch.compile", _delta_cfg(seq, "chunkwise", d_model=dm, layers=ly), seq,
                  batch_size=args.batch_size, compile_model=True,
                  warmup=args.warmup, iters=args.iters, device=device)
        bench("mamba  sequential (oracle)", _mamba_cfg(seq, "sequential", d_model=dm, layers=ly), seq,
              batch_size=args.batch_size, compile_model=False,
              warmup=1, iters=oi, device=device)
        bench("mamba  chunkwise", _mamba_cfg(seq, "chunkwise", d_model=dm, layers=ly), seq,
              batch_size=args.batch_size, compile_model=False,
              warmup=args.warmup, iters=args.iters, device=device)


if __name__ == "__main__":
    main()
