"""Typed configuration dataclasses + YAML loader.

All hyperparameters live here. No hardcoded values in code paths.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------


@dataclass
class ModelConfig:
    """Architecture hyperparameters shared across all registered models."""

    name: str = "transformer"
    vocab_size: int = 256
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 2048
    max_seq_len: int = 512
    dropout: float = 0.1
    tie_embeddings: bool = True
    use_rope: bool = True
    activation_checkpointing: bool = False

    # Hook for downstream cards: custom embedding init callable name (resolved
    # by the registry; None = default nn.Embedding init).
    embedding_init: str | None = None

    # --- Selective-SSM (Mamba) baseline hyperparameters (card 424e3a8e) ---
    # Ignored by the Transformer baseline; consumed by models/baselines/mamba.py.
    d_state: int = 16                  # SSM state dimension N
    d_conv: int = 4                    # depthwise causal conv kernel width
    expand: int = 2                    # inner expansion factor (d_inner = expand * d_model)
    dt_rank: int | str = "auto"        # rank of the dt projection ("auto" = ceil(d_model/16))
    dt_min: float = 0.001              # min init value for dt softplus bias
    dt_max: float = 0.1                # max init value for dt softplus bias
    # Selective-scan implementation (card 18b14615): "chunkwise" (default — the
    # fast chunked parallel scan: cumulative-decay matmuls within each chunk +
    # inter-chunk state recurrence, T/C sequential steps instead of T) or
    # "sequential" (the original O(T) Python-loop recurrence, kept as the
    # validated reference/oracle).  Both produce the SAME (loss, logits); this
    # only selects the internal scan.  torch.compile is impractical on the raw
    # T-loop (it unrolls the data-dependent loop -> pathological compile time),
    # so chunking is the effective fix.
    mamba_scan: str = "chunkwise"
    # Chunk size for the chunked selective scan.  Smaller than the delta scan's C
    # because the chunked SSM materialises a dense (B, C, C, d_inner, d_state)
    # decay-ratio tensor per chunk — peak VRAM grows ~linearly in C, so C=8 keeps
    # memory modest (fits an 8 GB card at seq 1024, where C=32 OOMs) while still
    # cutting the Python loop length 8x.  Raise it on larger GPUs for more speed.
    mamba_chunk_size: int = 8

    # --- Factorized bilinear (MFB) front-end hyperparameters (card 86347418) ---
    # Ignored by the baselines; consumed by models/components/bilinear_frontend.py
    # and models/bilinear_lm.py.  The 128-d embedding (d_model) is intentionally
    # small; the bulk of the parameter budget lives in the front-end + post-mixer.
    bilinear_window: int = 16          # W: each token interacts with itself + W-1 trailing neighbours
    bilinear_k: int = 2                # MFB factor k (sum-pool group size; k=1 == MLB)
    bilinear_o: int = 4096             # per-token MFB output dim o (default 4096 == 64x64)
    # Interaction mode: "factorized_mfb" (default), "control_linear" (matched-param
    # mixer with NO multiplicative interaction == the ablation null), or
    # "materialized_cnn" (reduce emb -> materialized_reduce_dim, form the small
    # reduce x reduce x W interaction, 2-D CNN over it).
    interaction_mode: str = "factorized_mfb"
    front_end_dropout: float = 0.1     # MFB dropout (applied after power+L2 norm)
    bilinear_offset_weighting: str = "learned"  # "sum" | "learned" aggregation across the W offsets
    # Post-mixer (depthwise-separable 1-D CNN over the sequence + gated MLP):
    post_mixer_layers: int = 4         # depth of the post-mixer stack
    post_mixer_kernel: int = 7         # causal depthwise conv kernel width
    post_mixer_ff_mult: int = 4        # gated-MLP inner expansion (d_ff = mult * trunk width)
    # Trunk width. The embedding (d_model) is intentionally tiny; the post-mixer
    # may run wider so depth*width can scale the model to GPT-1/2 budgets without
    # bloating the embedding table. 0 == "use d_model" (keeps small configs simple).
    post_mixer_width: int = 0
    # materialized_cnn mode only: emb is reduced to this dim before forming the
    # small dense interaction (32x32 == 16x lower memory than the 128x128 naive).
    materialized_reduce_dim: int = 32
    materialized_cnn_channels: int = 32  # 2-D CNN hidden channels over the interaction map

    # --- Gated-DeltaNet delta-rule matrix memory hyperparameters (card e2c6ea95) ---
    # Ignored by the baselines + bilinear_lm; consumed by
    # models/components/delta_memory.py and models/delta_memory_lm.py.  The memory
    # is a fixed-size per-head matrix S of shape (d_k, d_v) updated by a delta rule
    # + forget gate; its size is INDEPENDENT of sequence length (the bounded-memory
    # property).  Scale params via delta_n_heads x delta_layers x delta_head_*_dim.
    delta_layers: int = 6              # depth of the GatedDeltaMemory stack
    delta_n_heads: int = 8             # number of independent memory heads per layer
    delta_head_k_dim: int = 64         # per-head key/query dim d_k (memory capacity <= d_k)
    delta_head_v_dim: int = 64         # per-head value dim d_v (state S is d_k x d_v)
    # Feature map phi applied to keys/queries: "l2" (L2-normalise, the Gated-DeltaNet
    # choice — bounds ||phi(k)||=1 so the delta step is a well-scaled GD step) or
    # "silu_l2" (SiLU then L2-normalise) or "identity" (no map; for the math tests).
    delta_feature_map: str = "l2"
    delta_use_forget_gate: bool = True  # scalar per-head forget gate alpha_t (Gated-DeltaNet);
    #                                     False == ungated DeltaNet (alpha_t == 1).
    delta_ff_mult: int = 4             # gated-MLP inner expansion between memory layers
    delta_dropout: float = 0.0         # dropout inside the memory mixer + MLP
    # Scan implementation (card 18b14615): "chunkwise" (default — the fast
    # chunkwise-parallel DeltaNet scan: intra-chunk parallel matmuls + inter-chunk
    # state recurrence, T/C sequential steps instead of T), "sequential" (the
    # original O(T) Python-loop recurrence kept as the validated reference/oracle —
    # the chunkwise path is proven bit-equivalent to it within tolerance), or
    # "auto" (== "chunkwise"; alias kept for forward-compat).  Both produce the
    # SAME (loss, logits); this only selects the internal scan.
    delta_scan: str = "chunkwise"
    # Chunk size C for the chunkwise scan.  T/C sequential steps; intra-chunk work
    # is batched matmuls.  32 keeps the per-chunk cumulative forget-gate decay in a
    # safe fp32 dynamic range (the gated WY math divides by cumulative decay) while
    # cutting the sequential loop length by 32x.
    delta_chunk_size: int = 32


@dataclass
class DataConfig:
    """Dataset and tokenization settings."""

    # source: "synthetic" | "enwik8" | "text8" | "wikitext103" | "tinystories"
    # enwik8/text8 use the canonical 90M/5M/5M byte split; wikitext103 is loaded
    # byte-level (BPB) for now (token-level ppl seam documented in loader.py).
    source: str = "synthetic"
    encoder: str = "byte"              # "byte" | "bpe" (card e1644700)
    bpe_tokenizer_path: str | None = None  # path to saved BPETokenizer JSON
    data_dir: str = "data/"
    seq_len: int = 512
    batch_size: int = 8
    val_fraction: float = 0.1
    split: str = "train"               # "train" | "val" | "test" (real-corpus loaders)


@dataclass
class TrainConfig:
    """Training-loop and 12-GB toolkit settings."""

    seed: int = 42
    max_steps: int = 10_000
    grad_accumulation_steps: int = 1
    grad_clip: float = 1.0
    lr: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 400
    lr_schedule: str = "cosine"        # "cosine" | "constant"
    mixed_precision: str = "no"        # "no" | "fp16" | "bf16"
    checkpoint_dir: str = "checkpoints/"
    resume_from: str | None = None
    log_every: int = 100


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------


@dataclass
class Config:
    """Root configuration object."""

    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


# ---------------------------------------------------------------------------
# Loader helpers
# ---------------------------------------------------------------------------


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge *override* into a copy of *base*."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _dict_to_config(d: dict[str, Any]) -> Config:
    """Populate Config dataclasses from a raw nested dict."""
    cfg = Config()
    if "model" in d:
        for k, v in d["model"].items():
            if hasattr(cfg.model, k):
                setattr(cfg.model, k, v)
    if "data" in d:
        for k, v in d["data"].items():
            if hasattr(cfg.data, k):
                setattr(cfg.data, k, v)
    if "train" in d:
        for k, v in d["train"].items():
            if hasattr(cfg.train, k):
                setattr(cfg.train, k, v)
    return cfg


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> Config:
    """Load a YAML config file and apply optional dict overrides.

    Args:
        path: Path to a YAML config file.
        overrides: Nested dict of values to override after loading.

    Returns:
        A populated :class:`Config` instance.
    """
    with open(path) as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}
    if overrides:
        raw = _deep_merge(raw, overrides)
    return _dict_to_config(raw)


def config_to_dict(cfg: Config) -> dict[str, Any]:
    """Serialize a Config to a plain nested dict."""
    return asdict(cfg)
