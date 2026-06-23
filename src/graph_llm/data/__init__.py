"""Dataset loading and encoding.

Phase 0 supports three *source* values (set via DataConfig.source):

* ``"synthetic"`` — fully offline, deterministic random token sequences.
  Always works with no network or file I/O.  Use this for smoke tests.
* ``"enwik8"`` — 100 MB byte-level Wikipedia dump (lazy download via
  ``datasets`` library; cached under DataConfig.data_dir).
* ``"tinystories"`` — roneneldan/TinyStories (lazy download; needs HuggingFace
  Hub access).

Real tokenizer training and BPE are deferred to card e1644700.  Here we use a
trivial **byte-level** (vocab size 256) or **character-level** encoder just to
exercise the pipeline.
"""

from __future__ import annotations

from .loader import ByteLevelEncoder, SyntheticDataset, build_dataloaders

__all__ = ["build_dataloaders", "SyntheticDataset", "ByteLevelEncoder"]
