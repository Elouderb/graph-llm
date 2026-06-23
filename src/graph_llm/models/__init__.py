"""Model package.

Import :func:`build_model` and :func:`register_model` from here.
Individual model classes are importable from their submodules.

Importing this package registers the novel ``"bilinear_lm"`` (card 86347418) and
``"delta_memory_lm"`` (card e2c6ea95) models as an import side-effect, mirroring
how the baselines self-register through ``graph_llm.models.baselines``.  Both
reuse ``RMSNorm`` from ``baselines.transformer``, so importing them also
transitively registers the ``"transformer"`` and ``"mamba"`` baselines.
"""

from .bilinear_lm import BilinearLM  # noqa: F401 — side-effect: registers "bilinear_lm"
from .delta_memory_lm import DeltaMemoryLM  # noqa: F401 — registers "delta_memory_lm"
from .registry import build_model, register_model

__all__ = ["register_model", "build_model", "BilinearLM", "DeltaMemoryLM"]
