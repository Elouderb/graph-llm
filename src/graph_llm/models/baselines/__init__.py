"""Baseline model implementations.

Importing this package registers all baselines with the model registry:
``"transformer"`` (attention) and ``"mamba"`` (recurrent-state / selective SSM).
"""

from .mamba import MambaBaseline  # noqa: F401 — side-effect: registers "mamba"
from .sizing import count_params, match_params
from .transformer import TransformerBaseline  # noqa: F401 — registers "transformer"

__all__ = [
    "TransformerBaseline",
    "MambaBaseline",
    "count_params",
    "match_params",
]
