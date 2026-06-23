"""Model registry: register-by-name and build_model.

Usage
-----
Register a model class::

    @register_model("my_model")
    class MyModel(nn.Module):
        def __init__(self, cfg: ModelConfig) -> None: ...
        def forward(self, x: Tensor) -> tuple[Tensor, Tensor]: ...

Build from config::

    model = build_model(cfg)        # cfg.model.name resolves the class

Embedding injection hook
------------------------
``ModelConfig.embedding_init`` is an optional string name of a registered
*embedding initialiser* callable.  This is the hook for card e1644700
(phonological init).  Resolving and calling it is left to individual model
constructors; the registry itself is agnostic.  To add a custom init later:

1. Register it with :func:`register_embedding_init`.
2. Set ``model.embedding_init: "phonological"`` in the YAML config.
3. The model constructor calls :func:`get_embedding_init` and applies it to
   its ``nn.Embedding`` weight.  Zero trainer changes required.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import torch.nn as nn

if TYPE_CHECKING:
    from graph_llm.config import Config

_REGISTRY: dict[str, type[nn.Module]] = {}
_EMBEDDING_INITS: dict[str, Callable] = {}


def register_model(name: str) -> Callable[[type[nn.Module]], type[nn.Module]]:
    """Class decorator that registers a model under *name*.

    Example::

        @register_model("transformer")
        class TransformerBaseline(nn.Module): ...
    """

    def decorator(cls: type[nn.Module]) -> type[nn.Module]:
        if name in _REGISTRY:
            raise KeyError(f"Model '{name}' is already registered.")
        _REGISTRY[name] = cls
        return cls

    return decorator


def build_model(cfg: Config) -> nn.Module:
    """Instantiate and return the model named by ``cfg.model.name``.

    The model constructor receives the full :class:`~graph_llm.config.Config`
    so it can read both ``cfg.model`` and any other sub-config it needs.
    The Trainer only ever sees the returned ``nn.Module``; no model-specific
    branches exist in the training loop.

    Args:
        cfg: Full :class:`~graph_llm.config.Config` instance.

    Returns:
        An ``nn.Module`` whose ``forward`` returns ``(loss, logits)``.

    Raises:
        KeyError: If ``cfg.model.name`` is not registered.
    """
    name = cfg.model.name
    if name not in _REGISTRY:
        available = sorted(_REGISTRY.keys())
        raise KeyError(
            f"Model '{name}' not found in registry. "
            f"Available: {available}. "
            "Did you forget to import the module that registers it?"
        )
    return _REGISTRY[name](cfg)


def list_models() -> list[str]:
    """Return a sorted list of all registered model names."""
    return sorted(_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Embedding init hook (for card e1644700)
# ---------------------------------------------------------------------------


def register_embedding_init(name: str) -> Callable[[Callable], Callable]:
    """Decorator to register a custom embedding initialiser.

    The callable signature is ``(weight: Tensor, vocab_size: int, d_model: int) -> None``.
    """

    def decorator(fn: Callable) -> Callable:
        _EMBEDDING_INITS[name] = fn
        return fn

    return decorator


def get_embedding_init(name: str) -> Callable:
    """Return the embedding init callable registered under *name*.

    Raises:
        KeyError: If *name* is not found in the embedding-init registry.
    """
    try:
        return _EMBEDDING_INITS[name]
    except KeyError:
        available = sorted(_EMBEDDING_INITS.keys())
        raise KeyError(
            f"Embedding init '{name}' not found in registry. "
            f"Available: {available}. "
            "Did you forget to import the module that registers it?"
        ) from None
