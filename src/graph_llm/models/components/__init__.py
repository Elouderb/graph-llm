"""Novel architecture components.

Currently houses the Phase 2 factorized bilinear (MFB) front-end
(:class:`BilinearFrontEnd`).  Future components (reasoning GNN, memory GNN)
belong here too.

Each component is a plain ``torch.nn.Module`` and must NOT be imported by the
Trainer; the Trainer is model-agnostic and interacts only through the
``forward(x) -> (loss, logits)`` contract of the *registered model* that wraps
the component.  Register a component-backed model with ``@register_model`` (see
``models/bilinear_lm.py``) and select it in the YAML config.  Zero trainer
changes.
"""

from .bilinear_frontend import BilinearFrontEnd

__all__ = ["BilinearFrontEnd"]
