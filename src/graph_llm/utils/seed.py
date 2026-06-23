"""Deterministic seeding for reproducible experiments."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Set seeds for Python, NumPy, and PyTorch (CPU + CUDA).

    Also sets environment variables for cuDNN determinism.  When
    ``torch.cuda.is_available()`` is False the CUDA calls are skipped.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # cuDNN determinism (may reduce throughput slightly)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    # Python hash randomization
    os.environ["PYTHONHASHSEED"] = str(seed)
