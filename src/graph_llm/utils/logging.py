"""Simple structured logger for training/eval runs.

Writes JSON-serialisable lines so logs can be post-processed easily.
Falls back to plain text if the structured format is not desired.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from typing import Any

_logger = logging.getLogger("graph_llm")


def setup_logging(level: str = "INFO") -> None:
    """Configure the root graph_llm logger.

    Call once at the top of a script.  Subsequent calls are no-ops.
    """
    if _logger.handlers:
        return  # already configured
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_StructuredFormatter())
    _logger.addHandler(handler)
    _logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    _logger.propagate = False


class _StructuredFormatter(logging.Formatter):
    """Emit JSON lines for structured fields, plain text otherwise."""

    def format(self, record: logging.LogRecord) -> str:
        base = {
            "time": time.strftime("%H:%M:%S"),
            "level": record.levelname,
            "msg": record.getMessage(),
        }
        # Extra fields attached via log_metrics()
        extras = getattr(record, "metrics", None)
        if extras:
            base.update(extras)
        return json.dumps(base)


def log_metrics(step: int, **kwargs: Any) -> None:
    """Log a dict of scalar metrics at *step*.

    Example::

        log_metrics(100, loss=2.31, lr=3e-4, tokens_per_sec=12500)
    """
    record = _logger.makeRecord(
        _logger.name,
        logging.INFO,
        fn="",
        lno=0,
        msg="step",
        args=(),
        exc_info=None,
    )
    record.metrics = {"step": step, **kwargs}
    _logger.handle(record)


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a child of the graph_llm logger."""
    return logging.getLogger(f"graph_llm.{name}" if name else "graph_llm")
