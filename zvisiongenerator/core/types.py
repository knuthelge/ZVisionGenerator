"""Core types — StageOutcome enum shared by all generation pipelines."""

from __future__ import annotations

from enum import Enum, auto


class StageOutcome(Enum):
    """Explicit outcome of a workflow stage."""

    success = auto()
    skipped = auto()
    retry = auto()
    failed = auto()
