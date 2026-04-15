"""GenerationWorkflow — named, ordered sequence of generation stages.

Unified workflow engine for both image and video generation pipelines.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from zvisiongenerator.core.types import StageOutcome


@dataclass
class GenerationWorkflow:
    """Named, ordered sequence of generation stages. Works for image and video."""

    name: str
    stages: list[Callable[..., StageOutcome]]

    def run(self, request: Any, artifacts: Any) -> StageOutcome:
        """Run all stages in order. Stop on non-success outcome."""
        for stage in self.stages:
            outcome = stage(request, artifacts)
            if outcome is not StageOutcome.success:
                return outcome
        return StageOutcome.success
