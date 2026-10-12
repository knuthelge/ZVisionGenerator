"""Shared mechanics of a review before upscaling: announce the pause and wait for the decision."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from zvisiongenerator.core.progress_events import ProgressCallback, emit_progress
from zvisiongenerator.utils.interactive import SkipSignal

__all__ = ["REVIEW_PROMPT", "ReviewOutcome", "await_review_decision", "review_enabled"]

REVIEW_PROMPT = "Approve upscale? [y] yes  [n] discard  [q] quit"


@dataclass
class ReviewOutcome:
    """Working state of the review within one generation attempt."""

    wait_seconds: float = 0.0
    decision: str | None = None

    @property
    def declined(self) -> bool:
        """Whether the review ended the attempt: the user discarded the render, skipped the image or quit."""
        return self.decision in ("discard", "skip", "quit")


def review_enabled(args: Any) -> bool:
    """Return whether *args* ask for a review before the upscale: an upscale is on and ``review`` is set."""
    return bool(getattr(args, "upscale", None) and getattr(args, "review", False))


def await_review_decision(
    skip: SkipSignal,
    *,
    progress_callback: ProgressCallback | None,
    mode: str,
    completed_iterations: int,
    total_iterations: int,
    review: dict[str, Any],
    **paused_extra: Any,
) -> str:
    """Emit ``job_paused``, wait for the review decision, emit ``job_resumed`` and return the decision.

    Timing the wait is up to the caller, as it differs between the image and the video path.

    Args:
        skip: The job's control signal; its review state carries the decision.
        progress_callback: Receives the events; ``None`` emits nothing.
        mode: ``"image"`` or ``"video"``, for the events.
        completed_iterations: Finished iterations when the review starts.
        total_iterations: Iterations in the batch.
        review: The ``review`` payload of ``job_paused``.
        **paused_extra: Further keys of ``job_paused``, such as the ``preview`` image.

    Returns:
        ``"approve"``, ``"discard"``, ``"skip"`` (Next) or ``"quit"``.
    """
    skip.begin_review()
    emit_progress(
        progress_callback,
        "job_paused",
        mode=mode,
        completed_iterations=completed_iterations,
        total_iterations=total_iterations,
        review=review,
        **paused_extra,
    )
    decision = skip.wait_for_review()
    emit_progress(progress_callback, "job_resumed", mode=mode, completed_iterations=completed_iterations, total_iterations=total_iterations)
    return decision
