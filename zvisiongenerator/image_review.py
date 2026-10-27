"""Review of an image's small render before the upscale: show it, write it to a temporary file, report the decision."""

from __future__ import annotations

import os
import tempfile
import time
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from zvisiongenerator.core.progress_events import ProgressCallback
from zvisiongenerator.review import REVIEW_PROMPT, ReviewOutcome, await_review_decision
from zvisiongenerator.utils.interactive import SkipSignal

if TYPE_CHECKING:
    from PIL import Image

__all__ = ["make_image_review"]


def _remove_quietly(path: str | os.PathLike[str]) -> None:
    """Delete the temporary preview file; a file that is already gone is fine."""
    try:
        Path(path).unlink(missing_ok=True)
    except OSError as exc:
        warnings.warn(f"Could not delete {path}: {exc}", stacklevel=2)


def _write_preview_file(image: Image.Image) -> str:
    """Save *image* as a PNG in the system temp folder, outside the output folder, and return its path."""
    fd, path = tempfile.mkstemp(prefix="ziv-review-", suffix=".png")
    os.close(fd)
    try:
        image.save(path, format="PNG")
    except BaseException:
        os.unlink(path)
        raise
    return path


def make_image_review(
    skip: SkipSignal,
    outcome: ReviewOutcome,
    *,
    progress_callback: ProgressCallback | None,
    ran_iterations: int,
    attempt: int,
    completed_iterations: Callable[[], int],
    total_iterations: int,
) -> Callable[[Image.Image], bool]:
    """Build the ``review`` callable of one image generation.

    The callable shows the render, waits for the decision and records it on *outcome*, with the time spent waiting.
    It returns whether to go on with the upscale. On Quit it returns False and leaves the Quit queued, so the
    runner's skipped branch ends the batch.

    Args:
        skip: The job's control signal; its review state carries the decision.
        outcome: Receives the decision and the wait.
        progress_callback: Receives ``job_paused`` (with ``review`` and the render as ``preview``) and ``job_resumed``.
            Without one the job runs in a terminal: the render goes to a temporary PNG whose path is printed.
        ran_iterations: The 1-based iteration under review.
        attempt: The 1-based attempt within that iteration (a Repeat or retry starts the next one).
        completed_iterations: Returns the finished iterations when the review starts.
        total_iterations: Iterations in the batch.
    """

    def _review(image: Image.Image) -> bool:
        path = _write_preview_file(image) if progress_callback is None else None
        started = time.monotonic()
        try:
            if path is not None:
                print(f"\nSmall render: {path}", flush=True)
                print(REVIEW_PROMPT, flush=True)
            outcome.decision = await_review_decision(
                skip,
                progress_callback=progress_callback,
                mode="image",
                completed_iterations=completed_iterations(),
                total_iterations=total_iterations,
                review={"iteration": ran_iterations, "attempt": attempt},
                preview=image,
            )
        finally:
            outcome.wait_seconds += time.monotonic() - started
            if path is not None:
                _remove_quietly(path)
        return outcome.decision == "approve"

    return _review
