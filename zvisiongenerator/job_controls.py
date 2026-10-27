"""Shared steps of the runners' job controls: the pause and the repeat."""

from __future__ import annotations

import random
from typing import Any

from zvisiongenerator.core.progress_events import ProgressCallback, emit_progress
from zvisiongenerator.utils.interactive import SkipSignal

__all__ = ["announce_repeat", "pause_until_resumed", "seed_range"]


def seed_range(config: dict[str, Any]) -> tuple[int, int]:
    """Return the ``(seed_min, seed_max)`` new seeds are drawn from, as set under ``generation`` in *config*."""
    generation = config.get("generation", {})
    return generation.get("seed_min", 4), generation.get("seed_max", 2**32 - 1)


def pause_until_resumed(
    control: SkipSignal,
    *,
    progress_callback: ProgressCallback | None,
    mode: str,
    completed_iterations: int,
    total_iterations: int,
) -> str | None:
    """Wait for a key press or Resume, emitting ``job_paused`` and ``job_resumed``; return the action queued meanwhile."""
    emit_progress(progress_callback, "job_paused", mode=mode, completed_iterations=completed_iterations, total_iterations=total_iterations)
    print("\n⏸ Paused. Press any key to continue...")
    control.wait_for_key()
    resume_action = control.consume()
    emit_progress(progress_callback, "job_resumed", mode=mode, completed_iterations=completed_iterations, total_iterations=total_iterations)
    print("▶ Resumed.\n")
    return resume_action


def announce_repeat(
    progress_callback: ProgressCallback | None,
    *,
    mode: str,
    completed_iterations: int,
    total_iterations: int,
    seed_min: int,
    seed_max: int,
) -> int:
    """Emit ``generation_repeat`` and return the new seed to draw the repeated prompt with."""
    seed = random.randint(seed_min, seed_max)
    emit_progress(progress_callback, "generation_repeat", mode=mode, completed_iterations=completed_iterations, total_iterations=total_iterations)
    print("\n🔁 Repeating prompt with new seed...\n")
    return seed
