"""Job plan value types — the per-iteration text and seeds that preflight hands to generation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from zvisiongenerator.utils.prompt_enhance import EnhanceSettings


class EnhanceStatus(StrEnum):
    """Outcome of auto prompt enhancement for one iteration."""

    OFF = "off"
    ENHANCED = "enhanced"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass(frozen=True)
class IterationPlan:
    """Planned inputs for one generation iteration (one run × set × prompt).

    Attributes:
        run_index: Zero-based run number.
        set_name: Prompt set the iteration belongs to.
        prompt_index: Zero-based index of the prompt within its set.
        prompt: The prompt template as written (event payloads keep using it).
        seed: Seed for the first attempt.
        resolved_prompt: The prompt with ``{a|b}`` choices expanded; verbatim for JSON captions.
        enhance: Enhancement settings that applied, or ``None`` when not requested.
        enhanced_prompt: The rewrite, set only when ``enhance_status`` is ``enhanced``.
        enhance_status: Outcome of the rewrite.
    """

    run_index: int
    set_name: str
    prompt_index: int
    prompt: str
    seed: int
    resolved_prompt: str
    enhance: EnhanceSettings | None
    enhanced_prompt: str | None = None
    enhance_status: EnhanceStatus = EnhanceStatus.OFF

    def __post_init__(self) -> None:
        """Reject a status that contradicts the stored rewrite or settings."""
        if (self.enhanced_prompt is not None) != (self.enhance_status is EnhanceStatus.ENHANCED):
            raise ValueError(f"IterationPlan: enhanced_prompt must be set exactly when enhance_status is 'enhanced' (got status '{self.enhance_status}').")
        if self.enhance is None and self.enhance_status is not EnhanceStatus.OFF:
            raise ValueError(f"IterationPlan: enhance_status must be 'off' when no enhancement was requested (got '{self.enhance_status}').")


@dataclass(frozen=True)
class JobPlan:
    """Every planned iteration of a job, in run → set → prompt order.

    Attributes:
        iterations: One plan per iteration.
        cancelled: The user quit during preflight; generation must not run.
    """

    iterations: tuple[IterationPlan, ...]
    cancelled: bool = False

    @property
    def has_rewrites(self) -> bool:
        """Return whether any iteration carries an enhanced prompt."""
        return any(iteration.enhance_status is EnhanceStatus.ENHANCED for iteration in self.iterations)
