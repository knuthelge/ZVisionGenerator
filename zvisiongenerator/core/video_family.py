"""Describe a video model family: the frame counts it accepts, the steps it offers and what it can do.

These are plain values read from ``video_model_presets.<family>`` in the config, so a caller asks the family instead of
branching on its name. Everything here is pure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = ["ANY_FRAMES", "UNDESCRIBED_UPSCALE_STEPS", "FrameRule", "SettledSteps", "VideoCapabilities", "VideoFamily", "settle_video_steps"]


@dataclass(frozen=True)
class FrameRule:
    """The frame counts a family accepts: ``alignment * k + offset`` for ``k >= 1``.

    Attributes:
        alignment: The step between valid counts.
        offset: The count added to the multiple.
        round: How an invalid count moves to a valid one: ``"nearest"`` (halves to even) or ``"up"``.
    """

    alignment: int
    offset: int
    round: str

    def to_json(self) -> dict[str, Any]:
        """Return the rule as the JSON object the Web UI reads."""
        return {"alignment": self.alignment, "offset": self.offset, "round": self.round}


# The rule of a model with no family description: every count is valid except 1, which moves to 2.
ANY_FRAMES = FrameRule(1, 1, "nearest")

# The steps of an upscaled clip of a model with no family description.
UNDESCRIBED_UPSCALE_STEPS = 8


@dataclass(frozen=True)
class VideoCapabilities:
    """What a family can do. A platform may take some of these away (see ``VideoFamily.platform_capabilities``)."""

    i2v: bool
    end_frame: bool
    extend: bool
    upscale: bool
    upscale_review: bool
    lora: bool
    audio: bool
    low_memory: bool


@dataclass(frozen=True)
class VideoFamily:
    """A video model family as the config describes it.

    Attributes:
        name: The family id (the ``video_model_presets`` key).
        label: How the family reads in messages.
        frame_rule: The frame counts it accepts.
        capabilities: What it can do on every platform.
        platform_capabilities: Per ``sys.platform``, the capabilities that differ there.
        default_steps: The steps a clip runs at when none are given.
        max_steps: The most steps it runs, or ``None`` for no cap.
        step_options: The only step counts it offers (ascending), or ``None`` for any.
        default_frames: The frame count of a clip when none is given, or ``None`` to take the size table's.
        extend_frame_step: The step between the lengths an extension can add.
        carry_frames: The frames of the source's end an extension carries into its window (the window is these plus the new frames).
        default_upscale_steps: The steps of an upscaled clip when none are given.
        extend_fps: The one frame rate an extension source must run at, or ``None`` when the family extends a clip at its own rate.
    """

    name: str
    label: str
    frame_rule: FrameRule
    capabilities: VideoCapabilities
    platform_capabilities: dict[str, dict[str, bool]]
    default_steps: int
    max_steps: int | None
    step_options: tuple[int, ...] | None
    default_frames: int | None
    extend_frame_step: int
    carry_frames: int
    default_upscale_steps: int
    extend_fps: int | None = None


@dataclass(frozen=True)
class SettledSteps:
    """The steps a job runs at, the steps of its upscale pass, and a message when the request was changed."""

    steps: int | None
    upscale_steps: int | None
    note: str | None


def settle_video_steps(steps: int | None, *, upscale: bool, family: VideoFamily | None, default_upscale_steps: int) -> SettledSteps:
    """Settle the step counts of a video job.

    An upscaled clip takes *default_upscale_steps* when no steps were given and runs its upscale pass at the settled steps.
    A family that lists step options moves the steps to the nearest option (the higher one on a tie); otherwise a family
    with a cap lowers them to it. Without a family nothing is changed.

    Args:
        steps: The requested steps, or ``None`` for the default.
        upscale: Whether the clip is upscaled.
        family: The model's family, or ``None`` when it has no description.
        default_upscale_steps: The steps of an upscaled clip when none were given.

    Returns:
        The settled steps (``None`` only when there were none and no family default), the upscale steps (``None`` without
        upscaling) and a message when the steps were changed.
    """
    if steps is None:
        steps = default_upscale_steps if upscale else (family.default_steps if family is not None else None)
    note = None
    if steps is not None and family is not None:
        given = steps
        if family.step_options:
            steps = min(family.step_options, key=lambda option: (abs(option - given), -option))
            if steps != given:
                note = f"{family.label} offers {list(family.step_options)} steps; using {steps} instead of {given}"
        elif family.max_steps is not None and given > family.max_steps:
            steps = family.max_steps
            note = f"{family.label} supports max {family.max_steps} denoising steps; capping {given} -> {steps}"
    return SettledSteps(steps=steps, upscale_steps=steps if upscale else None, note=note)
