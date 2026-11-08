"""Rules of the LTX video file upscale: which clips it takes, its refine steps, its menu option and how its progress is counted."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from zvisiongenerator.utils.prompt_compose import compose_video_prompt
from zvisiongenerator.utils.upscale import SEEDVR2_MEMORY_WARNING, VIDEO_UPSCALE_FACTOR, WARNING_MEMORY, UpscaleOption

if TYPE_CHECKING:
    from zvisiongenerator.utils.model_memory import MemoryBudget

# Pixels per latent cell of the LTX video VAE: an upscaled clip's sides are multiples of it.
_LTX_CELL_PIXELS = 32
# LTX clips have 8k+1 frames: the VAE packs 8 frames into each latent frame after the first.
_LTX_FRAME_STEP = 8


def ltx_upscale_problem(width: int, height: int, frame_count: int) -> str | None:
    """Return why a clip of this size and length can't be upscaled with LTX, or ``None`` when it can.

    LTX refines the clip's own latent, so the clip must already be on its grid: sides that are multiples of 32 and
    8k+1 frames (every LTX render and extension is).
    """
    if width % _LTX_CELL_PIXELS or height % _LTX_CELL_PIXELS:
        return f"Its size, {width}×{height}, isn't a multiple of {_LTX_CELL_PIXELS} on each side, as LTX videos are."
    if frame_count < _LTX_FRAME_STEP + 1 or (frame_count - 1) % _LTX_FRAME_STEP:
        return f"It has {frame_count} frames; LTX videos have 8k+1 (9, 17, … 121, …)."
    return None


def upscale_progress_units(steps: int) -> int:
    """Return the progress units of an LTX upscale with *steps* refine steps: the encode, each step and the decode."""
    return steps + 2


def ltx_upscale_prompt(scene: str | None, prompt: str | None, *, extended: bool) -> str:
    """Return what the refine of an LTX upscale is told the clip shows: the scene, then the prompt.

    An extended clip records only its last part's action, which is wrong for the frames before it, so it gets the
    scene alone (empty without one).
    """
    return compose_video_prompt(scene, "" if extended else prompt or "")


# Refine steps of an LTX file upscale when the config sets none, and the most it may run (the length of the stage-2 schedule).
DEFAULT_LTX_UPSCALE_STEPS = 2
MAX_LTX_UPSCALE_STEPS = 3


def ltx_upscale_steps(config: dict[str, Any]) -> int:
    """Return the refine steps of an LTX file upscale: ``video_model_presets.ltx.upscale.existing_video_steps``, or the default.

    Raises:
        ValueError: If the value is not an integer from 1 to ``MAX_LTX_UPSCALE_STEPS``.
    """
    upscale = ((config.get("video_model_presets") or {}).get("ltx") or {}).get("upscale") or {}
    steps = upscale.get("existing_video_steps", DEFAULT_LTX_UPSCALE_STEPS) if isinstance(upscale, dict) else DEFAULT_LTX_UPSCALE_STEPS
    if isinstance(steps, bool) or not isinstance(steps, int) or not 1 <= steps <= MAX_LTX_UPSCALE_STEPS:
        raise ValueError(f"video_model_presets.ltx.upscale.existing_video_steps must be an integer from 1 to {MAX_LTX_UPSCALE_STEPS}, got {steps!r}.")
    return steps


def ltx_video_option(width: int, height: int, frame_count: int | None, *, model_ref: str, model_name: str, budget: MemoryBudget | None) -> UpscaleOption:
    """Return the one 2× option of an LTX video upscale, warned when the clip may not fit this computer's memory.

    The warning comes from ``clip_memory_fit`` for a two-stage render at the output size, which covers the upscale's
    encode, refine and decode. Without a frame count or a memory budget there is no warning. The option is never disabled.
    """
    from zvisiongenerator.utils.model_memory import clip_memory_fit

    out_width, out_height = width * VIDEO_UPSCALE_FACTOR, height * VIDEO_UPSCALE_FACTOR
    option = UpscaleOption(factor=VIDEO_UPSCALE_FACTOR, width=out_width, height=out_height, allowed=True)
    if frame_count is None or budget is None:
        return option
    fit = clip_memory_fit(model_ref, width=out_width, height=out_height, frames=frame_count, low_memory=True, upscale=True, budget=budget, model_name=model_name)
    if fit is not None and fit.get("status") in ("tight", "too_large"):
        return replace(option, warning=SEEDVR2_MEMORY_WARNING, warning_code=WARNING_MEMORY)
    return option


# The decode of an LTX upscale takes about this share of one refine step (measured 0.45-0.6 at s and m).
_DECODE_STEP_SHARE = 0.5


def ltx_upscale_eta(finished: Sequence[float], total: int) -> float | None:
    """Return the seconds an LTX upscale has left, from when each of its units finished, or ``None`` before a refine step has.

    The units are the encode, the refine steps and the decode (``upscale_progress_units``). The first unit also holds
    the model loads and the prompt encode and says nothing about the steps, so the estimate starts after the first
    step: the steps' average time for each step left, plus about half a step for the decode.

    Args:
        finished: Seconds from the start of the upscale at which each finished unit ended, in order.
        total: The upscale's progress units.
    """
    done = len(finished)
    if done >= total:
        return 0.0
    if done < 2:
        return None
    per_step = (finished[-1] - finished[0]) / (done - 1)
    steps_left = total - 1 - done
    return per_step * (steps_left + _DECODE_STEP_SHARE)
