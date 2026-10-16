"""Ideogram 4 settings both image backends share: the first-step sigma, JSON captions and the default sampler preset."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

__all__ = [
    "IDEOGRAM4_INITIAL_SIGMA",
    "Ideogram4Sampler",
    "caption_text",
    "sampler_settings",
    "wrap_ideogram4_prompt",
]

# First-step sigma override applied to every Ideogram 4 generation to reduce
# false positives from the safety filter on harmless prompts (the grey
# "Image blocked by safety filter" frame).
# Overrides only the first denoising step's noise level (above 1.0, slightly noisier than pure noise).
# Best-effort mitigation (not guaranteed); retune or set to None to disable.
IDEOGRAM4_INITIAL_SIGMA: float | None = 1.004

# The default sampler preset (mflux's ``V4_DEFAULT_20``): 20 steps, guidance 7.0 with 3.0 on the last two "polish" steps,
# and a logit-normal noise schedule with mean 0.0 and standard deviation 1.75.
_DEFAULT_STEPS = 20
_MAIN_GUIDANCE = 7.0
_POLISH_GUIDANCE = 3.0
_POLISH_STEPS = 2
_DEFAULT_MU = 0.0
_DEFAULT_STD = 1.75


@dataclass(frozen=True)
class Ideogram4Sampler:
    """The sampler arguments of Ideogram 4's diffusers pipeline.

    Attributes:
        steps: Number of denoising steps.
        guidance_schedule: Guidance per step, first step first, or ``None`` when *guidance_scale* is set.
        guidance_scale: A constant guidance for every step, or ``None`` when *guidance_schedule* is set.
        mu: Mean of the logit-normal noise schedule, before its resolution shift.
        std: Standard deviation of the noise schedule.
    """

    steps: int
    guidance_schedule: tuple[float, ...] | None
    guidance_scale: float | None
    mu: float = _DEFAULT_MU
    std: float = _DEFAULT_STD


def wrap_ideogram4_prompt(prompt: str) -> str | dict[str, Any]:
    """Wrap a plain-text prompt into Ideogram 4's minimal structured JSON caption.

    Prompts that already look like JSON captions (start with "{") are returned
    unchanged so callers retain full control.

    Args:
        prompt: The user-supplied prompt text.

    Returns:
        The original prompt when it is already a JSON caption, otherwise a
        minimal structured caption dict.
    """
    if prompt.strip().startswith("{"):
        return prompt
    return {
        "high_level_description": prompt,
        "compositional_deconstruction": {"background": prompt, "elements": []},
    }


def caption_text(prompt: str) -> str:
    """Return the text a backend that takes a string feeds Ideogram 4: JSON captions verbatim, plain text wrapped and serialized compactly."""
    wrapped = wrap_ideogram4_prompt(prompt)
    return wrapped if isinstance(wrapped, str) else json.dumps(wrapped, ensure_ascii=False, separators=(",", ":"))


def sampler_settings(steps: int, guidance: float, *, steps_explicit: bool, guidance_explicit: bool) -> Ideogram4Sampler:
    """Return the sampler arguments for a run, keeping the default preset unless steps or guidance were asked for.

    Without either, the run is the default preset. With explicit *steps* the guidance keeps the preset's shape (3.0 on the
    last two steps), and with explicit *guidance* it is that value on every step.

    Args:
        steps: The step count from the config or the command line.
        guidance: The guidance from the config or the command line.
        steps_explicit: Whether *steps* was given for this run.
        guidance_explicit: Whether *guidance* was given for this run.
    """
    count = steps if steps_explicit else _DEFAULT_STEPS
    if guidance_explicit:
        return Ideogram4Sampler(steps=count, guidance_schedule=None, guidance_scale=float(guidance))
    polish = min(_POLISH_STEPS, count - 1) if count > 1 else 0
    schedule = (_MAIN_GUIDANCE,) * (count - polish) + (_POLISH_GUIDANCE,) * polish
    return Ideogram4Sampler(steps=count, guidance_schedule=schedule, guidance_scale=None)
