"""Ideogram 4 Instant settings both image backends share: the first-step sigma, JSON captions and the sampler."""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from typing import Any

__all__ = [
    "IDEOGRAM4_INITIAL_SIGMA",
    "INSTANT_GUIDANCE",
    "Ideogram4Sampler",
    "caption_text",
    "compact_caption",
    "is_json_caption",
    "sampler_settings",
    "wrap_ideogram4_prompt",
]

# First-step sigma override applied to every Ideogram 4 generation to reduce
# false positives from the safety filter on harmless prompts (the grey
# "Image blocked by safety filter" frame).
# Overrides only the first denoising step's noise level (above 1.0, slightly noisier than pure noise).
# Best-effort mitigation (not guaranteed); retune or set to None to disable.
IDEOGRAM4_INITIAL_SIGMA: float | None = 1.004

# Ideogram 4 Instant (fal's distilled model) has its guidance built in: one conditional transformer pass per step, which
# is the pipeline's guidance at 1.0 (any other value only scales the prediction). Its noise schedule is the base
# model's logit-normal one, mean 0.0 and standard deviation 1.75.
INSTANT_GUIDANCE = 1.0
_DEFAULT_MU = 0.0
_DEFAULT_STD = 1.75


@dataclass(frozen=True)
class Ideogram4Sampler:
    """The sampler arguments of an Ideogram 4 Instant run.

    Attributes:
        steps: Number of denoising steps.
        mu: Mean of the logit-normal noise schedule, before its resolution shift.
        std: Standard deviation of the noise schedule.
    """

    steps: int
    mu: float = _DEFAULT_MU
    std: float = _DEFAULT_STD


def wrap_ideogram4_prompt(prompt: str) -> str | dict[str, Any]:
    """Wrap a plain-text prompt into Ideogram 4's minimal structured JSON caption.

    Prompts that already look like JSON captions (start with "{") keep their content so callers retain full
    control; a valid one is laid out on a single line, the form Ideogram 4 reads, wherever it came from.

    Args:
        prompt: The user-supplied prompt text.

    Returns:
        The caption as compact JSON text when it is already a JSON caption (as written when it does not parse),
        otherwise a minimal structured caption dict.
    """
    if prompt.strip().startswith("{"):
        return compact_caption(json.loads(prompt)) if is_json_caption(prompt) else prompt
    return {
        "high_level_description": prompt,
        "compositional_deconstruction": {"background": prompt, "elements": []},
    }


def caption_text(prompt: str) -> str:
    """Return the text a backend that takes a string feeds Ideogram 4: compact JSON, with plain text wrapped first."""
    wrapped = wrap_ideogram4_prompt(prompt)
    return wrapped if isinstance(wrapped, str) else compact_caption(wrapped)


def compact_caption(caption: dict[str, Any]) -> str:
    """Return *caption* as the single-line JSON Ideogram 4 reads (non-ASCII characters kept as-is)."""
    return json.dumps(caption, ensure_ascii=False, separators=(",", ":"))


def is_json_caption(text: str) -> bool:
    """Return whether *text* is a JSON object, i.e. a structured caption rather than a plain prompt."""
    if not text.strip().startswith("{"):
        return False
    try:
        return isinstance(json.loads(text), dict)
    except json.JSONDecodeError:
        return False


def sampler_settings(steps: int, guidance: float) -> Ideogram4Sampler:
    """Return the sampler arguments for a run of *steps*; a *guidance* other than 1.0 is ignored with a warning.

    Args:
        steps: The step count from the config or the command line.
        guidance: The guidance from the config or the command line.
    """
    if guidance != INSTANT_GUIDANCE:
        warnings.warn(f"Ideogram 4 Instant has its guidance built in; guidance {guidance:g} is ignored.", stacklevel=2)
    return Ideogram4Sampler(steps=steps)
