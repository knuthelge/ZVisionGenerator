"""The checks of an audio request that ``ziv audio`` and the Web UI share: the model, the length of a sound and the limits of speech.

All are pure functions of the config or a preset and the values the user gave. They raise ``ValueError`` with the
message the user sees; the CLI turns it into a usage error and the Web UI into a 422.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from zvisiongenerator.utils.audio_preset import parse_audio_presets
from zvisiongenerator.utils.audio_track import probe_track_seconds
from zvisiongenerator.utils.speech_text import split_speech_text

if TYPE_CHECKING:
    from zvisiongenerator.core.audio_types import AudioGenerationRequest
    from zvisiongenerator.utils.audio_preset import AudioLimits, AudioPreset

__all__ = ["DEFAULT_SOUND_SECONDS", "MIN_SOUND_SECONDS", "check_seed", "check_speech_request", "find_audio_preset", "resolve_sound_seconds"]

DEFAULT_SOUND_SECONDS = 10.0
"""The length of a sound when none is given."""

MIN_SOUND_SECONDS = 1.0
"""The shortest sound a model makes, in seconds."""

_MAX_SEED = 2**63 - 1
"""The largest seed every audio model takes (YuE2's generator needs one below 2**63)."""


def check_seed(seed: int | None) -> None:
    """Refuse a seed no audio model takes, before anything loads; ``None`` (a random seed per take) passes.

    Raises:
        ValueError: If *seed* is below 0 or above :data:`_MAX_SEED`.
    """
    if seed is not None and not 0 <= seed <= _MAX_SEED:
        raise ValueError(f"The seed must be from 0 to {_MAX_SEED}.")


def find_audio_preset(config: Mapping[str, Any], model: str | None, *, missing: str) -> AudioPreset:
    """Return the preset *model* names among the configured audio models.

    Args:
        config: The loaded config.
        model: The model id the user gave, or ``None``.
        missing: What to say when no model was given, such as ``"--model is required"``; the ids are added after it.

    Raises:
        ValueError: If no audio models are configured, if *model* is ``None``, or if it names no preset. The message lists the ids.
    """
    presets = parse_audio_presets(config)
    if not presets:
        raise ValueError("No audio models are configured. Add presets under audio_model_presets in your config to use ziv audio.")
    ids = ", ".join(preset.id for preset in presets)
    if model is None:
        raise ValueError(f"{missing}. Choose one of: {ids}.")
    preset = next((item for item in presets if item.id == model), None)
    if preset is None:
        raise ValueError(f"Unknown audio model '{model}'. Choose one of: {ids}.")
    return preset


def resolve_sound_seconds(seconds: float | None, preset: AudioPreset, *, name: str) -> float | None:
    """Return the length of a sound (the default when none is given), or ``None`` for a model that sets its own length.

    Args:
        seconds: The length the user asked for, or ``None``.
        preset: The audio model.
        name: How the field is called in the message (``--seconds`` or ``seconds``).

    Raises:
        ValueError: If the length is not a number within ``MIN_SOUND_SECONDS`` to the model's ``max_seconds``.
    """
    if not preset.sets_length:
        return None
    chosen = DEFAULT_SOUND_SECONDS if seconds is None else seconds
    longest = float(preset.limits.max_seconds)
    if not math.isfinite(chosen) or not MIN_SOUND_SECONDS <= chosen <= longest:
        raise ValueError(f"{name} must be between {MIN_SOUND_SECONDS:g} and {longest:g}; {preset.label} makes sounds of that length.")
    return float(chosen)


def check_speech_request(request: AudioGenerationRequest, limits: AudioLimits, *, check_voice_file: bool = True) -> None:
    """Check *request* against the speech limits, so a bad request fails before the model loads.

    Args:
        request: The take to make.
        limits: The preset's limits (``part_chars``, ``max_text_chars``, ``min_voice_seconds``, ``max_voice_seconds``).
        check_voice_file: Whether to read and measure the voice recording; ``False`` checks the text only.

    Raises:
        ValueError: If the text is empty or longer than ``max_text_chars``, or the voice recording cannot be read
            (``utils.audio_track.TRACK_UNREADABLE``) or is outside ``min_voice_seconds``..``max_voice_seconds``.
    """
    if not request.prompt.strip():
        raise ValueError("Write the text to read.")
    split_speech_text(request.prompt, int(limits.part_chars), max_text_chars=int(limits.max_text_chars))
    if request.voice is None or not check_voice_file:
        return
    seconds = probe_track_seconds(request.voice.path)
    low, high = float(limits.min_voice_seconds), float(limits.max_voice_seconds)
    if not low <= seconds <= high:
        raise ValueError(f"The voice recording is {seconds:.1f} seconds long; use one between {low:g} and {high:g} seconds.")
