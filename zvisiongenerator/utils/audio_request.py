"""The checks of an audio request that ``ziv audio`` and the Web UI share: the model, the fields it takes and the length of a sound.

All are pure functions of the config or a preset and the values the user gave. They raise ``ValueError`` with the
message the user sees; the CLI turns it into a usage error and the Web UI into a 422.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from zvisiongenerator.utils.audio_preset import parse_audio_presets, runnable_audio_presets
from zvisiongenerator.utils.audio_track import probe_track_seconds
from zvisiongenerator.utils.speech_text import split_speech_text

if TYPE_CHECKING:
    from zvisiongenerator.core.audio_types import AudioGenerationRequest
    from zvisiongenerator.utils.audio_preset import AudioLimits, AudioPreset

__all__ = ["DEFAULT_SOUND_SECONDS", "MIN_SOUND_SECONDS", "check_speech_request", "check_supported_fields", "find_runnable_preset", "resolve_sound_seconds"]

DEFAULT_SOUND_SECONDS = 10.0
"""The length of a sound when none is given."""

MIN_SOUND_SECONDS = 1.0
"""The shortest sound a model makes, in seconds."""

_NO_SECONDS_REASON = {
    "music": "sets the length from the lyrics",
    "speech": "reads the whole text",
    "sound": "sets its own length",
}


def find_runnable_preset(config: Mapping[str, Any], platform_key: str, model: str | None, *, missing: str) -> AudioPreset:
    """Return the preset *model* names among those that run on *platform_key*.

    Args:
        config: The loaded config.
        platform_key: The machine's key in a preset's ``platforms`` (``backends.audio_platform_key()``).
        model: The model id the user gave, or ``None``.
        missing: What to say when no model was given, such as ``"--model is required"``; the ids are added after it.

    Raises:
        ValueError: If no audio models are configured or none runs on this platform, if *model* is ``None``, or if it names no
            runnable preset. The message lists the runnable ids.
    """
    if not parse_audio_presets(config):
        raise ValueError("No audio models are configured. Add presets under audio_model_presets in your config to use ziv audio.")
    runnable = runnable_audio_presets(config, platform_key)
    if not runnable:
        raise ValueError(f"None of the configured audio models run on this platform ({platform_key}).")
    ids = ", ".join(preset.id for preset in runnable)
    if model is None:
        raise ValueError(f"{missing}. Choose one of: {ids}.")
    preset = next((item for item in runnable if item.id == model), None)
    if preset is None:
        raise ValueError(f"Unknown audio model '{model}'. Choose one of: {ids}.")
    return preset


def check_supported_fields(preset: AudioPreset, *, lyrics: bool, seconds: bool, voice: bool, drop: Mapping[str, str]) -> None:
    """Refuse a field that was given but that *preset* does not take.

    Args:
        preset: The audio model.
        lyrics: Whether lyrics were given.
        seconds: Whether a length was given.
        voice: Whether a voice recording was given.
        drop: What to tell the user to drop for each of ``lyrics``, ``seconds`` and ``voice``: a CLI flag, or a field name.

    Raises:
        ValueError: With the first refusal, such as ``"YuE2 3B sets the length from the lyrics; drop --seconds"``.
    """
    caps = preset.capabilities
    if lyrics and not caps.lyrics:
        raise ValueError(f"{preset.label} takes no lyrics; drop {drop['lyrics']}")
    if seconds and not caps.duration:
        raise ValueError(f"{preset.label} {_NO_SECONDS_REASON[preset.kind]}; drop {drop['seconds']}")
    if voice and not caps.voice_clone:
        raise ValueError(f"{preset.label} cannot follow a voice; drop {drop['voice']}")


def resolve_sound_seconds(seconds: float | None, preset: AudioPreset, *, name: str) -> float | None:
    """Return the length of a sound (the default when none is given), or ``None`` for a model that sets its own length.

    Args:
        seconds: The length the user asked for, or ``None``.
        preset: The audio model.
        name: How the field is called in the message (``--seconds`` or ``seconds``).

    Raises:
        ValueError: If the length is not a number within ``MIN_SOUND_SECONDS`` to the model's ``max_seconds``.
    """
    if not preset.capabilities.duration:
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
