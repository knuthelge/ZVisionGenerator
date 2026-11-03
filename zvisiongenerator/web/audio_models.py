"""The audio models the Web UI offers: the presets that run on this machine, and the payload of each for the Workspace."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from zvisiongenerator.backends import audio_platform_key
from zvisiongenerator.downloads.presence import target_present_offline
from zvisiongenerator.downloads.target import DownloadTarget
from zvisiongenerator.utils.audio_preset import AudioPreset, runnable_audio_presets
from zvisiongenerator.utils.audio_request import DEFAULT_SOUND_SECONDS, MIN_SOUND_SECONDS
from zvisiongenerator.utils.licences import LicenceRecord, licence_payload, load_licences
from zvisiongenerator.utils.model_memory import MemoryBudget
from zvisiongenerator.web.model_status import describe_audio_status
from zvisiongenerator.web.workspace_contract import hidden_audio_controls

__all__ = ["audio_model_rows", "runnable_audio_models"]

# The limits the Workspace reads, by name: the length of a sound, the size of the text, and the recordings a voice may be.
_LIMITS = ("max_seconds", "max_text_chars", "min_voice_seconds", "max_voice_seconds")


def runnable_audio_models(config: Mapping[str, Any]) -> tuple[tuple[AudioPreset, ...], str | None]:
    """Return the audio presets that run on this machine, in config order, with the platform key they run under.

    A machine with neither CUDA nor MPS runs none and has no key (``None``).
    """
    try:
        platform_key = audio_platform_key()
    except RuntimeError:
        return (), None
    return runnable_audio_presets(config, platform_key), platform_key


def audio_model_rows(
    config: Mapping[str, Any],
    budget: MemoryBudget | None,
    *,
    target_present: Callable[[DownloadTarget], bool] | None = None,
) -> list[dict[str, Any]]:
    """Return one Workspace row per audio model that runs on this machine, in config order.

    A row holds the model's ``id``, ``label``, ``kind`` (``music``, ``speech`` or ``sound``), ``type`` (``audio``),
    ``capabilities``, ``limits`` (the ones the Workspace reads, for the kind), ``default_seconds`` (the length of a sound
    when none is chosen; ``None`` for a model that sets its own), ``instrumental_sections`` (the section list an empty Lyrics box
    stands for, for a music model; ``None`` for the others), ``hidden_controls`` (the workflow controls the model's
    capabilities hide), ``licences``, ``downloaded`` and ``memory_fit`` (see :func:`describe_audio_status`).

    Args:
        config: The loaded config.
        budget: This machine's memory budget, or ``None`` when it is not estimated.
        target_present: Tells whether one download target is fully on disk; the offline cache check by default.
    """
    presets, platform_key = runnable_audio_models(config)
    if platform_key is None:
        return []
    present = target_present or target_present_offline
    licences = load_licences()
    return [_row(licences, preset, platform_key, budget, present) for preset in presets]


def _row(licences: Sequence[LicenceRecord], preset: AudioPreset, platform_key: str, budget: MemoryBudget | None, target_present: Callable[[DownloadTarget], bool]) -> dict[str, Any]:
    """Build the Workspace row of one audio model."""
    caps = preset.capabilities
    limits: dict[str, float | int] = {name: value for name in _LIMITS if (value := getattr(preset.limits, name)) is not None}
    if caps.duration:
        limits["min_seconds"] = MIN_SOUND_SECONDS
    return {
        "id": preset.id,
        "label": preset.label,
        "kind": preset.kind,
        "type": "audio",
        "capabilities": {"lyrics": caps.lyrics, "duration": caps.duration, "voice_clone": caps.voice_clone},
        "limits": limits,
        "default_seconds": DEFAULT_SOUND_SECONDS if caps.duration else None,
        "instrumental_sections": _instrumental_sections(preset),
        "hidden_controls": hidden_audio_controls(caps),
        "licences": licence_payload(licences, preset.id),
        **_status(preset, platform_key, budget, target_present),
    }


def _instrumental_sections(preset: AudioPreset) -> str | None:
    """Return the section list an empty Lyrics box stands for (a music model's ``instrumental_sections`` setting), or ``None``."""
    sections = preset.settings.get("instrumental_sections") if preset.kind == "music" else None
    return sections if isinstance(sections, str) else None


def _status(preset: AudioPreset, platform_key: str, budget: MemoryBudget | None, target_present: Callable[[DownloadTarget], bool]) -> dict[str, Any]:
    """Return the model's ``downloaded`` and ``memory_fit``; unknown (``None``) when they cannot be determined, since one model must not break the listing."""
    try:
        return describe_audio_status(preset, platform_key, budget=budget, target_present=target_present)
    except Exception as exc:  # noqa: BLE001 - one unreadable model must not break the whole listing
        warnings.warn(f"Could not determine status for audio model '{preset.id}': {exc}", stacklevel=2)
        return {"downloaded": None, "memory_fit": None}
