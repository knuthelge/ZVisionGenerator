"""The audio models the Web UI offers and the rows that describe them to the Workspace and the Models page."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping
from typing import Any

from zvisiongenerator.backends import audio_platform_key
from zvisiongenerator.downloads.plans import audio_plan
from zvisiongenerator.downloads.presence import target_present_offline
from zvisiongenerator.downloads.target import DownloadTarget
from zvisiongenerator.utils.audio_memory import audio_memory_fit
from zvisiongenerator.utils.audio_preset import AudioPreset, parse_audio_presets
from zvisiongenerator.utils.audio_request import DEFAULT_SOUND_SECONDS, MIN_SOUND_SECONDS, find_audio_preset
from zvisiongenerator.utils.licences import licence_payload, load_licences
from zvisiongenerator.utils.model_memory import MemoryBudget
from zvisiongenerator.web.model_status import NO_QUANTIZE_KEY

__all__ = ["audio_model_rows", "audio_rows", "find_audio_model", "runnable_audio_models"]

# The limits the Workspace reads, by name: the length of a sound, the size of the text, and the recordings a voice may be
# (a sound model adds the shortest and the default length, which are the same for every model).
_LIMITS = ("max_seconds", "max_text_chars", "min_voice_seconds", "max_voice_seconds")

type RowFields = Callable[[AudioPreset, dict[str, Any]], dict[str, Any]]
"""Adds a kind of row's own keys to the shared ones of one audio model (the second argument)."""


def _platform_key() -> str:
    """Return the key this machine runs audio models under.

    Raises:
        ValueError: If the machine has neither CUDA nor MPS.
    """
    try:
        return audio_platform_key()
    except RuntimeError as exc:
        raise ValueError(str(exc)) from exc


def runnable_audio_models(config: Mapping[str, Any]) -> tuple[AudioPreset, ...]:
    """Return the configured audio presets, in config order; a machine with neither CUDA nor MPS runs none."""
    try:
        _platform_key()
    except ValueError:
        return ()
    return parse_audio_presets(config)


def find_audio_model(config: Mapping[str, Any], model: str | None) -> AudioPreset:
    """Return the audio preset *model* names.

    Raises:
        ValueError: If this machine runs no audio models, *model* is ``None`` (a form without its ``model`` field), or it
            names no preset; the message lists the ids.
    """
    _platform_key()
    return find_audio_preset(config, model, missing="Field 'model' is required")


def audio_rows(
    config: Mapping[str, Any],
    budget: MemoryBudget | None,
    fields: RowFields,
    *,
    target_present: Callable[[DownloadTarget], bool] | None = None,
) -> list[dict[str, Any]]:
    """Return one row per audio model that runs on this machine, in config order: the keys both pages share, then *fields*' own.

    The shared keys are ``label``, ``kind``, ``licences``, ``downloaded`` (every repository of the plan is on disk) and
    ``memory_fit``. *target_present* defaults to the offline check.
    """
    try:
        platform_key = _platform_key()
    except ValueError:
        return []
    present = target_present or target_present_offline
    licences = load_licences()
    rows = []
    for preset in parse_audio_presets(config):
        shared = {"label": preset.label, "kind": preset.kind, "licences": licence_payload(licences, preset.id), **_status(preset, platform_key, budget, present)}
        rows.append({**shared, **fields(preset, shared)})
    return rows


def audio_model_rows(
    config: Mapping[str, Any],
    budget: MemoryBudget | None,
    *,
    target_present: Callable[[DownloadTarget], bool] | None = None,
) -> list[dict[str, Any]]:
    """Return one Workspace row per audio model that runs on this machine, in config order."""
    return audio_rows(config, budget, _workspace_fields, target_present=target_present)


def _workspace_fields(preset: AudioPreset, _shared: dict[str, Any]) -> dict[str, Any]:
    """Return the keys only the Workspace row has: the id, the limits the form reads and the section list an empty Lyrics box stands for."""
    limits: dict[str, float | int] = {name: value for name in _LIMITS if (value := getattr(preset.limits, name)) is not None}
    if preset.sets_length:
        limits["min_seconds"] = MIN_SOUND_SECONDS
        limits["default_seconds"] = DEFAULT_SOUND_SECONDS
    return {"id": preset.id, "type": "audio", "limits": limits, "instrumental_sections": _instrumental_sections(preset)}


def _instrumental_sections(preset: AudioPreset) -> str | None:
    """Return the section list an empty Lyrics box stands for (a music model's ``instrumental_sections`` setting), or ``None``."""
    sections = preset.settings.get("instrumental_sections") if preset.takes_lyrics else None
    return sections if isinstance(sections, str) else None


def _status(preset: AudioPreset, platform_key: str, budget: MemoryBudget | None, present: Callable[[DownloadTarget], bool]) -> dict[str, Any]:
    """Return the model's ``downloaded`` and ``memory_fit``; unknown (``None``) when they cannot be determined, since one model must not break the listing.

    ``memory_fit`` is the model's measured peak on this platform against *budget*, in the shape of an image model's
    (``none`` is its only ``by_quantize`` key); ``None`` without a budget or a measured peak.
    """
    try:
        fit = audio_memory_fit(preset, platform_key, budget)
        memory_fit = None if fit is None else {"kind": fit["kind"], "budget_gb": fit["budget_gb"], "by_quantize": {NO_QUANTIZE_KEY: {"status": fit["status"], "required_gb": fit["required_gb"]}}}
        return {"downloaded": all(present(target) for target in audio_plan(preset)), "memory_fit": memory_fit}
    except Exception as exc:  # noqa: BLE001 - one unreadable model must not break the whole listing
        warnings.warn(f"Could not determine status for audio model '{preset.id}': {exc}", stacklevel=2)
        return {"downloaded": None, "memory_fit": None}
