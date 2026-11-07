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
from zvisiongenerator.utils.audio_request import MIN_SOUND_SECONDS, auto_seconds, find_audio_preset
from zvisiongenerator.utils.licences import licence_payload, load_licences
from zvisiongenerator.utils.model_memory import MemoryBudget
from zvisiongenerator.web.model_status import NO_QUANTIZE_KEY

__all__ = ["audio_model_rows", "audio_rows", "find_audio_model", "runnable_audio_models"]

# The limits the Workspace reads, by name: the longest take, the size of the text, and the recordings a voice may be
# (a length-setting row adds the shortest and the default length).
_LIMITS = ("max_seconds", "max_text_chars", "min_voice_seconds", "max_voice_seconds")

type RowFields = Callable[[AudioPreset, dict[str, Any]], dict[str, Any]]
"""Adds a Models-page row's own keys to the shared ones of one audio model (the second argument)."""


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
        platform_key = _platform_key()
    except ValueError:
        return ()
    return tuple(preset.on_platform(platform_key) for preset in parse_audio_presets(config))


def find_audio_model(config: Mapping[str, Any], model: str | None) -> AudioPreset:
    """Return the audio preset *model* names, as this machine runs it.

    Raises:
        ValueError: If this machine runs no audio models, *model* is ``None`` (a form without its ``model`` field), or it
            names no preset; the message lists the ids.
    """
    platform_key = _platform_key()
    return find_audio_preset(config, model, missing="Field 'model' is required").on_platform(platform_key)


def _audio_entries(
    config: Mapping[str, Any],
    budget: MemoryBudget | None,
    target_present: Callable[[DownloadTarget], bool] | None,
) -> list[tuple[AudioPreset, dict[str, Any]]]:
    """Return each audio preset that runs on this machine, as this machine runs it, with the keys both pages share, in config order.

    The shared keys are ``label``, ``kinds``, ``licences``, ``downloaded`` (every repository of the plan is on disk) and
    ``memory_fit``. *target_present* defaults to the offline check.
    """
    try:
        platform_key = _platform_key()
    except ValueError:
        return []
    present = target_present or target_present_offline
    licences = load_licences()
    entries = []
    for preset in (item.on_platform(platform_key) for item in parse_audio_presets(config)):
        shared = {"label": preset.label, "kinds": list(preset.kinds), "licences": licence_payload(licences, preset.id), **_status(preset, platform_key, budget, present)}
        entries.append((preset, shared))
    return entries


def audio_rows(
    config: Mapping[str, Any],
    budget: MemoryBudget | None,
    fields: RowFields,
    *,
    target_present: Callable[[DownloadTarget], bool] | None = None,
) -> list[dict[str, Any]]:
    """Return one row per audio model that runs on this machine, in config order: the keys both pages share (see :func:`_audio_entries`), then *fields*' own."""
    return [{**shared, **fields(preset, shared)} for preset, shared in _audio_entries(config, budget, target_present)]


def audio_model_rows(
    config: Mapping[str, Any],
    budget: MemoryBudget | None,
    *,
    target_present: Callable[[DownloadTarget], bool] | None = None,
) -> list[dict[str, Any]]:
    """Return the Workspace rows of the audio models that run on this machine, in config order: one per (model, kind), in the model's kind order."""
    return [_workspace_row(preset, shared, kind) for preset, shared in _audio_entries(config, budget, target_present) for kind in preset.kinds]


def _workspace_row(preset: AudioPreset, shared: dict[str, Any], kind: str) -> dict[str, Any]:
    """Return the Workspace row of *preset* as a *kind* model: the shared keys (the model's ``kinds`` become this row's ``kind``), the id, the limits the form reads, the controls it hides and, for speech, the built-in voices.

    A length-setting row also carries the shortest and the default length; a row whose kind takes lyrics carries the section list an empty Lyrics box stands for.
    """
    limits: dict[str, float | int] = {name: value for name in _LIMITS if (value := getattr(preset.limits, name)) is not None}
    if preset.sets_length(kind):
        limits["min_seconds"] = MIN_SOUND_SECONDS
        limits["default_seconds"] = auto_seconds(kind)
    row: dict[str, Any] = {key: value for key, value in shared.items() if key != "kinds"}
    row.update({"id": preset.id, "type": "audio", "kind": kind, "limits": limits, "hidden_controls": _hidden_controls(preset, kind), "instrumental_sections": _instrumental_sections(preset, kind)})
    if kind == "speech":
        row["voices"] = [{"id": voice.id, "label": voice.label, "language": voice.language} for voice in preset.voices]
        row["default_voice"] = preset.default_voice
    else:
        row.pop("voice_download", None)
    return row


def _hidden_controls(preset: AudioPreset, kind: str) -> list[str]:
    """Return the Workspace controls of the *kind* workflow that this model does not have: a music model without lyrics has no Lyrics box, and one that follows its lyrics has no length."""
    if kind != "music":
        return []
    return [control for control, has in (("lyrics", preset.takes_lyrics(kind)), ("length", preset.sets_length(kind))) if not has]


def _instrumental_sections(preset: AudioPreset, kind: str) -> str | None:
    """Return the section list an empty Lyrics box stands for (the ``instrumental_sections`` setting of a (model, kind) that takes lyrics), or ``None``."""
    sections = preset.settings.get("instrumental_sections") if preset.takes_lyrics(kind) else None
    return sections if isinstance(sections, str) else None


def _status(preset: AudioPreset, platform_key: str, budget: MemoryBudget | None, present: Callable[[DownloadTarget], bool]) -> dict[str, Any]:
    """Return the model's ``downloaded`` and ``memory_fit``; unknown (``None``) when they cannot be determined, since one model must not break the listing.

    ``memory_fit`` is the model's measured peak on this platform against *budget*, in the shape of an image model's
    (``none`` is its only ``by_quantize`` key); ``None`` without a budget or a measured peak. ``downloaded`` covers the model's
    ``files`` only; a speech model also gets ``voice_download``: ``{"downloaded": bool}`` for its cloning files, or ``None`` when it has none.
    """
    try:
        fit = audio_memory_fit(preset, platform_key, budget)
        memory_fit = None if fit is None else {"kind": fit["kind"], "budget_gb": fit["budget_gb"], "by_quantize": {NO_QUANTIZE_KEY: {"status": fit["status"], "required_gb": fit["required_gb"]}}}
        status: dict[str, Any] = {"downloaded": all(present(target) for target in audio_plan(preset)), "memory_fit": memory_fit}
        if "speech" in preset.kinds:
            status["voice_download"] = {"downloaded": all(present(DownloadTarget(item.repo_id, item.revision, item.patterns)) for item in preset.voice_files)} if preset.voice_files else None
        return status
    except Exception as exc:  # noqa: BLE001 - one unreadable model must not break the whole listing
        warnings.warn(f"Could not determine status for audio model '{preset.id}': {exc}", stacklevel=2)
        return {"downloaded": None, "memory_fit": None, **({"voice_download": None} if "speech" in preset.kinds else {})}
