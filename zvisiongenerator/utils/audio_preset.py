"""Audio model presets: the audio models of the model metadata, joined with the settings of the config, as frozen values.

A preset is one audio model, keyed by its model id (``-m yue2``). The metadata (the ``audio`` entries of ``models``) names
the kinds of audio the model makes (``kinds``, the first one the default), the backend family and the pinned files it runs
from, its own ceilings (``limits``) and, per accelerator (``cuda``, ``mps``), the measured memory peak and an optional backend
family. The config (``audio_model_presets.<id>``) holds the model knobs (``settings``), shared or overridden per accelerator.
A speech model may list built-in ``voices``, the ``voice_files`` that cloning a recording needs on top of ``files`` and the
``design_files`` that designing a voice from a description needs. What a
model can do is decided per (model, kind): speech follows a voice, sound takes a length, and music takes a length unless the
model sings lyrics (``lyrics: true``), in which case the lyrics set it. The functions here read a config
``dict`` and the packaged metadata (parsed once) and touch nothing else: no other file, no network.
"""

from __future__ import annotations

import copy
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, Final

from zvisiongenerator.downloads.declaration import model_targets, role_targets
from zvisiongenerator.downloads.target import DownloadTarget
from zvisiongenerator.utils.model_metadata import model_entry, model_ids

__all__ = [
    "AUDIO_KINDS",
    "AUDIO_PLATFORMS",
    "AudioFiles",
    "AudioLimits",
    "AudioPlatform",
    "AudioPreset",
    "AudioVoice",
    "parse_audio_presets",
]

AUDIO_KINDS: Final = ("music", "speech", "sound")
"""The kinds of audio a preset can make."""

AUDIO_PLATFORMS: Final = ("cuda", "mps")
"""The platform keys of a preset's ``platforms`` mapping: a CUDA card, or Apple MPS."""

_SECTION: Final = "audio_model_presets"
_AUDIO_KIND: Final = "audio"
_VOICE_ROLE: Final = "voice"
_DESIGN_ROLE: Final = "design"


@dataclass(frozen=True)
class AudioLimits:
    """The ceilings of a model; each kind sets only its own.

    Attributes:
        max_seconds: The longest take, in seconds (music and sound).
        max_new_tokens: The most tokens one part of speech generates (speech).
        part_chars: The most characters one part of speech reads (speech).
        max_text_chars: The most characters one take reads (speech).
        min_voice_seconds: The shortest voice recording, in seconds (speech).
        max_voice_seconds: The longest voice recording, in seconds (speech).
    """

    max_seconds: float | None = None
    max_new_tokens: int | None = None
    part_chars: int | None = None
    max_text_chars: int | None = None
    min_voice_seconds: float | None = None
    max_voice_seconds: float | None = None


@dataclass(frozen=True)
class AudioFiles:
    """The files of one Hugging Face repository at a pinned commit.

    Attributes:
        repo_id: The repository, ``owner/name``.
        revision: The 40-character commit.
        patterns: The file names or glob patterns to fetch.
    """

    repo_id: str
    revision: str
    patterns: tuple[str, ...]


@dataclass(frozen=True)
class AudioVoice:
    """One built-in voice of a speech model.

    Attributes:
        id: The id the model takes, such as ``ryan``.
        label: The name the Web UI shows.
        language: The voice's native language, as the model card names it.
    """

    id: str
    label: str
    language: str


@dataclass(frozen=True)
class AudioPlatform:
    """What a preset overrides or records for one platform.

    Attributes:
        memory_gb: The GPU memory the model peaked at when measured on this platform, in GiB, or ``None`` while unmeasured.
        settings: The settings this platform overrides; see :meth:`AudioPreset.settings_for`.
        family: The backend family this platform runs the model with, or ``None`` for the preset's own.
    """

    memory_gb: float | None = None
    settings: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    family: str | None = None


@dataclass(frozen=True)
class AudioPreset:
    """One audio model.

    Attributes:
        id: The model id, the key of its ``models`` entry and of its ``audio_model_presets`` settings.
        label: The name the Web UI shows.
        kinds: The kinds of audio the model makes (``music``, ``speech``, ``sound``), unique and not empty; the first is the default.
        family: The backend family, such as ``yue2`` (``get_audio_backend`` resolves it).
        files: The pinned files to download, in order.
        limits: The model's ceilings: the union of what each of its kinds sets.
        settings: The model knobs every platform shares, an open mapping.
        platforms: The platform keys (``cuda``, ``mps``) the preset says something about; the others use the shared values.
        voices: The built-in voices of a speech model, the first one the default; empty for a model without any.
        voice_files: The pinned files a take that follows a voice recording also needs (speech), downloaded only for such a take; empty when
            ``files`` already serve cloning.
        design_files: The pinned files that designing a voice from a description needs (speech), downloaded only when a voice is designed;
            empty when the model cannot design a voice.
        lyrics: Whether the model sings lyrics, so that its music takes lyrics and their sections set the length.
    """

    id: str
    label: str
    kinds: tuple[str, ...]
    family: str
    files: tuple[AudioFiles, ...]
    limits: AudioLimits
    settings: Mapping[str, Any]
    platforms: Mapping[str, AudioPlatform]
    voices: tuple[AudioVoice, ...] = ()
    voice_files: tuple[AudioFiles, ...] = ()
    design_files: tuple[AudioFiles, ...] = ()
    lyrics: bool = False

    @property
    def default_kind(self) -> str:
        """The kind a take is made as when none is named: the first of ``kinds``."""
        return self.kinds[0]

    @property
    def default_voice(self) -> str | None:
        """The id of the voice a speech take reads in when it names none: the first of ``voices``, or ``None`` without any."""
        return self.voices[0].id if self.voices else None

    def has_kind(self, kind: str) -> bool:
        """Return whether the model makes *kind*."""
        return kind in self.kinds

    def takes_lyrics(self, kind: str) -> bool:
        """Return whether the prompt of a *kind* take is a style and the model also sings lyrics (music of a model with ``lyrics``)."""
        return kind == "music" and self.lyrics and self.has_kind(kind)

    def sets_length(self, kind: str) -> bool:
        """Return whether the user picks the length in seconds for a *kind* take (sound, or music without lyrics); the lyrics set a song's length and the text a speech's."""
        return self.has_kind(kind) and (kind == "sound" or (kind == "music" and not self.lyrics))

    def follows_voice(self, kind: str) -> bool:
        """Return whether a *kind* take can follow a voice recording (speech)."""
        return kind == "speech" and self.has_kind(kind)

    def settings_for(self, platform_key: str) -> dict[str, Any]:
        """Return the settings that apply on *platform_key* (``cuda`` or ``mps``): the shared ones, overridden by the platform's."""
        override = self.platforms[platform_key].settings if platform_key in self.platforms else {}
        return {**copy.deepcopy(dict(self.settings)), **copy.deepcopy(dict(override))}

    def on_platform(self, platform_key: str) -> AudioPreset:
        """Return this preset as *platform_key* (``cuda`` or ``mps``) runs it: with that platform's ``family`` when it names one."""
        platform = self.platforms.get(platform_key)
        if platform is None or platform.family is None:
            return self
        return replace(self, family=platform.family)

    def memory_gb_for(self, platform_key: str) -> float | None:
        """Return the GPU peak measured on *platform_key* in GiB, or ``None`` while unmeasured there."""
        return self.platforms[platform_key].memory_gb if platform_key in self.platforms else None


def parse_audio_presets(config: Mapping[str, Any]) -> tuple[AudioPreset, ...]:
    """Return one frozen preset per audio model of the metadata, in the order the metadata lists them.

    A preset joins the model's facts (the ``models`` entry and its declared downloads) with its settings from
    ``audio_model_presets.<id>`` of *config*: ``settings``, and ``platforms.<cuda|mps>.settings`` per accelerator. A config id
    that names no audio model is ignored, and so is any other key of an entry. Nothing is written into *config* or the metadata.

    Args:
        config: The loaded config. A missing section means no settings.

    Returns:
        One frozen preset per audio model.

    Raises:
        ValueError: If a settings block is not a mapping, or a model that sings lyrics has no ``instrumental_sections`` setting.
    """
    section = config.get(_SECTION)
    if section is None:
        section = {}
    if not isinstance(section, Mapping):
        raise ValueError(f"config '{_SECTION}' must be a mapping of model id to settings, got {type(section).__name__}.")
    return tuple(_preset(model_id, section.get(model_id)) for model_id in model_ids(_AUDIO_KIND))


def _preset(model_id: str, raw: object) -> AudioPreset:
    """Build one preset from the metadata entry of *model_id* and its config block *raw* (``None`` when the config has none)."""
    where = f"config '{_SECTION}.{model_id}'"
    entry = model_entry(model_id)
    block = {} if raw is None else _mapping(raw, where)
    settings = _settings(block.get("settings"), f"{where}.settings")
    pairs = model_targets(model_id, sys.platform) or ()
    voice_files = tuple(_files(target) for target in role_targets(pairs, _VOICE_ROLE))
    design_files = tuple(_files(target) for target in role_targets(pairs, _DESIGN_ROLE))
    files = tuple(_files(target) for role, target in pairs if role not in (_VOICE_ROLE, _DESIGN_ROLE))
    kinds = tuple(entry["kinds"])
    lyrics = bool(entry.get("lyrics", False))
    if lyrics:
        _require_sections(settings, where)
    return AudioPreset(
        id=model_id,
        label=entry["name"],
        kinds=kinds,
        family=entry["family"],
        files=files,
        limits=AudioLimits(**entry["limits"]),
        settings=settings,
        platforms=_platforms(entry.get("platforms"), block.get("platforms"), where),
        voices=tuple(AudioVoice(voice["id"], voice["label"], voice["language"]) for voice in entry.get("voices", ())),
        voice_files=voice_files,
        design_files=design_files,
        lyrics=lyrics,
    )


def _require_sections(settings: Mapping[str, Any], where: str) -> None:
    """Raise ``ValueError`` unless *settings* has the ``instrumental_sections`` setting a model that sings lyrics needs: the section list an empty Lyrics box stands for."""
    sections = settings.get("instrumental_sections")
    if not isinstance(sections, str) or not sections.strip():
        raise ValueError(f"{where}.settings.instrumental_sections is required for a model that sings lyrics: the section list an empty Lyrics box stands for.")


def _mapping(value: object, where: str) -> Mapping[str, Any]:
    """Return *value* when it is a mapping."""
    if not isinstance(value, Mapping):
        raise ValueError(f"{where} must be a mapping, got {type(value).__name__}.")
    return value


def _settings(value: object, where: str) -> Mapping[str, Any]:
    """Return a read-only copy of a ``settings`` mapping, an open mapping of model knobs; a missing one is empty."""
    if value is None:
        return MappingProxyType({})
    return MappingProxyType(copy.deepcopy(dict(_mapping(value, where))))


def _platforms(facts: object, configured: object, where: str) -> Mapping[str, AudioPlatform]:
    """Join the measured ``platforms`` facts of an entry with the ``platforms`` settings of its config block.

    The result has a key for each accelerator (``cuda``, ``mps``) either of them names.
    """
    configured_platforms = {} if configured is None else _mapping(configured, f"{where}.platforms")
    facts_by_platform = facts or {}
    keys = [key for key in AUDIO_PLATFORMS if key in facts_by_platform or key in configured_platforms]
    return MappingProxyType({key: _platform(facts_by_platform.get(key) or {}, configured_platforms.get(key), f"{where}.platforms.{key}") for key in keys})


def _platform(facts: Mapping[str, Any], configured: object, where: str) -> AudioPlatform:
    """Build one platform entry: the measured ``memory_gb`` and own ``family`` of the metadata, the ``settings`` of the config."""
    block = {} if configured is None else _mapping(configured, where)
    memory = facts.get("memory_gb")
    return AudioPlatform(
        memory_gb=None if memory is None else float(memory),
        settings=_settings(block.get("settings"), f"{where}.settings"),
        family=facts.get("family"),
    )


def _files(target: DownloadTarget) -> AudioFiles:
    """Return the audio files a declared download target names."""
    return AudioFiles(repo_id=target.repo_id, revision=target.revision, patterns=target.allow_patterns)
