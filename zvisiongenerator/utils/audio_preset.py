"""Audio model presets: the ``audio_model_presets`` section of the config, parsed and validated into frozen values.

A preset is one audio model, keyed by its model id (``-m yue2``). It names what the model is (``kind``), what it can do
(``capabilities``), its own ceilings (``limits``), its model knobs (``settings``) and, per platform, the backend
family and the pinned files it runs from. The functions here are pure: they read a config ``dict`` and do no I/O.
"""

from __future__ import annotations

import copy
import re
from collections.abc import Mapping
from dataclasses import dataclass, fields
from types import MappingProxyType
from typing import Any, Final

from zvisiongenerator.utils.paths import split_huggingface_revision

__all__ = [
    "AUDIO_KINDS",
    "AUDIO_PLATFORMS",
    "AudioCapabilities",
    "AudioFiles",
    "AudioLimits",
    "AudioMemory",
    "AudioPlatform",
    "AudioPreset",
    "parse_audio_presets",
    "runnable_audio_presets",
]

AUDIO_KINDS: Final = ("music", "speech", "sound")
"""The kinds of audio a preset can make."""

AUDIO_PLATFORMS: Final = ("cuda", "mps")
"""The platform keys of a preset's ``platforms`` mapping: a CUDA card, or Apple MPS."""

_SECTION: Final = "audio_model_presets"
_COMMIT: Final = re.compile(r"[0-9a-f]{40}")

# The limits each kind must give, and may give: a limit another kind owns is an unknown key.
_LIMITS_BY_KIND: Final = {
    "music": ("max_seconds",),
    "sound": ("max_seconds",),
    "speech": ("max_new_tokens", "part_chars", "max_text_chars", "min_voice_seconds", "max_voice_seconds"),
}


@dataclass(frozen=True)
class AudioCapabilities:
    """What a model can do.

    Attributes:
        lyrics: The prompt is a style and the model also sings lyrics.
        duration: The user picks the length in seconds.
        voice_clone: The model can follow a voice recording.
    """

    lyrics: bool
    duration: bool
    voice_clone: bool


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
class AudioMemory:
    """The GPU memory a model needs, in GiB.

    Attributes:
        cuda_gb: The peak on a CUDA card.
        mps_gb: The peak on Apple MPS, or ``None`` while unmeasured.
    """

    cuda_gb: float
    mps_gb: float | None


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
class AudioPlatform:
    """How a preset runs on one platform.

    Attributes:
        family: The backend family, such as ``yue2`` (``get_audio_backend`` resolves it).
        files: The pinned files to download, in order.
        settings: The settings this platform overrides; see :meth:`AudioPreset.settings_for`.
    """

    family: str
    files: tuple[AudioFiles, ...]
    settings: Mapping[str, Any]


@dataclass(frozen=True)
class AudioPreset:
    """One audio model.

    Attributes:
        id: The model id, the key in ``audio_model_presets``.
        label: The name the Web UI shows.
        kind: ``music``, ``speech`` or ``sound``.
        capabilities: What the model can do.
        limits: The model's ceilings.
        settings: The model knobs every platform shares, an open mapping.
        memory: The GPU memory it needs.
        platforms: The platform keys (``cuda``, ``mps``) the model runs on, with how.
    """

    id: str
    label: str
    kind: str
    capabilities: AudioCapabilities
    limits: AudioLimits
    settings: Mapping[str, Any]
    memory: AudioMemory
    platforms: Mapping[str, AudioPlatform]

    def settings_for(self, platform_key: str) -> dict[str, Any]:
        """Return the settings that apply on *platform_key*: the shared ones, overridden by the platform's.

        Raises:
            ValueError: If the preset does not run on *platform_key*.
        """
        platform = self.platforms.get(platform_key)
        if platform is None:
            raise ValueError(f"Audio model '{self.id}' does not run on '{platform_key}'. It runs on: {', '.join(self.platforms)}.")
        return {**copy.deepcopy(dict(self.settings)), **copy.deepcopy(dict(platform.settings))}


def parse_audio_presets(config: Mapping[str, Any]) -> tuple[AudioPreset, ...]:
    """Parse and validate the ``audio_model_presets`` section of *config*, in the order it is written.

    Args:
        config: The loaded config. A missing section means no presets.

    Returns:
        One frozen preset per entry.

    Raises:
        ValueError: If the section or a preset is malformed: not a mapping, a missing or non-boolean capability, a
            missing limit for the preset's kind, an unknown kind or platform, or a platform without a family or files.
    """
    section = config.get(_SECTION)
    if section is None:
        return ()
    if not isinstance(section, Mapping):
        raise ValueError(f"config '{_SECTION}' must be a mapping of model id to preset, got {type(section).__name__}.")
    return tuple(_parse_preset(str(model_id), raw) for model_id, raw in section.items())


def runnable_audio_presets(config: Mapping[str, Any], platform_key: str) -> tuple[AudioPreset, ...]:
    """Return the presets that have an entry for *platform_key* (``cuda`` or ``mps``), in config order."""
    return tuple(preset for preset in parse_audio_presets(config) if platform_key in preset.platforms)


def _parse_preset(model_id: str, raw: object) -> AudioPreset:
    """Validate one preset and return it."""
    where = f"config '{_SECTION}.{model_id}'"
    mapping = _mapping(raw, where)
    for key in ("label", "kind", "capabilities", "limits", "memory", "platforms"):
        if key not in mapping:
            raise ValueError(f"{where} is missing the required key '{key}'.")
    kind = mapping["kind"]
    if kind not in AUDIO_KINDS:
        raise ValueError(f"{where}.kind must be one of {', '.join(AUDIO_KINDS)}, got {kind!r}.")
    return AudioPreset(
        id=model_id,
        label=_text(mapping["label"], f"{where}.label"),
        kind=kind,
        capabilities=_capabilities(mapping["capabilities"], f"{where}.capabilities"),
        limits=_limits(mapping["limits"], kind, f"{where}.limits"),
        settings=_settings(mapping.get("settings"), f"{where}.settings"),
        memory=_memory(mapping["memory"], f"{where}.memory"),
        platforms=MappingProxyType({key: _platform(value, f"{where}.platforms.{key}") for key, value in _platform_entries(mapping["platforms"], f"{where}.platforms")}),
    )


def _mapping(value: object, where: str) -> Mapping[str, Any]:
    """Return *value* when it is a mapping."""
    if not isinstance(value, Mapping):
        raise ValueError(f"{where} must be a mapping, got {type(value).__name__}.")
    return value


def _text(value: object, where: str) -> str:
    """Return *value* when it is a non-empty string."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{where} must be a non-empty string.")
    return value.strip()


def _only_keys(mapping: Mapping[str, Any], allowed: tuple[str, ...], where: str) -> None:
    """Raise ``ValueError`` when *mapping* has a key outside *allowed*."""
    for key in mapping:
        if key not in allowed:
            raise ValueError(f"{where}.{key} is not a known key (expected {', '.join(allowed)}).")


def _number(value: object, where: str, *, integer: bool = False) -> float | int:
    """Return *value* when it is a number above zero (an integer when *integer*)."""
    kinds = int if integer else (int, float)
    if isinstance(value, bool) or not isinstance(value, kinds) or value <= 0:
        raise ValueError(f"{where} must be {'an integer' if integer else 'a number'} above 0.")
    return value


def _capabilities(value: object, where: str) -> AudioCapabilities:
    """Parse a ``capabilities`` mapping: every capability, each a boolean."""
    caps = _mapping(value, where)
    names = tuple(field.name for field in fields(AudioCapabilities))
    _only_keys(caps, names, where)
    for name in names:
        if name not in caps:
            raise ValueError(f"{where}.{name} is required.")
        if not isinstance(caps[name], bool):
            raise ValueError(f"{where}.{name} must be true or false.")
    return AudioCapabilities(**caps)


def _limits(value: object, kind: str, where: str) -> AudioLimits:
    """Parse a ``limits`` mapping: the limits of *kind*, each required."""
    limits = _mapping(value, where)
    names = _LIMITS_BY_KIND[kind]
    _only_keys(limits, names, where)
    parsed: dict[str, float | int] = {}
    for name in names:
        if name not in limits:
            raise ValueError(f"{where}.{name} is required for a {kind} model.")
        parsed[name] = _number(limits[name], f"{where}.{name}", integer=name in ("max_new_tokens", "part_chars", "max_text_chars"))
    return AudioLimits(**parsed)


def _settings(value: object, where: str) -> Mapping[str, Any]:
    """Return a read-only copy of a ``settings`` mapping, an open mapping of model knobs; a missing one is empty."""
    if value is None:
        return MappingProxyType({})
    return MappingProxyType(copy.deepcopy(dict(_mapping(value, where))))


def _memory(value: object, where: str) -> AudioMemory:
    """Parse a ``memory`` mapping: ``cuda_gb`` and ``mps_gb`` (``null`` while unmeasured)."""
    memory = _mapping(value, where)
    _only_keys(memory, ("cuda_gb", "mps_gb"), where)
    for key in ("cuda_gb", "mps_gb"):
        if key not in memory:
            raise ValueError(f"{where}.{key} is required (use null for an unmeasured mps_gb).")
    mps = memory["mps_gb"]
    return AudioMemory(float(_number(memory["cuda_gb"], f"{where}.cuda_gb")), None if mps is None else float(_number(mps, f"{where}.mps_gb")))


def _platform_entries(value: object, where: str) -> list[tuple[str, Any]]:
    """Return the entries of a ``platforms`` mapping: at least one, each keyed by a known platform."""
    platforms = _mapping(value, where)
    if not platforms:
        raise ValueError(f"{where} must name at least one platform ({', '.join(AUDIO_PLATFORMS)}).")
    _only_keys(platforms, AUDIO_PLATFORMS, where)
    return list(platforms.items())


def _platform(value: object, where: str) -> AudioPlatform:
    """Parse one platform entry: a backend ``family``, the pinned ``files`` and the ``settings`` it overrides."""
    entry = _mapping(value, where)
    _only_keys(entry, ("family", "files", "settings"), where)
    for key in ("family", "files"):
        if key not in entry:
            raise ValueError(f"{where} is missing the required key '{key}'.")
    files = entry["files"]
    if not isinstance(files, list) or not files:
        raise ValueError(f"{where}.files must be a non-empty list of {{repo, files}} entries.")
    return AudioPlatform(
        family=_text(entry["family"], f"{where}.family"),
        files=tuple(_files(item, f"{where}.files[{index}]") for index, item in enumerate(files)),
        settings=_settings(entry.get("settings"), f"{where}.settings"),
    )


def _files(value: object, where: str) -> AudioFiles:
    """Parse one ``{repo: owner/name@commit, files: [...]}`` entry."""
    entry = _mapping(value, where)
    _only_keys(entry, ("repo", "files"), where)
    reference = _text(entry.get("repo"), f"{where}.repo")
    repo_id, revision = split_huggingface_revision(reference)
    if revision is None or not _COMMIT.fullmatch(revision):
        raise ValueError(f"{where}.repo must be pinned to a 40-character commit (owner/name@commit), got {reference!r}.")
    patterns = entry.get("files")
    if not isinstance(patterns, list) or not patterns or not all(isinstance(item, str) and item.strip() for item in patterns):
        raise ValueError(f"{where}.files must be a non-empty list of file names or patterns.")
    return AudioFiles(repo_id=repo_id, revision=revision, patterns=tuple(patterns))
