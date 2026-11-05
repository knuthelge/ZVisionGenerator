"""Audio model presets: the ``audio_model_presets`` section of the config, parsed and validated into frozen values.

A preset is one audio model, keyed by its model id (``-m yue2``). It names what the model is (``kind``), the backend
family and the pinned files it runs from, its own ceilings (``limits``) and its model knobs (``settings``); a
``platforms`` entry per accelerator (``cuda``, ``mps``) overrides settings, records the measured memory peak, and may name
its own backend family and files. What a model can do follows from its kind: music takes lyrics, sound takes a length,
speech follows a voice. The functions here are pure: they read a config ``dict`` and do no I/O.
"""

from __future__ import annotations

import copy
import re
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, Final

from zvisiongenerator.utils.paths import split_huggingface_revision

__all__ = [
    "AUDIO_KINDS",
    "AUDIO_PLATFORMS",
    "AudioFiles",
    "AudioLimits",
    "AudioPlatform",
    "AudioPreset",
    "parse_audio_presets",
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
class AudioPlatform:
    """What a preset overrides or records for one platform.

    Attributes:
        memory_gb: The GPU memory the model peaked at when measured on this platform, in GiB, or ``None`` while unmeasured.
        settings: The settings this platform overrides; see :meth:`AudioPreset.settings_for`.
        family: The backend family this platform runs the model with, or ``None`` for the preset's own.
        files: The pinned files this platform downloads instead of the preset's, or ``None`` for the preset's own.
    """

    memory_gb: float | None = None
    settings: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    family: str | None = None
    files: tuple[AudioFiles, ...] | None = None


@dataclass(frozen=True)
class AudioPreset:
    """One audio model.

    Attributes:
        id: The model id, the key in ``audio_model_presets``.
        label: The name the Web UI shows.
        kind: ``music``, ``speech`` or ``sound``.
        family: The backend family, such as ``yue2`` (``get_audio_backend`` resolves it).
        files: The pinned files to download, in order.
        limits: The model's ceilings.
        settings: The model knobs every platform shares, an open mapping.
        platforms: The platform keys (``cuda``, ``mps``) the preset says something about; the others use the shared values.
    """

    id: str
    label: str
    kind: str
    family: str
    files: tuple[AudioFiles, ...]
    limits: AudioLimits
    settings: Mapping[str, Any]
    platforms: Mapping[str, AudioPlatform]

    @property
    def takes_lyrics(self) -> bool:
        """Whether the prompt is a style and the model also sings lyrics (music)."""
        return self.kind == "music"

    @property
    def sets_length(self) -> bool:
        """Whether the user picks the length in seconds (sound); music takes its length from the lyrics and speech from the text."""
        return self.kind == "sound"

    @property
    def follows_voice(self) -> bool:
        """Whether the model can follow a voice recording (speech)."""
        return self.kind == "speech"

    def settings_for(self, platform_key: str) -> dict[str, Any]:
        """Return the settings that apply on *platform_key* (``cuda`` or ``mps``): the shared ones, overridden by the platform's."""
        override = self.platforms[platform_key].settings if platform_key in self.platforms else {}
        return {**copy.deepcopy(dict(self.settings)), **copy.deepcopy(dict(override))}

    def on_platform(self, platform_key: str) -> AudioPreset:
        """Return this preset as *platform_key* (``cuda`` or ``mps``) runs it: with that platform's ``family`` and ``files`` when it names them."""
        platform = self.platforms.get(platform_key)
        if platform is None or (platform.family is None and platform.files is None):
            return self
        return replace(self, family=platform.family or self.family, files=platform.files or self.files)

    def memory_gb_for(self, platform_key: str) -> float | None:
        """Return the GPU peak measured on *platform_key* in GiB, or ``None`` while unmeasured there."""
        return self.platforms[platform_key].memory_gb if platform_key in self.platforms else None


_cache: tuple[Mapping[str, Any], tuple[AudioPreset, ...]] | None = None


def parse_audio_presets(config: Mapping[str, Any]) -> tuple[AudioPreset, ...]:
    """Parse and validate the ``audio_model_presets`` section of *config*, in the order it is written.

    The last result is kept and returned again while the section is equal to the one it was parsed from, so the callers
    that ask for the presets per request, per row or per gallery asset parse them once.

    Args:
        config: The loaded config. A missing section means no presets.

    Returns:
        One frozen preset per entry.

    Raises:
        ValueError: If the section or a preset is malformed: not a mapping, a missing or unknown key, a missing limit
            for the preset's kind, an unknown kind or platform, or files that are not pinned to a commit.
    """
    global _cache
    section = config.get(_SECTION)
    if section is None:
        return ()
    cached = _cache
    if cached is not None and cached[0] == section:
        return cached[1]
    if not isinstance(section, Mapping):
        raise ValueError(f"config '{_SECTION}' must be a mapping of model id to preset, got {type(section).__name__}.")
    presets = tuple(_parse_preset(str(model_id), raw) for model_id, raw in section.items())
    _cache = (copy.deepcopy(section), presets)
    return presets


def _parse_preset(model_id: str, raw: object) -> AudioPreset:
    """Validate one preset and return it."""
    where = f"config '{_SECTION}.{model_id}'"
    mapping = _mapping(raw, where)
    _only_keys(mapping, ("label", "kind", "family", "files", "limits", "settings", "platforms"), where)
    for key in ("label", "kind", "family", "files", "limits"):
        if key not in mapping:
            raise ValueError(f"{where} is missing the required key '{key}'.")
    kind = mapping["kind"]
    if kind not in AUDIO_KINDS:
        raise ValueError(f"{where}.kind must be one of {', '.join(AUDIO_KINDS)}, got {kind!r}.")
    return AudioPreset(
        id=model_id,
        label=_text(mapping["label"], f"{where}.label"),
        kind=kind,
        family=_text(mapping["family"], f"{where}.family"),
        files=_file_list(mapping["files"], f"{where}.files"),
        limits=_limits(mapping["limits"], kind, f"{where}.limits"),
        settings=_settings(mapping.get("settings"), f"{where}.settings"),
        platforms=_platforms(mapping.get("platforms"), f"{where}.platforms"),
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


def _platforms(value: object, where: str) -> Mapping[str, AudioPlatform]:
    """Parse a ``platforms`` mapping (optional): each key a known platform, each entry its ``memory_gb`` and the ``settings`` it overrides."""
    if value is None:
        return MappingProxyType({})
    platforms = _mapping(value, where)
    _only_keys(platforms, AUDIO_PLATFORMS, where)
    return MappingProxyType({key: _platform(entry, f"{where}.{key}") for key, entry in platforms.items()})


def _platform(value: object, where: str) -> AudioPlatform:
    """Parse one platform entry: the measured ``memory_gb`` peak, the ``settings`` it overrides, and its own ``family`` and ``files``, each optional."""
    entry = _mapping(value, where)
    _only_keys(entry, ("memory_gb", "settings", "family", "files"), where)
    memory = entry.get("memory_gb")
    return AudioPlatform(
        memory_gb=None if memory is None else float(_number(memory, f"{where}.memory_gb")),
        settings=_settings(entry.get("settings"), f"{where}.settings"),
        family=None if entry.get("family") is None else _text(entry["family"], f"{where}.family"),
        files=None if entry.get("files") is None else _file_list(entry["files"], f"{where}.files"),
    )


def _file_list(value: object, where: str) -> tuple[AudioFiles, ...]:
    """Parse the ``files`` list: at least one ``{repo, files}`` entry."""
    if not isinstance(value, list) or not value:
        raise ValueError(f"{where} must be a non-empty list of {{repo, files}} entries.")
    return tuple(_files(item, f"{where}[{index}]") for index, item in enumerate(value))


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
