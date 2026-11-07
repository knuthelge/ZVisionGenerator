"""Load and validate the bench cases from YAML."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from zvisiongenerator.utils.audio_writers import (
    FILL_FIELDS,
    FillSettings,
    LyricsSettings,
    ScriptSettings,
    check_fill_request,
    check_lyrics_request,
    check_script_request,
    fill_settings_from_mapping,
    lyrics_settings_from_mapping,
    script_settings_from_mapping,
)
from zvisiongenerator.utils.prompt_enhance import EnhanceSettings, settings_from_mapping, validate_settings
from zvisiongenerator.utils.prompt_enhance_audio import AudioEnhanceContext, AudioEnhanceSettings, audio_settings_from_mapping

CASES_PATH = Path(__file__).with_name("cases.yaml")
AUDIO_MODES = ("audio-sound", "audio-style", "audio-lyrics", "audio-script", "audio-fill")
CASE_MODES = ("image", "video", "caption", *AUDIO_MODES)
CONTEXT_KEYS = frozenset({"ratio", "width", "height", "frames", "fps", "audio", "reference_image", "extends", "previous", "extract_from", "new_frames", "prompt_is_original"})
AUDIO_CONTEXT_KEYS = frozenset({"lyrics", "instrumental", "tempo", "key", "seconds", "sound_type"})
LYRICS_CONTEXT_KEYS = frozenset({"style"})
SCRIPT_CONTEXT_KEYS: frozenset[str] = frozenset()
FILL_CONTEXT_KEYS = frozenset({"style", "lyrics", "text", "tempo", "key"})
_SOUND_TYPES = ("auto", "ambience", "effect", "one-shot")
# Seeds are at least eight digits and stable per case id, so reordering or adding cases never changes another case's seeds.
_SEED_FLOOR = 10_000_000
_SEED_SPAN = 900_000_000
_CASE_KEYS = frozenset({"id", "tags", "mode", "prompt", "scene", "settings", "context", "family", "keep", "count", "forbid"})


@dataclass(frozen=True)
class Case:
    """One bench case: the input, its enhance settings and job context, and what the output must keep or avoid."""

    id: str
    tags: tuple[str, ...]
    mode: str
    prompt: str
    scene: str
    settings: EnhanceSettings | AudioEnhanceSettings | LyricsSettings | ScriptSettings | FillSettings
    raw_settings: dict[str, Any]
    context: dict[str, Any]
    family: str | None
    keep: tuple[str, ...]
    count: str | None
    forbid: tuple[str, ...]
    audio_context: AudioEnhanceContext | None = None

    @property
    def media(self) -> str:
        """Return the enhancer mode the case runs in: ``audio``, ``video`` or ``image`` (captions are images)."""
        if self.is_audio:
            return "audio"
        return "video" if self.mode == "video" else "image"

    @property
    def is_audio(self) -> bool:
        """Return whether the case rewrites audio text (a sound, a music style, a song's lyrics, a speech script or a fill from an idea)."""
        return self.mode in AUDIO_MODES

    @property
    def task(self) -> str:
        """Return the audio task of the case: ``sound``, ``style``, ``lyrics``, ``script`` or ``fill``."""
        return self.mode.removeprefix("audio-")

    @property
    def source(self) -> str:
        """Return the text the rewrite works from: the prompt."""
        return self.prompt.strip()


def case_seeds(case_id: str, count: int) -> tuple[int, ...]:
    """Return *count* consecutive seeds for *case_id*, starting at a stable eight- or nine-digit base."""
    digest = int.from_bytes(hashlib.sha256(case_id.encode()).digest()[:8], "big")
    base = _SEED_FLOOR + digest % _SEED_SPAN
    return tuple(base + offset for offset in range(count))


def load_cases(path: Path = CASES_PATH) -> tuple[Case, ...]:
    """Load every case from *path*.

    Raises:
        ValueError: Naming the case when one is malformed or two share an id.
    """
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    entries = data.get("cases")
    if not isinstance(entries, list):
        raise ValueError(f"{path.name}: expected a top-level 'cases' list.")
    cases = tuple(parse_case(entry) for entry in entries)
    seen: set[str] = set()
    for case in cases:
        if case.id in seen:
            raise ValueError(f"Duplicate case id '{case.id}'.")
        seen.add(case.id)
    return cases


def parse_case(entry: Any) -> Case:
    """Build one case from its YAML mapping.

    Raises:
        ValueError: For unknown keys, a bad mode, invalid settings or context, or no text to enhance.
    """
    if not isinstance(entry, dict) or not isinstance(entry.get("id"), str):
        raise ValueError(f"Every case needs a mapping with a string 'id', got {entry!r}.")
    case_id = entry["id"]
    unknown = sorted(set(entry) - _CASE_KEYS)
    if unknown:
        raise ValueError(f"Case '{case_id}': unknown key(s) {', '.join(unknown)}.")
    mode = entry.get("mode", "image")
    if mode not in CASE_MODES:
        raise ValueError(f"Case '{case_id}': mode must be one of {', '.join(CASE_MODES)}.")
    audio = mode in AUDIO_MODES
    allowed_context = {"audio-lyrics": LYRICS_CONTEXT_KEYS, "audio-script": SCRIPT_CONTEXT_KEYS, "audio-fill": FILL_CONTEXT_KEYS}.get(mode, AUDIO_CONTEXT_KEYS) if audio else CONTEXT_KEYS
    context = entry.get("context") or {}
    if not isinstance(context, dict) or set(context) - allowed_context:
        raise ValueError(f"Case '{case_id}': context keys must be among {', '.join(sorted(allowed_context))}.")
    if audio and not isinstance(entry.get("family"), str):
        raise ValueError(f"Case '{case_id}': an audio case needs a family (the model whose caption rules it runs).")
    prompt = str(entry.get("prompt") or "")
    scene = str(entry.get("scene") or "")
    if mode != "video" and scene:
        raise ValueError(f"Case '{case_id}': only video cases take a scene.")
    if not (prompt.strip() or scene.strip() or (mode == "video" and context.get("extends"))):
        raise ValueError(f"Case '{case_id}': needs a prompt or a scene.")
    raw_settings = dict(entry.get("settings") or {})
    media = "video" if mode == "video" else "image"
    audio_context = None
    try:
        if mode == "audio-lyrics":
            settings = lyrics_settings_from_mapping(raw_settings)
            check_lyrics_request(prompt, settings)
            audio_context = _audio_context(context)
        elif mode == "audio-fill":
            settings = fill_settings_from_mapping(raw_settings, kind=_fill_kind(raw_settings))
            check_fill_request(prompt)
            audio_context = _audio_context(context)
        elif mode == "audio-script":
            settings = script_settings_from_mapping(raw_settings)
            check_script_request(prompt)
            audio_context = _audio_context(context)
        elif audio:
            settings = audio_settings_from_mapping(raw_settings, task=mode.removeprefix("audio-"))
            audio_context = _audio_context(context)
        else:
            settings = settings_from_mapping(raw_settings, mode=media)
            validate_settings(settings, mode=media)
    except ValueError as exc:
        raise ValueError(f"Case '{case_id}': {exc}") from exc
    return Case(
        id=case_id,
        tags=tuple(str(tag) for tag in entry.get("tags") or ()),
        mode=mode,
        prompt=prompt,
        scene=scene,
        settings=settings,
        raw_settings=raw_settings,
        context=dict(context),
        family=entry.get("family"),
        keep=tuple(str(item) for item in entry.get("keep") or ()),
        count=entry.get("count"),
        forbid=tuple(str(item) for item in entry.get("forbid") or ()),
        audio_context=audio_context,
    )


def _fill_kind(raw_settings: dict[str, Any]) -> str:
    """Return the kind a fill case runs for: speech when it asks for a Vibe or a Text, else music."""
    fields = raw_settings.get("fields")
    speech = isinstance(fields, list) and any(field in FILL_FIELDS["speech"] for field in fields)
    return "speech" if speech else "music"


def _audio_context(context: dict[str, Any]) -> AudioEnhanceContext:
    """Return the audio rewrite context for a case's ``context`` mapping.

    Raises:
        ValueError: For a value of the wrong type or an unknown ``sound_type``.
    """
    for key in ("lyrics", "style", "key"):
        if key in context and not isinstance(context[key], str):
            raise ValueError(f"context '{key}' must be a string.")
    if "instrumental" in context and not isinstance(context["instrumental"], bool):
        raise ValueError("context 'instrumental' must be true or false.")
    for key in ("tempo", "seconds"):
        if key in context and (isinstance(context[key], bool) or not isinstance(context[key], int | float) or context[key] <= 0):
            raise ValueError(f"context '{key}' must be a positive number.")
    if context.get("sound_type", "auto") not in _SOUND_TYPES:
        raise ValueError(f"context 'sound_type' must be one of {', '.join(_SOUND_TYPES)}.")
    return AudioEnhanceContext(**context)


def select_cases(cases: tuple[Case, ...], *, tags: set[str] | None = None, ids: set[str] | None = None) -> tuple[Case, ...]:
    """Return the cases that carry any of *tags* and whose id is in *ids*; ``None`` means no filter."""
    return tuple(case for case in cases if (not tags or tags & set(case.tags)) and (not ids or case.id in ids))
