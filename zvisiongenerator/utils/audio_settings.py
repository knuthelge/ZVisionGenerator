"""The vocabulary of the audio settings that steer a take: tempo and key, speech delivery and the type of a sound.

All values are plain data and the parsers are pure. A parser raises ``ValueError`` in the words the user sees; the Web UI
turns it into a 422 and ``ziv audio`` into a usage error. The Web UI reads the same vocabulary from the Workspace payload
(:func:`audio_settings_contract`), so a list is written once.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from zvisiongenerator.core.audio_types import SpeechDelivery

__all__ = [
    "AUTO_SOUND_SECONDS",
    "DEFAULT_MUSIC_SECONDS",
    "DELIVERY_FIELDS",
    "EMOTIONS",
    "EMOTION_GROUPS",
    "LEVELS",
    "MANNERS",
    "MARKS",
    "MUSIC_KEYS",
    "PACES",
    "PACE_READING_FACTOR",
    "READING_CHARS_PER_SECOND",
    "SOUND_TYPES",
    "TEMPO_MAX",
    "TEMPO_MIN",
    "VIBES",
    "audio_settings_contract",
    "format_reading_time",
    "key_words",
    "parse_delivery",
    "parse_key",
    "parse_sound_type",
    "parse_tempo",
    "parse_vibe",
    "reading_characters",
    "reading_seconds",
]

TEMPO_MIN = 40
TEMPO_MAX = 220
"""The slowest and fastest tempo, in beats per minute, that a take can ask for."""

MUSIC_KEYS: tuple[str, ...] = (
    *("C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"),
    *("Cm", "C#m", "Dm", "Ebm", "Em", "Fm", "F#m", "Gm", "G#m", "Am", "Bbm", "Bm"),
)
"""The 24 keys as ABC ids: the twelve majors, then the twelve minors."""

EMOTION_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Bright", ("elated", "amused", "enthusiastic", "proud")),
    ("Warm", ("content", "affectionate", "relieved", "longing")),
    ("Thoughtful", ("thoughtful", "determined", "awed", "surprised", "confused")),
    ("Heavy", ("sad", "bitter", "ashamed", "helpless", "afraid", "angry", "disgusted")),
)
"""The emotions of speech delivery under the group a picker shows them in; ``neutral`` is in none."""

EMOTIONS: tuple[str, ...] = ("neutral", *(emotion for _, emotions in EMOTION_GROUPS for emotion in emotions))
MANNERS: tuple[str, ...] = ("natural", "whisper", "shout")
PACES: tuple[str, ...] = ("very-slow", "slow", "normal", "fast", "very-fast")
LEVELS: tuple[str, ...] = ("low", "normal", "high")
"""The values of Pitch and Expressiveness."""

DELIVERY_FIELDS: tuple[str, ...] = ("emotion", "manner", "pace", "pitch", "expressiveness")
"""The five delivery settings, in the order of :class:`SpeechDelivery`; also their names in a form, a saved take and a Reuse link."""

_DELIVERY_CHOICES: dict[str, tuple[str, ...]] = {"emotion": EMOTIONS, "manner": MANNERS, "pace": PACES, "pitch": LEVELS, "expressiveness": LEVELS}

VIBES: dict[str, SpeechDelivery] = {
    "calm-narrator": SpeechDelivery("thoughtful", "natural", "slow", "normal", "normal"),
    "storyteller": SpeechDelivery("content", "natural", "normal", "normal", "high"),
    "excited-host": SpeechDelivery("enthusiastic", "natural", "fast", "high", "high"),
    "announcer": SpeechDelivery("determined", "natural", "normal", "normal", "low"),
    "sad-confession": SpeechDelivery("sad", "natural", "slow", "low", "normal"),
    "whispered-secret": SpeechDelivery("neutral", "whisper", "slow", "low", "low"),
}
"""Named starting points for the five delivery settings."""

MARKS: dict[str, tuple[str, str] | None] = {
    **{emotion: ("emotion", emotion) for emotion in EMOTIONS[1:]},
    "whisper": ("manner", "whisper"),
    "shout": ("manner", "shout"),
    "as set": None,
}
"""The ``[mark]`` words of a speech text: each sets one delivery field from there on; ``as set`` returns to the Settings."""

SOUND_TYPES: tuple[str, ...] = ("auto", "ambience", "effect", "one-shot")
_SOUND_TYPE_LABELS = {"auto": "Auto", "ambience": "Ambience", "effect": "Effect", "one-shot": "One-shot"}

AUTO_SOUND_SECONDS: dict[str, float] = {"auto": 12.0, "ambience": 12.0, "effect": 5.0, "one-shot": 2.0}
"""The length of a sound whose length is left on Auto, by its type."""

DEFAULT_MUSIC_SECONDS = 30.0
"""The length of music from a model that sets its length (a model without lyrics) when none is given."""

READING_CHARS_PER_SECOND = 14.0
"""How many characters of speech are read in a second at a Normal pace."""

PACE_READING_FACTOR: dict[str, float] = {"very-slow": 0.72, "slow": 0.81, "normal": 1.0, "fast": 1.17, "very-fast": 1.22}
"""How much faster (above 1) or slower (below 1) than Normal each pace reads."""


def reading_seconds(characters: int, pace: str = "normal") -> float:
    """Return about how many seconds *characters* characters of speech take to read at *pace*; a pace not listed reads at Normal."""
    return characters / (READING_CHARS_PER_SECOND * PACE_READING_FACTOR.get(pace, 1.0))


def reading_characters(seconds: float, pace: str = "normal") -> int:
    """Return about how many characters of speech are read in *seconds* at *pace*, the inverse of :func:`reading_seconds`."""
    return round(seconds * READING_CHARS_PER_SECOND * PACE_READING_FACTOR.get(pace, 1.0))


def format_reading_time(seconds: float) -> str:
    """Return a reading time as the Web UI writes it: ``1:24``, rounded to the second."""
    whole = round(seconds)
    return f"{whole // 60}:{whole % 60:02d}"


def key_words(key: str) -> str:
    """Return a key id in words, such as ``F# minor`` for ``F#m`` and ``C major`` for ``C``."""
    return f"{key[:-1]} minor" if key.endswith("m") else f"{key} major"


def parse_tempo(value: object) -> int | None:
    """Return the tempo in BPM *value* names, or ``None`` for Auto (nothing given or a blank).

    Raises:
        ValueError: If *value* is not a whole number from ``TEMPO_MIN`` to ``TEMPO_MAX``.
    """
    wrong = ValueError(f"Tempo must be {TEMPO_MIN} to {TEMPO_MAX} BPM.")
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if isinstance(value, bool):
        raise wrong
    try:
        tempo = int(value.strip() if isinstance(value, str) else value)  # type: ignore[call-overload]
    except TypeError, ValueError:
        raise wrong from None
    if isinstance(value, float) and value != tempo:
        raise wrong
    if not TEMPO_MIN <= tempo <= TEMPO_MAX:
        raise wrong
    return tempo


def parse_key(value: object) -> str | None:
    """Return the key id *value* names, or ``None`` for Auto (nothing given or a blank).

    Raises:
        ValueError: If *value* is not one of ``MUSIC_KEYS``.
    """
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if not isinstance(value, str) or value.strip() not in MUSIC_KEYS:
        raise ValueError(f"Key must be one of: {', '.join(MUSIC_KEYS)}.")
    return value.strip()


def parse_delivery(values: Mapping[str, object]) -> SpeechDelivery:
    """Return the delivery *values* name (keys ``emotion``, ``manner``, ``pace``, ``pitch`` and ``expressiveness``); a missing or blank one is the default.

    Raises:
        ValueError: If a value is not one of its setting's choices.
    """
    chosen: dict[str, str] = {}
    for field, choices in _DELIVERY_CHOICES.items():
        value = values.get(field)
        if value is None or (isinstance(value, str) and not value.strip()):
            continue
        if not isinstance(value, str) or value.strip() not in choices:
            raise ValueError(f"{field.capitalize()} must be one of: {', '.join(choices)}.")
        chosen[field] = value.strip()
    return SpeechDelivery(**chosen)


def parse_vibe(value: object) -> str | None:
    """Return the Vibe id *value* names, or ``None`` for none (nothing given or a blank).

    Raises:
        ValueError: If *value* is not one of ``VIBES``.
    """
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if not isinstance(value, str) or value.strip() not in VIBES:
        raise ValueError(f"Vibe must be one of: {', '.join(VIBES)}.")
    return value.strip()


def parse_sound_type(value: object) -> str:
    """Return the sound type *value* names; nothing given or a blank is ``auto``.

    Raises:
        ValueError: If *value* is not one of ``SOUND_TYPES``.
    """
    if value is None or (isinstance(value, str) and not value.strip()):
        return "auto"
    if not isinstance(value, str) or value.strip() not in SOUND_TYPES:
        raise ValueError(f"Sound type must be one of: {', '.join(SOUND_TYPES)}.")
    return value.strip()


def _label(identifier: str) -> str:
    """Return an id as a label: ``very-slow`` is ``Very slow``."""
    return identifier.replace("-", " ").capitalize()


def _options(ids: tuple[str, ...]) -> list[dict[str, str]]:
    """Return *ids* as picker options with their labels."""
    return [{"id": item, "label": _label(item)} for item in ids]


def audio_settings_contract() -> dict[str, Any]:
    """Return the vocabulary of the audio settings as JSON-ready data, for the Workspace payload (``audio_settings``).

    Keys: ``tempo`` (``min``, ``max``), ``keys`` (``id``, ``words``, ``mode``), ``emotions`` (``id``, ``label``,
    ``group`` or ``None``), ``manners``, ``paces``, ``levels`` (each ``id`` and ``label``), ``delivery_defaults``, ``vibes``
    (``id``, ``label``, ``delivery``), ``marks`` (``id``, ``field``, ``value``; both ``None`` for ``as set``),
    ``sound_types`` (``id``, ``label``), ``auto_seconds`` and ``reading`` (``chars_per_second``,
    ``pace_factor``).
    """
    groups = {emotion: group for group, emotions in EMOTION_GROUPS for emotion in emotions}
    return {
        "tempo": {"min": TEMPO_MIN, "max": TEMPO_MAX},
        "keys": [{"id": key, "words": key_words(key), "mode": "minor" if key.endswith("m") else "major"} for key in MUSIC_KEYS],
        "emotions": [{"id": emotion, "label": _label(emotion), "group": groups.get(emotion)} for emotion in EMOTIONS],
        "manners": _options(MANNERS),
        "paces": _options(PACES),
        "levels": _options(LEVELS),
        "delivery_defaults": _delivery_values(SpeechDelivery()),
        "vibes": [{"id": vibe, "label": _label(vibe), "delivery": _delivery_values(delivery)} for vibe, delivery in VIBES.items()],
        "marks": [{"id": mark, "field": target[0] if target else None, "value": target[1] if target else None} for mark, target in MARKS.items()],
        "sound_types": [{"id": item, "label": _SOUND_TYPE_LABELS[item]} for item in SOUND_TYPES],
        "auto_seconds": dict(AUTO_SOUND_SECONDS),
        "reading": {"chars_per_second": READING_CHARS_PER_SECOND, "pace_factor": dict(PACE_READING_FACTOR)},
    }


def _delivery_values(delivery: SpeechDelivery) -> dict[str, str]:
    """Return *delivery* as a mapping of its five settings."""
    return {field: getattr(delivery, field) for field in DELIVERY_FIELDS}
