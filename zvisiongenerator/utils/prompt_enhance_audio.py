"""Rewrite the text of an audio take: option axes, caption rules per model family, chat messages, output cleaning and the run (no model I/O).

Two tasks work on one text: ``sound`` rewrites a Sound prompt and ``style`` rewrites a Music style. Both aim the rewrite
at the model that will read it: ``_CAPTION_RULES`` holds one :class:`CaptionRules` per backend family (YuE2 reads comma
descriptors, Stable Audio one short sentence), and an unknown family gets plain prose. A later model adds an entry, for
example one that writes three headings with BPM and key as metadata. Tempo, key and length are settings the backend adds at
Generate; a family that writes metadata states a tempo or key only when one is set.
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from zvisiongenerator.utils.audio_settings import key_words
from zvisiongenerator.utils.prompt_enhance import (
    DEFAULT_TEMPERATURE,
    KEEP,
    NO_USABLE_MESSAGE,
    PRESERVATION_RULES,
    QUOTES_CHANGED_WARNING,
    EnhanceAxis,
    EnhanceOption,
    EnhanceResult,
    generate_raw,
    is_unusable,
    labelled_prompt,
    note_rules,
    parse_axis_value,
    parse_note,
    parse_spec_pairs,
    prompt_rule,
    strip_reasoning,
    strip_wrapping_quotes,
    system_message,
)
from zvisiongenerator.utils.prompt_quotes import missing_quotes, quoted_texts

if TYPE_CHECKING:
    from zvisiongenerator.core.audio_types import AudioGenerationRequest
    from zvisiongenerator.core.prompt_enhancer import PromptEnhancer

__all__ = [
    "AUDIO_AXES",
    "AUDIO_MODE",
    "AUDIO_TASKS",
    "DESCRIBE_MAX_CHARS",
    "LONG_TEXT_MAX_CHARS",
    "LYRICS_CONTEXT_CHARS",
    "SHORT_TEXT_MAX_CHARS",
    "TASK_KINDS",
    "TASK_LABELS",
    "AudioAxis",
    "AudioEnhanceContext",
    "AudioEnhanceSettings",
    "CaptionRules",
    "audio_enhance_context",
    "audio_matrix_contract",
    "audio_settings_from_mapping",
    "audio_settings_from_spec",
    "audio_settings_summary",
    "audio_settings_to_mapping",
    "build_audio_messages",
    "caption_rules",
    "caption_task",
    "clean_audio_output",
    "enhance_audio_caption",
    "style_rule_lines",
    "without_phrase",
]

AUDIO_MODE = "audio"
"""The ``mode`` of an audio request to the enhancer."""

AUDIO_TASKS: tuple[str, ...] = ("sound", "style")
"""The tasks the enhancer does on audio text; a later task is added here with its own rules."""

TASK_KINDS: dict[str, str] = {"sound": "sound", "style": "music"}
"""The kind of audio each task writes for."""

TASK_LABELS: dict[str, str] = {"sound": "Sound", "style": "Style"}
"""How a task reads in a summary of an enhanced take."""

SHORT_TEXT_MAX_CHARS = 1000
"""The longest Sound prompt or Music style that is rewritten."""

DESCRIBE_MAX_CHARS = 1000
"""The longest Describe it text."""

LONG_TEXT_MAX_CHARS = 4000
"""The longest lyrics or speech script that a writer task edits."""

LYRICS_CONTEXT_CHARS = 600
"""How much of the lyrics a style rewrite gets as context."""

_LYRICS_CONTEXT_NOTE = "The song's lyrics (context only: never copy them into the style)"


# ── Option axes ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class AudioAxis:
    """One audio option axis: the shared axis, the kind it belongs to and whether it needs a singing voice.

    Attributes:
        axis: The choices, with the same fields an image or video axis has.
        kind: ``music`` or ``sound``: the kind whose Enhance offers it.
        vocal: Whether the axis only applies when the take has a singing voice (Singer).
    """

    axis: EnhanceAxis
    kind: str
    vocal: bool = False


def _keep(label: str) -> EnhanceOption:
    return EnhanceOption(KEEP, "Keep", f"{label}: keep what the user wrote; do not impose one.", "Don't impose one; keep what your text says")


def _single(key: str, label: str, options: tuple[EnhanceOption, ...]) -> EnhanceAxis:
    return EnhanceAxis(key=key, label=label, multi=False, video_only=False, options=(_keep(label), *options), default=(KEEP,))


def _multi(key: str, label: str, options: tuple[EnhanceOption, ...]) -> EnhanceAxis:
    return EnhanceAxis(key=key, label=label, multi=True, video_only=False, options=options, default=())


_GENRES: tuple[tuple[str, tuple[tuple[str, str], ...]], ...] = (
    ("Pop & rock", (("pop", "Pop"), ("synth-pop", "Synth-pop"), ("indie-rock", "Indie rock"), ("rock", "Rock"))),
    ("Electronic", (("house", "House"), ("techno", "Techno"), ("drum-and-bass", "Drum and bass"), ("synthwave", "Synthwave"))),
    ("Hip-hop & R&B", (("hip-hop", "Hip-hop"), ("trap", "Trap"), ("r-and-b", "R&B"), ("soul", "Soul"))),
    ("Jazz & blues", (("jazz", "Jazz"), ("smooth-jazz", "Smooth jazz"), ("swing", "Swing"), ("blues", "Blues"))),
    ("Folk & country", (("folk", "Folk"), ("country", "Country"), ("bluegrass", "Bluegrass"), ("singer-songwriter", "Singer-songwriter"))),
    ("Classical & cinematic", (("classical", "Classical"), ("orchestral", "Orchestral"), ("film-score", "Film score"), ("neoclassical", "Neoclassical"))),
    ("World", (("latin", "Latin"), ("reggae", "Reggae"), ("afrobeat", "Afrobeat"), ("flamenco", "Flamenco"))),
    ("Lo-fi & ambient", (("lo-fi", "Lo-fi"), ("ambient", "Ambient"), ("chillout", "Chillout"), ("downtempo", "Downtempo"))),
)
GENRE_AXIS = _single(
    "genre",
    "Genre",
    tuple(EnhanceOption(slug, label, f"Genre: {label}.", f"Write it as {label}", group) for group, genres in _GENRES for slug, label in genres),
)

_MOODS: tuple[tuple[str, str, str], ...] = (
    ("serene", "Serene", "calm and serene, unhurried"),
    ("joyful", "Joyful", "joyful and upbeat"),
    ("romantic", "Romantic", "romantic and tender"),
    ("nostalgic", "Nostalgic", "nostalgic, a fond memory"),
    ("melancholic", "Melancholic", "melancholic and wistful"),
    ("mysterious", "Mysterious", "mysterious, with an air of suspense"),
    ("dark", "Dark", "dark and brooding"),
    ("dramatic", "Dramatic", "dramatic and tense"),
    ("epic", "Epic", "epic and awe-inspiring"),
    ("hopeful", "Hopeful", "hopeful and uplifting"),
    ("energetic", "Energetic", "energetic and driving"),
    ("dreamy", "Dreamy", "dreamy and floating"),
)
MUSIC_MOOD_AXIS = _single("music_mood", "Mood", tuple(EnhanceOption(slug, label, f"Mood: {phrase}.", phrase.capitalize()) for slug, label, phrase in _MOODS))

_INSTRUMENTS = ("guitar", "piano", "drums", "bass", "strings", "synth", "brass", "saxophone", "violin", "flute", "organ", "percussion")


def _instruments(what: str) -> tuple[EnhanceOption, ...]:
    return tuple(EnhanceOption(name, name.capitalize(), name, f"{what} {name}") for name in _INSTRUMENTS)


INSTRUMENTS_AXIS = _multi("instruments", "Instruments", _instruments("Feature"))
WITHOUT_AXIS = _multi("without", "Without", _instruments("Leave out"))
SINGER_AXIS = _single(
    "singer",
    "Singer",
    (
        EnhanceOption("female", "Female", "Vocal: a female lead vocal.", "A female lead vocal"),
        EnhanceOption("male", "Male", "Vocal: a male lead vocal.", "A male lead vocal"),
        EnhanceOption("duet", "Duet", "Vocal: a duet vocal, one female and one male voice.", "A female and a male voice"),
        EnhanceOption("choir", "Choir", "Vocal: a choir vocal, many voices singing together.", "Many voices together"),
    ),
)
PRODUCTION_AXIS = _single(
    "production",
    "Production",
    (
        EnhanceOption("polished", "Polished", "Production: polished, clean and modern.", "Clean, modern, radio-ready"),
        EnhanceOption("lo-fi", "Lo-fi", "Production: lo-fi, with tape warmth, soft noise and a muffled top end.", "Tape warmth, soft noise, muffled"),
        EnhanceOption("live", "Live", "Production: a live recording in a room, with natural ambience.", "Played together in a room"),
        EnhanceOption("vintage", "Vintage", "Production: vintage, in the sound of an old analog record.", "The sound of an old record"),
        EnhanceOption("cinematic", "Cinematic", "Production: cinematic, wide and large with a lot of depth.", "Wide, large, deep"),
    ),
)
SPACE_AXIS = _single(
    "space",
    "Space",
    (
        EnhanceOption("small-room", "Small room", "Space: a small room, dry with short reflections.", "Dry, close walls"),
        EnhanceOption("large-hall", "Large hall", "Space: a large hall with a long, open reverb.", "Long, open reverb"),
        EnhanceOption("street", "Street", "Space: an outdoor city street with echoes off buildings.", "Outdoors between buildings"),
        EnhanceOption("open-country", "Open country", "Space: open country, outdoors with almost no reflections.", "Outdoors, wide and dry"),
        EnhanceOption("forest", "Forest", "Space: a forest, outdoors with soft, scattered reflections.", "Soft, scattered reflections"),
        EnhanceOption("underwater", "Underwater", "Space: underwater, muffled and dense.", "Muffled and dense"),
    ),
)
DISTANCE_AXIS = _single(
    "distance",
    "Distance",
    (
        EnhanceOption("close", "Close", "Distance: the source is close to the listener.", "Right next to you"),
        EnhanceOption("medium", "Medium", "Distance: the source is at a medium distance.", "A few steps away"),
        EnhanceOption("far", "Far", "Distance: the source is far away.", "Far off, faint"),
    ),
)
OVER_TIME_AXIS = _single(
    "over_time",
    "Over time",
    (
        EnhanceOption("steady", "Steady", "Over time: steady, it does not change.", "Stays the same"),
        EnhanceOption("builds", "Builds", "Over time: it builds, growing louder or denser.", "Grows louder or denser"),
        EnhanceOption("hit-then-tail", "Hit then tail", "Over time: a sudden hit followed by a fading tail.", "A sharp start, then it fades"),
    ),
)
MOVEMENT_AXIS = _single(
    "movement",
    "Movement",
    (
        EnhanceOption("still", "Still", "Movement: the source stays in place.", "Stays in place"),
        EnhanceOption("passing", "Passing", "Movement: the source passes by the listener.", "Goes past you"),
        EnhanceOption("approaching", "Approaching", "Movement: the source approaches the listener.", "Comes toward you"),
        EnhanceOption("receding", "Receding", "Movement: the source moves away from the listener.", "Moves away"),
    ),
)

AUDIO_AXES: tuple[AudioAxis, ...] = (
    AudioAxis(GENRE_AXIS, "music"),
    AudioAxis(MUSIC_MOOD_AXIS, "music"),
    AudioAxis(INSTRUMENTS_AXIS, "music"),
    AudioAxis(WITHOUT_AXIS, "music"),
    AudioAxis(SINGER_AXIS, "music", vocal=True),
    AudioAxis(PRODUCTION_AXIS, "music"),
    AudioAxis(SPACE_AXIS, "sound"),
    AudioAxis(DISTANCE_AXIS, "sound"),
    AudioAxis(OVER_TIME_AXIS, "sound"),
    AudioAxis(MOVEMENT_AXIS, "sound"),
)
"""Every audio axis in display order: Music's six, then Sound's four."""

_AXES_BY_KEY = {item.axis.key: item for item in AUDIO_AXES}
_NOTE_KEY = "note"


@dataclass(frozen=True)
class AudioEnhanceSettings:
    """A choice on every audio axis, plus the free-text direction.

    The axes of the other kind stay at their defaults: a ``sound`` rewrite reads only Sound's four, a ``style`` rewrite Music's six.
    """

    genre: str = GENRE_AXIS.default[0]
    music_mood: str = MUSIC_MOOD_AXIS.default[0]
    instruments: tuple[str, ...] = INSTRUMENTS_AXIS.default
    without: tuple[str, ...] = WITHOUT_AXIS.default
    singer: str = SINGER_AXIS.default[0]
    production: str = PRODUCTION_AXIS.default[0]
    space: str = SPACE_AXIS.default[0]
    distance: str = DISTANCE_AXIS.default[0]
    over_time: str = OVER_TIME_AXIS.default[0]
    movement: str = MOVEMENT_AXIS.default[0]
    note: str = ""


@dataclass(frozen=True)
class AudioEnhanceContext:
    """Facts about the take that shape a rewrite.

    Attributes:
        lyrics: The start of the song's lyrics (at most ``LYRICS_CONTEXT_CHARS``), so the style fits the words.
        text: The start of the speech text (at most ``LYRICS_CONTEXT_CHARS``), so Fill in picks a Vibe that suits the words.
        style: The song's style text (the lyrics task's context, so the words suit it); a style rewrite does not read it.
        instrumental: Whether the song has no singing, so no singer is described.
        tempo: The tempo in BPM when one is set; only a family that writes metadata states it.
        key: The key as an ABC id when one is set; only a family that writes metadata states it.
        seconds: The length of the take when known; a sound's rewrite describes only what fits in it.
        sound_type: ``auto``, ``ambience``, ``effect`` or ``one-shot`` for a sound.
    """

    lyrics: str = ""
    text: str = ""
    style: str = ""
    instrumental: bool = False
    tempo: int | None = None
    key: str | None = None
    seconds: float | None = None
    sound_type: str = "auto"


def caption_task(kind: str) -> str | None:
    """Return the task that rewrites the text of a *kind* take (``sound`` for a sound, ``style`` for music), or ``None`` for a kind whose text is read as written (speech)."""
    return next((task for task, task_kind in TASK_KINDS.items() if task_kind == kind), None)


def audio_enhance_context(request: AudioGenerationRequest, *, task: str, sings: bool) -> AudioEnhanceContext:
    """Return the facts of *request* that shape a *task* rewrite, as the Web UI's Enhance sends them.

    A sound is told its type and its length. A music style is told whether the song is instrumental (*sings* false: the model takes no lyrics,
    or the lyrics hold no words), the start of the lyrics of a song that is sung, and the tempo and key when set.
    """
    _check_task(task)
    if task == "sound":
        return AudioEnhanceContext(sound_type=request.sound_type or "auto", seconds=request.seconds)
    return AudioEnhanceContext(instrumental=not sings, lyrics=request.lyrics[:LYRICS_CONTEXT_CHARS] if sings else "", tempo=request.tempo, key=request.key)


def _task_axes(task: str) -> tuple[AudioAxis, ...]:
    """Return the axes *task* reads (those of its kind)."""
    _check_task(task)
    return tuple(item for item in AUDIO_AXES if item.kind == TASK_KINDS[task])


def _check_task(task: str) -> None:
    if task not in AUDIO_TASKS:
        raise ValueError(f"Unknown audio task '{task}'. Valid values: {', '.join(AUDIO_TASKS)}.")


def audio_settings_from_mapping(data: Any, *, task: str) -> AudioEnhanceSettings:
    """Build settings from a mapping; omitted keys keep their defaults.

    Args:
        data: Mapping with any of the axis keys of *task*'s kind (see ``AUDIO_AXES``) and ``note``.
        task: ``sound`` or ``style``: the axes of the other kind are refused.

    Raises:
        ValueError: On a non-mapping, an unknown key or slug, an axis of the other kind, a direction that is too long, or an instrument that is
            in both Instruments and Without.
    """
    allowed = {item.axis.key for item in _task_axes(task)}
    if not isinstance(data, dict):
        raise ValueError(f"Enhancement settings must be a mapping, got {type(data).__name__}.")
    unknown = sorted(set(data) - {*_AXES_BY_KEY, _NOTE_KEY})
    if unknown:
        raise ValueError(f"Unknown enhancement setting(s): {', '.join(unknown)}. Valid keys: {', '.join(sorted(allowed | {_NOTE_KEY}))}.")
    values: dict[str, Any] = {}
    for key, raw in data.items():
        if key == _NOTE_KEY:
            values[key] = parse_note(raw)
        elif key in allowed:
            values[key] = parse_axis_value(_AXES_BY_KEY[key].axis, raw)
        else:
            raise ValueError(f"'{key}' only applies to {_AXES_BY_KEY[key].kind} prompts, not to the {task} task.")
    both = [slug for slug in values.get("instruments", ()) if slug in values.get("without", ())]
    if both:
        raise ValueError(f"Instruments and Without both list: {', '.join(both)}.")
    return AudioEnhanceSettings(**values)


def audio_settings_from_spec(spec: str | None, *, task: str, note: str | None = None) -> AudioEnhanceSettings:
    """Parse a CLI spec such as ``genre=jazz,instruments=piano+saxophone`` (the keys and slugs of *task*'s axes, ``+`` between several) and an optional *note*.

    An empty or ``None`` spec is the defaults.

    Raises:
        ValueError: On a malformed part, an unknown key or slug, an axis of the other kind, a direction that is too long, or an instrument in both Instruments and Without.
    """
    data: dict[str, Any] = parse_spec_pairs(spec)
    if note is not None:
        data[_NOTE_KEY] = note
    return audio_settings_from_mapping(data, task=task)


def audio_settings_to_mapping(settings: AudioEnhanceSettings, *, changed_only: bool = False) -> dict[str, Any]:
    """Return *settings* as a JSON-friendly mapping: every axis and ``note``, or with *changed_only* just what differs from the defaults."""
    defaults = AudioEnhanceSettings()
    mapping: dict[str, Any] = {}
    for key in (*_AXES_BY_KEY, _NOTE_KEY):
        value = getattr(settings, key)
        if changed_only and value == getattr(defaults, key):
            continue
        mapping[key] = list(value) if isinstance(value, tuple) else value
    return mapping


def audio_matrix_contract() -> dict[str, Any]:
    """Serialize the audio option axes and the defaults for the Web UI (the Workspace's ``prompt_enhancer.audio_matrix``).

    Each axis carries its ``key``, ``label``, ``multi``, the ``kind`` that offers it, ``vocal`` (it needs a singing voice), its ``options``
    (``slug``, ``label``, ``description``, ``group``) and ``default``.
    """
    return {
        "axes": [
            {
                "key": item.axis.key,
                "label": item.axis.label,
                "multi": item.axis.multi,
                "kind": item.kind,
                "vocal": item.vocal,
                "options": [{"slug": option.slug, "label": option.label, "description": option.description, "group": option.group} for option in item.axis.options],
                "default": list(item.axis.default),
            }
            for item in AUDIO_AXES
        ],
        "defaults": audio_settings_to_mapping(AudioEnhanceSettings()),
    }


def audio_settings_summary(settings: dict[str, Any]) -> list[str]:
    """Return the labels of the audio options a recorded *settings* mapping picks, in axis order, then the quoted direction."""
    picked: list[str] = []
    for item in AUDIO_AXES:
        value = settings.get(item.axis.key)
        for slug in value if isinstance(value, list) else [value]:
            if slug is not None and slug != KEEP and slug in item.axis.slugs():
                picked.append(item.axis.option(slug).label)
    if settings.get(_NOTE_KEY):
        picked.append(f'"{settings[_NOTE_KEY]}"')
    return picked


def without_phrase(slugs: tuple[str, ...]) -> str:
    """Return the Without choices as the caption writes them: ``no guitar, no drums``."""
    return ", ".join(f"no {WITHOUT_AXIS.option(slug).instruction}" for slug in slugs)


# ── Caption rules per model family ─────────────────────────────────────────────


@dataclass(frozen=True)
class CaptionRules:
    """How one model family wants its text written.

    Attributes:
        form: The rules for the shape of the text (descriptors, a sentence, prose), the same for music and sound.
        vocals: Whether the model can sing; without it no singer, vocal type or lyrics are described.
        max_tokens: The token budget of one rewrite for this shape.
        headings: When set, the text is sections under these headings, in this order (a caption with a metadata section).
        metadata: Whether tempo and key are written into the text when set (and never invented when not).
        closing: The last line of the system message: the output format.
        example: An example of the text for a take without a singing voice (empty for none); it holds no vocal descriptor and no "no X".
        vocal_example: An example of the text for a song the model sings (empty for none).
    """

    form: tuple[str, ...]
    vocals: bool = True
    max_tokens: int = 200
    headings: tuple[str, ...] = ()
    metadata: bool = False
    closing: str = "Output only the text, as written: no preamble, no headings, no quotes around the whole text."
    example: str = ""
    vocal_example: str = ""


_CAPTION_RULES: dict[str, CaptionRules] = {
    "yue2": CaptionRules(
        form=(
            "Write one line of comma-separated descriptors.",
            "Use short noun phrases only: no full sentences, no verbs, no period at the end.",
            'Write what to leave out as "no" followed by the instrument.',
            "Do not write a tempo, BPM or key: they are set elsewhere. Do not write lyrics, titles or section tags.",
        ),
        max_tokens=160,
        closing="Output only the comma-separated descriptors on one line: no preamble, no headings, no quotes around the whole text.",
        example="Jazz, piano, tenor saxophone, upright bass, brushed drums, warm",
        vocal_example="Jazz, expressive lead vocal, piano, tenor saxophone, upright bass, brushed drums, warm",
    ),
    "stable_audio_3": CaptionRules(
        form=(
            "Write one sentence of at most 45 words, aiming for 25 to 40.",
            "Use plain English with no brackets, no markdown and no lists.",
            "The model cannot sing: even for a song, never describe a voice or singing. Name the instruments that carry the melody instead.",
            'Do not write "TrackType", "Length", a tempo, BPM or key: they are added separately.',
        ),
        vocals=False,
        max_tokens=160,
        closing="Output only the one sentence: no preamble, no headings, no quotes around the whole text.",
    ),
}
_PLAIN_PROSE = CaptionRules(form=("Write one or two plain sentences of at most 60 words, with no brackets and no markdown.",), max_tokens=200)


def caption_rules(family: str | None) -> CaptionRules:
    """Return the caption rules of backend *family*; an unknown family (or ``None``) gets plain prose."""
    return _CAPTION_RULES.get(family or "", _PLAIN_PROSE)


# ── Chat messages ──────────────────────────────────────────────────────────────

_SOUND_TYPE_RULES = {
    "ambience": "This is an ambience: a continuous background sound of a place, steady with slow variation.",
    "effect": "This is a sound effect: one event or action that lasts a few seconds.",
    "one-shot": "This is a one-shot: a single short sound of one to three seconds, its attack and its decay.",
}
_SUBJECTS = {"sound": "the sound", "style": "the music's style"}
_INTROS = {"sound": "You rewrite sound descriptions for a text-to-audio model.", "style": "You rewrite music style descriptions for a text-to-music model."}
_KIND_RULES = {
    "sound": (
        "Describe the sound: its source, the material it is made of or hits, the space it is in (indoor or outdoor, dry or reverberant, close or distant), how it changes over time and how it moves.",
        "Describe only what is heard: no thoughts, backstory or narrative. Add no music, speech or singing unless the user asks for it.",
    ),
    "music": ("Describe the music's style: genre, mood, instruments{vocal_type} and production. Never write a title, lyrics or a story.",),
}


def _kind_rules(kind: str, *, vocals: bool) -> list[str]:
    """Return the rules for the kind of audio; the music line names a vocal type only when the take has a singing voice."""
    return [rule.replace("{vocal_type}", ", vocal type" if vocals else "") for rule in _KIND_RULES[kind]]


def _example_rules(rules: CaptionRules, *, vocals: bool) -> list[str]:
    """Return the example line of the family: the vocal example only for a take with a singing voice, else the plain one."""
    example = (rules.vocal_example or rules.example) if vocals else rules.example
    return [f'For example: "{example}".'] if example else []


def style_rule_lines(family: str | None, *, vocals: bool) -> list[str]:
    """Return the rules that shape a Music style for the model of *family*: what a style describes, its form and an example.

    *vocals* is whether the take has a singing voice; a family that cannot sing never describes one.
    """
    rules = caption_rules(family)
    sings = rules.vocals and vocals
    return [*_kind_rules("music", vocals=sings), *rules.form, *_example_rules(rules, vocals=sings)]


def _quote_rule(spans: tuple[str, ...]) -> list[str]:
    """Return the rule that keeps every quoted text, a name included, exactly as written with its quote marks (none without quotes)."""
    if not spans:
        return []
    listed = ", ".join(f'"{span}"' for span in spans)
    return [f"Keep every quoted text exactly as written, with its quote marks, even a name: {listed}."]


def _axis_rules(settings: AudioEnhanceSettings, task: str, *, vocals: bool) -> list[str]:
    """Return one rule per axis of the task's kind that is not on Keep; Singer is skipped without a singing voice."""
    rules: list[str] = []
    for item in _task_axes(task):
        key = item.axis.key
        value = getattr(settings, key)
        if (item.vocal and not vocals) or value == getattr(AudioEnhanceSettings(), key):
            continue
        if key == INSTRUMENTS_AXIS.key:
            rules.append("Instruments to feature: " + ", ".join(INSTRUMENTS_AXIS.option(slug).instruction for slug in value) + ".")
        elif key == WITHOUT_AXIS.key:
            rules.append(f'Leave these out and write them as: "{without_phrase(value)}".')
        else:
            rules.append(item.axis.option(value).instruction)
    return rules


def _context_rules(rules: CaptionRules, context: AudioEnhanceContext, task: str) -> list[str]:
    """Return the rules the take's facts add: instrumental, lyrics, metadata, sound type and length."""
    out: list[str] = []
    if task == "style":
        if context.instrumental:
            out.append("The song is instrumental: describe no singer, voice, vocals or lyrics.")
        elif not rules.vocals:
            out.append("Write it as instrumental music even if the text says song: describe no singer, voice, vocals or lyrics.")
        elif rules.vocals and context.lyrics.strip():
            out.append(f"{_LYRICS_CONTEXT_NOTE}: {' '.join(context.lyrics.split())[:LYRICS_CONTEXT_CHARS]}")
        if rules.metadata:
            out.extend(_metadata_rules(context))
        else:
            out.extend(_setting_rules(context))
    else:
        if context.sound_type in _SOUND_TYPE_RULES:
            out.append(_SOUND_TYPE_RULES[context.sound_type])
        if context.seconds:
            out.append(f"The sound runs about {context.seconds:g} seconds: describe only what fits in that time, and do not write the length.")
    return out


def _setting_rules(context: AudioEnhanceContext) -> list[str]:
    """Return the rules for a tempo or key that is set but not written into the text: the setting handles it, so the text has no such word."""
    out = []
    if context.tempo:
        out.append("The tempo is set separately: write no tempo, BPM or speed words (such as fast or slow).")
    if context.key:
        out.append("The key is set separately: write no key.")
    return out


def _metadata_rules(context: AudioEnhanceContext) -> list[str]:
    """Return the rules that put a set tempo and key into the text as metadata, and write none that is Auto."""
    out = []
    if context.tempo:
        out.append(f'State the tempo as "BPM: {context.tempo}".')
    if context.key:
        out.append(f'State the key as "Key: {key_words(context.key)}".')
    missing = [name for name, value in (("BPM", context.tempo), ("key", context.key)) if not value]
    if missing:
        out.append(f"Never invent a {' or '.join(missing)}: write none that is not given.")
    return out


def build_audio_messages(text: str, settings: AudioEnhanceSettings, *, task: str, rules: CaptionRules, context: AudioEnhanceContext) -> list[dict[str, str]]:
    """Build the system + user chat messages for one audio rewrite of *text*."""
    _check_task(task)
    source = text.strip()
    vocals = rules.vocals and not context.instrumental
    system_rules = [
        prompt_rule(_SUBJECTS[task]),
        *PRESERVATION_RULES,
        "Copy names, dates and places exactly as the user wrote them.",
        *_kind_rules(TASK_KINDS[task], vocals=vocals),
        *rules.form,
        *(_example_rules(rules, vocals=vocals) if task == "style" else []),
    ]
    if rules.headings:
        system_rules.append(f"Write these headings, each on its own line followed by its text, in this order: {', '.join(rules.headings)}.")
    system_rules.extend(_axis_rules(settings, task, vocals=vocals))
    system_rules.extend(_context_rules(rules, context, task))
    system_rules.extend(note_rules(settings.note))
    system_rules.extend(_quote_rule(quoted_texts(source)))
    return [{"role": "system", "content": system_message(_INTROS[task], system_rules, rules.closing)}, {"role": "user", "content": labelled_prompt(source)}]


# ── Output cleaning and the run ────────────────────────────────────────────────

_CHAT_TOKEN_RE = re.compile(r"<\|[^|>]*\|>")
_FENCE_RE = re.compile(r"^\s*```[\w-]*\s*\n?|\n?\s*```\s*$")
_LABEL_RE = re.compile(r"^\s*(?:rewritten (?:prompt|text)|enhanced (?:prompt|text)|style|caption|prompt|sound|description)\s*:\s*", re.I)


def clean_audio_output(text: str, *, strip_quotes: bool = True, strip_label: bool = True) -> str:
    """Strip reasoning blocks, chat special tokens, code fences and a leading label from model output, keeping marks and section tags.

    Square-bracket marks (``[sad]``, ``[Verse 1]``) and the quotation marks inside the text stay. With *strip_quotes* a pair of double
    quotes that wraps all of the text is removed too (the model quoting its answer); a text that is itself speech passes ``False``.
    With *strip_label* a leading ``Style:``-like label is removed; an answer made of labelled blocks passes ``False``.
    """
    text = _CHAT_TOKEN_RE.sub("", strip_reasoning(text))
    text = _FENCE_RE.sub("", text.strip()).strip()
    if strip_label:
        text = _LABEL_RE.sub("", text).strip()
    return strip_wrapping_quotes(text) if strip_quotes else text


def enhance_audio_caption(
    enhancer: PromptEnhancer,
    text: str,
    settings: AudioEnhanceSettings,
    *,
    task: str,
    family: str | None,
    context: AudioEnhanceContext | None = None,
    seed: int,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int | None = None,
    on_text: Callable[[str], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> EnhanceResult:
    """Rewrite a Sound prompt or a Music style for the model of *family*: generate, clean, and retry once if unusable or a quote changed.

    Names, dates, places and quoted text are copied exactly.
    When no attempt keeps every quoted text, the last usable one is returned with ``QUOTES_CHANGED_WARNING`` (also issued as a warning).
    Expand ``{a|b}`` choices before calling.

    Args:
        task: ``sound`` or ``style``.
        family: The backend family of the audio model (``yue2``, ``stable_audio_3``); an unknown one gets plain prose.
        max_tokens: The token budget; ``None`` takes the budget of the family's shape.
        on_text: Receives the full in-progress text at most every ``TEXT_UPDATE_INTERVAL_SECONDS``, then the final text; a retry starts over from an empty string.
        cancelled: Polled per token; when it returns True generation stops and ``RuntimeError`` is raised.

    Raises:
        ValueError: For an unknown task, or when there is nothing to rewrite.
        RuntimeError: When cancelled, or when both attempts are unusable.
    """
    _check_task(task)
    context = context or AudioEnhanceContext()
    source = text.strip()
    if not source:
        raise ValueError("Enter a sound to enhance." if task == "sound" else "Enter a style to enhance.")
    rules = caption_rules(family)
    messages = build_audio_messages(text, settings, task=task, rules=rules, context=context)
    budget = max_tokens if max_tokens is not None else rules.max_tokens
    spans = quoted_texts(source)
    flawed: str | None = None
    for attempt in range(2):
        raw = generate_raw(enhancer, messages, seed=seed + attempt, max_tokens=budget, temperature=temperature, on_text=on_text, cancelled=cancelled, clean=clean_audio_output)
        output = clean_audio_output(raw)
        if is_unusable(output, source):
            continue
        if missing_quotes(spans, output):
            flawed = output
            continue
        return _final(EnhanceResult(prompt=output, clamped=False), on_text)
    if flawed is None:
        raise RuntimeError(NO_USABLE_MESSAGE)
    warnings.warn(QUOTES_CHANGED_WARNING, stacklevel=2)
    return _final(EnhanceResult(prompt=flawed, clamped=False, warning=QUOTES_CHANGED_WARNING), on_text)


def _final(result: EnhanceResult, on_text: Callable[[str], None] | None) -> EnhanceResult:
    """Report *result*'s text as the final text and return it."""
    if on_text is not None:
        on_text(result.prompt)
    return result
