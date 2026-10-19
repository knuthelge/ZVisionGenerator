"""Build, run, and post-process local-LLM prompt enhancement (pure logic, no model I/O)."""

from __future__ import annotations

import re
import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any


from zvisiongenerator.utils.prompt_quotes import lettering_spans, missing_quotes, quoted_texts

if TYPE_CHECKING:
    from zvisiongenerator.core.prompt_enhancer import PromptEnhancer

ENHANCE_MODES = ("image", "video")
# What an image rewrite writes: a prompt (``text``) or a structured JSON caption (``json``) for models that take one.
ENHANCE_FORMATS = ("text", "json")
DEFAULT_LENGTH_CONFIG: dict[str, Any] = {
    "percent": {"shorter": 50, "same": 100, "longer": 200, "extra": 300},
    "floor_words": {"shorter": 12, "same": 25, "longer": 40, "extra": 80},
    "max_words": 300,
}
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_NEW_TOKENS = 700
DEFAULT_IDLE_RELEASE_SECONDS = 120.0
TEXT_UPDATE_INTERVAL_SECONDS = 0.1

_THINK_RE = re.compile(r"<think>.*?(?:</think>|$)", re.S)
_SPECIAL_TOKEN_RE = re.compile(r"<\|[^|>]*\|>")
_LABEL_RE = re.compile(r"^\s*(?:rewritten prompt|enhanced prompt|prompt)\s*:\s*", re.I)
_REFUSAL_RE = re.compile(r"^\s*(?:I can['’]t|I cannot|I won['’]t|I'm sorry|I’m sorry|I am sorry|As an AI)\b", re.I)
# Only double quotes: single quotes double as apostrophes ("Moe's"), so stripping them would cut real text.
_QUOTE_PAIRS = {'"': '"', "“": "”"}
PRESERVATION_RULES = (
    "EVERY DETAIL IS IMPORTANT: keep the user's subject, intent and every element they mention.",
    "Never add new people, animals, characters or major objects.",
)
QUOTES_CHANGED_WARNING = "A quoted text was changed in the rewrite."
NO_USABLE_MESSAGE = "The enhancer returned no usable rewrite (empty, unchanged, or a refusal). Try different settings."


@dataclass(frozen=True)
class EnhanceOption:
    """One selectable value on an enhancement axis."""

    slug: str
    label: str
    instruction: str


@dataclass(frozen=True)
class EnhanceAxis:
    """One axis of the enhancement option matrix."""

    key: str
    label: str
    multi: bool
    video_only: bool
    options: tuple[EnhanceOption, ...]
    default: tuple[str, ...]

    def slugs(self) -> tuple[str, ...]:
        """Return the valid option slugs in display order."""
        return tuple(option.slug for option in self.options)

    def option(self, slug: str) -> EnhanceOption:
        """Return the option for *slug*."""
        for option in self.options:
            if option.slug == slug:
                return option
        raise ValueError(f"Unknown {self.key} '{slug}'. Valid values: {', '.join(self.slugs())}.")


STYLE_AXIS = EnhanceAxis(
    key="style",
    label="Style",
    multi=False,
    video_only=False,
    options=(
        EnhanceOption("keep", "Keep", "Style: keep the user's existing style; do not impose a new one."),
        EnhanceOption("photo", "Photographic", "Style: photographic. Describe it as a photograph."),
        EnhanceOption(
            "candid",
            "Candid",
            "Style: candid snapshot. Describe it as an unposed, casual phone photo in an everyday setting with natural available light; no studio lighting, retouching or cinematic color grading.",
        ),
        EnhanceOption(
            "street",
            "Street photography",
            "Style: street photography. Describe it as a documentary 35mm photograph of everyday public life, with passersby and available light.",
        ),
        EnhanceOption("film", "Analog film", "Style: analog film photograph. Name the film stock and its grain and color rendering."),
        EnhanceOption(
            "bw",
            "Black & white",
            "Style: black-and-white photograph. Describe the tonal range, contrast and light instead of colors.",
        ),
        EnhanceOption(
            "portrait",
            "Studio portrait",
            "Style: studio portrait or fashion editorial photograph. Describe the lighting setup and backdrop.",
        ),
        EnhanceOption(
            "product",
            "Product shot",
            "Style: commercial product photograph. Describe the studio lighting, surface and backdrop.",
        ),
        EnhanceOption("cinematic", "Cinematic", "Style: cinematic. Describe it as a shot from a film."),
        EnhanceOption("illustration", "Illustration", "Style: digital illustration. Name the style explicitly."),
        EnhanceOption("anime", "Anime", "Style: anime illustration. Name the style explicitly."),
        EnhanceOption("comic", "Comic", "Style: comic book art with inked line work. Name the style explicitly."),
        EnhanceOption("3d", "3D render", "Style: 3D render. Name the style explicitly."),
        EnhanceOption("painterly", "Painterly", "Style: classical oil painting. Name the medium explicitly."),
    ),
    default=("keep",),
)
MOOD_AXIS = EnhanceAxis(
    key="mood",
    label="Mood",
    multi=False,
    video_only=False,
    options=(
        EnhanceOption("keep", "Keep", "Mood: keep the user's existing mood; do not impose a new one."),
        EnhanceOption("serene", "Serene", "Mood: calm and serene. Convey stillness and peace through soft light, gentle color and quiet surroundings."),
        EnhanceOption("joyful", "Joyful", "Mood: joyful and upbeat. Convey warmth and energy through bright light, lively color and open expressions."),
        EnhanceOption("romantic", "Romantic", "Mood: romantic and intimate. Convey tenderness through warm, soft light and close, gentle framing."),
        EnhanceOption("melancholic", "Melancholic", "Mood: melancholic and wistful. Convey solitude through muted color, subdued light and empty space."),
        EnhanceOption("mysterious", "Mysterious", "Mood: mysterious. Convey intrigue through shadow, haze and partly hidden detail."),
        EnhanceOption("eerie", "Eerie", "Mood: eerie and unsettling. Convey unease through cold tones, odd stillness and ominous shadow."),
        EnhanceOption("dramatic", "Dramatic", "Mood: dramatic and tense. Convey intensity through hard contrast, charged light and a decisive moment."),
        EnhanceOption("epic", "Epic", "Mood: epic and awe-inspiring. Convey grandeur through vast scale, sweeping depth and majestic light."),
        EnhanceOption("whimsical", "Whimsical", "Mood: whimsical and playful. Convey a dreamlike lightness through gentle color and charming detail."),
        EnhanceOption("nostalgic", "Nostalgic", "Mood: nostalgic. Convey a fond memory through warm, faded color and soft, golden light."),
    ),
    default=("keep",),
)
DETAILS_AXIS = EnhanceAxis(
    key="details",
    label="Details",
    multi=True,
    video_only=False,
    options=(
        EnhanceOption("lighting", "Lighting", "lighting"),
        EnhanceOption("composition", "Composition", "composition and framing"),
        EnhanceOption("camera", "Camera & lens", "camera and lens"),
        EnhanceOption("materials", "Materials & textures", "materials and textures"),
        EnhanceOption("color", "Color & mood", "color and mood"),
        EnhanceOption("environment", "Environment", "environment and background"),
        EnhanceOption("subject", "Subject", "the subject's appearance, clothing and pose"),
    ),
    default=("lighting", "composition"),
)
LENGTH_AXIS = EnhanceAxis(
    key="length",
    label="Length",
    multi=False,
    video_only=False,
    options=(
        EnhanceOption("shorter", "Shorter", "shorter"),
        EnhanceOption("same", "Same", "same"),
        EnhanceOption("longer", "Longer", "longer"),
        EnhanceOption("extra", "Extra long", "extra"),
    ),
    default=("same",),
)
MOTION_AXIS = EnhanceAxis(
    key="motion",
    label="Motion",
    multi=True,
    video_only=True,
    options=(
        EnhanceOption("action", "Action / sequence", "the action as a clear sequence of events"),
        EnhanceOption("camera-move", "Camera movement", "camera movement"),
        EnhanceOption("pacing", "Pacing", "pacing and timing"),
    ),
    default=("action",),
)
SOUND_AXIS = EnhanceAxis(
    key="sound",
    label="Sound",
    multi=True,
    video_only=True,
    options=(
        EnhanceOption("ambient", "Ambience", "the ambient soundscape of the setting"),
        EnhanceOption("effects", "Sound effects", "sound effects synced to the visible action"),
        EnhanceOption(
            "dialogue",
            "Dialogue",
            "speech: keep any quoted lines word for word, or write short lines that fit the scene when there are none, and say who speaks and in what voice",
        ),
        EnhanceOption("music", "Music", "music: whether it plays in the scene or as a score, its genre and tempo"),
    ),
    default=(),
)
# Display order in the Web UI: Length comes last, next to the Direction field.
ENHANCE_AXES: tuple[EnhanceAxis, ...] = (STYLE_AXIS, MOOD_AXIS, DETAILS_AXIS, MOTION_AXIS, SOUND_AXIS, LENGTH_AXIS)
_AXES_BY_KEY = {axis.key: axis for axis in ENHANCE_AXES}
# Axes that only apply when the video has an audio track.
AUDIO_AXES = frozenset({SOUND_AXIS.key})
NOTE_KEY = "note"
NOTE_MAX_CHARS = 300
# Which parts of a video prompt the rewrite writes: ``both`` (scene and action) or ``action`` (the scene is kept as written).
# It is video only and not an option axis: ``None`` on the settings means "auto" (see ``resolve_parts``).
PARTS_KEY = "parts"
PARTS_OPTIONS = (("both", "Scene and action"), ("action", "Action only"))
PARTS_VALUES = tuple(slug for slug, _label in PARTS_OPTIONS)
_SETTING_KEYS = (*_AXES_BY_KEY, NOTE_KEY, PARTS_KEY)
_DURATION_BEATS = ((5.0, "one continuous action"), (10.0, "two or three beats at most"))


@dataclass(frozen=True)
class EnhanceSettings:
    """A choice on every axis of the option matrix, plus the video-only ``parts`` choice.

    ``parts`` is ``"both"`` (rewrite scene and action), ``"action"`` (keep the scene, rewrite only the action) or
    ``None`` for auto, which ``resolve_parts`` turns into ``"action"`` when extending a clip that has a scene, or
    starting from an image with a scene, and
    ``"both"`` otherwise. Image rewrites ignore it.
    """

    style: str = STYLE_AXIS.default[0]
    mood: str = MOOD_AXIS.default[0]
    details: tuple[str, ...] = DETAILS_AXIS.default
    length: str = LENGTH_AXIS.default[0]
    motion: tuple[str, ...] = MOTION_AXIS.default
    sound: tuple[str, ...] = SOUND_AXIS.default
    note: str = ""
    parts: str | None = None


@dataclass(frozen=True)
class EnhanceContext:
    """Facts about the job that shape a rewrite: audio track, clip length, frame shape and reference image.

    The last five fields describe extending an existing clip. ``seconds`` is then the whole window (the carried
    frames plus the new ones) and ``new_seconds`` the part the new action fills. ``previous`` is the source's
    recorded prompt ("what just happened"); ``extract_from`` is set for a source recorded without a scene;
    ``prompt_is_original`` says the user's prompt still equals that old prompt.
    """

    audio: bool = False
    seconds: float | None = None
    orientation: str | None = None
    reference_image: bool = False
    extends: bool = False
    previous: str | None = None
    extract_from: str | None = None
    prompt_is_original: bool = False
    new_seconds: float | None = None


@dataclass(frozen=True)
class LengthPlan:
    """The word target and accepted range sent to the model for one enhancement."""

    length: str
    target: int
    lo: int
    hi: int
    clamped: bool


@dataclass(frozen=True)
class EnhanceResult:
    """The final enhanced prompt, whether its length request was clamped, and an optional notice (a changed quote)."""

    prompt: str
    clamped: bool
    warning: str | None = None


def matrix_contract() -> dict[str, Any]:
    """Serialize the option matrix and defaults for the Web UI.

    ``parts`` describes the video-only Rewrite choice; its ``default`` is ``None`` (auto: ``action`` when extending a
    clip that has a scene or starting from an image with a scene, else ``both``), and ``defaults.parts`` is ``None`` too.
    """
    return {
        "parts": {
            "key": PARTS_KEY,
            "label": "Rewrite",
            "video_only": True,
            "options": [{"slug": slug, "label": label} for slug, label in PARTS_OPTIONS],
            "default": None,
        },
        "axes": [
            {
                "key": axis.key,
                "label": axis.label,
                "multi": axis.multi,
                "video_only": axis.video_only,
                "audio_only": axis.key in AUDIO_AXES,
                "options": [{"slug": option.slug, "label": option.label} for option in axis.options],
                "default": list(axis.default),
            }
            for axis in ENHANCE_AXES
        ],
        "defaults": settings_to_mapping(EnhanceSettings()),
    }


def settings_to_mapping(settings: EnhanceSettings) -> dict[str, Any]:
    """Return *settings* as a JSON-friendly mapping."""
    return {
        "style": settings.style,
        "mood": settings.mood,
        "details": list(settings.details),
        "length": settings.length,
        "motion": list(settings.motion),
        "sound": list(settings.sound),
        "note": settings.note,
        "parts": settings.parts,
    }


def enhance_context(
    *,
    mode: str,
    width: int | None = None,
    height: int | None = None,
    ratio: str | None = None,
    frames: int | None = None,
    fps: float | None = None,
    audio: bool = False,
    reference_image: bool = False,
    extends: bool = False,
    previous: str | None = None,
    extract_from: str | None = None,
    prompt_is_original: bool = False,
    new_frames: int | None = None,
) -> EnhanceContext:
    """Build the rewrite context from a job's shape; audio and duration apply to video only.

    The orientation comes from *width* and *height* when both are set, else from a ``W:H`` *ratio*.

    Args:
        frames: The clip's frame count; when *extends*, the whole window (carried plus new frames).
        extends: Whether the job extends an existing video (video only); the fields below apply only then.
        previous: The source video's recorded prompt, "what just happened".
        extract_from: The source's prompt when it was recorded without a scene.
        prompt_is_original: Whether the user's prompt still equals the source's recorded prompt.
        new_frames: The frames the new action fills; with *fps* it gives ``new_seconds``.
    """
    _check_mode(mode)
    video = mode == "video"
    seconds = frames / fps if video and frames and fps else None
    extending = video and extends
    return EnhanceContext(
        audio=video and audio,
        seconds=seconds,
        orientation=_orientation(width, height, ratio),
        reference_image=reference_image,
        extends=extending,
        previous=previous if extending else None,
        extract_from=extract_from if extending else None,
        prompt_is_original=extending and prompt_is_original,
        new_seconds=new_frames / fps if extending and new_frames and fps else None,
    )


def _orientation(width: float | None, height: float | None, ratio: str | None) -> str | None:
    """Return ``portrait``, ``landscape`` or ``square`` for the frame, or None when its shape is unknown."""
    if not (width and height) and ratio:
        w_text, _sep, h_text = ratio.partition(":")
        try:
            width, height = float(w_text), float(h_text)
        except ValueError:
            return None
    if not width or not height or width <= 0 or height <= 0:
        return None
    if width == height:
        return "square"
    return "portrait" if height > width else "landscape"


def _check_mode(mode: str) -> None:
    if mode not in ENHANCE_MODES:
        raise ValueError(f"Unknown enhancement mode '{mode}'. Valid values: {', '.join(ENHANCE_MODES)}.")


def _parse_axis_value(axis: EnhanceAxis, raw: Any) -> str | tuple[str, ...]:
    if axis.multi:
        if raw is None:
            items: list[str] = []
        elif isinstance(raw, str):
            items = [part.strip() for part in raw.split("+") if part.strip()]
        elif isinstance(raw, (list, tuple)):
            items = [str(part).strip() for part in raw if str(part).strip()]
        else:
            raise ValueError(f"{axis.label} must be a list of values. Valid values: {', '.join(axis.slugs())}.")
        for item in items:
            axis.option(item)
        return tuple(dict.fromkeys(items))
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"{axis.label} must be one of: {', '.join(axis.slugs())}.")
    return axis.option(raw.strip()).slug


def _parse_parts(raw: Any) -> str | None:
    """Return the ``parts`` slug, or None (auto) for a missing value."""
    if raw is None:
        return None
    if not isinstance(raw, str) or raw.strip() not in PARTS_VALUES:
        raise ValueError(f"Rewrite must be one of: {', '.join(PARTS_VALUES)}.")
    return raw.strip()


def _parse_note(raw: Any) -> str:
    """Return the direction note as one trimmed line."""
    if raw is None:
        return ""
    if not isinstance(raw, str):
        raise ValueError(f"Direction must be text, got {type(raw).__name__}.")
    note = " ".join(raw.split())
    if len(note) > NOTE_MAX_CHARS:
        raise ValueError(f"Direction is too long ({len(note)} characters); keep it to {NOTE_MAX_CHARS}.")
    return note


def settings_from_mapping(data: dict[str, Any], *, mode: str, motion_in_image: str = "error") -> EnhanceSettings:
    """Build settings from a mapping; omitted keys keep their defaults.

    Args:
        data: Mapping with any of ``style``, ``mood``, ``details``, ``length``, ``motion``, ``sound``, ``note``, ``parts``.
        mode: ``"image"`` or ``"video"``.
        motion_in_image: ``"error"`` to reject the video-only keys (``motion``, ``sound``, ``parts``) in image mode,
            ``"warn"`` to warn and ignore them.

    Raises:
        ValueError: On unknown keys or values, or a video-only key in image mode with ``motion_in_image="error"``.
    """
    _check_mode(mode)
    if not isinstance(data, dict):
        raise ValueError(f"Enhancement settings must be a mapping, got {type(data).__name__}.")
    unknown = sorted(set(data) - set(_SETTING_KEYS))
    if unknown:
        raise ValueError(f"Unknown enhancement setting(s): {', '.join(unknown)}. Valid keys: {', '.join(_SETTING_KEYS)}.")
    values: dict[str, Any] = {}
    for key, raw in data.items():
        if key == NOTE_KEY:
            values[key] = _parse_note(raw)
            continue
        if (key == PARTS_KEY or _AXES_BY_KEY[key].video_only) and mode != "video":
            if motion_in_image == "warn":
                warnings.warn(f"'{key}' only applies to video prompts and is ignored for images.", stacklevel=2)
                continue
            raise ValueError(f"'{key}' only applies to video prompts.")
        values[key] = _parse_parts(raw) if key == PARTS_KEY else _parse_axis_value(_AXES_BY_KEY[key], raw)
    return EnhanceSettings(**values)


def parse_enhance_spec(spec: str | None, *, mode: str) -> EnhanceSettings:
    """Parse a CLI spec such as ``style=cinematic,details=lighting+camera,length=longer``.

    An empty or ``None`` spec returns the defaults. The free-text ``note`` is not part of the
    grammar (it may contain commas); the CLIs take it from ``--enhance-note``.
    """
    _check_mode(mode)
    if spec is None or not spec.strip():
        return EnhanceSettings()
    data: dict[str, str] = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        key, sep, value = part.partition("=")
        if not sep:
            raise ValueError(f"Invalid enhancement setting '{part}'. Use key=value, e.g. style=cinematic,details=lighting+camera,length=longer.")
        if key.strip() == NOTE_KEY:
            raise ValueError("Set the direction with --enhance-note, not in the --enhance spec.")
        data[key.strip()] = value.strip()
    return settings_from_mapping(data, mode=mode)


def format_enhance_spec(settings: EnhanceSettings, *, mode: str = "video") -> str:
    """Serialize *settings* to the CLI spec grammar (``motion``, a non-empty ``sound`` and a set ``parts`` only for video; ``note`` is left out)."""
    parts = [f"style={settings.style}", f"mood={settings.mood}", f"details={'+'.join(settings.details)}", f"length={settings.length}"]
    if mode == "video":
        parts.append(f"motion={'+'.join(settings.motion)}")
        if settings.sound:
            parts.append(f"sound={'+'.join(settings.sound)}")
        if settings.parts is not None:
            parts.append(f"{PARTS_KEY}={settings.parts}")
    return ",".join(parts)


def parse_enhance_entry(value: Any, *, mode: str, where: str) -> EnhanceSettings | None:
    """Parse a prompt-file entry's ``enhance:`` value (``true``, ``false``, or a mapping).

    Raises:
        ValueError: Naming *where* when the value is invalid.
    """
    if value is None or value is False:
        return None
    if value is True:
        return EnhanceSettings()
    if isinstance(value, dict):
        try:
            return settings_from_mapping(value, mode=mode, motion_in_image="warn")
        except ValueError as exc:
            raise ValueError(f"Invalid 'enhance' in {where}: {exc}") from exc
    raise ValueError(f"Invalid 'enhance' in {where}: expected true, false, or a mapping, got {type(value).__name__}.")


def validate_settings(settings: EnhanceSettings, *, mode: str, split: bool = False, caption: bool = False) -> None:
    """Reject unknown slugs and the no-op combination (keep style and mood, no details, same length, no motion, sound or direction).

    Args:
        split: Whether a video prompt has no scene, so the rewrite splits it into scene and action (never a no-op).
            ``parts="action"`` is never a no-op either.
        caption: Whether the rewrite writes a JSON caption, which is never a no-op.
    """
    _check_mode(mode)
    if settings.parts is not None and settings.parts not in PARTS_VALUES:
        raise ValueError(f"Rewrite must be one of: {', '.join(PARTS_VALUES)}.")
    STYLE_AXIS.option(settings.style)
    MOOD_AXIS.option(settings.mood)
    LENGTH_AXIS.option(settings.length)
    for slug in settings.details:
        DETAILS_AXIS.option(slug)
    for slug in settings.motion:
        MOTION_AXIS.option(slug)
    for slug in settings.sound:
        SOUND_AXIS.option(slug)
    if not caption and is_noop(settings, mode=mode, split=split):
        raise ValueError("Nothing to enhance: pick a style, a mood, a detail, a length, or write a direction.")


def resolve_enhance_ceiling(config: dict[str, Any], *, family: str | None, mode: str) -> int:
    """Return the max enhanced-prompt word count for a model family (per-family override or global)."""
    presets_key = "video_model_presets" if mode == "video" else "model_presets"
    preset = (config.get(presets_key) or {}).get(family or "", {}) or {}
    value = preset.get("enhance_max_words")
    if isinstance(value, int) and value > 0:
        return value
    return int(length_config(config)["max_words"])


def length_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return the ``prompt_enhancer.length`` section merged over the built-in defaults."""
    section = ((config.get("prompt_enhancer") or {}).get("length") or {}) if isinstance(config, dict) else {}
    return {
        "percent": {**DEFAULT_LENGTH_CONFIG["percent"], **(section.get("percent") or {})},
        "floor_words": {**DEFAULT_LENGTH_CONFIG["floor_words"], **(section.get("floor_words") or {})},
        "max_words": section.get("max_words", DEFAULT_LENGTH_CONFIG["max_words"]),
    }


def enhance_options(config: dict[str, Any]) -> dict[str, Any]:
    """Return enhancer knobs (temperature, max tokens, idle release, length config) from ``prompt_enhancer`` config."""
    section = (config.get("prompt_enhancer") or {}) if isinstance(config, dict) else {}
    return {
        "temperature": float(section.get("temperature", DEFAULT_TEMPERATURE)),
        "max_new_tokens": int(section.get("max_new_tokens", DEFAULT_MAX_NEW_TOKENS)),
        "idle_release_seconds": float(section.get("idle_release_seconds", DEFAULT_IDLE_RELEASE_SECONDS)),
        "length": length_config(config),
    }


def resolve_item_enhance(*, disabled: bool, override: EnhanceSettings | None, entry: EnhanceSettings | None) -> EnhanceSettings | None:
    """Apply enhancement precedence for one prompt: disabled > job-wide override > entry YAML > off."""
    if disabled:
        return None
    if override is not None:
        return override
    return entry


def entry_enhance(enhance_by_set: dict[str, list[EnhanceSettings | None]] | None, set_name: str, index: int) -> EnhanceSettings | None:
    """Return the YAML ``enhance:`` settings for prompt *index* of *set_name*, if any."""
    entries = (enhance_by_set or {}).get(set_name) or []
    return entries[index] if 0 <= index < len(entries) else None


def entry_scene(scene_by_set: dict[str, list[str | None]] | None, set_name: str, index: int, job_scene: str | None) -> str | None:
    """Return the trimmed ``scene:`` for prompt *index* of *set_name*, else the trimmed *job_scene*, else ``None``; blank texts are skipped."""
    scenes = (scene_by_set or {}).get(set_name) or []
    own = scenes[index] if 0 <= index < len(scenes) else None
    for candidate in (own, job_scene):
        if candidate and candidate.strip():
            return candidate.strip()
    return None


def resolve_parts(settings: EnhanceSettings, *, scene: str | None, extends: bool = False, reference_image: bool = False) -> str:
    """Return ``"action"`` or ``"both"``: which parts of a video prompt one rewrite writes.

    An explicit ``parts`` wins, except that ``"action"`` needs a scene to keep: with a blank *scene* it falls back to
    ``"both"`` (the text is split into scene and action), without error. With ``parts`` unset (auto), an extension of a
    clip that has a scene writes only the action, and so does an image-to-video job with a scene (the scene describes
    the starting image, which the rewrite must not restyle); every other video job writes both.
    """
    has_scene = bool((scene or "").strip())
    wanted = settings.parts if settings.parts is not None else ("action" if extends or reference_image else "both")
    return "action" if wanted == "action" and has_scene else "both"


def is_noop(settings: EnhanceSettings, *, mode: str, split: bool = False) -> bool:
    """Return whether *settings* ask for no change in *mode* (keep style and mood, no details, same length, no direction, no video motion or sound).

    A video prompt that *split* into scene and action (blank scene, non-blank prompt) is always a change, and so is
    ``parts="action"``: the action is rewritten to fit the scene (or the text is split when there is no scene).
    """
    if mode == "video" and (split or settings.parts == "action"):
        return False
    has_video_axes = mode == "video" and bool(settings.motion or settings.sound)
    return settings.style == "keep" and settings.mood == "keep" and not settings.details and settings.length == "same" and not settings.note and not has_video_axes


def enhance_by_set_for_mode(
    enhance_by_set: dict[str, list[EnhanceSettings | None]] | None,
    *,
    mode: str,
    scene_by_set: dict[str, list[str | None]] | None = None,
    job_scene: str | None = None,
    caption: bool = False,
) -> dict[str, list[EnhanceSettings | None]] | None:
    """Drop prompt-file entries that change nothing in *mode* (e.g. motion-only entries in an image run), warning once.

    Args:
        scene_by_set: Each entry's scene, aligned with *enhance_by_set*; video entries without a scene split their prompt into scene and action.
        job_scene: The scene used by entries that have none of their own.
        caption: The job writes JSON captions, which always change the prompt, so nothing is dropped.
    """
    if not enhance_by_set or caption:
        return enhance_by_set
    dropped = 0
    result: dict[str, list[EnhanceSettings | None]] = {}
    for set_name, entries in enhance_by_set.items():
        kept: list[EnhanceSettings | None] = []
        for index, entry in enumerate(entries):
            split = entry_scene(scene_by_set, set_name, index, job_scene) is None
            if entry is not None and is_noop(entry, mode=mode, split=split):
                dropped += 1
                entry = None
            kept.append(entry)
        result[set_name] = kept
    if dropped:
        noun = "entry asks" if dropped == 1 else "entries ask"
        media = "images" if mode == "image" else "videos"
        warnings.warn(f"{dropped} prompt-file 'enhance:' {noun} for no change in {media} (keep style and mood, no details, same length, no direction); not enhanced.", stacklevel=2)
    return result


def enhancement_requested(*, disabled: bool, override: EnhanceSettings | None, enhance_by_set: dict[str, list[EnhanceSettings | None]] | None) -> bool:
    """Return whether any prompt in the job will be enhanced."""
    if disabled:
        return False
    return override is not None or any(entry is not None for entries in (enhance_by_set or {}).values() for entry in entries)


def plan_length(in_words: int, length: str, *, ceiling: int, length_cfg: dict[str, Any] | None = None) -> LengthPlan:
    """Turn a Length choice into a word target and range (percent of input, floors, ceiling, clamp-to-same).

    Same keeps the input's length but grows a short prompt to its floor, since the rewrite adds the chosen details.
    A Shorter, Longer or Extra request that cannot apply keeps the input's own length, without the floor.
    """
    LENGTH_AXIS.option(length)
    cfg = length_cfg or DEFAULT_LENGTH_CONFIG
    in_words = max(1, in_words)
    same_target = min(in_words, ceiling)
    clamped = False
    if length == "same":
        target = min(max(in_words, int(cfg["floor_words"].get("same", 1))), ceiling)
    else:
        target = round(in_words * cfg["percent"][length] / 100)
        target = max(target, int(cfg["floor_words"].get(length, 1)))
        target = min(target, ceiling)
        if (length == "shorter" and target >= in_words) or (length in ("longer", "extra") and target <= in_words):
            target, clamped = same_target, True
    span = max(3, round(target * 0.15))
    # The upper bound is what the model is told not to exceed, so it must respect the text-encoder ceiling too.
    return LengthPlan(length=length, target=target, lo=max(1, target - span), hi=min(target + span, max(ceiling, target)), clamped=clamped)


def length_instruction(plan: LengthPlan, label: str = "Length") -> str:
    if plan.length == "shorter" and not plan.clamped:
        return f"{label}: at most {plan.hi} words (aim for {plan.target}). Cut only redundancy and filler, never an element the user named."
    if plan.length == "same" or plan.clamped:
        return f"{label}: {plan.lo}-{plan.hi} words. Do not exceed {plan.hi} words."
    return f"{label}: {plan.lo}-{plan.hi} words."


def word_count(text: str) -> int:
    """Count whitespace-separated tokens that contain a letter or digit (stray punctuation is not a word)."""
    return sum(1 for token in text.split() if any(char.isalnum() for char in token))


def strip_reasoning(text: str) -> str:
    """Return *text* without ``<think>`` reasoning blocks, including one that is still open."""
    return _THINK_RE.sub("", text)


def duration_beats(seconds: float) -> str:
    """Return how many beats of action fit *seconds*: one continuous action, two or three beats, or a short sequence."""
    return next((text for limit, text in _DURATION_BEATS if seconds <= limit), "a short sequence of beats")


def clean_output(text: str) -> str:
    """Strip reasoning blocks, special tokens, leading labels, and wrapping quotes from model output."""
    text = strip_reasoning(text)
    text = _SPECIAL_TOKEN_RE.sub("", text)
    text = _LABEL_RE.sub("", text.strip())
    text = text.strip()
    while len(text) >= 2 and _QUOTE_PAIRS.get(text[0]) == text[-1] and not _has_inner_quote(text):
        text = text[1:-1].strip()
    return text


def _has_inner_quote(text: str) -> bool:
    """Return whether the wrapping quote characters also appear inside, so they are not one pair around everything."""
    inner = text[1:-1]
    return text[0] in inner or text[-1] in inner


def normalize_text(text: str) -> str:
    return " ".join(text.lower().split())


def is_unusable(output: str, original: str) -> bool:
    """Return whether *output* is empty, an echo of *original*, or starts with a refusal."""
    if not output.strip():
        return True
    if normalize_text(output) == normalize_text(original):
        return True
    return bool(_REFUSAL_RE.match(output))


def perception_rule(mode: str, context: EnhanceContext) -> str:
    """Return the rule on which senses the rewrite may describe (sound only for video with audio)."""
    if mode == "video" and context.audio:
        return (
            "Describe only what is seen and heard (visible motion, camera movement and sounds, in time order): "
            "no smells, temperatures felt, thoughts, backstory or narrative commentary. "
            "Keep every sound and every line of quoted speech the user wrote, word for word."
        )
    visible = "Describe only what is visible"
    if mode == "video":
        visible += " (and for video: visible motion and camera movement, in time order)"
    return f"{visible}: no sounds, smells, temperatures felt, thoughts, backstory or narrative commentary."


def enforced_quotes(text: str, *, mode: str, context: EnhanceContext) -> tuple[str, ...]:
    """Return the quoted texts of *text* a rewrite must copy exactly; a silent video keeps lettering only, not speech."""
    return lettering_spans(text) if mode == "video" and not context.audio else quoted_texts(text)


def quote_rules(spans: tuple[str, ...]) -> list[str]:
    """Return the rule that lists the quoted texts to copy exactly, or none when there are none."""
    if not spans:
        return []
    listed = ", ".join(f'"{span}"' for span in spans)
    return [f"Text in quotes is lettering or speech: copy it exactly, with its case, punctuation and quote marks: {listed}."]


def frame_rule(orientation: str) -> str:
    """Return the rule that fits the composition to a ``portrait``, ``landscape`` or ``square`` frame without naming it."""
    return f"Compose for a {orientation} frame; never mention the frame or its shape."


def context_rules(mode: str, context: EnhanceContext) -> list[str]:
    """Return the rules for the reference image, frame shape and clip length."""
    rules: list[str] = []
    if context.reference_image:
        if mode == "video":
            rules.append(
                "The first frame is supplied as an image. Describe the motion, camera movement and changes over time; "
                "refer to the subject briefly and don't add appearance details the image already shows."
            )
        else:
            rules.append("A reference image supplies the subject and layout. Refer to the subject briefly; apply the style and mood without re-describing its appearance.")
    if context.orientation is not None:
        rules.append(frame_rule(context.orientation))
    if mode == "video" and context.seconds is not None:
        beats = duration_beats(context.seconds)
        rules.append(f"Duration: about {max(1, round(context.seconds))} seconds; fit the action to it: {beats}.")
    return rules


def action_rules(settings: EnhanceSettings, context: EnhanceContext, *, for_pair: bool = False) -> list[str]:
    """Return the motion and sound rules of a video rewrite; sound only when the job has an audio track.

    They open with their own names ("Motion: describe ..."), or with "For the ACTION" in a scene-and-action rewrite.
    """
    motion_lead, sound_lead = ("For the ACTION", "For the ACTION") if for_pair else ("Motion", "Sound")
    rules = []
    if settings.motion:
        rules.append(f"{motion_lead}: describe {', '.join(MOTION_AXIS.option(slug).instruction for slug in settings.motion)}.")
    if context.audio and settings.sound:
        sounds = "; ".join(SOUND_AXIS.option(slug).instruction for slug in settings.sound)
        rules.append(f"{sound_lead}: describe {sounds}. Put spoken lines in double quotes; only people already in the scene speak.")
    return rules


def direction_rules(settings: EnhanceSettings, *, both_parts: bool = False) -> list[str]:
    """Return the rule that carries the user's free-text direction, or none when there is no direction."""
    if not settings.note:
        return []
    return [f"Direction from the user (follow it{' in both parts' if both_parts else ''}): {settings.note}"]


def system_message(intro: str, rules: list[str], closing: str) -> str:
    """Return a rewrite's system message: *intro*, the *rules* as a bulleted list, then *closing* (the output format)."""
    return f"{intro}\nRules:\n" + "\n".join(f"- {rule}" for rule in rules) + f"\n{closing}"


def build_messages(prompt: str, settings: EnhanceSettings, *, mode: str, plan: LengthPlan, context: EnhanceContext | None = None) -> list[dict[str, str]]:
    """Build the system + user chat messages for one enhancement."""
    _check_mode(mode)
    context = context or EnhanceContext()
    kind = "text-to-video" if mode == "video" else "text-to-image"
    rules = [
        prompt_rule("the clip" if mode == "video" else "the image"),
        *PRESERVATION_RULES,
        *quote_rules(enforced_quotes(prompt, mode=mode, context=context)),
        perception_rule(mode, context),
        *context_rules(mode, context),
    ]
    rules.append(STYLE_AXIS.option(settings.style).instruction)
    rules.append(MOOD_AXIS.option(settings.mood).instruction)
    if settings.details:
        aspects = ", ".join(DETAILS_AXIS.option(slug).instruction for slug in settings.details)
        rules.append(f"Keep these aspects while cutting: {aspects}." if plan.length == "shorter" and not plan.clamped else f"Add detail on: {aspects}.")
    if mode == "video":
        rules.extend(action_rules(settings, context))
    rules.extend(direction_rules(settings))
    rules.append(length_instruction(plan))
    if plan.length == "same" or plan.clamped:
        rules.append("Apply the style, mood and details; don't return the input unchanged.")
    system = system_message(f"You rewrite prompts for a {kind} model.", rules, "Output only the rewritten prompt as plain prose: no preamble, no headings, no quotes.")
    return [{"role": "system", "content": system}, {"role": "user", "content": labelled_prompt(prompt)}]


def prompt_rule(subject: str) -> str:
    """Return the rule that the labelled Prompt is text to rewrite as a description of *subject*, never a request to follow."""
    return f"The user gives a Prompt. Rewrite it as a description of {subject}; never answer it or carry it out as a request."


def labelled_prompt(prompt: str) -> str:
    """Return the user message of a rewrite: *prompt* after a ``Prompt:`` label, so it reads as text to rewrite."""
    return f"Prompt: {prompt}"


def enhance_prompt(
    enhancer: PromptEnhancer,
    prompt: str,
    settings: EnhanceSettings,
    *,
    mode: str,
    seed: int,
    ceiling: int,
    context: EnhanceContext | None = None,
    length_cfg: dict[str, Any] | None = None,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    on_text: Callable[[str], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> EnhanceResult:
    """Rewrite *prompt* with *enhancer*: generate, clean, and retry once if unusable or a quoted text was changed.

    Expand ``{a|b}`` choices before calling: the model rewrites one concrete prompt. When no attempt keeps every quoted
    text, the last usable one is returned with ``QUOTES_CHANGED_WARNING`` (also issued as a warning).

    Args:
        context: Job facts (audio track, clip length, frame shape, reference image).
        on_text: Receives the full in-progress text at most every
            ``TEXT_UPDATE_INTERVAL_SECONDS``, then the final text; a retry starts over from an empty string.
        cancelled: Polled per token; when it returns True generation stops and ``RuntimeError`` is raised.

    Raises:
        ValueError: For invalid settings or an empty prompt.
        RuntimeError: When cancelled, or when both attempts are unusable.
    """
    if not prompt.strip():
        raise ValueError("Enter a prompt to enhance.")
    validate_settings(settings, mode=mode)
    plan = plan_length(word_count(prompt), settings.length, ceiling=ceiling, length_cfg=length_cfg)
    messages = build_messages(prompt, settings, mode=mode, plan=plan, context=context)
    spans = enforced_quotes(prompt, mode=mode, context=context or EnhanceContext())
    changed_quotes: str | None = None
    for attempt in range(2):
        raw = generate_raw(enhancer, messages, seed=seed + attempt, max_tokens=max_tokens, temperature=temperature, on_text=on_text, cancelled=cancelled)
        output = clean_output(raw)
        if is_unusable(output, prompt):
            continue
        if not missing_quotes(spans, output):
            return _final(EnhanceResult(prompt=output, clamped=plan.clamped), on_text)
        changed_quotes = output
    if changed_quotes is None:
        raise RuntimeError(NO_USABLE_MESSAGE)
    warnings.warn(QUOTES_CHANGED_WARNING, stacklevel=2)
    return _final(EnhanceResult(prompt=changed_quotes, clamped=plan.clamped, warning=QUOTES_CHANGED_WARNING), on_text)


def _final(result: EnhanceResult, on_text: Callable[[str], None] | None) -> EnhanceResult:
    """Report *result*'s prompt as the final text and return it."""
    if on_text is not None:
        on_text(result.prompt)
    return result


def generate_raw(
    enhancer: PromptEnhancer,
    messages: list[dict[str, str]],
    *,
    seed: int,
    max_tokens: int,
    temperature: float,
    on_text: Callable[[str], None] | None,
    cancelled: Callable[[], bool] | None,
) -> str:
    """Stream one generation, reporting throttled cleaned text to *on_text*, and return the raw text.

    Raises:
        RuntimeError: When *cancelled* returns True.
    """
    raw = ""
    last_update = float("-inf")
    if on_text is not None:
        on_text("")
    for delta in enhancer.generate(messages, seed=seed, max_tokens=max_tokens, temperature=temperature, cancelled=cancelled):
        raw += delta
        now = time.monotonic()
        if on_text is not None and now - last_update >= TEXT_UPDATE_INTERVAL_SECONDS:
            last_update = now
            on_text(clean_output(raw))
    if cancelled is not None and cancelled():
        raise RuntimeError("Prompt enhancement was cancelled.")
    return raw
