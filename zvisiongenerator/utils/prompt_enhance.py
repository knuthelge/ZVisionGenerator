"""Build, run, and post-process local-LLM prompt enhancement (pure logic, no model I/O)."""

from __future__ import annotations

import re
import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from zvisiongenerator.core.prompt_enhancer import PromptEnhancer

ENHANCE_MODES = ("image", "video")
DEFAULT_LENGTH_CONFIG: dict[str, Any] = {
    "percent": {"shorter": 50, "same": 100, "longer": 200, "extra": 300},
    "floor_words": {"shorter": 12, "longer": 40, "extra": 80},
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
ENHANCE_AXES: tuple[EnhanceAxis, ...] = (STYLE_AXIS, DETAILS_AXIS, LENGTH_AXIS, MOTION_AXIS)
_AXES_BY_KEY = {axis.key: axis for axis in ENHANCE_AXES}


@dataclass(frozen=True)
class EnhanceSettings:
    """A choice on every axis of the option matrix."""

    style: str = STYLE_AXIS.default[0]
    details: tuple[str, ...] = DETAILS_AXIS.default
    length: str = LENGTH_AXIS.default[0]
    motion: tuple[str, ...] = MOTION_AXIS.default


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
    """The final enhanced prompt and whether its length request was clamped."""

    prompt: str
    clamped: bool


def matrix_contract() -> dict[str, Any]:
    """Serialize the option matrix and defaults for the Web UI."""
    return {
        "axes": [
            {
                "key": axis.key,
                "label": axis.label,
                "multi": axis.multi,
                "video_only": axis.video_only,
                "options": [{"slug": option.slug, "label": option.label} for option in axis.options],
                "default": list(axis.default),
            }
            for axis in ENHANCE_AXES
        ],
        "defaults": settings_to_mapping(EnhanceSettings()),
    }


def settings_to_mapping(settings: EnhanceSettings) -> dict[str, Any]:
    """Return *settings* as a JSON-friendly mapping."""
    return {"style": settings.style, "details": list(settings.details), "length": settings.length, "motion": list(settings.motion)}


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


def settings_from_mapping(data: dict[str, Any], *, mode: str, motion_in_image: str = "error") -> EnhanceSettings:
    """Build settings from a mapping; omitted axes keep their defaults.

    Args:
        data: Mapping with any of ``style``, ``details``, ``length``, ``motion``.
        mode: ``"image"`` or ``"video"``.
        motion_in_image: ``"error"`` to reject ``motion`` in image mode, ``"warn"`` to warn and ignore it.

    Raises:
        ValueError: On unknown keys or values, or ``motion`` in image mode with ``motion_in_image="error"``.
    """
    _check_mode(mode)
    if not isinstance(data, dict):
        raise ValueError(f"Enhancement settings must be a mapping, got {type(data).__name__}.")
    unknown = sorted(set(data) - set(_AXES_BY_KEY))
    if unknown:
        raise ValueError(f"Unknown enhancement setting(s): {', '.join(unknown)}. Valid keys: {', '.join(_AXES_BY_KEY)}.")
    values: dict[str, Any] = {}
    for key, raw in data.items():
        axis = _AXES_BY_KEY[key]
        if axis.video_only and mode != "video":
            if motion_in_image == "warn":
                warnings.warn(f"'{key}' only applies to video prompts and is ignored for images.", stacklevel=2)
                continue
            raise ValueError(f"'{key}' only applies to video prompts.")
        values[key] = _parse_axis_value(axis, raw)
    return EnhanceSettings(**values)


def parse_enhance_spec(spec: str | None, *, mode: str) -> EnhanceSettings:
    """Parse a CLI spec such as ``style=cinematic,details=lighting+camera,length=longer``.

    An empty or ``None`` spec returns the defaults.
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
        data[key.strip()] = value.strip()
    return settings_from_mapping(data, mode=mode)


def format_enhance_spec(settings: EnhanceSettings, *, mode: str = "video") -> str:
    """Serialize *settings* to the CLI spec grammar (``motion`` only for video)."""
    parts = [f"style={settings.style}", f"details={'+'.join(settings.details)}", f"length={settings.length}"]
    if mode == "video":
        parts.append(f"motion={'+'.join(settings.motion)}")
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


def validate_settings(settings: EnhanceSettings, *, mode: str) -> None:
    """Reject unknown slugs and the no-op combination (keep style, no details, same length, no motion)."""
    _check_mode(mode)
    STYLE_AXIS.option(settings.style)
    LENGTH_AXIS.option(settings.length)
    for slug in settings.details:
        DETAILS_AXIS.option(slug)
    for slug in settings.motion:
        MOTION_AXIS.option(slug)
    if is_noop(settings, mode=mode):
        raise ValueError("Nothing to enhance: pick a style, a detail, or a length.")


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


def is_noop(settings: EnhanceSettings, *, mode: str) -> bool:
    """Return whether *settings* ask for no change in *mode* (keep style, no details, same length, no video motion)."""
    has_motion = mode == "video" and bool(settings.motion)
    return settings.style == "keep" and not settings.details and settings.length == "same" and not has_motion


def enhance_by_set_for_mode(enhance_by_set: dict[str, list[EnhanceSettings | None]] | None, *, mode: str) -> dict[str, list[EnhanceSettings | None]] | None:
    """Drop prompt-file entries that change nothing in *mode* (e.g. motion-only entries in an image run), warning once."""
    if not enhance_by_set:
        return enhance_by_set
    dropped = 0
    result: dict[str, list[EnhanceSettings | None]] = {}
    for set_name, entries in enhance_by_set.items():
        kept: list[EnhanceSettings | None] = []
        for entry in entries:
            if entry is not None and is_noop(entry, mode=mode):
                dropped += 1
                entry = None
            kept.append(entry)
        result[set_name] = kept
    if dropped:
        noun = "entry asks" if dropped == 1 else "entries ask"
        warnings.warn(f"{dropped} prompt-file 'enhance:' {noun} for no change in images (keep style, no details, same length); not enhanced.", stacklevel=2)
    return result


def enhancement_requested(*, disabled: bool, override: EnhanceSettings | None, enhance_by_set: dict[str, list[EnhanceSettings | None]] | None) -> bool:
    """Return whether any prompt in the job will be enhanced."""
    if disabled:
        return False
    return override is not None or any(entry is not None for entries in (enhance_by_set or {}).values() for entry in entries)


def plan_length(in_words: int, length: str, *, ceiling: int, length_cfg: dict[str, Any] | None = None) -> LengthPlan:
    """Turn a Length choice into a word target and range (percent of input, floors, ceiling, clamp-to-same)."""
    LENGTH_AXIS.option(length)
    cfg = length_cfg or DEFAULT_LENGTH_CONFIG
    in_words = max(1, in_words)
    same_target = min(in_words, ceiling)
    clamped = False
    if length == "same":
        target = same_target
    else:
        target = round(in_words * cfg["percent"][length] / 100)
        target = max(target, int(cfg["floor_words"].get(length, 1)))
        target = min(target, ceiling)
        if (length == "shorter" and target >= in_words) or (length in ("longer", "extra") and target <= in_words):
            target, clamped = same_target, True
    span = max(3, round(target * 0.15))
    # The upper bound is what the model is told not to exceed, so it must respect the text-encoder ceiling too.
    return LengthPlan(length=length, target=target, lo=max(1, target - span), hi=min(target + span, max(ceiling, target)), clamped=clamped)


def _length_instruction(plan: LengthPlan) -> str:
    if plan.length == "shorter" and not plan.clamped:
        return f"Length: at most {plan.hi} words (aim for {plan.target}). Cut only redundancy and filler, never an element the user named."
    if plan.length == "same" or plan.clamped:
        return f"Length: {plan.lo}-{plan.hi} words. Do not exceed {plan.hi} words."
    return f"Length: {plan.lo}-{plan.hi} words."


def _word_count(text: str) -> int:
    """Count whitespace-separated tokens that contain a letter or digit (stray punctuation is not a word)."""
    return sum(1 for token in text.split() if any(char.isalnum() for char in token))


def clean_output(text: str) -> str:
    """Strip reasoning blocks, special tokens, leading labels, and wrapping quotes from model output."""
    text = _THINK_RE.sub("", text)
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


def _normalized(text: str) -> str:
    return " ".join(text.lower().split())


def is_unusable(output: str, original: str) -> bool:
    """Return whether *output* is empty, an echo of *original*, or starts with a refusal."""
    if not output.strip():
        return True
    if _normalized(output) == _normalized(original):
        return True
    return bool(_REFUSAL_RE.match(output))


def build_messages(prompt: str, settings: EnhanceSettings, *, mode: str, plan: LengthPlan) -> list[dict[str, str]]:
    """Build the system + user chat messages for one enhancement."""
    _check_mode(mode)
    kind = "text-to-video" if mode == "video" else "text-to-image"
    visible = "Describe only what is visible"
    if mode == "video":
        visible += " (and for video: visible motion and camera movement, in time order)"
    rules = [
        "EVERY DETAIL IS IMPORTANT: keep the user's subject, intent and every element they mention.",
        "Never add new people, animals, characters or major objects.",
        f"{visible}: no sounds, smells, temperatures felt, thoughts, backstory or narrative commentary.",
    ]
    rules.append(STYLE_AXIS.option(settings.style).instruction)
    if settings.details:
        aspects = ", ".join(DETAILS_AXIS.option(slug).instruction for slug in settings.details)
        rules.append(f"Keep these aspects while cutting: {aspects}." if plan.length == "shorter" and not plan.clamped else f"Add detail on: {aspects}.")
    if mode == "video" and settings.motion:
        rules.append(f"Motion: describe {', '.join(MOTION_AXIS.option(slug).instruction for slug in settings.motion)}.")
    rules.append(_length_instruction(plan))
    if plan.length == "same" or plan.clamped:
        rules.append("Apply the style and details; don't return the input unchanged.")
    system = f"You rewrite prompts for a {kind} model.\nRules:\n" + "\n".join(f"- {rule}" for rule in rules)
    system += "\nOutput only the rewritten prompt as plain prose: no preamble, no headings, no quotes."
    return [{"role": "system", "content": system}, {"role": "user", "content": prompt}]


def enhance_prompt(
    enhancer: PromptEnhancer,
    prompt: str,
    settings: EnhanceSettings,
    *,
    mode: str,
    seed: int,
    ceiling: int,
    length_cfg: dict[str, Any] | None = None,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    on_text: Callable[[str], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> EnhanceResult:
    """Rewrite *prompt* with *enhancer*: generate, clean, and retry once if unusable.

    Expand ``{a|b}`` choices before calling: the model rewrites one concrete prompt.

    Args:
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
    plan = plan_length(_word_count(prompt), settings.length, ceiling=ceiling, length_cfg=length_cfg)
    messages = build_messages(prompt, settings, mode=mode, plan=plan)
    for attempt in range(2):
        raw = ""
        last_update = float("-inf")
        if on_text is not None:
            on_text("")
        for delta in enhancer.generate(messages, seed=seed + attempt, max_tokens=max_tokens, temperature=temperature, cancelled=cancelled):
            raw += delta
            now = time.monotonic()
            if on_text is not None and now - last_update >= TEXT_UPDATE_INTERVAL_SECONDS:
                last_update = now
                on_text(clean_output(raw))
        if cancelled is not None and cancelled():
            raise RuntimeError("Prompt enhancement was cancelled.")
        output = clean_output(raw)
        if not is_unusable(output, prompt):
            if on_text is not None:
                on_text(output)
            return EnhanceResult(prompt=output, clamped=plan.clamped)
    raise RuntimeError("The enhancer returned no usable rewrite (empty, unchanged, or a refusal). Try different settings.")
