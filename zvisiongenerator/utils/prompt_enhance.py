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

_WORD_RE = re.compile(r"\w+")
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
REFERENCES_DROPPED_WARNING = "A reference image named in the prompt was dropped from the rewrite."
_REFERENCE_MENTION_RE = re.compile(r"\bimage\s+(\d+)\b", re.I)
NO_USABLE_MESSAGE = "The enhancer returned no usable rewrite (empty, unchanged, or a refusal). Try different settings."


@dataclass(frozen=True)
class EnhanceOption:
    """One selectable value on an enhancement axis.

    Attributes:
        description: One line for the Web UI menu: what the option asks the enhancer for.
        group: The menu group it is listed under (Style: Photo or Art; Theme: its five groups), or None.
    """

    slug: str
    label: str
    instruction: str
    description: str = ""
    group: str | None = None


@dataclass(frozen=True)
class EnhanceAxis:
    """One axis of the enhancement option matrix; a single-choice axis starts with its ``keep`` option where it has one."""

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


KEEP = "keep"
# Every Photo style asks for a real photograph: the camera, the light source and natural skin and surface texture, without
# the words that push image models toward a glossy, waxy, rendered look. Listing pores, wear or dust as examples made the
# enhancer age people and dirty scenes (bench picture sheet), so the rule names the look to avoid instead.
PHOTO_REALISM = (
    "Write it as a real photograph: say what camera or lens took it and where the light comes from. "
    "Show skin and surfaces as a camera sees them: natural skin texture, not airbrushed or waxy; "
    "never make people older, dirtier or more worn than the prompt says. "
    'Use plain, specific words; never "hyperrealistic", "8k", "masterpiece", "flawless", "perfect" or "smooth skin".'
)


def _photo(slug: str, label: str, what: str, description: str) -> EnhanceOption:
    return EnhanceOption(slug, label, f"Style: {what} {PHOTO_REALISM}", description, "Photo")


def _art(slug: str, label: str, what: str, description: str) -> EnhanceOption:
    return EnhanceOption(slug, label, f"Style: {what}", description, "Art")


def _keep(axis: str) -> EnhanceOption:
    return EnhanceOption(KEEP, "Keep", f"{axis}: keep the user's existing {axis.lower()}; do not impose a new one.", "Don't impose one; keep what your prompt says")


STYLE_AXIS = EnhanceAxis(
    key="style",
    label="Style",
    multi=False,
    video_only=False,
    options=(
        _keep("Style"),
        _photo("photo", "Natural photo", "a natural photograph in real, natural light; nothing staged.", "A real photograph in natural light; real skin and surface texture"),
        _photo(
            "candid",
            "Candid",
            "a candid snapshot: an unposed, casual phone photo of everyday life with available light or a phone flash and slightly imperfect framing; no studio lighting, retouching or color grading.",
            "An unposed phone snapshot of everyday life",
        ),
        _photo("street", "Street", "street photography: a documentary 35mm photograph of public life, with available light and a moment caught in motion.", "A 35mm documentary photo of public life"),
        _photo(
            "film",
            "Analog film",
            "an analog film photograph: name the film stock (such as Portra 400 or Tri-X), its grain, color and slight softness.",
            "A named film stock: its grain, color and softness",
        ),
        _photo("bw", "Black & white", "a black-and-white photograph: describe the tonal range, contrast and light instead of colors.", "Tones, contrast and light instead of color"),
        _photo(
            "portrait",
            "Studio portrait",
            "a studio portrait or fashion editorial: describe the lighting setup (key, fill, rim), the backdrop, the lens and the pose.",
            "Lighting setup, backdrop, lens and pose",
        ),
        _photo("product", "Product", "a commercial product photograph: describe the studio lighting, the surface, the backdrop and the reflections.", "Studio light, surface, backdrop, reflections"),
        _photo("cinematic", "Cinematic", "a still from a film: name the shot size, the lens look, the lighting setup and the color grade.", "A film still: shot size, lens look, lighting, grade"),
        _art("illustration", "Illustration", "a digital illustration: name the technique, the line and edge quality and the palette.", "Digital illustration: technique, line, palette"),
        _art("anime", "Anime", "anime art: name the line work, the cel shading, the palette and the background treatment.", "Line work, cel shading, palette, background"),
        _art("comic", "Comic", "comic book art: inked line work, flat color or halftone, and panel-like framing.", "Inked lines, flat color or halftone"),
        _art("3d", "3D render", "a 3D render: name the render style (stylized or realistic), the materials, the lighting and the camera.", "Render style, materials, lighting, camera"),
        _art("painterly", "Oil painting", "a classical oil painting: name the brushwork, the palette and the light.", "Brushwork, palette and light"),
        _art("watercolor", "Watercolor", "a watercolor painting: soft washes, colors bleeding at the edges, visible paper texture.", "Soft washes, bleeding edges, paper"),
        _art("gouache", "Gouache", "a gouache painting: opaque matte paint and flat, layered shapes with soft edges.", "Opaque matte paint, flat layered shapes"),
        _art("sketch", "Sketch", "a pencil or ink sketch: line weight, hatching and the paper, with little or no color.", "Pencil or ink: line weight, hatching"),
        _art("pixel", "Pixel art", "pixel art: a low-resolution look, a limited palette and crisp, visible pixels.", "Limited palette, crisp pixels"),
        _art("concept", "Concept art", "digital concept art: painterly forms, dramatic mood lighting and a sense of scale.", "Painterly forms, mood light, scale"),
        _art("graphic", "Graphic design", "graphic design or a poster: a clear layout, flat shapes, areas for type and a limited palette.", "Layout, flat shapes, type"),
        _art("isometric", "Isometric", "an isometric 3D diorama: a small, clean world seen from above at an angle, with neat edges.", "A small-world diorama seen at an angle"),
        _art("claymation", "Claymation", "claymation: sculpted clay figures and sets with fingerprints and a stop-motion look.", "Clay figures, fingerprints, stop-motion"),
        _art("ukiyo-e", "Ukiyo-e", "a ukiyo-e woodblock print: bold outlines, flat color areas and visible paper grain.", "Woodblock print: outlines, flat color"),
        _art("papercraft", "Papercraft", "papercraft: layered cut paper with soft shadows between the layers.", "Layered cut paper, soft shadows"),
    ),
    default=(KEEP,),
)


def _mood(slug: str, label: str, instruction: str, description: str) -> EnhanceOption:
    return EnhanceOption(slug, label, instruction, description)


MOOD_AXIS = EnhanceAxis(
    key="mood",
    label="Mood",
    multi=False,
    video_only=False,
    options=(
        _keep("Mood"),
        _mood("serene", "Serene", "Mood: calm and serene. Convey stillness and peace through soft light, gentle color and quiet surroundings.", "Stillness, soft light, gentle color"),
        _mood("cozy", "Cozy", "Mood: cozy and warm. Convey comfort through warm low light, soft textures and a sheltered, lived-in space.", "Warm low light, soft textures, enclosed"),
        _mood("joyful", "Joyful", "Mood: joyful and upbeat. Convey warmth and energy through bright light, lively color and open expressions.", "Bright light, lively color, open faces"),
        _mood("romantic", "Romantic", "Mood: romantic and intimate. Convey tenderness through warm, soft light and close, gentle framing.", "Warm soft light, close tender framing"),
        _mood("whimsical", "Whimsical", "Mood: whimsical and playful. Convey a dreamlike lightness through gentle color and charming detail.", "Playful, dreamlike, charming detail"),
        _mood("nostalgic", "Nostalgic", "Mood: nostalgic. Convey a fond memory through warm, faded color and soft, golden light.", "Warm faded color, golden light"),
        _mood("melancholic", "Melancholic", "Mood: melancholic and wistful. Convey solitude through muted color, subdued light and empty space.", "Muted color, subdued light, empty space"),
        _mood("mysterious", "Mysterious", "Mood: mysterious. Convey intrigue through shadow, haze and partly hidden detail.", "Shadow, haze, partly hidden detail"),
        _mood("eerie", "Eerie", "Mood: eerie and unsettling. Convey unease through cold tones, odd stillness and ominous shadow.", "Cold tones, odd stillness, ominous shadow"),
        _mood("dramatic", "Dramatic", "Mood: dramatic and tense. Convey intensity through hard contrast, charged light and a decisive moment.", "Hard contrast, charged light, a decisive moment"),
        _mood("epic", "Epic", "Mood: epic and awe-inspiring. Convey grandeur through vast scale, sweeping depth and majestic light.", "Vast scale, sweeping depth, majestic light"),
    ),
    default=(KEEP,),
)


def _theme(slug: str, label: str, cues: str, description: str, group: str) -> EnhanceOption:
    instruction = f"Theme: {label.lower()}. Carry it through {cues}, in the styling, materials, color, composition and small details; never add people, animals or major objects for it."
    return EnhanceOption(slug, label, instruction, description, group)


THEME_AXIS = EnhanceAxis(
    key="theme",
    label="Theme",
    multi=False,
    video_only=False,
    options=(
        _keep("Theme"),
        _theme("zen", "Zen", "stillness, simplicity, natural materials and balanced empty space", "Stillness, simplicity, natural materials, empty space", "Calm & sacred"),
        _theme("spiritual", "Spiritual", "reverence, soft radiant light and sacred, ritual details", "Reverence, radiant light, sacred detail", "Calm & sacred"),
        _theme("rustic", "Rustic", "worn wood, handmade things and earthy textures", "Worn wood, handmade things, earthy textures", "Calm & sacred"),
        _theme("sensual", "Sensual", "intimacy and allure: pose, light on skin, soft fabrics and closeness", "Intimacy and allure: pose, light on skin, soft fabric", "Allure"),
        _theme("glamorous", "Glamorous", "luxury: polished surfaces, fashion and elegant styling", "Luxury: polish, fashion, elegant styling", "Allure"),
        _theme("gothic", "Gothic", "dark romance: ornate architecture, candles and deep shadow", "Dark romance: ornate, candles, deep shadow", "Dark"),
        _theme("noir", "Noir", "hard shadows, smoke, rain-slick streets and a 1940s crime feel", "Hard shadows, smoke, 1940s crime", "Dark"),
        _theme("horror", "Horror", "dread: unsettling details, a sense of wrongness and darkness at the edges", "Dread, wrongness, darkness at the edges", "Dark"),
        _theme("dystopian", "Dystopian", "decay, control, brutal architecture and a broken future", "Decay, control, a broken future", "Dark"),
        _theme("fairytale", "Fairytale", "enchantment, storybook charm and soft magic", "Enchantment, storybook charm", "Wonder"),
        _theme("mythic", "Mythic", "legend: ancient grandeur, heroic scale and timeless symbols", "Legend: ancient grandeur, heroic scale", "Wonder"),
        _theme("surreal", "Surreal", "dream logic: impossible scale and odd juxtapositions", "Dream logic, impossible juxtapositions", "Wonder"),
        _theme("retro", "Retro", "a past era's design and faded palette (the era the prompt names, or mid-century)", "A past era's design and faded palette", "Era & tech"),
        _theme("futuristic", "Futuristic", "sleek forms, advanced technology and clean light", "Sleek forms, advanced technology", "Era & tech"),
        _theme("cyberpunk", "Cyberpunk", "neon tech-noir, a dense city, high tech and low life", "Neon tech-noir, dense city", "Era & tech"),
        _theme("steampunk", "Steampunk", "brass, gears and Victorian machinery", "Brass, gears, Victorian machinery", "Era & tech"),
    ),
    default=(KEEP,),
)
DETAILS_AXIS = EnhanceAxis(
    key="details",
    label="Focus on",
    multi=True,
    video_only=False,
    options=(
        EnhanceOption("lighting", "Lighting", "lighting", "The light: its source, direction, quality and color"),
        EnhanceOption("composition", "Composition", "composition and framing", "Framing: what is where, foreground and background"),
        EnhanceOption(
            "camera",
            "Camera & viewpoint",
            "camera and viewpoint (for a photo the camera, lens, angle and depth of field; for art the viewpoint and framing)",
            "Camera, lens and angle; viewpoint for art",
        ),
        EnhanceOption("materials", "Materials & textures", "materials and textures", "What things are made of and how they feel"),
        EnhanceOption("color", "Color palette", "the color palette and where each color appears", "The palette and where each color appears"),
        EnhanceOption("environment", "Environment", "environment and background", "The setting and background"),
        EnhanceOption("subject", "Subject", "the subject's appearance, clothing and pose", "The main subject's looks, clothing and pose"),
        EnhanceOption("expression", "Expression", "faces, gaze and body language: the emotion the people show", "Faces, gaze and body language"),
    ),
    default=(),
)
LENGTH_AXIS = EnhanceAxis(
    key="length",
    label="Length",
    multi=False,
    video_only=False,
    options=(
        EnhanceOption("shorter", "Shorter", "shorter", "About half, keeping every element you named"),
        EnhanceOption("same", "Same", "same", "Your length; short prompts grow to about 25 words"),
        EnhanceOption("longer", "Longer", "longer", "About double, at least 40 words"),
        EnhanceOption("extra", "Extra long", "extra", "About triple, at least 80 words"),
    ),
    default=("same",),
)
MOTION_AXIS = EnhanceAxis(
    key="motion",
    label="Describe",
    multi=True,
    video_only=True,
    options=(
        EnhanceOption("action", "Action sequence", "the action as a clear sequence of events", "What happens, as clear beats in time order"),
        EnhanceOption("secondary", "Secondary motion", "the secondary motion that moves along with it: hair, cloth, water, smoke, leaves or dust", "Hair, cloth, water, smoke, leaves, dust"),
    ),
    default=(),
)


def _camera(slug: str, label: str, move: str, description: str) -> EnhanceOption:
    return EnhanceOption(slug, label, move, description)


CAMERA_AXIS = EnhanceAxis(
    key="camera",
    label="Camera",
    multi=False,
    video_only=True,
    options=(
        EnhanceOption(KEEP, "Keep", "", "Only what your prompt says; nothing added"),
        _camera("auto", "Auto", "describe one camera movement that suits the action.", "The enhancer picks a move that suits the action"),
        _camera("static", "Static", "a locked-off camera that does not move; only the subject moves.", "Locked off; only the subject moves"),
        _camera("push-in", "Push in", "a slow dolly push in toward the subject.", "A slow dolly toward the subject"),
        _camera("pull-back", "Pull back", "a slow dolly pulling back from the subject, revealing more of the scene.", "A slow dolly away, revealing more"),
        _camera("pan", "Pan", "the camera pans sideways across the scene.", "The camera turns sideways"),
        _camera("tilt", "Tilt", "the camera tilts up or down to reveal the scene.", "The camera turns up or down"),
        _camera("tracking", "Tracking", "a tracking shot that follows the subject alongside.", "The camera follows alongside"),
        _camera("orbit", "Orbit", "the camera orbits around the subject.", "The camera circles the subject"),
        _camera("handheld", "Handheld", "a handheld camera, slightly shaky and close, documentary style.", "Slightly shaky, close, documentary"),
        _camera("aerial", "Aerial", "an aerial drone shot moving over the scene from above.", "A drone shot moving over the scene"),
    ),
    default=(KEEP,),
)
PACE_AXIS = EnhanceAxis(
    key="pace",
    label="Pace",
    multi=False,
    video_only=True,
    options=(
        EnhanceOption(KEEP, "Keep", "", "Natural speed"),
        EnhanceOption("slow-motion", "Slow motion", "slow motion; time is stretched so every movement and detail is visible.", "Time stretched; every detail visible"),
        EnhanceOption("slow", "Slow", "slow, unhurried movement.", "Unhurried movement"),
        EnhanceOption("brisk", "Brisk", "brisk, lively movement.", "Lively movement"),
        EnhanceOption("fast", "Fast", "fast, quick action.", "Quick action, more beats in the clip"),
    ),
    default=(KEEP,),
)
# How much action fits a clip at each pace: the clip's seconds are scaled before they are turned into beats.
_PACE_BEAT_SCALE = {KEEP: 1.0, "slow-motion": 0.5, "slow": 0.75, "brisk": 1.5, "fast": 2.0}
SOUND_AXIS = EnhanceAxis(
    key="sound",
    label="Sound",
    multi=True,
    video_only=True,
    options=(
        EnhanceOption("ambient", "Ambience", "the ambient soundscape of the setting", "The setting's soundscape"),
        EnhanceOption("effects", "Sound effects", "sound effects synced to the visible action", "Sounds synced to the action"),
        EnhanceOption(
            "dialogue",
            "Dialogue",
            "speech: keep any quoted lines word for word, or write short lines that fit the scene when there are none, and say who speaks and in what voice",
            "Your quoted lines, or short lines that fit",
        ),
        EnhanceOption("music", "Music", "music: whether it plays in the scene or as a score, its genre and tempo", "In the scene or as a score: genre, tempo"),
    ),
    default=(),
)
# Display order in the Web UI: Look (Style, Mood, Theme), Focus on, Motion (Camera, Pace, Describe), Sound, then Length.
ENHANCE_AXES: tuple[EnhanceAxis, ...] = (STYLE_AXIS, MOOD_AXIS, THEME_AXIS, DETAILS_AXIS, CAMERA_AXIS, PACE_AXIS, MOTION_AXIS, SOUND_AXIS, LENGTH_AXIS)
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
PARTS_DESCRIPTIONS = {"both": "Rewrite both parts; a blank scene is written from your text", "action": "Keep your scene exactly; write the action to fit it"}
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
    theme: str = THEME_AXIS.default[0]
    details: tuple[str, ...] = DETAILS_AXIS.default
    length: str = LENGTH_AXIS.default[0]
    camera: str = CAMERA_AXIS.default[0]
    pace: str = PACE_AXIS.default[0]
    motion: tuple[str, ...] = MOTION_AXIS.default
    sound: tuple[str, ...] = SOUND_AXIS.default
    note: str = ""
    parts: str | None = None


@dataclass(frozen=True)
class EnhanceContext:
    """Facts about the job that shape a rewrite: audio track, clip length, frame shape, start image, references and last frame.

    ``reference_image`` is the start image (img2img's image, or a video's first frame); ``references`` counts the
    reference images the prompt names as "image 1", "image 2".

    The last five fields describe extending an existing clip. ``seconds`` is then the whole window (the carried
    frames plus the new ones) and ``new_seconds`` the part the new action fills. ``previous`` is the source's
    recorded prompt ("what just happened"); ``extract_from`` is set for a source recorded without a scene;
    ``prompt_is_original`` says the user's prompt still equals that old prompt.
    """

    audio: bool = False
    seconds: float | None = None
    orientation: str | None = None
    reference_image: bool = False
    references: int = 0
    end_image: bool = False
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
            "options": [{"slug": slug, "label": label, "description": PARTS_DESCRIPTIONS[slug]} for slug, label in PARTS_OPTIONS],
            "default": None,
        },
        "axes": [
            {
                "key": axis.key,
                "label": axis.label,
                "multi": axis.multi,
                "video_only": axis.video_only,
                "audio_only": axis.key in AUDIO_AXES,
                "options": [{"slug": option.slug, "label": option.label, "description": option.description, "group": option.group} for option in axis.options],
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
        "theme": settings.theme,
        "details": list(settings.details),
        "length": settings.length,
        "camera": settings.camera,
        "pace": settings.pace,
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
    references: int = 0,
    end_image: bool = False,
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
        reference_image: Whether a start image (img2img's image, or a video's first frame) is supplied.
        references: How many reference images are supplied; the prompt names them "image 1", "image 2".
        end_image: Whether the clip also ends on a supplied image (video only; the rewrite then knows both frames are fixed).
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
        references=references,
        extends=extending,
        previous=previous if extending else None,
        extract_from=extract_from if extending else None,
        prompt_is_original=extending and prompt_is_original,
        new_seconds=new_frames / fps if extending and new_frames and fps else None,
        end_image=video and end_image,
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
        data: Mapping with any of ``style``, ``mood``, ``theme``, ``details``, ``length``, ``camera``, ``pace``, ``motion``, ``sound``, ``note``, ``parts``.
        mode: ``"image"`` or ``"video"``.
        motion_in_image: ``"error"`` to reject the video-only keys (``camera``, ``pace``, ``motion``, ``sound``, ``parts``) in image mode,
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
    """Serialize *settings* to the CLI spec grammar, compactly: only the values that differ from the defaults.

    Video-only values (``camera``, ``pace``, ``motion``, ``sound``, ``parts``) are written for video only and ``note`` never.
    Settings that are all defaults give ``style=keep``, so the result is never empty.
    """
    current, defaults = settings_to_mapping(settings), settings_to_mapping(EnhanceSettings())
    parts = []
    for axis in ENHANCE_AXES:
        if (axis.video_only and mode != "video") or current[axis.key] == defaults[axis.key]:
            continue
        value = current[axis.key]
        parts.append(f"{axis.key}={'+'.join(value) if axis.multi else value}")
    if mode == "video" and settings.parts is not None:
        parts.append(f"{PARTS_KEY}={settings.parts}")
    return ",".join(parts) or f"{STYLE_AXIS.key}={KEEP}"


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
    """Reject unknown slugs. Every combination is a rewrite: with nothing picked it is the general rewrite (see ``detail_rules``)."""
    _check_mode(mode)
    if settings.parts is not None and settings.parts not in PARTS_VALUES:
        raise ValueError(f"Rewrite must be one of: {', '.join(PARTS_VALUES)}.")
    for axis in ENHANCE_AXES:
        value = getattr(settings, axis.key)
        for slug in value if axis.multi else (value,):
            axis.option(slug)


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


def only_video_options(settings: EnhanceSettings) -> bool:
    """Return whether *settings* change only video options (camera, pace, motion, sound, Rewrite) and leave every other at its default."""
    current, defaults = settings_to_mapping(settings), settings_to_mapping(EnhanceSettings())
    video_keys = {axis.key for axis in ENHANCE_AXES if axis.video_only} | {PARTS_KEY}
    changed = {key for key in current if current[key] != defaults[key]}
    return bool(changed) and changed <= video_keys


def image_enhance_entries(enhance_by_set: dict[str, list[EnhanceSettings | None]] | None, *, caption: bool = False) -> dict[str, list[EnhanceSettings | None]] | None:
    """Return the prompt-file ``enhance:`` entries of an image run: entries that set only video options are not enhanced, with one warning.

    Such an entry asks for a video rewrite; in an image run its options do nothing, and enhancing it anyway would give a
    general rewrite nobody asked for. ``enhance: true`` (nothing set) still gives the general rewrite.

    Args:
        caption: The run writes JSON captions. Every entry is then enhanced: its prompt becomes a caption either way.
    """
    if not enhance_by_set or caption:
        return enhance_by_set
    dropped = sum(1 for entries in enhance_by_set.values() for entry in entries if entry is not None and only_video_options(entry))
    if not dropped:
        return enhance_by_set
    noun = "entry sets" if dropped == 1 else "entries set"
    warnings.warn(f"{dropped} prompt-file 'enhance:' {noun} only video options (camera, pace, motion, sound or parts); not enhanced in this image run.", stacklevel=2)
    return {name: [None if entry is not None and only_video_options(entry) else entry for entry in entries] for name, entries in enhance_by_set.items()}


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


def duration_beats(seconds: float, pace: str = KEEP) -> str:
    """Return how many beats of action fit *seconds* at *pace*: one continuous action, two or three beats, or a short sequence.

    A faster pace fits more action into the same time, slow motion less.
    """
    seconds *= _PACE_BEAT_SCALE.get(pace, 1.0)
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


def _echo_key(text: str) -> str:
    """Return the words of *text* in lower case, without punctuation, so a rewrite that only recases or repunctuates is an echo."""
    return " ".join(_WORD_RE.findall(text.casefold()))


def is_unusable(output: str, original: str) -> bool:
    """Return whether *output* is empty, an echo of *original* (case and punctuation aside), or starts with a refusal."""
    if not output.strip():
        return True
    if _echo_key(output) == _echo_key(original):
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


def references_rule(count: int) -> str:
    """Return the rule for *count* reference images: keep the prompt's "image N" phrases and describe the new scene, not the references."""
    names = [f'"image {number}"' for number in range(1, count + 1)]
    listed = names[0] if count == 1 else f"{', '.join(names[:-1])} and {names[-1]}"
    return (
        f"{count} reference image{'s are' if count > 1 else ' is'} supplied, which the prompt names {listed}. "
        "Keep those phrases exactly as written. You can't see the references, so don't describe what they show: "
        "describe the new scene, action and composition around them."
    )


def _drops_reference(prompt: str, output: str, count: int) -> bool:
    """Return whether *prompt* names a reference from 1 to *count* as "image N" that *output* does not."""

    def named(text: str) -> set[int]:
        return {number for number in map(int, _REFERENCE_MENTION_RE.findall(text)) if 1 <= number <= count}

    return not named(prompt) <= named(output)


def frame_rule(orientation: str) -> str:
    """Return the rule that fits the composition to a ``portrait``, ``landscape`` or ``square`` frame without naming it."""
    return f"Compose for a {orientation} frame; never mention the frame or its shape."


def context_rules(mode: str, context: EnhanceContext, *, pace: str = KEEP) -> list[str]:
    """Return the rules for the start image, references, frame shape and clip length (how much action fits depends on *pace*)."""
    rules: list[str] = []
    if context.reference_image:
        if mode == "video" and context.end_image:
            rules.append(
                "The first and last frames are supplied as images. Describe the motion, camera movement and changes that lead from the first to the last; "
                "refer to the subject briefly, don't add appearance details the images show, and don't invent a different ending."
            )
        elif mode == "video":
            rules.append(
                "The first frame is supplied as an image. Describe the motion, camera movement and changes over time; "
                "refer to the subject briefly and don't add appearance details the image already shows."
            )
        else:
            rules.append("A start image supplies the layout and colours. Refer to what it shows briefly; apply the style and mood without re-describing its appearance.")
    if context.references:
        rules.append(references_rule(context.references))
    if context.orientation is not None:
        rules.append(frame_rule(context.orientation))
    if mode == "video" and context.seconds is not None:
        beats = duration_beats(context.seconds, pace)
        rules.append(f"Duration: about {max(1, round(context.seconds))} seconds; fit the action to it: {beats}.")
    return rules


def action_rules(settings: EnhanceSettings, context: EnhanceContext, *, for_pair: bool = False) -> list[str]:
    """Return the camera, pace, motion and sound rules of a video rewrite; sound only when the job has an audio track.

    They open with their own names ("Camera: ...", "Motion: describe ..."), or with "For the ACTION" in a scene-and-action rewrite.
    """
    motion_lead, sound_lead = ("For the ACTION", "For the ACTION") if for_pair else ("Motion", "Sound")
    camera_lead, pace_lead = ("For the ACTION, camera", "For the ACTION, pace") if for_pair else ("Camera", "Pace")
    rules = []
    if settings.camera != KEEP:
        rules.append(f"{camera_lead}: {CAMERA_AXIS.option(settings.camera).instruction}")
    if settings.pace != KEEP:
        rules.append(f"{pace_lead}: {PACE_AXIS.option(settings.pace).instruction}")
    if settings.motion:
        rules.append(f"{motion_lead}: describe {', '.join(MOTION_AXIS.option(slug).instruction for slug in settings.motion)}.")
    if context.audio and settings.sound:
        sounds = "; ".join(SOUND_AXIS.option(slug).instruction for slug in settings.sound)
        rules.append(f"{sound_lead}: describe {sounds}. Put spoken lines in double quotes; only people already in the scene speak.")
    return rules


def theme_rules(settings: EnhanceSettings, *, lead: str = "") -> list[str]:
    """Return the Theme rule, or none when the theme is kept; *lead* ("For the SCENE: ") prefixes it."""
    if settings.theme == KEEP:
        return []
    return [f"{lead}{THEME_AXIS.option(settings.theme).instruction}"]


def detail_rules(settings: EnhanceSettings, *, shorter: bool, lead: str = "") -> list[str]:
    """Return the Focus on rule: where to add detail (or what to keep while cutting), or a general rewrite when nothing is picked.

    Args:
        shorter: The rewrite cuts (Length Shorter, not clamped): picked aspects are kept, and nothing picked adds no rule.
        lead: Prefixes the rule ("For the SCENE: "); without it the rule starts with a capital.
    """
    if settings.details:
        aspects = ", ".join(DETAILS_AXIS.option(slug).instruction for slug in settings.details)
        text = f"keep these aspects while cutting: {aspects}." if shorter else f"add detail on: {aspects}."
    elif shorter:
        return []
    else:
        text = "add concrete visual detail where the prompt is thin: the subject, the setting and the light."
    return [f"{lead}{text}" if lead else text[0].upper() + text[1:]]


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
        *context_rules(mode, context, pace=settings.pace),
    ]
    rules.append(STYLE_AXIS.option(settings.style).instruction)
    rules.append(MOOD_AXIS.option(settings.mood).instruction)
    rules.extend(theme_rules(settings))
    rules.extend(detail_rules(settings, shorter=plan.length == "shorter" and not plan.clamped))
    if mode == "video":
        rules.extend(action_rules(settings, context))
    rules.extend(direction_rules(settings))
    rules.append(length_instruction(plan))
    if plan.length == "same" or plan.clamped:
        rules.append("Apply the options above; don't return the input unchanged.")
    system = system_message(f"You rewrite prompts for a {kind} model.", rules, "Output only the rewritten prompt as plain prose: no preamble, no headings, no quotes around it.")
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
    """Rewrite *prompt* with *enhancer*: generate, clean, and retry once if unusable, a quoted text was changed or a referenced "image N" was dropped.

    Expand ``{a|b}`` choices before calling: the model rewrites one concrete prompt. When no attempt keeps every quoted
    text and every "image N" the prompt names, the last usable one is returned with ``QUOTES_CHANGED_WARNING`` or
    ``REFERENCES_DROPPED_WARNING`` (also issued as a warning).

    Args:
        context: Job facts (audio track, clip length, frame shape, start image, references).
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
    references = context.references if context is not None else 0
    flawed: tuple[str, str] | None = None
    for attempt in range(2):
        raw = generate_raw(enhancer, messages, seed=seed + attempt, max_tokens=max_tokens, temperature=temperature, on_text=on_text, cancelled=cancelled)
        output = clean_output(raw)
        if is_unusable(output, prompt):
            continue
        if missing_quotes(spans, output):
            problem = QUOTES_CHANGED_WARNING
        elif _drops_reference(prompt, output, references):
            problem = REFERENCES_DROPPED_WARNING
        else:
            return _final(EnhanceResult(prompt=output, clamped=plan.clamped), on_text)
        flawed = (output, problem)
    if flawed is None:
        raise RuntimeError(NO_USABLE_MESSAGE)
    output, problem = flawed
    warnings.warn(problem, stacklevel=2)
    return _final(EnhanceResult(prompt=output, clamped=plan.clamped, warning=problem), on_text)


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
