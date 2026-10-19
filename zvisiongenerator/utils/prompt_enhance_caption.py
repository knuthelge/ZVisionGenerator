"""Rewrite a prompt as an Ideogram 4 JSON caption: chat messages, labelled-line parsing, caption assembly and the model run.

The model never writes JSON. It answers in labelled lines (``SUMMARY:``, ``LIGHTING:``, ``OBJECT:`` ...) and code builds
the caption: key order, escaping, the photo or art branch, the medium, colors and the user's own lettering.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from zvisiongenerator.core.ideogram4 import compact_caption
from zvisiongenerator.utils.prompt_enhance import (
    DEFAULT_TEMPERATURE,
    MOOD_AXIS,
    PRESERVATION_RULES,
    STYLE_AXIS,
    EnhanceContext,
    EnhanceResult,
    EnhanceSettings,
    context_rules,
    direction_rules,
    generate_raw,
    perception_rule,
    plan_length,
    strip_reasoning,
    system_message,
    word_count,
)

if TYPE_CHECKING:
    from zvisiongenerator.core.prompt_enhancer import PromptEnhancer

__all__ = ["CAPTION_NO_USABLE_MESSAGE", "enhance_caption"]

CAPTION_NO_USABLE_MESSAGE = "The enhancer returned no usable JSON caption. Try different settings or write as text."
_CAPTION_MAX_TOKENS = 1400
# Sparse captions push Ideogram 4 toward flat "asset" images, so every Length keeps a minimum of words in the caption.
_FLOOR_WORDS = {"shorter": 45, "same": 70, "longer": 110, "extra": 170}
_OBJECT_LINES = (3, 8)
_MAX_INVENTED_TEXTS = 3

_ART_MEDIUM = {"illustration": "illustration", "anime": "illustration", "comic": "illustration", "3d": "3d_render", "painterly": "painting"}
# Prompt words that point to an art medium when the style is kept, checked in order.
# Words that are also everyday things ("olive oil", "a canvas tent", "a kitchen blender") count only in a medium phrase.
_MEDIUM_WORDS = (
    ("3d_render", re.compile(r"\b(?:3d|render(?:ed|ing)?|cgi|octane render|blender render)\b", re.I)),
    ("painting", re.compile(r"\b(?:paint(?:ing|ed)?|oil (?:painting|on canvas|portrait)|in oils?|on canvas|watercolou?r|gouache|acrylic)\b", re.I)),
    ("illustration", re.compile(r"\b(?:illustrat\w*|drawing|drawn|sketch\w*|cartoon|anime|manga|comic|vector|pixel art|line art)\b", re.I)),
)
# Words that make a prompt a photo even when it names a medium; "portrait" and "shot" fit a painting or a render too.
_PHOTO_WORDS = re.compile(r"\b(?:photo\w*|camera|lens|dslr|35mm|snapshot|bokeh)\b", re.I)
# A design the image IS (not one shown inside a scene): the noun opens the prompt, after at most four words that describe
# it ("a vintage travel poster", "birthday card"). A preposition, a second article or a verb before it ("men playing cards",
# "a girl reading a menu") makes it a thing in a scene; -ing words that name or describe a design are allowed.
_DESIGN_RE = re.compile(
    r"^((?:[\w'-]+\s+){0,4}?)(poster|flyer|logo|label|book cover|album cover|cover|infographic|packaging|card|invitation|sticker|banner|menu)s?\b",
    re.I,
)
_DESIGN_STOP_WORDS = frozenset({"with", "on", "in", "of", "at"})
_ARTICLES = frozenset({"a", "an", "the"})
_DESIGN_ING_WORDS = frozenset(
    {"wedding", "greeting", "opening", "clothing", "housewarming", "christening", "trading", "recording", "stunning", "striking", "amazing", "charming", "inviting", "eye-catching"}
)
# Lettering inside a scene whose words the user did not give: the model may write a few.
_SIGN_RE = re.compile(r"\b(?:signs?|signage|signboards?|neon|billboards?|headlines?|titles?|graffiti|placards?|marquees?)\b", re.I)
# Languages a prompt may ask lettering to be in, with a character class the lettering must use.
_SCRIPTS = (
    ("japanese", r"[぀-ヿ一-鿿]"),
    ("chinese", r"[一-鿿]"),
    ("korean", r"[가-힯]"),
    ("arabic", r"[؀-ۿ]"),
    ("hebrew", r"[֐-׿]"),
    ("russian", r"[Ѐ-ӿ]"),
    ("greek", r"[Ͱ-Ͽ]"),
    ("thai", r"[฀-๿]"),
    ("hindi", r"[ऀ-ॿ]"),
)
# Quoted text: paired quotes, guillemets and CJK brackets anywhere; single quotes only around a phrase of two or more
# characters, which may hold an apostrophe between letters ('Joe's Diner'), while an apostrophe on its own as in "Moe's" or
# "rock 'n' roll" is not a quote; an all-caps phrase after "text", "reads" or "says", which may go on with numbers and
# single letters (OPEN 24 HOURS, PLAN B).
_QUOTE_RES = (
    re.compile(r'"([^"\n]+)"|“([^”\n]+)”|«\s*([^»\n]+?)\s*»|「([^」\n]+)」|『([^』\n]+)』'),
    re.compile(r"(?<![\w'‘])['‘]((?:[^'’\n]|(?<=\w)['’](?=\w)){2,})['’](?!\w)"),
    re.compile(r"\b(?:text|word|words|reads|reading|says|saying|lettering)\s+([A-Z][A-Z0-9]+(?:\s(?:[A-Z][A-Z0-9]*|[0-9]+))*)\b"),
)
# A capitalized name after "called", "named" or "titled": lettering only on a design ("a logo for a shop called Moonbean"),
# never in a scene ("a dog named Max").
_NAME_RE = re.compile(r"\b(?:called|named|titled)\s+([A-Z][\w'’&]*(?:\s+[A-Z0-9][\w'’&]*)*)")
_LETTERING_RE = re.compile(r"\b(?:text|letter\w*|font|typeface|typograph\w*|title|word\w*|serif|sans)\b", re.I)
# Words that tie lettering to a thing carrying it ("a mug with the text ...", "a sign on the door reading ...").
_CARRIER_RE = re.compile(r"\b(?:with|on|of|bearing|showing|displaying|printed|painted|engraved|carved|stamped)\b", re.I)
_DOUBLE_QUOTED_RE = re.compile(r'"([^"]+)"|“([^”]+)”')

_LABELS = {
    "SUMMARY": "summary",
    "AESTHETICS": "aesthetics",
    "LIGHTING": "lighting",
    "CAMERA": "camera",
    "ART STYLE": "art_style",
    "COLORS": "colors",
    "BACKGROUND": "background",
    "OBJECT": "object",
    "TEXT": "text",
}
# A label opens a line, after optional bullets, markdown and a number ("**OBJECT 2:**", "- TEXT:").
_LINE_RE = re.compile(r"^[\s*#\->•]*(?:\d+[.)]\s*)?(" + "|".join(sorted(map(re.escape, _LABELS), key=len, reverse=True)) + r")\s*\d*\s*\**\s*[:：]\**\s*(.*)$", re.I)
_HEX_RE = re.compile(r"#([0-9A-Fa-f]{6})\b")
_STYLE_COLORS_MAX = 16
_COLOR_NAMES = {
    "black": "#111111",
    "charcoal": "#36454F",
    "grey": "#808080",
    "gray": "#808080",
    "silver": "#C0C0C0",
    "white": "#F5F5F5",
    "cream": "#F3E9D2",
    "beige": "#D8C8A8",
    "brown": "#6B4226",
    "red": "#C62828",
    "crimson": "#9B111E",
    "pink": "#F48FB1",
    "orange": "#EF6C00",
    "gold": "#D4AF37",
    "golden": "#D4AF37",
    "yellow": "#FBC02D",
    "green": "#2E7D32",
    "teal": "#00897B",
    "cyan": "#00BCD4",
    "blue": "#1E5AA8",
    "navy": "#1B2A4A",
    "purple": "#6A1B9A",
}

_DETAIL_ROUTES = {
    "lighting": "LIGHTING: name the light source, its direction and quality.",
    "composition": "Order the OBJECT lines from most to least important and say where each sits in the frame (left, right, center, foreground, background).",
    "camera": "",  # Folded into the CAMERA or ART STYLE rule.
    "materials": "Name the materials and surface textures in each OBJECT line.",
    "color": "COLORS: 4 to 6 colors, and name the main colors in each OBJECT line.",
    "environment": "BACKGROUND: describe the setting in depth: the place, surfaces, distance and weather.",
    "subject": "In the main subject's OBJECT line describe its appearance, clothing and pose.",
}
_EXAMPLE_HEAD = "SUMMARY: A white lighthouse stands on a rocky headland above a stormy sea at dusk.\nAESTHETICS: moody, windswept, coastal\nLIGHTING: low dusk light from the left under heavy clouds, the lamp glowing warm\n"
_EXAMPLE_TAIL = (
    "COLORS: slate grey, white, deep teal, warm amber\n"
    "BACKGROUND: A dark, cloudy sky over a rough sea that fades to the horizon.\n"
    "OBJECT: A tall white lighthouse with a red lamp room, its lamp glowing, standing on the right of the frame.\n"
    "OBJECT: Jagged black rocks of the headland below the lighthouse, wet with spray.\n"
    "OBJECT: Rough teal waves breaking into white foam in the foreground."
)
_EXAMPLE = {
    "photo": _EXAMPLE_HEAD + "CAMERA: eye-level, 35mm lens, deep depth of field\n" + _EXAMPLE_TAIL,
    "art": _EXAMPLE_HEAD + "ART STYLE: gouache illustration, flat layered shapes, visible brush texture\n" + _EXAMPLE_TAIL,
}


@dataclass(frozen=True)
class _Draft:
    """The labelled fields the model wrote; ``items`` are ``("obj", desc)`` or ``("text", words, look)`` in order."""

    summary: str = ""
    aesthetics: str = ""
    lighting: str = ""
    camera: str = ""
    art_style: str = ""
    colors: str = ""
    background: str = ""
    items: tuple[tuple[str, ...], ...] = ()


@dataclass(frozen=True)
class _Lettering:
    """What lettering a caption may hold, decided once from the prompt and the direction.

    ``mode`` is ``quotes`` (only the user's quoted texts), ``design`` (a design whose words come from the prompt),
    ``signs`` (signs named without words: the model may write a few, in ``script`` when one is named) or ``none``.
    """

    quotes: tuple[str, ...]
    mode: str
    script: tuple[str, str] | None
    source: str


def enhance_caption(
    enhancer: PromptEnhancer,
    prompt: str,
    settings: EnhanceSettings,
    *,
    seed: int,
    ceiling: int,
    context: EnhanceContext | None = None,
    length_cfg: dict[str, Any] | None = None,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = _CAPTION_MAX_TOKENS,
    on_text: Callable[[str], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> EnhanceResult:
    """Rewrite *prompt* as a compact Ideogram 4 JSON caption; the model writes labelled lines and code builds the caption.

    Expand ``{a|b}`` choices and validate *settings* before calling. *on_text* receives the labelled lines as they stream.

    Raises:
        ValueError: For an empty prompt.
        RuntimeError: When cancelled, or with ``CAPTION_NO_USABLE_MESSAGE`` when both attempts are unusable.
    """
    if not prompt.strip():
        raise ValueError("Enter a prompt to enhance.")
    lettering = _lettering(prompt, settings.note)
    messages = _messages(prompt, settings, lettering=lettering, target=_target_words(prompt, settings, ceiling, length_cfg), context=context or EnhanceContext())
    for attempt in range(2):
        raw = generate_raw(enhancer, messages, seed=seed + attempt, max_tokens=max(max_tokens, _CAPTION_MAX_TOKENS), temperature=temperature, on_text=on_text, cancelled=cancelled)
        draft = _parse(raw)
        if draft.summary and draft.background and draft.items:
            return EnhanceResult(prompt=compact_caption(_assemble(draft, prompt, settings, lettering)), clamped=False)
    raise RuntimeError(CAPTION_NO_USABLE_MESSAGE)


def _target_words(prompt: str, settings: EnhanceSettings, ceiling: int, length_cfg: dict[str, Any] | None) -> int:
    """Return the caption's word target: the usual Length plan, raised to the caption floor and kept under *ceiling*."""
    planned = plan_length(word_count(prompt), settings.length, ceiling=ceiling, length_cfg=length_cfg).target
    return min(max(planned, _FLOOR_WORDS[settings.length]), max(ceiling, _FLOOR_WORDS["shorter"]))


def _quoted_texts(text: str, *, names: bool) -> tuple[str, ...]:
    """Return the texts the user quoted, plus called/named names when *names*, in order of appearance and without repeats."""
    found: list[tuple[int, str]] = []
    taken: list[range] = []
    for pattern in (*_QUOTE_RES, _NAME_RE) if names else _QUOTE_RES:
        for match in pattern.finditer(text):
            group = next(index for index in range(1, (match.lastindex or 1) + 1) if match.group(index) is not None)
            words = " ".join(match.group(group).split())
            if words and not any(match.start(group) in span for span in taken):
                found.append((match.start(group), words))
                taken.append(range(match.start(group), match.end(group)))
    return tuple(dict.fromkeys(words for _start, words in sorted(found)))


def _design(prompt: str) -> str | None:
    """Return the design noun when the image is a design (a poster, logo, card ...), else None."""
    match = _DESIGN_RE.match(prompt.strip())
    if match is None:
        return None
    for index, word in enumerate(match.group(1).lower().split()):
        if word in _DESIGN_STOP_WORDS or (index and word in _ARTICLES) or (word.endswith("ing") and word not in _DESIGN_ING_WORDS):
            return None
    return match.group(2).lower()


def _lettering(prompt: str, note: str) -> _Lettering:
    """Decide what lettering the caption may hold from the prompt and the user's direction."""
    source = f"{prompt}\n{note}"
    design = _design(prompt) is not None
    quotes = _quoted_texts(source, names=design)
    script = next(((language, pattern) for language, pattern in _SCRIPTS if re.search(rf"\b{language}\b", source, re.I)), None)
    mode = "quotes" if quotes else "design" if design else "signs" if _SIGN_RE.search(source) else "none"
    return _Lettering(quotes=quotes, mode=mode, script=script, source=source)


def _lettering_rule(lettering: _Lettering) -> str:
    """Return the rule that tells the model which TEXT lines to write."""
    if lettering.mode == "quotes":
        listed = ", ".join(f'"{words}"' for words in lettering.quotes)
        return f"Lettering: write one TEXT line for each of these texts, copied exactly: {listed}. Write no other TEXT lines. Lettering goes only in TEXT lines, never in OBJECT lines."
    if lettering.mode == "design":
        return "Lettering: if the user gives words to write (a name, a greeting, a title), write each as a TEXT line copied exactly from the prompt. Never invent other words. Lettering goes only in TEXT lines."
    if lettering.mode == "signs":
        language = f" written in {lettering.script[0].capitalize()}" if lettering.script else ""
        return (
            f"Lettering: the user names signs or titles without their words; you may write at most {_MAX_INVENTED_TEXTS} TEXT lines with short words{language} that fit the place. "
            "Lettering goes only in TEXT lines; never put words on anything else."
        )
    return "Lettering: the image has no readable text."


def _look(settings: EnhanceSettings, prompt: str) -> tuple[str, str]:
    """Return the branch (``photo`` or ``art``) and the ``medium``: from the chosen style, or for "keep" from the prompt."""
    if settings.style != "keep":
        medium = _ART_MEDIUM.get(settings.style)
        return ("art", medium) if medium else ("photo", "photograph")
    if _design(prompt):
        return "art", "graphic_design"
    medium = next((name for name, pattern in _MEDIUM_WORDS if pattern.search(prompt)), None)
    return ("art", medium) if medium and not _PHOTO_WORDS.search(prompt) else ("photo", "photograph")


def _messages(prompt: str, settings: EnhanceSettings, *, lettering: _Lettering, target: int, context: EnhanceContext) -> list[dict[str, str]]:
    """Build the system + user messages: the rig's rules pointed at the caption lines, plus object, lettering and length rules."""
    branch, _medium = _look(settings, prompt)
    camera = "camera" in settings.details
    rules = [
        *PRESERVATION_RULES,
        "Give every person, animal and thing the user names its own OBJECT line, one thing per line; never put two things in one line.",
        "Something the user names once is exactly one OBJECT line: never repeat or multiply it. A group the user names (a crowd, two cats) is one OBJECT line.",
        "A person's face, hair, expression and clothing belong in that person's OBJECT line, never in lines of their own. Light belongs in LIGHTING, never in an OBJECT line.",
        "OBJECT lines: the main subject first, then every other thing the user names, even small ones, then one or two for the ground, surfaces and scenery around them "
        f"({_OBJECT_LINES[0]} to {_OBJECT_LINES[1]} lines in total). Never add new people, animals, characters or objects.",
        _lettering_rule(lettering),
        perception_rule("image", context),
        *context_rules("image", context),
        STYLE_AXIS.option(settings.style).instruction,
        "Write a CAMERA line: " + ("camera angle, lens and depth of field." if camera else "camera angle and lens in a few words.")
        if branch == "photo"
        else "Write an ART STYLE line naming the style, technique and texture" + (", plus the viewpoint and framing." if camera else "."),
    ]
    if settings.style == "bw":
        rules.append("COLORS: only black, white and greys.")
    rules.append(MOOD_AXIS.option(settings.mood).instruction + " Show it in AESTHETICS, LIGHTING and COLORS.")
    design = _design(prompt)
    if design and branch == "art":
        rules.append(
            f"The image is the {design} itself, filling the whole frame edge to edge: describe its design, never a {design} shown on a wall or background. "
            "BACKGROUND is the design's own background (paper, color, texture)."
        )
    rules.extend(route for slug in settings.details if (route := _DETAIL_ROUTES[slug]))
    rules.extend(direction_rules(settings))
    summary, look, background = (max(8, round(target * share)) for share in (0.12, 0.22, 0.14))
    per_object = max(10, round(max(30, target - summary - look - background) / 4))
    rules.append(
        f"Length: about {target} words in total. SUMMARY about {summary} words; AESTHETICS, LIGHTING and the CAMERA or ART STYLE line about {look} words together; "
        f"BACKGROUND about {background} words; each OBJECT line about {per_object} words."
    )
    format_lines = [
        "SUMMARY: one sentence describing the whole image",
        "AESTHETICS: the look and mood in a few words",
        "LIGHTING: how the scene is lit",
        "CAMERA: angle, lens and depth of field" if branch == "photo" else "ART STYLE: style, technique and texture",
        "COLORS: the main colors, comma-separated (names or #RRGGBB codes)",
        "BACKGROUND: what fills the background",
        "OBJECT: one thing in the image, where it is and how it looks (one line per thing)",
    ]
    if lettering.mode != "none":
        format_lines.append('TEXT: "the exact words" | how the lettering looks and where it is')
    closing = "Reply with these labelled lines in this order and nothing else (no JSON, no markdown, no preamble):\n" + "\n".join(format_lines)
    closing += "\n\nExample of the format for a different prompt (never copy its content):\n" + _EXAMPLE[branch]
    system = system_message("You turn prompts into structured captions for a text-to-image model.", rules, closing)
    return [{"role": "system", "content": system}, {"role": "user", "content": prompt}]


def _parse(text: str) -> _Draft:
    """Parse labelled lines; unlabelled lines continue the previous field, and a repeated single field appends."""
    values: dict[str, str] = {}
    items: list[list[str]] = []
    current: str | None = None
    for raw in strip_reasoning(text).splitlines():
        line = raw.strip()
        if not line or line.startswith("```"):
            continue
        match = _LINE_RE.match(line)
        if match:
            current = _LABELS[" ".join(match.group(1).upper().split())]
            value = match.group(2).strip().strip("*").strip()
            if current == "object":
                items.append(["obj", value])
            elif current == "text":
                items.append(["text", *_split_text_line(value)])
            else:
                values[current] = f"{values[current]} {value}".strip() if current in values else value
        elif current in ("object", "text") and items:
            items[-1][-1] = f"{items[-1][-1]} {line.strip('*').strip()}".strip()
        elif current is not None:
            values[current] = f"{values.get(current, '')} {line.strip('*').strip()}".strip()
    nested = re.match(r"^ART\s*STYLE\s*[:：]\s*(.*)$", values.get("camera", ""), re.I)
    if nested:  # "CAMERA: ART STYLE: ..." -- the model filled the template's look line with the other label.
        values["art_style"], values["camera"] = values.get("art_style") or nested.group(1), ""
    return _Draft(**values, items=tuple(tuple(item) for item in items if item[-1] or item[0] == "text"))


def _split_text_line(value: str) -> tuple[str, str]:
    """Split ``"words" | look`` into the words and the look; unquoted words end at the first ``|``."""
    match = re.match(r'^\s*["“「『«]\s*(.+?)\s*["”」』»]\s*[|:\-–—,]?\s*(.*)$', value)
    if match:
        return " ".join(match.group(1).split()), match.group(2).strip()
    words, _sep, look = value.partition("|")
    return " ".join(words.split()), look.strip()


def _assemble(draft: _Draft, prompt: str, settings: EnhanceSettings, lettering: _Lettering) -> dict[str, Any]:
    """Build the caption from *draft*: code sets the key order, branch, medium, colors and the user's lettering."""
    branch, medium = _look(settings, prompt)
    style: dict[str, Any] = {"aesthetics": draft.aesthetics or "detailed, natural", "lighting": draft.lighting or "natural light"}
    if branch == "photo":
        style |= {"photo": draft.camera or "eye-level, natural perspective", "medium": medium}
    else:
        style |= {"medium": medium, "art_style": draft.art_style or medium.replace("_", " ")}
    palette = [] if settings.style == "bw" else _colors(draft.colors)
    if palette:
        style["color_palette"] = palette
    return {
        "high_level_description": draft.summary,
        "style_description": style,
        "compositional_deconstruction": {"background": draft.background, "elements": _elements(draft, lettering)},
    }


def _colors(text: str) -> list[str]:
    """Return uppercase ``#RRGGBB`` colors from hex codes or common color names in *text*."""
    colors: list[str] = []
    for item in re.split(r"[,;/\n]| and ", text):
        hex_match = _HEX_RE.search(item)
        words = [word if word in _COLOR_NAMES else word.removesuffix("s") for word in re.findall(r"[a-z]+", item.lower())]
        color = f"#{hex_match.group(1).upper()}" if hex_match else next((_COLOR_NAMES[word] for word in reversed(words) if word in _COLOR_NAMES), None)
        if color is not None and color not in colors:
            colors.append(color)
    return colors[:_STYLE_COLORS_MAX]


def _elements(draft: _Draft, lettering: _Lettering) -> list[dict[str, Any]]:
    """Return the caption elements: the model's objects and the lettering the prompt allows, with every user quote present."""
    by_key = {_key(words): words for words in lettering.quotes}
    used: set[str] = set()
    elements: list[dict[str, Any]] = []
    invented = 0
    for item in draft.items:
        if item[0] == "obj":
            spoken = _prompt_words_in_quotes(item[1], lettering.source) if lettering.mode == "design" else None
            if spoken:  # 'The words "Happy 80th Birthday Grandma" in script': the user's lettering written as an OBJECT line.
                if _key(spoken) not in used:
                    used.add(_key(spoken))
                    elements.append({"type": "text", "text": spoken, "desc": item[1]})
            elif not _lettering_only(item[1], lettering.quotes):
                elements.append({"type": "obj", "desc": item[1]})
            continue
        words, look = item[1], item[2] if len(item) > 2 else ""
        key = _key(words)
        prefix = None if key in by_key else next((quote for quote in by_key if key.startswith(f"{quote} ")), None)
        if prefix is not None and not look:  # 'TEXT: LISBOA in bold serif': the words, then the look, unquoted.
            key, look = prefix, words[len(by_key[prefix]) :].strip(" ,|-:")
        elif lettering.mode == "design" and not look:  # 'TEXT: Happy 80th Birthday Grandma displayed at the top': the prompt's words, then the look.
            words, look = _prompt_prefix(words, lettering.source)
            key = _key(words)
        if key in used:
            continue
        if key in by_key:
            used.add(key)
            elements.append({"type": "text", "text": by_key[key], "desc": look or f"The words {by_key[key]}."})
        elif lettering.mode == "design" and words and f" {key} " in f" {_key(lettering.source)} ":
            used.add(key)
            elements.append({"type": "text", "text": words, "desc": look or f"The words {words}."})
        elif lettering.mode == "signs" and words and invented < _MAX_INVENTED_TEXTS and (lettering.script is None or re.search(lettering.script[1], words)):
            invented += 1
            elements.append({"type": "text", "text": words, "desc": look or f"The words {words}."})
    elements.extend({"type": "text", "text": words, "desc": f"The words {words}, clearly legible."} for key, words in by_key.items() if key not in used)
    return elements


def _lettering_only(desc: str, quotes: tuple[str, ...]) -> bool:
    """Return whether an OBJECT line only describes the user's lettering ('Bold gold letters reading "LISBOA"').

    A line naming a thing that carries the text ('A white mug with the text "BEST DAD"') is an object and is kept.
    """
    folded = desc.casefold()
    starts = [folded.find(words.casefold()) for words in quotes if words.casefold() in folded]
    if not starts:
        return False
    head = desc[: min(starts)]
    return bool(_LETTERING_RE.search(head)) and not _CARRIER_RE.search(head)


def _prompt_prefix(text: str, source: str) -> tuple[str, str]:
    """Split *text* into its longest leading run of two or more words found in *source*, and the rest; else ``(text, "")``."""
    tokens = text.split()
    for end in range(len(tokens), 1, -1):
        head = " ".join(tokens[:end])
        if f" {_key(head)} " in f" {_key(source)} ":
            return head.strip(" ,.:;|-"), " ".join(tokens[end:]).strip(" ,|-:")
    return text, ""


def _prompt_words_in_quotes(desc: str, source: str) -> str | None:
    """Return a double-quoted span of *desc* whose words all appear in *source*, else None."""
    for match in _DOUBLE_QUOTED_RE.finditer(desc):
        span = match.group(1) or match.group(2)
        if _key(span) and f" {_key(span)} " in f" {_key(source)} ":
            return " ".join(span.split())
    return None


def _key(words: str) -> str:
    """Return *words* casefolded with punctuation dropped, for comparing texts."""
    return " ".join(re.sub(r"[^\w\s]", " ", words.casefold()).split())
