"""Score one enhancer output: length, quoted text, kept phrases, counts, forbidden additions and format (pure functions)."""

from __future__ import annotations

import json
import re
import unicodedata
from typing import Any

from zvisiongenerator.utils.prompt_enhance import EnhanceContext, LengthPlan, plan_length, resolve_parts, word_count
from zvisiongenerator.utils.prompt_quotes import lettering_spans, missing_quotes, quoted_texts
from zvisiongenerator.utils.prompt_enhance_video import _kept_scene_action_plan, _pair_word_counts, plan_pair_length, repeats_source

from bench.enhance.cases import Case

# The checks every case reports, in report order; a value is True/False, a 0-1 share (keep), or None when it does not apply.
CHECK_NAMES = ("usable", "first_try", "format", "length_in_range", "quotes_kept", "keep", "count", "forbid", "no_repeat")
_CAPTION_OBJECTS = (1, 8)
_NUMBER_WORDS = {
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
}
_DIGIT_WORDS = {value: word for word, value in _NUMBER_WORDS.items()}
_NON_WORD_RE = re.compile(r"[^\w\s]+")


def length_plan(case: Case, *, ceiling: int, length_cfg: dict[str, Any] | None) -> tuple[LengthPlan | None, str]:
    """Return the word plan production uses for *case*, and what it measures: ``prompt``, ``action`` or ``pair``; no plan for captions."""
    settings, context = case.settings, case.context
    if case.mode == "caption":
        return None, "caption"
    if case.mode == "image":
        return plan_length(word_count(case.prompt), settings.length, ceiling=ceiling, length_cfg=length_cfg), "prompt"
    scene, prompt = case.scene.strip(), case.prompt.strip()
    extends = bool(context.get("extends"))
    if resolve_parts(settings, scene=scene, extends=extends, reference_image=bool(context.get("reference_image"))) == "action":
        plan, _heavy = _kept_scene_action_plan(word_count(scene), word_count(prompt), settings.length, ceiling=ceiling, length_cfg=length_cfg, extends=extends)
        return plan, "action"
    scene_words, prompt_words = _pair_word_counts(scene, prompt, _enhance_context(context))
    return plan_pair_length(scene_words, prompt_words, settings.length, ceiling=ceiling, length_cfg=length_cfg).total, "pair"


def _enhance_context(context: dict[str, Any]) -> EnhanceContext:
    """Return the minimal ``EnhanceContext`` the pair word count reads (extension and its source prompt)."""
    return EnhanceContext(extends=bool(context.get("extends")), extract_from=context.get("extract_from"), prompt_is_original=bool(context.get("prompt_is_original")))


def measured_text(*, output: str, scene: str, measures: str) -> str:
    """Return the text the length plan measures: the action alone, the scene and action together, or the prompt."""
    return f"{scene} {output}".strip() if measures == "pair" else output


def length_check(words: int, plan: LengthPlan | None) -> tuple[bool | None, float | None]:
    """Return whether *words* lies within the plan's range, and its ratio to the target; both None without a plan."""
    if plan is None:
        return None, None
    return plan.lo <= words <= plan.hi, round(words / max(1, plan.target), 3)


def caption_texts(output: str) -> str | None:
    """Return every string value of a JSON caption joined by newlines, or None when *output* is not a JSON object."""
    try:
        data = json.loads(output)
    except json.JSONDecodeError, TypeError:
        return None
    if not isinstance(data, dict):
        return None
    return "\n".join(_strings(data))


def _strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [text for item in value.values() for text in _strings(item)]
    if isinstance(value, list):
        return [text for item in value for text in _strings(item)]
    return []


def caption_format(output: str) -> bool:
    """Return whether *output* is a caption with a description, a background and 1 to 8 object elements.

    The element count is not a content check: a design's lettering lines become text elements, so a poster may have one
    object. Missing subjects are what ``keep`` and ``count`` catch.
    """
    try:
        data = json.loads(output)
    except json.JSONDecodeError, TypeError:
        return False
    if not isinstance(data, dict) or not data.get("high_level_description"):
        return False
    composition = data.get("compositional_deconstruction") or {}
    elements = composition.get("elements") or []
    objects = sum(1 for element in elements if isinstance(element, dict) and element.get("type") == "obj")
    return bool(composition.get("background")) and _CAPTION_OBJECTS[0] <= objects <= _CAPTION_OBJECTS[1]


def quotes_kept(source: str, output: str, *, lettering_only: bool = False) -> bool | None:
    """Return whether every quoted span of *source* appears in *output* (spacing ignored); None when *source* quotes nothing.

    Args:
        lettering_only: Check only lettering, not quoted speech (a silent video drops speech by design).
    """
    spans = lettering_spans(source) if lettering_only else quoted_texts(source)
    if not spans:
        return None
    return not missing_quotes(spans, output)


def _normalize(text: str) -> str:
    """Return *text* case-folded, accents removed ("café" → "cafe") and punctuation turned into spaces."""
    folded = "".join(char for char in unicodedata.normalize("NFKD", text.casefold()) if not unicodedata.combining(char))
    return " ".join(_NON_WORD_RE.sub(" ", folded).split())


def _phrase_found(phrase: str, text: str, *, stem: bool = False, gap: int = 0) -> bool:
    """Return whether *phrase* occurs in *text* as whole words, case and punctuation ignored, its last word optionally plural.

    Args:
        stem: Let every word match as the start of a longer word ("river" finds "riverbank", "read" finds "reads").
        gap: Allow up to this many extra words after the first word ("three cats" finds "three fluffy cats").
    """
    words = _normalize(phrase).split()
    if not words:
        return False
    tail = r"\w*" if stem else r"(?:s|es)?"
    parts = [re.escape(word) + (r"\w*" if stem else "") for word in words[:-1]] + [re.escape(words[-1]) + tail]
    joints = [rf"(?:\s+\w+){{0,{gap}}}\s+"] + [r"\s+"] * (len(parts) - 2) if len(parts) > 1 else []
    pattern = parts[0] + "".join(joint + part for joint, part in zip(joints, parts[1:]))
    return re.search(rf"(?<!\w){pattern}(?!\w)", _normalize(text)) is not None


def _kept(item: str, text: str) -> bool:
    """Return whether any ``|``-separated alternative of a keep *item* is found, words matching as stems, up to two words apart."""
    return any(_phrase_found(option, text, stem=True, gap=2) for option in item.split("|"))


def keep_share(output: str, phrases: tuple[str, ...]) -> float | None:
    """Return the share of *phrases* found in *output*, or None when there are none.

    An item may list alternatives separated by ``|`` ("old man|elderly man"); words match as stems ("river" finds "riverbank").
    """
    if not phrases:
        return None
    return round(sum(_kept(phrase, output) for phrase in phrases) / len(phrases), 3)


def count_kept(output: str, phrase: str | None) -> bool | None:
    """Return whether a counted *phrase* ("three cats") survives in *output*, as a number word or a digit; None without one.

    Up to two words may sit between the number and the noun ("three fluffy cats").
    """
    if not phrase:
        return None
    first, _sep, rest = _normalize(phrase).partition(" ")
    variants = {phrase}
    if first in _NUMBER_WORDS:
        variants.add(f"{_NUMBER_WORDS[first]} {rest}")
    elif first.isdigit() and int(first) in _DIGIT_WORDS:
        variants.add(f"{_DIGIT_WORDS[int(first)]} {rest}")
    return any(_phrase_found(variant, output, gap=2) for variant in variants)


def forbid_clear(output: str, phrases: tuple[str, ...]) -> bool | None:
    """Return whether none of *phrases* occur in *output*, or None when there are none."""
    if not phrases:
        return None
    return not any(_phrase_found(phrase, output) for phrase in phrases)


def no_repeat(case: Case, action: str) -> bool | None:
    """Return whether an extension's action avoids repeating the clip's earlier prompt; None for other cases."""
    previous = case.context.get("previous") or case.context.get("extract_from")
    if case.mode != "video" or not case.context.get("extends") or not previous:
        return None
    return not repeats_source(action, previous=previous)


def score_output(
    case: Case,
    *,
    output: str,
    scene: str,
    attempts: int,
    format_ok: bool | None,
    plan: LengthPlan | None,
    measures: str,
) -> dict[str, Any]:
    """Return every check for one usable output, plus the measured word count and its ratio to the target."""
    text = caption_texts(output) if case.mode == "caption" else f"{scene}\n{output}".strip()
    text = text if text is not None else output
    words = word_count(measured_text(output=output, scene=scene, measures=measures)) if plan is not None else None
    in_range, ratio = length_check(words, plan) if words is not None else (None, None)
    silent_video = case.mode == "video" and not case.context.get("audio")
    return {
        "usable": True,
        "first_try": attempts == 1,
        "format": format_ok,
        "length_in_range": in_range,
        "length_ratio": ratio,
        "words": words,
        # A silent video drops quoted speech by design, so only its lettering is checked.
        "quotes_kept": quotes_kept(f"{case.scene}\n{case.prompt}", text, lettering_only=silent_video),
        "keep": keep_share(text, case.keep),
        "count": count_kept(text, case.count),
        "forbid": forbid_clear(text, case.forbid),
        "no_repeat": no_repeat(case, output),
    }


def failed_checks() -> dict[str, Any]:
    """Return the checks of a case whose enhancement failed: not usable, everything else not measured."""
    return {name: None for name in CHECK_NAMES} | {"usable": False, "length_ratio": None, "words": None}


def case_score(checks: dict[str, Any]) -> float:
    """Return the mean of the checks that apply (True = 1, a share as is); a failed enhancement scores 0."""
    if checks.get("usable") is False:
        return 0.0
    values = [float(checks[name]) for name in CHECK_NAMES if checks.get(name) is not None]
    return round(sum(values) / len(values), 3) if values else 0.0
