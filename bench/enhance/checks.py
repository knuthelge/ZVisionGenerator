"""Score one enhancer output: length, quoted text, kept phrases, counts, forbidden additions, format, audio shape, lyrics structure and speech scripts (pure functions)."""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from typing import Any

from zvisiongenerator.utils.audio_settings import MARKS, reading_characters
from zvisiongenerator.utils.audio_writers import lyric_quotes, parse_fill_answer, parse_song
from zvisiongenerator.utils.prompt_enhance import EnhanceContext, LengthPlan, plan_length, resolve_parts, word_count
from zvisiongenerator.utils.prompt_quotes import lettering_spans, missing_quotes, quoted_texts
from zvisiongenerator.utils.prompt_enhance_audio import CaptionRules, caption_rules
from zvisiongenerator.utils.prompt_enhance_video import _kept_scene_action_plan, _pair_word_counts, plan_pair_length, repeats_source
from zvisiongenerator.utils.speech_text import strip_marks, unknown_marks

from bench.enhance.cases import Case

# The checks every case reports, in report order; a value is True/False, a 0-1 share (keep), or None when it does not apply.
CHECK_NAMES = (
    "usable",
    "first_try",
    "format",
    "length_in_range",
    "quotes_kept",
    "keep",
    "count",
    "forbid",
    "no_repeat",
    "banned",
    "marks_kept",
    "tags_kept",
    "scope_kept",
    "words_kept",
    "marks_known",
    "fit_in_range",
    "fields_filled",
    "only_empty",
    "values_valid",
    "fill_tags",
)
FIT_TOLERANCE = 0.15
"""How far a fitted script may read from its target time, as a share of it, for the bench to count it as landed (the app itself only warns beyond 25%)."""
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
    """Return the word plan production uses for *case*, and what it measures: ``prompt``, ``action``, ``pair`` or ``audio``; no plan for captions."""
    settings, context = case.settings, case.context
    if case.is_audio:
        # Lyrics have no word range: their structure is what is scored.
        # Lyrics and scripts have no word range: their structure, words and reading time are what is scored.
        return (None, case.task) if case.task in ("lyrics", "script", "fill") else (audio_length_plan(case), "audio")
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


# ── Audio captions ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class AudioShape:
    """What a family's audio text must look like, read from its caption rules.

    Attributes:
        lo: The fewest words a good text has.
        target: The words the rules aim for.
        hi: The most words the rules allow.
        sentences: The most sentences; 0 means a comma list of short descriptors with no sentence at all.
        banned: Words the rules say not to write (metadata that is added separately).
    """

    lo: int
    target: int
    hi: int
    sentences: int
    banned: tuple[str, ...] = ()


_TEMPO_WORDS = ("bpm", "tempo")
_VOCAL_WORDS = ("vocal", "vocalist", "singer", "sing", "singing", "lyric")
_INSTRUMENTAL_WORDS = (*_VOCAL_WORDS, "voice")
_NEGATED_VOCAL_RE = re.compile(r"\b(?:no|without)\s+(?:any\s+)?(?:(?:vocals?|vocalists?|singers?|singing|voices?|lyrics?)\b[\s,]*(?:(?:or|and)\s+)?)+", re.IGNORECASE)
_DESCRIPTOR_MIN_PARTS = 3
_DESCRIPTOR_MAX_WORDS = 6
_AUDIO_SHAPES = {
    "descriptors": AudioShape(lo=3, target=12, hi=30, sentences=0, banned=_TEMPO_WORDS),
    "sentence": AudioShape(lo=5, target=33, hi=45, sentences=1, banned=(*_TEMPO_WORDS, "TrackType", "Length")),
    "prose": AudioShape(lo=5, target=35, hi=60, sentences=2),
}
_BRACKET_RE = re.compile(r"\[[^\[\]\n]+\]")
_MARKDOWN_RE = re.compile(r"[*`#]|^\s*(?:[-•]|\d+[.)])\s", re.MULTILINE)
_SENTENCE_BREAK_RE = re.compile(r"[.!?]+(?=\s+\S)")
_SENTENCE_END_RE = re.compile(r"[.!?]\s*$")


def audio_shape_name(rules: CaptionRules) -> str:
    """Return the shape of a family's text from its caption rules: ``descriptors``, ``sentence`` or ``prose``.

    A family whose example is a comma list writes descriptors; one that cannot sing writes one sentence; any other gets plain prose.
    """
    if "," in rules.example:
        return "descriptors"
    return "sentence" if not rules.vocals else "prose"


def audio_shape(case: Case) -> AudioShape:
    """Return the shape the family of audio *case* writes."""
    return _AUDIO_SHAPES[audio_shape_name(caption_rules(case.family))]


def audio_length_plan(case: Case) -> LengthPlan:
    """Return the word range of the family's text, as a length plan (the rules' own ceiling, not the image and video budget)."""
    shape = audio_shape(case)
    return LengthPlan(length="same", target=shape.target, lo=shape.lo, hi=shape.hi, clamped=False)


def marks_in(text: str) -> list[str]:
    """Return the square-bracket marks and section tags of *text* (``[sad]``, ``[Verse 1]``), spacing normalised."""
    return [" ".join(mark.split()) for mark in _BRACKET_RE.findall(text)]


def marks_kept(source: str, output: str) -> bool | None:
    """Return whether every mark and section tag of *source* appears in *output*; None when *source* has none."""
    marks = marks_in(source)
    if not marks:
        return None
    return all(mark in marks_in(output) for mark in marks)


# ── Lyrics ─────────────────────────────────────────────────────────────────────


def lyrics_format(output: str) -> bool:
    """Return whether *output* is plain lyrics: not empty and without markdown."""
    return bool(output.strip()) and _MARKDOWN_RE.search(output) is None


def tags_kept(case: Case, output: str) -> bool:
    """Return whether *output* has the same section tags as the input lyrics, in the same order, each alone on its line."""
    return [block.tag.strip() for block in parse_song(output).blocks] == [block.tag.strip() for block in parse_song(case.prompt).blocks]


def scope_kept(case: Case, output: str) -> bool:
    """Return whether the task stayed in its scope.

    A whole-song improvement leaves a section without words empty, leaves a sung one sung and keeps a word-for-word repeat (a chorus) a repeat.
    One section's improvement changes that section and no other; a Write fills the empty section and changes no other. A different number of
    sections fails the check, since there is no section to compare.
    """
    before, after = parse_song(case.prompt).blocks, parse_song(output).blocks
    if len(before) != len(after):
        return False
    settings = case.settings
    if settings.section is not None:
        others = all(was.text == now.text for at, (was, now) in enumerate(zip(before, after, strict=True)) if at != settings.section)
        target = before[settings.section].text.strip(), after[settings.section].text.strip()
        return others and bool(target[1]) and (settings.action == "write" or target[0] != target[1])
    if any(bool(was.text.strip()) != bool(now.text.strip()) for was, now in zip(before, after, strict=True)):
        return False
    repeats = [[at for at, block in enumerate(before) if block.text.strip() == text] for text in dict.fromkeys(block.text.strip() for block in before) if text]
    return all(len({after[at].text.strip() for at in group}) == 1 for group in repeats)


# ── Speech scripts ─────────────────────────────────────────────────────────────

_KNOWN_MARK_RE = re.compile(r"\[(?:" + "|".join(re.escape(mark) for mark in MARKS) + r")\]", re.IGNORECASE)


def script_words(text: str) -> str:
    """Return the words of a speech script: the known marks and the pauses taken out, one space between words."""
    return " ".join(_KNOWN_MARK_RE.sub(" ", text).replace("…", " ").split())


def words_kept(case: Case, output: str) -> bool | None:
    """Return whether an Add delivery left every word as it was: the script with its marks and pauses removed is the output's. None for a Fit."""
    if case.settings.actions != ("delivery",):
        return None
    return script_words(output) == script_words(case.prompt)


def marks_known(output: str) -> bool:
    """Return whether the output holds no bracketed word the server would refuse."""
    return not unknown_marks(output)


def fit_ratio(case: Case, output: str) -> float | None:
    """Return the share of the target a Fit's output reads for, by the app's reading rate (marks not counted); None for an Add delivery."""
    settings = case.settings
    if "fit" not in settings.actions:
        return None
    return round(len(strip_marks(output)) / reading_characters(settings.seconds, settings.pace), 3)


def script_format(output: str) -> bool:
    """Return whether *output* is a plain script: not empty and without markdown."""
    return bool(output.strip()) and _MARKDOWN_RE.search(output) is None


# ── Fill in ────────────────────────────────────────────────────────────────────

_FILL_LABEL_RE = re.compile(r"^[\s*#>_-]*(style|tempo|key|lyrics|text|vibe)[\s*_]*:", re.IGNORECASE | re.MULTILINE)
_LYRICS_BLOCK_RE = re.compile(r"^[\s*#>_-]*lyrics[\s*_]*:(.*)\Z", re.IGNORECASE | re.MULTILINE | re.DOTALL)
_TAG_LINE_RE = re.compile(r"^\[[^\[\]\n]+\]$")
_FILL_VALUE_FIELDS = ("tempo", "key", "vibe")


def fill_requested(case: Case) -> tuple[str, ...]:
    """Return the fields a fill case asks for."""
    return tuple(case.settings.fields)


def expects_instrumental(case: Case) -> bool:
    """Return whether a fill case's idea is an instrumental, so the right Lyrics is ``none``: the case is tagged ``instrumental``."""
    return "instrumental" in case.tags


def fill_format(case: Case, output: str) -> bool:
    """Return whether a fill's answer has no markdown in its Style or Text (a key such as ``F#m`` is not markdown) and a usable value for at least one field."""
    answer = parse_fill_answer(output, fill_requested(case))
    return bool(answer.fields) and not any(_MARKDOWN_RE.search(answer.fields.get(field, "")) for field in ("style", "text"))


def fields_filled(case: Case, output: str) -> float:
    """Return the share of the requested fields the answer filled; Lyrics of an instrumental is filled by ``none`` and is not filled by words."""
    answer = parse_fill_answer(output, fill_requested(case))
    instrumental = expects_instrumental(case)
    done = 0
    for field in fill_requested(case):
        if field == "lyrics" and instrumental:
            done += answer.instrumental and "lyrics" not in answer.fields
        else:
            done += field in answer.fields
    return round(done / len(fill_requested(case)), 3)


def only_empty(case: Case, first_raw: str) -> bool:
    """Return whether the model's first answer wrote only the fields asked for: a field the user already set is not written again."""
    asked = set(fill_requested(case))
    return all(label.lower() in asked for label in _FILL_LABEL_RE.findall(first_raw))


def values_valid(case: Case, first_raw: str) -> float | None:
    """Return the share of the asked Tempo, Key and Vibe that the model's first answer got valid (in range, one of the 24, one the app serves); None when none was asked."""
    asked = tuple(field for field in _FILL_VALUE_FIELDS if field in fill_requested(case))
    if not asked:
        return None
    answer = parse_fill_answer(first_raw, asked)
    return round(sum(field in answer.fields for field in asked) / len(asked), 3)


def fill_tags(case: Case, first_raw: str) -> bool | None:
    """Return whether the model's first lyrics put every section tag alone on its own line, with a tag first; None when no lyrics were asked or the idea is an instrumental."""
    if "lyrics" not in fill_requested(case) or expects_instrumental(case):
        return None
    found = _LYRICS_BLOCK_RE.search(first_raw)
    if found is None:
        return False
    lines = [line.strip() for line in found.group(1).replace("\r\n", "\n").split("\n") if line.strip()]
    tags = [line for line in lines if line.startswith("[")]
    return bool(tags) and all(_TAG_LINE_RE.match(line) for line in tags) and lines[0] == tags[0]


def fill_banned(case: Case, output: str) -> bool | None:
    """Return whether a fill's Style avoids the words its family rules ban (a tempo, BPM, vocals for a model that cannot sing); None without a Style."""
    style = parse_fill_answer(output, ("style",)).fields.get("style")
    return None if style is None else banned_clear(case, style)


def banned_clear(case: Case, output: str) -> bool:
    """Return whether *output* avoids the words the family's rules ban (metadata, and vocals where there is no singing voice).

    A word the user's own text contains is not held against the rewrite, and a vocal word is fine when it is negated ("no vocals",
    "without singing"): the rewrite then says there is no voice, which is what the family wants.
    """
    rules = caption_rules(case.family)
    banned = list(audio_shape(case).banned)
    vocal: tuple[str, ...] = ()
    if case.task == "style" and case.audio_context is not None and case.audio_context.instrumental:
        vocal = _INSTRUMENTAL_WORDS
    elif not rules.vocals:
        vocal = _VOCAL_WORDS
    unnegated = _NEGATED_VOCAL_RE.sub(" ", output)
    found = [word for word in banned if _phrase_found(word, output)] + [word for word in vocal if _phrase_found(word, unnegated)]
    return not any(not _phrase_found(word, case.source) for word in found)


def audio_format(case: Case, output: str) -> bool:
    """Return whether *output* has the family's shape: one line, no markdown or brackets, and a comma list or at most N sentences."""
    shape = audio_shape(case)
    text = output.strip()
    source_marks = marks_in(case.source)
    plain = _BRACKET_RE.sub(lambda found: "" if " ".join(found.group().split()) in source_marks else found.group(), text)
    if not text or "\n" in text or _MARKDOWN_RE.search(text) or "[" in plain or "]" in plain:
        return False
    if shape.sentences == 0:
        parts = [part.strip() for part in text.split(",")]
        short = all(part and len(part.split()) <= _DESCRIPTOR_MAX_WORDS for part in parts)
        return len(parts) >= _DESCRIPTOR_MIN_PARTS and short and not _SENTENCE_BREAK_RE.search(text) and not _SENTENCE_END_RE.search(text)
    return len(_SENTENCE_BREAK_RE.findall(text)) + 1 <= shape.sentences


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
    first_raw: str = "",
) -> dict[str, Any]:
    """Return every check for one usable output, plus the measured word count and its ratio to the target.

    *first_raw* is the model's first raw answer; the fill checks about what it wrote before any retry read it.
    """
    text = caption_texts(output) if case.mode == "caption" else f"{scene}\n{output}".strip()
    text = text if text is not None else output
    source = case.source if case.is_audio else f"{case.scene}\n{case.prompt}"
    lyrics = case.is_audio and case.task == "lyrics"
    script = case.is_audio and case.task == "script"
    fill = case.is_audio and case.task == "fill"
    words = word_count(measured_text(output=output, scene=scene, measures=measures)) if plan is not None else None
    in_range, ratio = length_check(words, plan) if words is not None else (None, None)
    if script:
        ratio = fit_ratio(case, output)
    silent_video = case.mode == "video" and not case.context.get("audio")
    return {
        "usable": True,
        "first_try": attempts == 1,
        "format": fill_format(case, output) if fill else script_format(output) if script else lyrics_format(output) if lyrics else audio_format(case, output) if case.is_audio else format_ok,
        "length_in_range": in_range,
        "length_ratio": ratio,
        "words": words,
        # A silent video drops quoted speech by design, so only its lettering is checked.
        "quotes_kept": quotes_kept(source, output, lettering_only=False)
        if script
        else (None if not lyric_quotes(source) else not missing_quotes(lyric_quotes(source), text))
        if lyrics
        else quotes_kept(source, text, lettering_only=silent_video),
        "keep": keep_share(text, case.keep),
        "count": count_kept(text, case.count),
        "forbid": forbid_clear(text, case.forbid),
        "no_repeat": no_repeat(case, output),
        "banned": fill_banned(case, output) if fill else banned_clear(case, output) if case.is_audio and not (lyrics or script) else None,
        "marks_kept": None if fill else marks_kept(source, output) if case.is_audio and not (script and "fit" in case.settings.actions) else None,
        "tags_kept": tags_kept(case, output) if lyrics else None,
        "scope_kept": scope_kept(case, output) if lyrics else None,
        "words_kept": words_kept(case, output) if script else None,
        "marks_known": marks_known(output) if script else None,
        "fit_in_range": None if ratio is None or not script else abs(ratio - 1) <= FIT_TOLERANCE,
        "fields_filled": fields_filled(case, output) if fill else None,
        "only_empty": only_empty(case, first_raw) if fill else None,
        "values_valid": values_valid(case, first_raw) if fill else None,
        "fill_tags": fill_tags(case, first_raw) if fill else None,
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
