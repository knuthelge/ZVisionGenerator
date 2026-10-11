"""Compose prompts from YAML snippets and expand random choices."""

from __future__ import annotations

import random
import re
from typing import Any

_SNIPPET_REF_RE = re.compile(r"\$([a-zA-Z_][a-zA-Z0-9_]*)")
_STANDALONE_REF_RE = re.compile(r"^\s*\$([a-zA-Z_][a-zA-Z0-9_]*)\s*$")
SEPARATOR = ". "


def flatten_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, dict):
        parts = []
        for k, v in value.items():
            flat = flatten_value(v)
            parts.append(f"{k}: {flat}" if flat else str(k))
        return SEPARATOR.join(parts)
    if isinstance(value, list):
        parts = [flatten_value(item) for item in value]
        return SEPARATOR.join(p for p in parts if p)
    return str(value)


def resolve_snippets(
    value: Any,
    snippets: dict[str, Any],
    _seen: frozenset[str] = frozenset(),
) -> Any:
    if value is None or isinstance(value, (int, float, bool)):
        return value

    if isinstance(value, str):
        standalone = _STANDALONE_REF_RE.match(value)
        if standalone:
            name = standalone.group(1)
            if name not in snippets:
                raise ValueError(f"Undefined snippet: '{name}'")
            if name in _seen:
                raise ValueError(f"Circular snippet reference: '{name}' already seen in {_seen}")
            return resolve_snippets(snippets[name], snippets, _seen | {name})

        def _replace_inline(match: re.Match) -> str:
            name = match.group(1)
            if name not in snippets:
                raise ValueError(f"Undefined snippet: '{name}'")
            if name in _seen:
                raise ValueError(f"Circular snippet reference: '{name}' already seen in {_seen}")
            resolved = resolve_snippets(snippets[name], snippets, _seen | {name})
            return flatten_value(resolved)

        return _SNIPPET_REF_RE.sub(_replace_inline, value)

    if isinstance(value, dict):
        return {k: resolve_snippets(v, snippets, _seen) for k, v in value.items()}

    if isinstance(value, list):
        return [resolve_snippets(item, snippets, _seen) for item in value]

    return value


def snippet_references(value: Any) -> list[str]:
    """Return every ``$name`` reference in *value*'s strings, in order, with repeats."""
    if isinstance(value, str):
        return _SNIPPET_REF_RE.findall(value)
    if isinstance(value, dict):
        return [name for item in value.values() for name in snippet_references(item)]
    if isinstance(value, list):
        return [name for item in value for name in snippet_references(item)]
    return []


_INNERMOST_RE = re.compile(r"\{([^{}]+)\}")


def expand_random_choices(text: str) -> str:
    """Expand ``{a|b|c}`` random choice blocks in *text*, supporting nesting.

    Innermost blocks are resolved first; the process repeats until no more
    blocks remain.

    Args:
        text: Input text containing zero or more ``{a|b|c}`` blocks.

    Returns:
        Text with all choice blocks resolved and whitespace stripped.
    """
    rng = random.Random()
    while _INNERMOST_RE.search(text):
        text = _INNERMOST_RE.sub(
            lambda m: rng.choice(m.group(1).split("|")),
            text,
        )
    return text.strip()


def compose_video_prompt(scene: str | None, prompt: str) -> str:
    """Return the text a video model receives: the scene, then the prompt.

    A blank *scene* returns *prompt* untouched (byte-identical), so runs that do not use a scene send exactly
    what they always did. Otherwise both parts are stripped and joined by one space; a blank part is dropped,
    so a scene-only run sends the scene alone.

    Args:
        scene: Who and where, look, style and camera; may be ``None`` or blank.
        prompt: What happens; may be blank when *scene* is not.
    """
    if not scene or not scene.strip():
        return prompt
    return " ".join(part.strip() for part in (scene, prompt) if part and part.strip())


_SENTENCE_END = frozenset(".!?…")
_CLOSING_QUOTES = frozenset('"”')
_QUOTE_CHARS = frozenset('"“”')
# Abbreviations whose period does not end a sentence.
_ABBREVIATIONS = frozenset({"mr.", "mrs.", "ms.", "dr.", "st.", "prof.", "jr.", "sr.", "vs.", "e.g.", "i.e."})


def normalize_prompt_text(text: str) -> str:
    """Return *text* with every whitespace run collapsed to one space and the ends trimmed.

    This is the one normalisation behind ``is_original_prompt``; the Web UI mirrors it by splitting on
    whitespace runs, dropping empty parts and joining with one space.
    """
    return " ".join(text.split())


def is_original_prompt(prompt_template: str, recorded_prompt: str) -> bool:
    """Return whether *prompt_template* still equals *recorded_prompt* once whitespace is normalised.

    Compare the user's template (before ``{a|b}`` expansion) with the prompt recorded in the source video.
    """
    return normalize_prompt_text(prompt_template) == normalize_prompt_text(recorded_prompt)


def strip_quoted_lines(text: str) -> str:
    """Drop every sentence of *text* that contains a double quote, and join the rest with single spaces.

    Sentences end at ``.``, ``!``, ``?`` or ``…`` followed by whitespace or the end, outside quotes. A terminator
    inside a quote ends the sentence too when the closing quote follows it and the next word does not start in
    lowercase (``"Hello." She waves.`` splits; ``"Run!" and sprints off.`` does not). ``"`` toggles the quote
    state, ``“`` opens and ``”`` closes it; apostrophes are ignored. Text without any double quote is returned as is.

    Args:
        text: A prompt that may contain quoted speech.
    """
    if not any(char in _QUOTE_CHARS for char in text):
        return text
    kept = [sentence for sentence, quoted in _split_sentences(text) if sentence and not quoted]
    return " ".join(kept)


def _split_sentences(text: str) -> list[tuple[str, bool]]:
    """Split *text* into stripped sentences, each with whether it contains a double quote."""
    sentences: list[tuple[str, bool]] = []
    buffer: list[str] = []
    quoted = False
    in_quote = False
    index = 0
    while index < len(text):
        char = text[index]
        buffer.append(char)
        if char in _QUOTE_CHARS:
            quoted = True
            in_quote = (not in_quote) if char == '"' else char == "“"
        elif char in _SENTENCE_END and _ends_sentence(text, index, in_quote):
            if in_quote:
                index += 1
                buffer.append(text[index])
                in_quote = False
            sentences.append(("".join(buffer).strip(), quoted))
            buffer, quoted = [], False
        index += 1
    sentences.append(("".join(buffer).strip(), quoted))
    return sentences


def _ends_sentence(text: str, index: int, in_quote: bool) -> bool:
    """Return whether the terminator at *index* ends a sentence, given the current quote state."""
    if text[index] == "." and _ends_with_abbreviation(text[: index + 1]):
        return False
    after = index + 1
    if not in_quote:
        return after >= len(text) or text[after].isspace()
    if after >= len(text) or text[after] not in _CLOSING_QUOTES:
        return False
    rest = text[after + 1 :]
    if rest and not rest[0].isspace():
        return False
    following = rest.lstrip()
    return not following or not following[0].islower()


def _ends_with_abbreviation(text: str) -> bool:
    """Return whether the last word of *text* (which ends in a period) is a common abbreviation such as ``Dr.``."""
    word = text.rsplit(None, 1)[-1].lower().lstrip("\"“'([")
    return word in _ABBREVIATIONS


def strip_quotes_while_original(resolved_prompt: str, *, template: str, recorded_prompt: str) -> str:
    """Return *resolved_prompt* without quoted sentences while *template* still equals *recorded_prompt*, else unchanged.

    Used when extending a clip recorded with a single prompt and no scene: re-sending its old dialogue would repeat the speech.

    Args:
        resolved_prompt: The prompt after snippet and ``{a|b}`` expansion.
        template: The user's prompt before expansion.
        recorded_prompt: The prompt recorded in the source video.
    """
    if is_original_prompt(template, recorded_prompt):
        return strip_quoted_lines(resolved_prompt)
    return resolved_prompt
