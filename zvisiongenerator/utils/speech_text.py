"""Split speech text into parts a speech model reads faithfully.

Higgs reads about 600 characters in one pass, so longer text is read in parts of at most ``max_chars`` characters and the
parts are joined into one take. A part holds whole sentences; a sentence that does not fit alone is cut into clauses,
and a clause that still does not fit is cut at a space, or at the limit when it has none. The functions here are pure.
"""

from __future__ import annotations

import re

__all__ = ["DEFAULT_MAX_TEXT_CHARS", "DEFAULT_PART_CHARS", "split_speech_text"]

DEFAULT_PART_CHARS = 500
"""The most characters one part holds."""

DEFAULT_MAX_TEXT_CHARS = 10_000
"""The most characters a take reads, about ten minutes of speech."""

# A sentence ends at one of these followed by a space, a line end or a closing quote; the full-width marks need no space.
_SENTENCE_MARKS = frozenset(".!?…。！？")
_SENTENCE_ALWAYS = frozenset("。！？")
# A clause ends at one of these; the dash and the full-width marks need no space around them.
_CLAUSE_MARKS = frozenset(";:—,；：，")
_CLAUSE_ALWAYS = frozenset("—；：，")
_CLOSERS = frozenset("\"'”’)]»")
_TAG = re.compile(r"<\|.*?\|>")
_PRIVATE_USE_START = 0xE000


def split_speech_text(text: str, max_chars: int = DEFAULT_PART_CHARS, *, max_text_chars: int = DEFAULT_MAX_TEXT_CHARS) -> tuple[str, ...]:
    """Split *text* into parts of at most *max_chars* characters, in reading order.

    Whole sentences are packed into a part while they fit. A sentence longer than *max_chars* is cut into clauses
    (after ``;`` ``:`` ``—`` ``,``), and the clauses are packed the same way, so a clause can share a part with the
    sentence before it. A clause longer than *max_chars* is cut at the last space before the limit, or exactly at the
    limit when it has no space. An expressive tag (``<|sfx:laughter|>``) is never cut. Empty parts are dropped.

    Args:
        text: The text to read.
        max_chars: The most characters one part holds.
        max_text_chars: The most characters *text* may hold.

    Returns:
        The parts, each stripped; empty for blank text.

    Raises:
        ValueError: If *text* is longer than *max_text_chars*.
    """
    if len(text) > max_text_chars:
        raise ValueError(f"Shorten the text: it's over {max_text_chars:,} characters, about {max_text_chars // 1000} minutes of speech.")
    protected, tags = _protect_tags(text)
    pieces: list[str] = []
    for sentence in _cut_after(protected, _SENTENCE_MARKS, _SENTENCE_ALWAYS):
        sentence = sentence.strip()
        if len(sentence) <= max_chars:
            pieces.append(sentence)
            continue
        for clause in _cut_after(sentence, _CLAUSE_MARKS, _CLAUSE_ALWAYS):
            clause = clause.strip()
            pieces.extend(piece.strip() for piece in ([clause] if len(clause) <= max_chars else _cut_at_spaces(clause, max_chars)))
    parts = (_restore_tags(part, tags) for part in _pack(pieces, max_chars))
    return tuple(part for part in parts if part)


def _protect_tags(text: str) -> tuple[str, dict[str, str]]:
    """Replace each ``<|…|>`` tag with a run of one private-use character of the same length, so no rule can cut inside it.

    Returns:
        The protected text and the map from each run back to its tag.
    """
    tags: dict[str, str] = {}

    def replace(match: re.Match[str]) -> str:
        run = chr(_PRIVATE_USE_START + len(tags)) * len(match.group(0))
        tags[run] = match.group(0)
        return run

    return _TAG.sub(replace, text), tags


def _restore_tags(text: str, tags: dict[str, str]) -> str:
    """Put the tags that :func:`_protect_tags` replaced back into *text*."""
    for run, tag in tags.items():
        text = text.replace(run, tag)
    return text


def _cut_after(text: str, marks: frozenset[str], always: frozenset[str]) -> list[str]:
    """Cut *text* after each run of *marks* (with the closing quotes after it) and after each line break.

    A run of marks ends a unit only before a space, a closing quote or the end of the text, so ``3.5`` and ``12:30``
    stay whole; a run holding a mark of *always* ends it wherever it is.
    """
    units: list[str] = []
    start = index = 0
    while index < len(text):
        if text[index] == "\n":
            index += 1
            units.append(text[start:index])
            start = index
        elif text[index] in marks:
            end = index + 1
            while end < len(text) and text[end] in marks:
                end += 1
            ends_unit = end == len(text) or text[end].isspace() or text[end] in _CLOSERS or any(mark in always for mark in text[index:end])
            while end < len(text) and text[end] in _CLOSERS:
                end += 1
            index = end
            if ends_unit:
                units.append(text[start:index])
                start = index
        else:
            index += 1
    if start < len(text):
        units.append(text[start:])
    return units


def _cut_at_spaces(clause: str, max_chars: int) -> list[str]:
    """Cut *clause* into pieces of at most *max_chars* (plus the space they end at): at the last space before the limit, else at the limit."""
    pieces: list[str] = []
    while len(clause) > max_chars:
        space = clause.rfind(" ", 0, max_chars + 1)
        cut = space + 1 if space > 0 else _clear_of_tag(clause, max_chars)
        pieces.append(clause[:cut])
        clause = clause[cut:]
    if clause:
        pieces.append(clause)
    return pieces


def _clear_of_tag(text: str, cut: int) -> int:
    """Return *cut* moved to the start (or, for a tag at the very start, the end) of the protected tag it falls inside; else *cut* itself."""
    if cut >= len(text) or text[cut] != text[cut - 1] or ord(text[cut]) < _PRIVATE_USE_START:
        return cut
    mark = text[cut]
    start = cut
    while start > 0 and text[start - 1] == mark:
        start -= 1
    if start > 0:
        return start
    end = cut
    while end < len(text) and text[end] == mark:
        end += 1
    return end


def _pack(pieces: list[str], max_chars: int) -> list[str]:
    """Pack *pieces* in order into parts joined by a space, each at most *max_chars* long where the pieces allow."""
    parts: list[str] = []
    current = ""
    for piece in pieces:
        if not piece:
            continue
        if current and len(current) + 1 + len(piece) <= max_chars:
            current = f"{current} {piece}"
        else:
            if current:
                parts.append(current)
            current = piece
    if current:
        parts.append(current)
    return parts
