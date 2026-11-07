"""Split speech text into the parts a speech model reads, and into the Delivery marks it holds.

A speech model reads a limited length in one pass, so longer text is read in parts of at most ``max_chars`` characters and the
parts are joined into one take. A part holds whole sentences; a sentence that does not fit alone is cut into clauses,
and a clause that still does not fit is cut at a space, or at the limit when it has none. A line break the writer put between
two units of a part stays a line break in it, because a model may read it as a pause.

A ``[mark]`` (``[sad]``, ``[whisper]``, ``[as set]``) changes the Delivery from there to the next mark or the end of the text.
Marks split the text into segments first, and no part crosses a segment. The functions here are pure.
"""

from __future__ import annotations

import re
from collections.abc import Collection
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from zvisiongenerator.core.audio_types import SpeechDelivery
from zvisiongenerator.utils.audio_settings import MARKS

if TYPE_CHECKING:
    from zvisiongenerator.core.audio_types import AudioGenerationRequest

__all__ = ["SpeechPart", "has_marks", "known_marks_only", "marks_used", "request_speech_parts", "speech_parts", "strip_marks", "unknown_mark_message", "unknown_marks", "without_trailing_marks"]

# A sentence ends at one of these followed by a space, a line end or a closing quote; the full-width marks need no space.
_SENTENCE_MARKS = frozenset(".!?…。！？")
_SENTENCE_ALWAYS = frozenset("。！？")
# A clause ends at one of these; the dash and the full-width marks need no space around them.
_CLAUSE_MARKS = frozenset(";:—,；：，")
_CLAUSE_ALWAYS = frozenset("—；：，")
_CLOSERS = frozenset("\"'”’)]»")

# A bracketed run of 2 to 20 letters and spaces is a mark when it names one, and a mistake when it does not.
_BRACKETED = re.compile(r"\[([A-Za-z ]{2,20})\]")
_SPACE_RUN = re.compile(r"[ \t]{2,}")
_EDGE_SPACE = re.compile(r"[ \t]+(\n)|(?<=\n)[ \t]+|^[ \t]+|[ \t]+\Z")

# What a mark people expect to exist is written with instead (the Insert menu adds both); any other unknown mark is only refused.
_PAUSE_WORDS = frozenset({"pause", "pauses"})
_LAUGH_WORDS = frozenset({"laugh", "laughs", "laughing"})

_Piece = tuple[str, bool]
"""A stretch of text and whether the writer ended it with a line break."""


@dataclass(frozen=True)
class SpeechPart:
    """One part of a speech text: what to read, and how.

    Attributes:
        text: The part's text, without marks.
        delivery: How to say it, or ``None`` for a voice recording, which is read as recorded.
    """

    text: str
    delivery: SpeechDelivery | None


def speech_parts(text: str, delivery: SpeechDelivery | None, *, part_chars: int, max_text_chars: int) -> tuple[SpeechPart, ...]:
    """Split *text* into the parts to read, in reading order, each with the Delivery it is read in.

    The marks in *text* split it into segments. A segment's Delivery is *delivery* with that one mark's value applied (marks do
    not stack, so ``[whisper]`` after ``[sad]`` is *delivery* plus whisper); the text before the first mark, and the text after
    ``[as set]``, has *delivery* itself. Each segment is then split into parts of at most *part_chars* characters, and a part never crosses a
    segment. With no *delivery* (a voice recording) the marks are only taken out and every part has none.

    Args:
        text: The text to read, with its marks.
        delivery: The Delivery set for the take, or ``None`` for a voice recording.
        part_chars: The most characters one part holds.
        max_text_chars: The most characters the text may hold, not counting its marks.

    Returns:
        The parts; empty when the text holds nothing to read.

    Raises:
        ValueError: If the text without its marks is longer than *max_text_chars*.
    """
    segments = _segments(text, delivery)
    _check_length(sum(len(segment) for segment, _ in segments), max_text_chars)
    return tuple(SpeechPart(piece, segment_delivery) for segment, segment_delivery in segments for piece in _split(segment, part_chars))


def request_speech_parts(request: AudioGenerationRequest, *, part_chars: int, max_text_chars: int) -> tuple[SpeechPart, ...]:
    """Return the parts *request*'s speech text is read in: with its Delivery for a built-in voice, without any for a voice recording.

    A built-in voice request with no Delivery of its own reads at the neutral Delivery, so its marks still work.

    Raises:
        ValueError: If the text without its marks is longer than *max_text_chars*.
    """
    delivery = None if request.voice is not None else (request.delivery or SpeechDelivery())
    return speech_parts(request.prompt, delivery, part_chars=part_chars, max_text_chars=max_text_chars)


def has_marks(text: str) -> bool:
    """Return whether *text* holds a Delivery mark."""
    return any(_mark_name(found) in MARKS for found in _BRACKETED.finditer(text))


def unknown_marks(text: str) -> tuple[str, ...]:
    """Return the bracketed runs in *text* that look like a mark but are not one, as written (``[sadd]``), in order.

    A run is 2 to 20 letters and spaces between brackets; any other brackets are plain text. A speech model would read an unknown one aloud.
    """
    return tuple(found.group(0) for found in _BRACKETED.finditer(text) if found.group(1).strip() and _mark_name(found) not in MARKS)


def unknown_mark_message(found: str) -> str:
    """Return what the user reads when *found* (a run from :func:`unknown_marks`, brackets included) is refused: it names the way to get a pause or a laugh when that is what was meant.

    The Web UI words this the same way (``unknownMarkMessage`` in ``speechDelivery.ts``).
    """
    refused = f"{found} isn't a Delivery mark."
    name = " ".join(found[1:-1].lower().split())
    if name in _PAUSE_WORDS:
        return f"{refused} For a pause, type … (the Insert menu adds one)."
    if name in _LAUGH_WORDS:
        return f"{refused} For a laugh, write haha (the Insert menu adds one)."
    return f"{refused} Use one of the Delivery marks, or remove the brackets."


def strip_marks(text: str) -> str:
    """Return *text* without its Delivery marks: the characters that are read. Any other bracketed text stays."""
    return _BRACKETED.sub(lambda found: "" if _mark_name(found) in MARKS else found.group(0), text)


def without_trailing_marks(text: str) -> str:
    """Return *text* without the Delivery marks at its very end, which change nothing because no words follow them."""
    stripped = text.rstrip()
    while True:
        found = next((found for found in _BRACKETED.finditer(stripped) if found.end() == len(stripped) and _mark_name(found) in MARKS), None)
        if found is None:
            return stripped if stripped != text.rstrip() else text
        stripped = stripped[: found.start()].rstrip()


def marks_used(text: str) -> tuple[str, ...]:
    """Return the names of the Delivery marks in *text* (``whisper``, ``as set``), each once, in order of first use."""
    return tuple(dict.fromkeys(_mark_name(found) for found in _BRACKETED.finditer(text) if _mark_name(found) in MARKS))


def known_marks_only(text: str, *, only: Collection[str] | None = None) -> str:
    """Return *text* with every Delivery mark in its canonical form (``[Sad]`` becomes ``[sad]``) and every other mark-like run dropped.

    A run is mark-like when :func:`unknown_marks` would report it, so the result is text the server accepts. With *only*, a Delivery mark
    whose name is not in it is dropped too.
    """
    dropped = False

    def repair(found: re.Match[str]) -> str:
        nonlocal dropped
        name = _mark_name(found)
        if name in MARKS and (only is None or name in only):
            return f"[{name}]"
        if name not in MARKS and not found.group(1).strip():
            return found.group(0)
        dropped = True
        return ""

    repaired = _BRACKETED.sub(repair, text)
    if not dropped:
        return repaired
    # The spaces the dropped run sat between close up, and no line starts or ends with one.
    return _EDGE_SPACE.sub(r"\1", _SPACE_RUN.sub(" ", repaired))


def _check_length(length: int, max_text_chars: int) -> None:
    """Refuse a text of *length* characters when it is over *max_text_chars*."""
    if length > max_text_chars:
        raise ValueError(f"Shorten the text: it's over {max_text_chars:,} characters, about {max_text_chars // 1000} minutes of speech.")


def _mark_name(found: re.Match[str]) -> str:
    """Return the bracketed run *found* as a mark word: lower case, one space between words."""
    return " ".join(found.group(1).lower().split())


def _segments(text: str, delivery: SpeechDelivery | None) -> list[tuple[str, SpeechDelivery | None]]:
    """Cut *text* at its marks into ``(text, delivery)`` segments, the marks taken out; one segment with no delivery when *delivery* is ``None``."""
    segments: list[tuple[str, SpeechDelivery | None]] = []
    current = delivery
    start = 0
    for found in _BRACKETED.finditer(text):
        name = _mark_name(found)
        if name not in MARKS:
            continue
        segments.append((text[start : found.start()], current))
        target = MARKS[name]
        current = delivery if delivery is None or target is None else replace(delivery, **{target[0]: target[1]})
        start = found.end()
    segments.append((text[start:], current))
    if delivery is None:
        return [("".join(segment for segment, _ in segments), None)]
    return segments


def _split(text: str, max_chars: int) -> tuple[str, ...]:
    """Split *text* into parts of at most *max_chars* characters, in reading order, each stripped; empty for blank text.

    Whole sentences are packed into a part while they fit. A sentence longer than *max_chars* is cut into clauses
    (after ``;`` ``:`` ``—`` ``,``), and the clauses are packed the same way, so a clause can share a part with the
    sentence before it. A clause longer than *max_chars* is cut at the last space before the limit, or exactly at the
    limit when it has no space. Units the writer ended with a line break are joined by that line break, the others by a
    space. Empty parts are dropped. Marks are not looked at.
    """
    pieces: list[_Piece] = []
    for unit in _cut_after(text, _SENTENCE_MARKS, _SENTENCE_ALWAYS):
        sentence = unit.strip()
        broke = "\n" in unit
        if not sentence:
            if broke and pieces:  # a blank line after a sentence: the break belongs to it
                pieces[-1] = (pieces[-1][0], True)
            continue
        if len(sentence) <= max_chars:
            pieces.append((sentence, broke))
            continue
        cut: list[str] = []
        for clause in _cut_after(sentence, _CLAUSE_MARKS, _CLAUSE_ALWAYS):
            clause = clause.strip()
            cut.extend(piece.strip() for piece in ([clause] if len(clause) <= max_chars else _cut_at_spaces(clause, max_chars)))
        cut = [piece for piece in cut if piece]
        pieces.extend((piece, broke and index == len(cut) - 1) for index, piece in enumerate(cut))
    return tuple(part for part in _pack(pieces, max_chars) if part)


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
        cut = space + 1 if space > 0 else max_chars
        pieces.append(clause[:cut])
        clause = clause[cut:]
    if clause:
        pieces.append(clause)
    return pieces


def _pack(pieces: list[_Piece], max_chars: int) -> list[str]:
    """Pack *pieces* in order into parts, each at most *max_chars* long where the pieces allow.

    Two pieces in a part are joined by a line break when the writer ended the first with one, else by a space.
    """
    parts: list[str] = []
    current = ""
    broke = False
    for piece, ends_line in pieces:
        joiner = "\n" if broke else " "
        if current and len(current) + len(joiner) + len(piece) <= max_chars:
            current = f"{current}{joiner}{piece}"
        else:
            if current:
                parts.append(current)
            current = piece
        broke = ends_line
    if current:
        parts.append(current)
    return parts
