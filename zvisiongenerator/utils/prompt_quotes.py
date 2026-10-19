"""Find the texts a prompt quotes (lettering or speech) and check that a rewrite keeps them exactly (pure functions)."""

from __future__ import annotations

import re

__all__ = ["QUOTE_PATTERNS", "lettering_spans", "missing_quotes", "quoted_texts"]

# Quoted text: paired quotes, guillemets and CJK brackets anywhere; single quotes only around a phrase of two or more
# characters, which may hold an apostrophe between letters ('Joe's Diner'), while an apostrophe on its own as in "Moe's" or
# "rock 'n' roll" is not a quote; an all-caps phrase after "text", "reads" or "says", which may go on with numbers and
# single letters (OPEN 24 HOURS, PLAN B).
QUOTE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r'"([^"\n]+)"|“([^”\n]+)”|«\s*([^»\n]+?)\s*»|「([^」\n]+)」|『([^』\n]+)』'),
    re.compile(r"(?<![\w'‘])['‘]((?:[^'’\n]|(?<=\w)['’](?=\w)){2,})['’](?!\w)"),
    re.compile(r"\b(?:text|word|words|reads|reading|says|saying|lettering)\s+([A-Z][A-Z0-9]+(?:\s(?:[A-Z][A-Z0-9]*|[0-9]+))*)\b"),
)
# A verb of speaking just before a quote makes it speech ("she says "Hi""); the quote mark may follow a colon or comma.
_SPEECH_BEFORE_RE = re.compile(
    r"\b(?:say|says|said|saying|shout\w*|yell\w*|ask\w*|whisper\w*|call\w*|repl\w*|exclaim\w*|mutter\w*|sing\w*|sang|cr(?:y|ies|ied)|tell\w*|told|speak\w*|spoke)\b[\s,:]*[\"“'‘«「『]?\s*$",
    re.I,
)
_SPEECH_LOOKBACK = 40
_SPACE_RE = re.compile(r"\s+")


def quoted_texts(text: str, *, extra: tuple[re.Pattern[str], ...] = ()) -> tuple[str, ...]:
    """Return the texts *text* quotes, in order of appearance, without repeats and with inner spacing collapsed.

    Args:
        extra: Further patterns whose first matching group is a quoted text (the caption's "called X" names).
    """
    found: list[tuple[int, str]] = []
    taken: list[range] = []
    for pattern in (*QUOTE_PATTERNS, *extra):
        for match in pattern.finditer(text):
            group = next(index for index in range(1, (match.lastindex or 1) + 1) if match.group(index) is not None)
            words = " ".join(match.group(group).split())
            if words and not any(match.start(group) in span for span in taken):
                found.append((match.start(group), words))
                taken.append(range(match.start(group), match.end(group)))
    return tuple(dict.fromkeys(words for _start, words in sorted(found)))


def lettering_spans(text: str) -> tuple[str, ...]:
    """Return the quoted texts of *text* that are not speech, i.e. not right after a verb of speaking ("says", "asks")."""
    flat = _SPACE_RE.sub(" ", text)
    kept = []
    for span in quoted_texts(text):
        start = flat.find(span)
        if start < 0 or not _SPEECH_BEFORE_RE.search(flat[max(0, start - _SPEECH_LOOKBACK) : start]):
            kept.append(span)
    return tuple(kept)


def missing_quotes(spans: tuple[str, ...], output: str) -> tuple[str, ...]:
    """Return the *spans* that do not appear in *output* exactly (case and punctuation kept, spacing ignored)."""
    flat = _SPACE_RE.sub(" ", output)
    return tuple(span for span in spans if span not in flat)
