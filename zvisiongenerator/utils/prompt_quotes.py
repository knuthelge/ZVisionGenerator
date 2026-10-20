"""Find the texts a prompt quotes (lettering or speech) and check that a rewrite keeps them exactly (pure functions)."""

from __future__ import annotations

import re

__all__ = ["QUOTE_PATTERNS", "contains_text", "find_text", "lettering_spans", "missing_quotes", "quoted_texts"]

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
# Scripts from here on (CJK, Hangul syllables and later) do not separate words with spaces, so a quote there may touch
# the letters around it.
_UNSPACED_SCRIPTS_START = 0x2E80


def _quotes(text: str, extra: tuple[re.Pattern[str], ...] = ()) -> list[tuple[int, str]]:
    """Return every quote in *text*, repeats included, as (where its words start in *text*, the words), in order."""
    found: list[tuple[int, str]] = []
    taken: list[range] = []
    for pattern in (*QUOTE_PATTERNS, *extra):
        for match in pattern.finditer(text):
            group = next(index for index in range(1, (match.lastindex or 1) + 1) if match.group(index) is not None)
            words = " ".join(match.group(group).split())
            if words and not any(match.start(group) in span for span in taken):
                found.append((match.start(group), words))
                taken.append(range(match.start(group), match.end(group)))
    return sorted(found)


def quoted_texts(text: str, *, extra: tuple[re.Pattern[str], ...] = ()) -> tuple[str, ...]:
    """Return the texts *text* quotes, in order of appearance, without repeats and with inner spacing collapsed.

    Args:
        extra: Further patterns whose first matching group is a quoted text (the caption's "called X" names).
    """
    return tuple(dict.fromkeys(words for _start, words in _quotes(text, extra)))


def lettering_spans(text: str) -> tuple[str, ...]:
    """Return the quoted texts of *text* that are not speech, i.e. not right after a verb of speaking ("says", "asks").

    Each quote is judged by the words just before it in *text*. A text quoted more than once is lettering when any of
    its quotes is (a sign that reads what someone also says).
    """
    return tuple(dict.fromkeys(words for start, words in _quotes(text) if not _is_speech(text, start)))


def _is_speech(text: str, start: int) -> bool:
    """Return whether the quote whose words start at *start* in *text* follows a verb of speaking."""
    before = _SPACE_RE.sub(" ", text[max(0, start - 4 * _SPEECH_LOOKBACK) : start])[-_SPEECH_LOOKBACK:]
    return _SPEECH_BEFORE_RE.search(before) is not None


def missing_quotes(spans: tuple[str, ...], output: str) -> tuple[str, ...]:
    """Return the *spans* that *output* does not keep exactly: case and punctuation kept, spacing ignored (see ``contains_text``)."""
    flat = _SPACE_RE.sub(" ", output)
    return tuple(span for span in spans if not contains_text(flat, span))


def contains_text(text: str, span: str) -> bool:
    """Return whether *span* appears in *text* on its own, not as part of a longer word (see ``find_text``)."""
    return find_text(text, span) >= 0


def find_text(text: str, span: str) -> int:
    """Return where *span* first appears in *text* on its own, not as part of a longer word ("OPEN" is not in "OPENING"); -1 when it does not.

    Letters of scripts written without spaces (Chinese, Japanese) may touch the span, since their words are not spaced.
    """
    if not span:
        return 0
    start = text.find(span)
    while start >= 0:
        end = start + len(span)
        if not (_joins(text[start - 1 : start], span[0]) or _joins(span[-1], text[end : end + 1])):
            return start
        start = text.find(span, start + 1)
    return -1


def _joins(left: str, right: str) -> bool:
    """Return whether the characters *left* and *right* belong to one spaced word, so a span cannot end between them."""
    return bool(left and right) and _spaced_letter(left) and _spaced_letter(right)


def _spaced_letter(char: str) -> bool:
    """Return whether *char* is a letter or digit of a script that separates words with spaces."""
    return char.isalnum() and ord(char) < _UNSPACED_SCRIPTS_START
