"""Add the delivery of a model's answer to a script without touching the script's own text (pure functions).

A model asked to add Delivery marks, pauses and line breaks to a script often changes the words around them: it straightens quotes,
re-punctuates, or turns a direction in brackets into a mark. :func:`realign_delivery` takes only what the answer adds between the words
(its known marks, its ``…`` pauses and its line breaks) and puts that into the script's own text, so the words come back exactly as the
user wrote them. The answer and the script are matched word by word, ignoring case and punctuation.
"""

from __future__ import annotations

import difflib
import re

from zvisiongenerator.utils.speech_text import has_marks, known_marks_only

__all__ = ["realign_delivery"]

_PAUSE = "…"
_ASCII_PAUSE = "..."
_BREAK = "\n"
# A token is a bracketed run, a pause, a line break or a run of other characters (a word with its punctuation).
_TOKEN = re.compile(r"\[[^\[\]\n]*\]|…|\n|[^\s…\[\]]+")
_NOT_A_LETTER = re.compile(r"[\W_]")
_SENTENCE_END = ".!?…\n:"
_CLOSING_QUOTES = ('"', "”")
_SPACE_BEFORE_BREAK = re.compile(r"[ \t]+\n")


def _is_delivery(token: str) -> bool:
    """Return whether *token* is something an answer may add between words: a known mark, a pause or a line break."""
    return token in (_PAUSE, _BREAK) or has_marks(token)


def _has_pause(text: str) -> bool:
    """Return whether *text* holds a pause: a ``…``, or a word or line that ends in ``...``."""
    return _PAUSE in text or text.rstrip().endswith(_ASCII_PAUSE)


def _is_word(token: str) -> bool:
    return not token.startswith("[") and token not in (_PAUSE, _BREAK)


def _key(word: str) -> str:
    """Return *word* as it is compared: lower case, without punctuation."""
    return _NOT_A_LETTER.sub("", word.casefold())


def _answer_words(answer: str) -> tuple[list[str], list[list[str]], list[str]]:
    """Cut *answer* into its words, what stands before each word, and what follows the last word.

    Any bracketed run that is not a known mark is dropped.
    """
    words: list[str] = []
    before: list[list[str]] = []
    pending: list[str] = []
    for token in _TOKEN.findall(answer):
        if _is_delivery(token):
            pending.append(known_marks_only(token))  # a mark in its canonical form; a pause or a break as it is
        elif _is_word(token):
            words.append(token)
            before.append(pending)
            pending = []
    return words, before, pending


def _additions(source_words: list[str], answer: str) -> tuple[list[list[str]], bool]:
    """Return what the answer adds before each of *source_words*, and whether it ends on a pause after its last word (an answer without words adds nothing)."""
    words, before, tail = _answer_words(answer)
    added: list[list[str]] = [[] for _ in source_words]
    matcher = difflib.SequenceMatcher(None, [_key(word) for word in source_words], [_key(word) for word in words], autojunk=False)
    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op in ("equal", "replace"):
            for offset in range(min(i2 - i1, j2 - j1)):
                added[i1 + offset] += before[j1 + offset]
        elif op == "insert" and i1 < len(source_words):
            # Words the model invented are not kept, but the delivery it put before them goes to the next word of the script.
            added[i1] += [token for at in range(j1, j2) for token in before[at]]
    return added, bool(words) and _PAUSE in tail


def _starts_a_sentence(before: str) -> bool:
    """Return whether *before* (the script up to a word) ends a sentence or a line, or opens the text, so a mark may start there."""
    text = before.rstrip(" \t")
    return not text or text[-1] in _SENTENCE_END or (text.endswith(_CLOSING_QUOTES) and text[-2:-1] in ".!?")


def _fresh(extra: list[str], gap: str, *, mark_allowed: bool) -> list[str]:
    """Return the tokens of *extra* that *gap* (the script's own text before the word) does not already hold, no pause twice in a row.

    A previous word that ends in ``...`` is a pause the script already holds, like a ``…``.
    """
    kept: list[str] = []
    for token in extra:
        if token.startswith("[") and not mark_allowed:
            continue
        if token == _PAUSE:
            held = _has_pause(gap)
        elif token == _BREAK:
            held = _BREAK in gap
        else:
            held = token.lower() in gap.lower()
        if held or (kept and kept[-1] == token):
            continue
        kept.append(token)
    return kept


def realign_delivery(source: str, answer: str) -> str:
    """Return *source* with the known marks, pauses and line breaks that *answer* adds between its words; the script's own text is kept exactly.

    A mark is added only at the start of a sentence or a line and only where the script has no mark already; a pause or a line break
    is never added where the script has one. Words the answer changed, added or dropped, and brackets it wrote that are not marks, are
    ignored. The result is *source* itself when the answer adds nothing.

    Args:
        source: The script as the user wrote it, brackets and all.
        answer: What the model returned.
    """
    found = [match for match in _TOKEN.finditer(source) if _is_word(match.group())]
    starts = [match.start() for match in found]
    words = [match.group() for match in found]
    added, ends_on_pause = _additions(words, answer)
    pieces: list[str] = []
    last = 0
    for start, word, extra in zip(starts, words, added, strict=True):
        gap = source[last:start]
        own_mark = has_marks(gap)
        mark_allowed = not own_mark and _starts_a_sentence(source[:start])
        # A bare punctuation mark that stands apart (after a bracket, or a dash between spaces) takes nothing before it.
        kept = _fresh(extra, gap, mark_allowed=mark_allowed) if _key(word) else []
        pieces.append(gap + "".join(token if token == _BREAK else f"{token} " for token in kept))
        last = start
    rest = source[last:]
    if ends_on_pause and not _has_pause(rest):
        text = rest.rstrip()
        rest = f"{text} {_PAUSE}{rest[len(text) :]}"
    pieces.append(rest)
    return _SPACE_BEFORE_BREAK.sub(_BREAK, "".join(pieces))
