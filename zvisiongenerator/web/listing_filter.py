"""Decide whether Discover lists a Hugging Face repository: the content filter.

The filter is a best-effort check on names, not a content rating. A name is split into words (see :func:`name_words`), and
a repository is hidden when one of its names uses a term:

- a **word** matches a whole word only, for a term that also occurs inside harmless words;
- a **stem** also matches inside a run-together word, for a long, distinctive term.

The terms ship in ``listing_filter.json``, base64-encoded only to keep them out of casual reading. :func:`is_filtered`
and :func:`parse_listing_filter` do no I/O; :func:`packaged_listing_filter` reads and caches the packaged file. The
``ui.discover_content_filter`` setting picks the level (see :func:`content_filter_level`).
"""

from __future__ import annotations

import base64
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from functools import cache
import json
from pathlib import Path
import re
from typing import Any
import warnings

__all__ = [
    "CONTENT_FILTER_LEVELS",
    "DATA_FILE",
    "ListingFilter",
    "content_filter_level",
    "is_filtered",
    "listing_filter_for",
    "name_words",
    "packaged_listing_filter",
    "parse_listing_filter",
]

DATA_FILE = Path(__file__).with_name("listing_filter.json")
CONTENT_FILTER_LEVELS = ("on", "off", "explicit")
_SEPARATORS = re.compile(r"[^A-Za-z0-9]+")
_CASE_CHANGE = re.compile(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
# A capital run before lower-case letters is a word of its own (``ABCv2``); the letters after it are not split off.
_CAPITAL_RUN = re.compile(r"[A-Z]{2,}(?=[a-z])")
_DIGIT_CHANGE = re.compile(r"(?<=[A-Za-z])(?=[0-9])|(?<=[0-9])(?=[A-Za-z])")


@dataclass(frozen=True)
class ListingFilter:
    """The content filter at one level: what hides a repository from Discover.

    Attributes:
        words: Terms matched as whole words.
        stems: Terms matched anywhere inside a run-together word.
        hide_adult_tagged: Whether repositories tagged ``not-for-all-audiences`` are hidden.
    """

    words: frozenset[str]
    stems: tuple[str, ...]
    hide_adult_tagged: bool = True


def name_words(text: str) -> set[str]:
    """Return the lower-cased words of *text*, split at separators and case changes, each also split between letters and digits.

    A capital run ends before a capitalised word, and a capital run before lower-case letters is also a word: ``MysticABC``
    gives ``mystic`` and ``abc``; ``ABCStyle`` gives ``abc`` and ``style``; ``ABCv2`` also gives ``abc``; ``krea2Term``
    gives ``krea2``, ``krea``, ``2`` and ``term``.
    """
    words: set[str] = set()
    for chunk in _SEPARATORS.split(text):
        words.update(run.lower() for run in _CAPITAL_RUN.findall(chunk))
        for word in _CASE_CHANGE.sub(" ", chunk).split():
            words.add(word.lower())
            words.update(part.lower() for part in _DIGIT_CHANGE.sub(" ", word).split())
    return words


def is_filtered(names: Iterable[str], rules: ListingFilter) -> bool:
    """Return whether any of *names* uses a word or a stem of *rules*."""
    for name in names:
        words = name_words(name)
        if not rules.words.isdisjoint(words):
            return True
        if any(stem in chunk for chunk in _SEPARATORS.split(name.lower()) for stem in rules.stems):
            return True
    return False


def parse_listing_filter(data: Mapping[str, Any]) -> ListingFilter:
    """Return the filter stored in *data*: base64-encoded, space-separated terms under ``words`` and ``stems``.

    Raises:
        ValueError: If either list is missing or is not base64-encoded text.
    """
    try:
        words, stems = (_decoded(data[key]) for key in ("words", "stems"))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("content filter data must hold base64-encoded 'words' and 'stems'.") from exc
    return ListingFilter(frozenset(words), stems)


def content_filter_level(setting: object) -> str:
    """Return the level the ``ui.discover_content_filter`` *setting* names: ``on``, ``off`` or ``explicit``.

    YAML reads a bare ``on`` and ``off`` as true and false, so booleans name those levels too; unset is ``on``. Any other
    value warns and is ``on``.
    """
    if setting is None or setting is True:
        return "on"
    if setting is False:
        return "off"
    if isinstance(setting, str) and setting.strip().lower() in CONTENT_FILTER_LEVELS:
        return setting.strip().lower()
    warnings.warn(f"ui.discover_content_filter must be one of {', '.join(CONTENT_FILTER_LEVELS)}; got {setting!r}, so it is on.", stacklevel=2)
    return "on"


def listing_filter_for(level: str, terms: ListingFilter) -> ListingFilter:
    """Return the filter for a content filter *level*: *terms* at ``on``, the adult tag alone at ``off``, nothing at ``explicit``."""
    if level == "on":
        return terms
    return ListingFilter(frozenset(), (), hide_adult_tagged=level == "off")


@cache
def packaged_listing_filter() -> ListingFilter:
    """Return the filter in the packaged ``listing_filter.json``, read once.

    Raises:
        RuntimeError: If the packaged file is missing or malformed, which is a fault of the installation.
    """
    try:
        return parse_listing_filter(json.loads(DATA_FILE.read_text(encoding="utf-8")))
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"The content filter could not be read from {DATA_FILE.name}.") from exc


def _decoded(value: object) -> tuple[str, ...]:
    """Return the lower-cased terms of one base64-encoded, space-separated list."""
    if not isinstance(value, str):
        raise TypeError("a term list must be a string.")
    return tuple(term.lower() for term in base64.b64decode(value.encode("ascii"), validate=True).decode("utf-8").split())
