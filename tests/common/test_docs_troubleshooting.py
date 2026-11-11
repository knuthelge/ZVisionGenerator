"""Tests that every message the Troubleshooting page quotes in a heading is still in the code.

Convention: a heading of the page (level 3 or 4) is a *quoted heading* when its whole text, after an optional platform
prefix (``macOS: ``, ``Windows: `` or ``Linux: ``) and before an optional ``{ #anchor }`` attribute, is one string in
straight double quotes. The string is the fixed opening of a message the app prints or shows. Each ``…`` in it stands
for a variable part, so the string is split at every ``…`` and each literal piece must appear in some source file under
``zvisiongenerator/`` or ``frontend/src/``. Headings in plain words, or with a quoted phrase inside a longer heading,
describe a problem that has no message of its own and are not checked.
"""

from __future__ import annotations

import functools
import re

import pytest

from tests._paths import REPO_ROOT

_PAGE = REPO_ROOT / "docs" / "reference" / "troubleshooting.md"
_SOURCE_ROOTS = (REPO_ROOT / "zvisiongenerator", REPO_ROOT / "frontend" / "src")
# The packaged Web UI is build output of frontend/src; caches and dependencies are not source.
_SKIPPED_DIRECTORIES = frozenset({"static", "node_modules", "__pycache__"})
_SKIPPED_SUFFIXES = (".test.ts", ".spec.ts")
_VARIABLE_PART = "…"
_QUOTED_HEADING = re.compile(r'^#{3,4} (?:(?:macOS|Windows|Linux): )?"(?P<message>[^"]+)"(?: \{ #[\w-]+ \})?$')


def _quoted_headings() -> list[str]:
    """Return the message of every quoted heading on the page, in page order."""
    messages = []
    for line in _PAGE.read_text(encoding="utf-8").splitlines():
        match = _QUOTED_HEADING.match(line)
        if match:
            messages.append(match.group("message"))
    return messages


def _fixed_pieces(message: str) -> list[str]:
    """Split *message* at its variable parts and return the non-empty literal pieces, trimmed."""
    return [piece.strip() for piece in message.split(_VARIABLE_PART) if piece.strip()]


@functools.cache
def _source_text() -> str:
    """Return the text of every source file under the source roots, joined with a NUL so no piece spans two files."""
    texts = []
    for root in _SOURCE_ROOTS:
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.name.endswith(_SKIPPED_SUFFIXES):
                continue
            if _SKIPPED_DIRECTORIES.intersection(path.relative_to(root).parts):
                continue
            data = path.read_bytes()
            if b"\0" in data:
                continue  # a binary file
            texts.append(data.decode("utf-8", errors="ignore"))
    return "\0".join(texts)


def test_page_quotes_messages() -> None:
    """The page has many quoted headings, so a change to the heading convention can't silently empty the check."""
    assert len(_quoted_headings()) >= 80


@pytest.mark.parametrize("message", _quoted_headings())
def test_quoted_heading_is_in_the_code(message: str) -> None:
    """Every fixed piece of a quoted heading's message appears in the Python or frontend source."""
    pieces = _fixed_pieces(message)
    assert pieces, f"heading has no fixed text: {message!r}"
    source = _source_text()
    missing = [piece for piece in pieces if piece not in source]
    assert not missing, f"not in zvisiongenerator/ or frontend/src/: {missing!r} (heading {message!r})"
