"""Turn free text into a file-name slug, and tell whether a string is one."""

from __future__ import annotations

import re
import unicodedata

__all__ = ["MAX_SLUG_LENGTH", "is_slug", "slugify"]

MAX_SLUG_LENGTH = 64
_SLUG_PATTERN = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")


def slugify(text: str) -> str:
    """Return the file slug of *text*: ASCII letters and digits in lowercase, other runs becoming one dash, at most 64 characters.

    Returns:
        The slug, or ``""`` when *text* holds no ASCII letter or digit; callers raise their own message or fall back.
    """
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii").lower()
    return re.sub(r"[^a-z0-9]+", "-", ascii_text)[:MAX_SLUG_LENGTH].strip("-")


def is_slug(text: str) -> bool:
    """Return whether *text* is a slug: lowercase ASCII letters and digits in groups separated by single dashes."""
    return _SLUG_PATTERN.fullmatch(text) is not None
