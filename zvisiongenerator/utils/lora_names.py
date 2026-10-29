"""Choose the local name a downloaded LoRA is saved under: a slug of its repository, made free of clashes."""

from __future__ import annotations

from collections.abc import Collection
import re

__all__ = ["MAX_NAME_LENGTH", "repo_slug", "unique_name"]

MAX_NAME_LENGTH = 60
_FALLBACK = "lora"
_NON_WORD = re.compile(r"[^a-z0-9]+")


def repo_slug(repo_id: str) -> str:
    """Return a file-name-safe, lower-case name from the repository part of *repo_id* (``owner/Some-Name`` gives ``some_name``).

    The slug uses only ``a-z``, ``0-9`` and ``_``, is at most ``MAX_NAME_LENGTH`` characters and is never empty.
    """
    repo = repo_id.rsplit("/", 1)[-1]
    slug = _NON_WORD.sub("_", repo.lower()).strip("_")[:MAX_NAME_LENGTH].strip("_")
    return slug or _FALLBACK


def unique_name(base: str, taken: Collection[str]) -> str:
    """Return *base*, or *base* with ``-2``, ``-3`` and so on, whichever is the first not in *taken*.

    Names compare case-insensitively, as the file systems the LoRA folder lives on do.
    """
    used = {name.casefold() for name in taken}
    if base.casefold() not in used:
        return base
    number = 2
    while f"{base}-{number}".casefold() in used:
        number += 1
    return f"{base}-{number}"
