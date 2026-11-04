"""Song or instrumental: what the Lyrics box of a music take makes. Pure; the Web UI's status line mirrors this rule."""

from __future__ import annotations

import re
from typing import Final

__all__ = ["INSTRUMENTAL", "music_mode"]

_SONG: Final = "song"
"""The kind of a take whose Lyrics hold words."""

INSTRUMENTAL: Final = "instrumental"
"""The kind of a take whose Lyrics are empty or hold only ``[section]`` tags."""

_TAG: Final = re.compile(r"\[[^\[\]\n]+\]")


def _section_tags(text: str) -> list[str]:
    """Return the ``[section]`` tags of *text*, in order and as written."""
    return _TAG.findall(text)


def music_mode(lyrics: str, default_sections: str) -> tuple[str, str, int]:
    """Decide whether *lyrics* make a song or an instrumental, and what lyrics the model is given.

    Words outside ``[tags]`` make a song from the lyrics as given. Empty lyrics make an instrumental from
    *default_sections*, and lyrics of only tags make one from those tags; the section list (one tag per paragraph) is
    what the model is given, since the list sets the piece's length.

    Returns:
        ``(lyrics, kind, section_count)``: the lyrics to give the model, ``song`` or ``instrumental``, and the number of
        ``[section]`` tags in those lyrics.
    """
    tags = _section_tags(lyrics)
    if _TAG.sub("", lyrics).strip():
        return lyrics, _SONG, len(tags)
    tags = tags or _section_tags(default_sections)
    return "\n\n".join(tags), INSTRUMENTAL, len(tags)
