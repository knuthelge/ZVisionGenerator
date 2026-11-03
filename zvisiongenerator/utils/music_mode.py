"""Song or instrumental: what the Lyrics box of a music take makes. Pure; the Web UI's status line mirrors this rule."""

from __future__ import annotations

import re
from typing import Final

__all__ = ["INSTRUMENTAL", "SONG", "music_mode", "section_tags"]

SONG: Final = "song"
"""The kind of a take whose Lyrics hold words."""

INSTRUMENTAL: Final = "instrumental"
"""The kind of a take whose Lyrics are empty or hold only ``[section]`` tags."""

_TAG: Final = re.compile(r"\[[^\[\]\n]+\]")


def section_tags(text: str) -> list[str]:
    """Return the ``[section]`` tags of *text*, in order and as written."""
    return _TAG.findall(text)


def music_mode(lyrics: str, default_sections: str, *, cot_song: str = "full", cot_instrumental: str = "melody") -> tuple[str, str, str, int]:
    """Decide whether *lyrics* make a song or an instrumental, and what the model is given.

    Words outside ``[tags]`` make a song: *cot_song* and the lyrics as given. Empty lyrics make an instrumental from
    *default_sections*, and lyrics of only tags make one from those tags; both use *cot_instrumental*, and the section
    list (one tag per paragraph) is what the model is given as lyrics, since the list sets the piece's length.

    Args:
        lyrics: The Lyrics box.
        default_sections: The section list an empty Lyrics box stands for, such as ``"[intro] [verse] [chorus]"``.
        cot_song: The planning mode of a song.
        cot_instrumental: The planning mode of an instrumental.

    Returns:
        ``(cot, lyrics, kind, section_count)``: the planning mode, the lyrics to give the model, ``song`` or
        ``instrumental``, and the number of ``[section]`` tags in those lyrics.

    Raises:
        ValueError: If the lyrics are empty and *default_sections* holds no ``[section]`` tag.
    """
    tags = section_tags(lyrics)
    if _TAG.sub("", lyrics).strip():
        return cot_song, lyrics, SONG, len(tags)
    if not tags:
        tags = section_tags(default_sections)
        if not tags:
            raise ValueError("The default instrumental section list holds no [section] tag.")
    return cot_instrumental, "\n\n".join(tags), INSTRUMENTAL, len(tags)
