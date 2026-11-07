"""The caption Stable Audio 3 is given: the metadata shell the model was trained on around the user's words.

The model's text encoder was trained on captions that start with ``TrackType: <type>,`` and end with ``Length: <n> seconds``.
This module wraps a prompt in that shell (music gets ``TrackType: Music``, a sound has no prefix) and adds the music settings as caption words. The prompt is never cut: the encoder
truncates at 256 tokens on its own, and the audio enhancer aims for a short caption, not this module.
"""

from __future__ import annotations

from zvisiongenerator.utils.audio_settings import key_words

__all__ = ["sa3_caption"]

_MUSIC_PREFIX = "TrackType: Music"
_KINDS = ("sound", "music")
_TRACK_TYPE_MARK = "tracktype:"
_LENGTH_MARK = "length:"


def _seconds_text(seconds: float) -> str:
    """Return *seconds* as ``30`` when whole, else with one decimal (``7.5``)."""
    return str(int(seconds)) if float(seconds).is_integer() else f"{seconds:.1f}"


def sa3_caption(prompt: str, *, kind: str, seconds: float, tempo: int | None = None, key: str | None = None) -> str:
    """Return *prompt* in the caption shape of Stable Audio 3 for a *kind* take of *seconds*.

    A sound reads ``{prompt}. Length: {seconds} seconds``. Music reads
    ``TrackType: Music, {prompt}, {tempo} BPM, {key in words}, no vocals. Length: {seconds} seconds``, with the tempo and the key
    only when set. One trailing period of the prompt is dropped. A music prompt that already starts with ``TrackType:`` gets no
    prefix and one that contains ``Length:`` gets no suffix (case-insensitive). The prompt is never shortened.

    Args:
        prompt: The user's description.
        kind: ``sound`` or ``music``.
        seconds: The resolved length of the take.
        tempo: The tempo of music in beats per minute, or ``None`` to leave it out.
        key: The key of music as an ABC id (``F#m``), or ``None`` to leave it out.

    Returns:
        The caption to send to the model.

    Raises:
        ValueError: If *kind* is not ``sound`` or ``music``.
    """
    if kind not in _KINDS:
        raise ValueError(f"Stable Audio 3 makes sound or music, not {kind!r}.")
    text = prompt.strip()
    if text.endswith("."):
        text = text[:-1].rstrip()
    lowered = text.lower()
    parts: list[str] = []
    if kind == "music" and not lowered.startswith(_TRACK_TYPE_MARK):
        parts.append(_MUSIC_PREFIX)
    if text:
        parts.append(text)
    if kind == "music":
        if tempo is not None:
            parts.append(f"{tempo} BPM")
        if key is not None:
            parts.append(key_words(key))
        parts.append("no vocals")
    caption = ", ".join(parts)
    if _LENGTH_MARK in lowered:
        return caption
    return f"{caption}. Length: {_seconds_text(seconds)} seconds"
