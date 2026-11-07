"""Edit YuE2's ABC score as text: write the header it plans from, read the tempo and meter it planned, silence the Vocal voice.

YuE2 plans a score before it writes the song, and the score decides the song's tempo, key and which voices play. Planning
can start from a header the caller wrote (:func:`score_header`), which the model then continues, and a finished score can be
handed back to the kit to be used as written. Everything here is plain text work on that score.
"""

from __future__ import annotations

import re
from typing import Final

__all__ = ["planned_meter", "planned_tempo", "reads_as_score", "score_header", "silence_vocal"]

_VOICE_LINES: Final = 'V: Vocal clef=treble name="Vocal Melody" snm="Vocal"\nV: Ins clef=treble name="Ins Melody" snm="Inst."\n'
"""The two voice declarations exactly as YuE2 writes them."""

_TEMPO: Final = re.compile(r"^Q:\s*1/4\s*=\s*(\d+)\s*$", re.MULTILINE)
_METER: Final = re.compile(r"^M:\s*(\d+/\d+)\s*$", re.MULTILINE)
_UNIT: Final = re.compile(r"^L:\s*(\d+/\d+)\s*$", re.MULTILINE)
_VOICE: Final = re.compile(r"V:\s*(\S+)")
_FIELD: Final = re.compile(r"[A-Za-z]:")
_MULTI_REST: Final = re.compile(r"Z(\d*)")
_BAR_MARKS: Final = re.compile(r"[\]\[:\d]*")
"""What can sit between two bar lines without being a bar: repeat colons, endings and the closing bracket of ``|]``."""

_VOCAL: Final = "Vocal"

_DEFAULT_METER: Final = "4/4"
_DEFAULT_UNIT: Final = "1/32"
"""The meter and note length of a header when the model has planned none."""


def score_header(tempo: int, key: str | None = None, meter: str = _DEFAULT_METER, unit: str = _DEFAULT_UNIT) -> str:
    """Return the start of a YuE2 score with *tempo* (and *key*) already written, for the model to continue.

    Args:
        tempo: Beats per minute.
        key: An ABC key id such as ``Am`` or ``F#``, or ``None`` to leave the key to the model; the header then ends after
            the voice declarations, where YuE2 writes the key.
        meter: The time signature, such as ``3/4``; the model's own choice is kept by passing :func:`planned_meter`'s.
        unit: The default note length, such as ``1/32``.

    Returns:
        The header text, ending in a line break.
    """
    header = f"X:1\nT:\nM:{meter}\nL:{unit}\nQ:1/4={tempo}\n{_VOICE_LINES}"
    return header if key is None else f"{header}K:{key}\n"


def planned_tempo(abc: str) -> int | None:
    """Return the tempo in beats per minute that the score *abc* declares, or ``None`` if it declares none."""
    match = _TEMPO.search(abc)
    return int(match.group(1)) if match else None


def planned_meter(abc: str) -> tuple[str, str]:
    """Return the time signature and default note length of the score *abc*, such as ``("3/4", "1/32")``.

    The first ``M:`` line and the first ``L:`` line count. A score without one gives the kit's usual ``4/4`` or ``1/32``
    for it, which is what a header carried before the model's own meter was kept.
    """
    meter, unit = _METER.search(abc), _UNIT.search(abc)
    return (meter.group(1) if meter else _DEFAULT_METER, unit.group(1) if unit else _DEFAULT_UNIT)


def reads_as_score(abc: str) -> bool:
    """Return whether *abc* has a key line followed by at least one line of music."""
    lines = abc.split("\n")
    body = next((index for index, line in enumerate(lines) if line.startswith("K:")), None)
    return body is not None and any(_is_music(line) for line in lines[body + 1 :])


def silence_vocal(abc: str) -> str:
    """Return *abc* with every line of the Vocal voice replaced by rests of the same number of bars.

    Every other line stays byte-identical, and so does the line structure, so the song keeps its sections and length.

    Raises:
        ValueError: If the score has no key line or no music for the Vocal voice, so it cannot be read as a YuE2 score.
    """
    lines = abc.split("\n")
    body = next((index for index, line in enumerate(lines) if line.startswith("K:")), None)
    if body is None:
        raise ValueError("The score has no key line.")
    voice = ""
    silenced = False
    for index in range(body + 1, len(lines)):
        line = lines[index]
        if (declared := _VOICE.match(line)) is not None:
            voice = declared.group(1)
        elif voice == _VOCAL and _is_music(line) and (bars := _bar_count(line)):
            lines[index] = f"Z{bars}|" if bars > 1 else "Z|"
            silenced = True
    if not silenced:
        raise ValueError("The score has no music for the Vocal voice.")
    return "\n".join(lines)


def _is_music(line: str) -> bool:
    """Return whether *line* holds notes: not blank, not a comment and not a field line such as ``V:`` or ``K:``."""
    return bool(line.strip()) and not line.startswith("%") and _FIELD.match(line) is None


def _bar_count(line: str) -> int:
    """Count the bars on a line of music; a multi-bar rest such as ``Z4`` counts as its bars."""
    count = 0
    for segment in line.split("|"):
        text = segment.strip()
        if not text or _BAR_MARKS.fullmatch(text):
            continue
        rest = _MULTI_REST.fullmatch(text)
        count += int(rest.group(1) or 1) if rest else 1
    return count
