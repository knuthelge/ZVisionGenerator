"""Name generated output files: a set-name prefix plus a timestamp, kept unique with a counter."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re


def generate_filename(set_name: str | None = None, *, now: datetime | None = None) -> str:
    """Return an output file stem: ``{set_name}_{YYYY-MM-DD_HH-MM-SS}``, or the timestamp alone.

    Generation settings are embedded in the file, so the name only says what it is and when it was made.
    """
    timestamp = (now or datetime.now()).strftime("%Y-%m-%d_%H-%M-%S")
    safe_name = _safe_set_name(set_name)
    return f"{safe_name}_{timestamp}" if safe_name else timestamp


def derived_stem(source: str | Path, tag: str) -> str:
    """Return the output stem for a ``tag`` derivative of the picture at *source*: its name plus ``_{tag}``, without stacking a second one.

    A source that already carries the tag, with the counter :func:`unique_output_path` may have added
    (``photo_adjusted``, ``photo_adjusted_2``), gives ``photo_adjusted`` again, so the counter is added afresh:
    ``photo_adjusted_2``, ``photo_adjusted_3``.
    """
    return f"{re.sub(rf'_{re.escape(tag)}(?:_\d+)?$', '', Path(source).stem)}_{tag}"


def edit_output_stem(source: str | Path) -> str:
    """Return the output stem for an edit of the picture at *source*: its name plus ``_edit``, without stacking a second one."""
    return derived_stem(source, "edit")


def unique_output_path(directory: str | Path, stem: str, suffix: str) -> Path:
    """Return ``directory/stem+suffix``, or the first free ``stem_N+suffix`` (N from 2) when it exists."""
    folder = Path(directory)
    candidate = folder / f"{stem}{suffix}"
    counter = 2
    while candidate.exists():
        candidate = folder / f"{stem}_{counter}{suffix}"
        counter += 1
    return candidate


def _safe_set_name(set_name: str | None) -> str | None:
    if not set_name:
        return None
    safe_name = re.sub(r'[/\\:*?"<>|]', "_", set_name)
    safe_name = safe_name.replace("..", "_")
    safe_name = safe_name.strip(". ")
    return safe_name or None
