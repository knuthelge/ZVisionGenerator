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


# A stem that is already an edit's: ``_edit`` with the counter ``unique_output_path`` may have added.
_EDITED_STEM = re.compile(r"_edit(?:_\d+)?$")


def edit_output_stem(source: str | Path) -> str:
    """Return the output stem for an edit of the picture at *source*: its name plus ``_edit``, without stacking a second one.

    A source that is already an edit (``photo_edit``, ``photo_edit_2``) gives ``photo_edit`` again, so
    :func:`unique_output_path` adds the counter: ``photo_edit_2``, ``photo_edit_3``.
    """
    return f"{_EDITED_STEM.sub('', Path(source).stem)}_edit"


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
