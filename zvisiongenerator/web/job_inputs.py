"""Record what a Web UI job starts from, as names the browser can show.

A job's context keeps one record per input (``job_input``): its role, the name to show and whether it is an image or a
video. A record never holds a path, so snapshots can carry it as it is.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Final, Literal

from zvisiongenerator.web.upload_folders import UPLOADS_FOLDER

__all__ = ["job_input"]

InputKind = Literal["image", "video"]

#: What an uploaded image is called when the only name left is the hash it was stored under.
UPLOADED_IMAGE_NAME: Final = "Uploaded image"

_STORED_UPLOAD_STEM: Final = re.compile(r"[0-9a-f]{64}")


def _is_stored_upload(path: Path) -> bool:
    """Whether *path* is an image saved by an upload: ``<sha256>.<ext>`` directly under the uploads folder."""
    return path.parent.name == UPLOADS_FOLDER and _STORED_UPLOAD_STEM.fullmatch(path.stem) is not None


def job_input(role: str, path: str | Path, kind: InputKind, *, name: str | None = None) -> dict[str, str]:
    """Return the record of one input for a job's context.

    Args:
        role: What the job uses it as: ``source``, ``start``, ``end`` or ``reference``.
        path: The file on the host; only its name is kept.
        kind: ``image`` or ``video``.
        name: The name to show; the file's own name when omitted. An upload keeps the name its sender gave it here. An
            image stored under its hash, with no sender's name to show (Reuse and Load settings send the stored path), is
            shown as an uploaded image rather than as the hash.
    """
    file = Path(path)
    shown = Path(name).name if name else ""
    if not shown:
        shown = UPLOADED_IMAGE_NAME if _is_stored_upload(file) else file.name
    return {"role": role, "name": shown, "kind": kind}
