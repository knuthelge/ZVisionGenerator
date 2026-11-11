"""Read the picture, the Extend request and the painted mask of an Edit form.

The Edit form's fields are ``edit_source_path`` or ``edit_source_file`` (the picture), ``edit_mask_file`` (a PNG of the
painted area, optional), ``edit_extend_side`` and ``edit_extend_amount`` (optional). The browser sends the mask at up to
4 MP; it is scaled to the picture's upright size here, once, and travels with the job as a bool array that is never
recorded in the job's settings or in the saved image.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from zvisiongenerator.utils.edit_request import decode_paint, parse_extend, source_is_cutout
from zvisiongenerator.utils.input_images import upright_size
from zvisiongenerator.web.planned_upload import PlannedUpload

__all__ = [
    "EDIT_MASK_FIELD",
    "EDIT_SOURCE_FIELD",
    "EDIT_WORKFLOW",
    "EditSource",
    "extend_from_fields",
    "read_edit_source",
    "read_job_paint",
]

EDIT_WORKFLOW = "edit"
EDIT_SOURCE_FIELD = "edit_source_path"
EDIT_MASK_FIELD = "edit_mask_file"


@dataclass(frozen=True)
class EditSource:
    """The picture an edit starts from.

    Attributes:
        path: The file on the Web UI host.
        size: The upright ``(width, height)``, or ``None`` when the file was not read (a download plan).
        cutout: Whether the picture has a transparent background; ``False`` when the file was not read.
    """

    path: str
    size: tuple[int, int] | None
    cutout: bool


def extend_from_fields(side: str | None, amount: str | None) -> tuple[str, int] | None:
    """Return the Extend request of the form's two fields as ``(side, amount)``, or ``None`` when neither is set.

    Raises:
        ValueError: One field is set without the other, or the side or amount is not allowed (see ``parse_extend``).
    """
    if side is None and amount is None:
        return None
    return parse_extend((side or "", amount if amount is not None else ""))


def read_edit_source(path: str, *, read: bool) -> EditSource:
    """Return the picture at *path* with its upright size and whether it is a cut-out; with ``read=False`` the file is left unread.

    Raises:
        ValueError: The file is not a readable image.
    """
    if not read:
        return EditSource(path, None, False)
    try:
        size = upright_size(path)
        cutout = source_is_cutout(path)
    except OSError as error:
        raise ValueError(f"The picture to edit is not a readable image: {error}") from error
    return EditSource(path, size, cutout)


def read_job_paint(upload: Any, size: tuple[int, int] | None) -> np.ndarray | None:
    """Return the painted area of the uploaded PNG as a bool array at *size* (the picture's upright size).

    The upload may be at another size of the picture's shape (the browser sends up to 4 MP). A download plan reads
    nothing: for its stand-in upload, or when the picture was not read, ``None`` is returned.

    Raises:
        ValueError: The upload is not a readable PNG, its shape differs from the picture's, or nothing is painted.
    """
    if isinstance(upload, PlannedUpload) or size is None:
        return None
    upload.file.seek(0)
    return decode_paint(upload.file.read(), *size)
