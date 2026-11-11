"""Read the painted mask of an Edit form.

The Edit form's fields are ``edit_source_path`` or ``edit_source_file`` (the picture), ``edit_mask_file`` (a PNG of the
painted area, optional), ``edit_extend_side`` and ``edit_extend_amount`` (optional). The browser sends the mask at up to
4 MP; it is scaled to the picture's upright size here, once, and travels with the job as a bool array that is never
recorded in the job's settings or in the saved image.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from zvisiongenerator.utils.edit_request import decode_paint
from zvisiongenerator.web.planned_upload import PlannedUpload

__all__ = [
    "EDIT_MASK_FIELD",
    "EDIT_SOURCE_FIELD",
    "EDIT_WORKFLOW",
    "read_job_paint",
]

EDIT_WORKFLOW = "edit"
EDIT_SOURCE_FIELD = "edit_source_path"
EDIT_MASK_FIELD = "edit_mask_file"


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
