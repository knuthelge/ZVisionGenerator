"""Stand-ins for the uploads of a generate form, so its download plan is made without reading or saving them.

The Workspace asks ``POST /api/generate/plan`` before it submits, and sends the fields only: each file part keeps its
field name and file name but has no body. The plan route swaps every uploaded file for a :class:`PlannedUpload`, and
the form's parser then skips what only matters for the file's content (saving it, decoding an image, probing a
recording). The submit that follows sends the real files and runs those checks.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from starlette.datastructures import FormData

from zvisiongenerator.web.upload_folders import UPLOADS_FOLDER

__all__ = ["PlannedUpload", "planned_form", "planned_upload_path"]

_PLANNED_DIR = Path(UPLOADS_FOLDER) / "planned"


@dataclass(frozen=True)
class PlannedUpload:
    """An uploaded file that is not read: its name stands for it.

    Attributes:
        filename: The name the browser gave the file.
    """

    filename: str


def planned_upload_path(upload: PlannedUpload, output_dir: str) -> str:
    """Return a path for *upload* under the output folder's planning folder; nothing is written there."""
    return str(Path(output_dir) / _PLANNED_DIR / Path(upload.filename).name)


def planned_form(form: Any) -> FormData:
    """Return a copy of *form* with every uploaded file replaced by a :class:`PlannedUpload`; text values are kept as they are.

    A file part without a name (a file input nothing was chosen for) is left alone: it was never an upload.
    """
    items = form.multi_items() if hasattr(form, "multi_items") else form.items()
    return FormData([(key, _stand_in(value)) for key, value in items])


def _stand_in(value: Any) -> Any:
    """Return a :class:`PlannedUpload` for an uploaded file with a name, else *value*."""
    filename = getattr(value, "filename", None)
    if filename and getattr(value, "file", None) is not None:
        return PlannedUpload(str(filename))
    return value
