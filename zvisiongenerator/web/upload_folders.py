"""Name the folder under the output folder that holds the files the Web UI saves for a job."""

from __future__ import annotations

from typing import Final

__all__ = ["UPLOADS_FOLDER"]

UPLOADS_FOLDER: Final = ".web_uploads"
"""The folder under the output folder that holds uploads and extracted frames; the Gallery does not list it."""
