"""Build the answer of the route that saves one frame of a video as a PNG under the output folder's uploads."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from zvisiongenerator.utils.ffmpeg import require_ffmpeg
from zvisiongenerator.utils.video_frame import save_video_frame
from zvisiongenerator.web.extend_api import source_video_path
from zvisiongenerator.web.json_fields import json_field
from zvisiongenerator.web.upload_folders import UPLOADS_FOLDER

__all__ = ["frame_response"]


def frame_response(payload: Any, output_dir: Path) -> dict[str, Any]:
    """Save the frame of a host-local video that a time lies in, and name the file.

    The PNG is full size and upright, named after the clip and the frame's time, and kept under a folder named by the hash of
    its content, so the same picture is saved once.

    Args:
        payload: The request body: ``path`` of the clip and ``time`` in seconds.
        output_dir: The output folder the PNG's folder is made in.

    Returns:
        ``path`` of the PNG.

    Raises:
        ValueError: If the body, path or time is invalid, ffmpeg is not installed, or the clip's frame rate or length can't be read.
        FileNotFoundError: If no file is at the path.
        RuntimeError: If ffprobe or ffmpeg fails to read the frame; the message says why.
    """
    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object.")
    path = source_video_path(payload.get("path"), key="path")
    seconds = json_field(payload, "time", (int, float), "a number")
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        raise ValueError("time must be a number of seconds, zero or more.")
    try:
        require_ffmpeg()
    except RuntimeError as exc:
        raise ValueError(str(exc)) from exc
    return {"path": str(save_video_frame(path, float(seconds), output_dir / UPLOADS_FOLDER))}
