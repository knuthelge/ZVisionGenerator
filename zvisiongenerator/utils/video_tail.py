"""Read the last frames of a clip, streamed through ffmpeg."""

from __future__ import annotations

import subprocess
import tempfile
from collections import deque
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

from zvisiongenerator.utils.ffmpeg import tool_error
from zvisiongenerator.utils.video_extend import CARRY_FRAMES

if TYPE_CHECKING:
    import numpy as np

    from zvisiongenerator.core.video_types import ExtendSource

# Seconds before the first wanted frame where the read starts, so the decoder is warmed up from a keyframe.
TAIL_MARGIN_SECONDS = 2.0


def _tail_start_seconds(frame_count: int, fps: float, video_start: float = 0.0) -> float:
    """Return where to start reading so the last ``CARRY_FRAMES`` frames are in the stream; 0.0 means from the beginning.

    Args:
        frame_count: Frames in the clip.
        fps: Frames per second.
        video_start: Seconds from the start of the file (ffmpeg's input ``-ss`` timeline) to the first video frame.
    """
    return max(0.0, video_start + (frame_count - CARRY_FRAMES) / fps - TAIL_MARGIN_SECONDS)


def _build_tail_command(path: str | Path, start: float) -> list[str]:
    """Build the ffmpeg command that writes a clip's frames as raw RGB at the displayed size.

    ffmpeg's auto-rotation stays on, so a clip stored sideways and rotated by its display matrix comes out upright.

    Args:
        path: The video file.
        start: Seconds to seek to before decoding; 0.0 reads from the beginning.
    """
    seek = ["-ss", f"{start:.6f}"] if start > 0 else []
    return ["ffmpeg", "-v", "error", *seek, "-i", str(path), "-map", "0:v:0", "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]


def _read_tail_frames(
    path: str | Path,
    *,
    width: int,
    height: int,
    fps: float,
    frame_count: int,
    video_start: float = 0.0,
) -> np.ndarray:
    """Return the last ``CARRY_FRAMES`` frames of a clip as a uint8 array of shape ``(CARRY_FRAMES, height, width, 3)``.

    Frames are streamed through a ring buffer, so a long clip is never held in memory. When the seek lands too late
    and the read comes up short, the clip is read again from the beginning.

    Args:
        path: The video file.
        width: Displayed width (the size after rotation), as in the probe.
        height: Displayed height.
        fps: Frames per second.
        frame_count: Frames in the clip.
        video_start: Seconds from the start of the file to the first video frame.

    Raises:
        RuntimeError: If ffmpeg fails, a frame has the wrong byte count, or the clip has fewer than ``CARRY_FRAMES`` frames.
    """
    import numpy as np

    start = _tail_start_seconds(frame_count, fps, video_start)
    for seek in (start, 0.0) if start > 0 else (0.0,):
        ring = deque(_stream_frames(_build_tail_command(path, seek), width, height), maxlen=CARRY_FRAMES)
        if len(ring) >= CARRY_FRAMES:
            return np.stack([np.frombuffer(frame, dtype=np.uint8).reshape(height, width, 3) for frame in ring])
    raise RuntimeError(f"Could not read the last {CARRY_FRAMES} frames of {path}: found only {len(ring)}.")


def read_source_tail(source: ExtendSource) -> np.ndarray:
    """Return the last ``CARRY_FRAMES`` frames of an extend source at its displayed size (see ``_read_tail_frames``)."""
    return _read_tail_frames(source.path, width=source.width, height=source.height, fps=source.fps, frame_count=source.frame_count, video_start=source.video_start)


def _stream_frames(command: list[str], width: int, height: int) -> Iterator[bytes]:
    """Run an ffmpeg command writing raw RGB frames and yield each frame's bytes.

    Raises:
        RuntimeError: If ffmpeg exits with an error, or its output is not a whole number of ``width`` x ``height`` frames.
    """
    frame_bytes = width * height * 3
    with tempfile.TemporaryFile() as stderr:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=stderr)
        try:
            assert process.stdout is not None
            while chunk := process.stdout.read(frame_bytes):
                if len(chunk) < frame_bytes:
                    raise RuntimeError(f"ffmpeg wrote a partial frame ({len(chunk)} of {frame_bytes} bytes for {width}x{height}); the displayed size does not match the video.")
                yield chunk
            returncode = process.wait()
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            if process.stdout is not None:
                process.stdout.close()
        if returncode != 0:
            stderr.seek(0)
            raise tool_error("ffmpeg could not read the video's frames", stderr.read())
