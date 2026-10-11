"""Shared helpers for the tests that make and read real clips with the installed ffmpeg."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

HAS_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
REAL = pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg and ffprobe are needed")


def ffmpeg(*args: str, data: bytes | None = None) -> bytes:
    """Run ffmpeg quietly, feeding ``data`` to its stdin, and return its stdout."""
    return subprocess.run(["ffmpeg", "-v", "error", "-y", *args], input=data, capture_output=True, check=True).stdout


def numbered_frames(count: int, width: int = 64, height: int = 32, offset: int = 0) -> np.ndarray:
    """Return flat-colour frames whose colour identifies the frame, with a red patch in the top-left corner."""
    frames = np.zeros((count, height, width, 3), dtype=np.uint8)
    for index in range(count):
        frames[index, :, :] = ((index + offset) % 256, ((index + offset) * 3) % 256, ((index + offset) * 7) % 256)
    frames[:, : height // 4, : width // 4] = (255, 0, 0)
    return frames


def encode_frames(frames: np.ndarray, path: Path, *args: str, quality: tuple[str, str] = ("-qp", "0"), pix_fmt: str = "yuv420p") -> None:
    """Encode raw frames into ``path`` with libx264 at 24 fps."""
    _, height, width, _ = frames.shape
    raw = ["-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", "24", "-i", "-"]
    ffmpeg(*raw, "-c:v", "libx264", *quality, "-pix_fmt", pix_fmt, *args, str(path), data=frames.tobytes())


def decode_frames(path: Path, width: int, height: int) -> np.ndarray:
    """Decode every frame with ffmpeg's own defaults (auto-rotation on)."""
    data = ffmpeg("-i", str(path), "-map", "0:v:0", "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "rgb24", "-")
    return np.frombuffer(data, dtype=np.uint8).reshape(-1, height, width, 3)


def first_frame(path: Path, width: int, height: int) -> np.ndarray:
    """Decode the first frame, which must have the displayed size."""
    data = ffmpeg("-i", str(path), "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-")
    return np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3)


def rotated_copy(source: Path, target: Path, angle: str) -> Path:
    """Copy ``source`` to ``target`` with a display matrix that turns it by ``angle`` degrees."""
    ffmpeg("-display_rotation", angle, "-i", str(source), "-c", "copy", str(target))
    return target
