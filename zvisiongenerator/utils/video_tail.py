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


def _tail_start_seconds(frame_count: int, fps: float, video_start: float = 0.0, frames: int = CARRY_FRAMES) -> float:
    """Return where to start reading so the last *frames* frames are in the stream; 0.0 means from the beginning.

    Args:
        frame_count: Frames in the clip.
        fps: Frames per second.
        video_start: Seconds from the start of the file (ffmpeg's input ``-ss`` timeline) to the first video frame.
        frames: How many frames at the end are wanted.
    """
    return max(0.0, video_start + (frame_count - frames) / fps - TAIL_MARGIN_SECONDS)


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
    frames: int = CARRY_FRAMES,
) -> np.ndarray:
    """Return the last *frames* frames of a clip as a uint8 array of shape ``(frames, height, width, 3)``.

    Frames are streamed through a ring buffer, so a long clip is never held in memory. When the seek lands too late
    and the read comes up short, the clip is read again from the beginning.

    Args:
        path: The video file.
        width: Displayed width (the size after rotation), as in the probe.
        height: Displayed height.
        fps: Frames per second.
        frame_count: Frames in the clip.
        video_start: Seconds from the start of the file to the first video frame.
        frames: How many frames at the end to return.

    Raises:
        RuntimeError: If ffmpeg fails, a frame has the wrong byte count, or the clip has fewer than *frames* frames.
    """
    import numpy as np

    start = _tail_start_seconds(frame_count, fps, video_start, frames)
    for seek in (start, 0.0) if start > 0 else (0.0,):
        ring = deque(_stream_frames(_build_tail_command(path, seek), width, height), maxlen=frames)
        if len(ring) >= frames:
            return np.stack([np.frombuffer(frame, dtype=np.uint8).reshape(height, width, 3) for frame in ring])
    raise RuntimeError(f"Could not read the last {frames} frames of {path}: found only {len(ring)}.")


def read_source_tail(source: ExtendSource, frames: int = CARRY_FRAMES) -> np.ndarray:
    """Return the last *frames* frames of an extend source at its displayed size (see ``_read_tail_frames``)."""
    return _read_tail_frames(source.path, width=source.width, height=source.height, fps=source.fps, frame_count=source.frame_count, video_start=source.video_start, frames=frames)


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


def build_tail_audio_command(source: ExtendSource, sample_rate: int, frames: int = CARRY_FRAMES) -> list[str]:
    """Build the ffmpeg command that writes the sound under a clip's last *frames* frames as stereo float32 PCM.

    The read starts at the tail's own position in the file and lasts as long as the tail, resampled to *sample_rate*.
    """
    start = source.video_start + (source.frame_count - frames) / source.fps
    duration = frames / source.fps
    return ["ffmpeg", "-v", "error", "-ss", f"{start:.6f}", "-i", str(source.path), "-t", f"{duration:.6f}", "-map", "0:a:0", "-f", "f32le", "-ac", "2", "-ar", str(sample_rate), "-"]


def read_source_tail_audio(source: ExtendSource, sample_rate: int, frames: int = CARRY_FRAMES) -> np.ndarray:
    """Return the sound under a clip's last *frames* frames as float32 of shape ``(2, samples)`` at *sample_rate*.

    Raises:
        RuntimeError: If ffmpeg fails or the clip has no sound at its end.
    """
    import numpy as np

    result = subprocess.run(build_tail_audio_command(source, sample_rate, frames), capture_output=True, check=False)
    if result.returncode != 0:
        raise tool_error("ffmpeg could not read the video's audio", result.stderr)
    samples = np.frombuffer(result.stdout[: len(result.stdout) // 8 * 8], dtype=np.float32).reshape(-1, 2).T.copy()
    if not samples.shape[1]:
        raise RuntimeError(f"Could not read the audio of {source.path} to continue it.")
    return samples
