"""Read one frame of a video as a PNG, snapped to the frame a time lies in, and save it under an upload folder."""

from __future__ import annotations

import hashlib
import re
import tempfile
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

from zvisiongenerator.utils.atomic_write import write_bytes_atomic
from zvisiongenerator.utils.ffmpeg import capture_ffmpeg
from zvisiongenerator.utils.video_point import frame_at, frame_seek_seconds, frame_total
from zvisiongenerator.utils.video_probe import VideoProbe, ffprobe_json, parse_probe

__all__ = [
    "FRAME_TIMEOUT",
    "FrameTarget",
    "extract_frame_png",
    "extract_last_frame_png",
    "frame_color_filter",
    "frame_file_name",
    "frame_png_command",
    "last_frame_png_command",
    "locate_frame",
    "locate_last_frame",
    "probe_for_frames",
    "save_video_frame",
]

FRAME_TIMEOUT = 60.0
"""Seconds ffprobe or ffmpeg may take to read one frame."""

# What a player does with a clip that names no colour matrix: it treats it as BT.709, while ffmpeg converts it as BT.601.
_UNTAGGED = frozenset({None, "", "unknown", "unspecified"})
_PLAYER_MATRIX_FILTER = "scale=in_color_matrix=bt709,format=rgb24"
_UNSAFE_NAME_CHARACTERS = re.compile(r"[^\w ()\-.]")
_NAME_LIMIT = 100


@dataclass(frozen=True)
class FrameTarget:
    """Describe the frame a time was snapped to: its index and the time to seek to."""

    frame: int
    seek: float


def probe_for_frames(path: Path, *, timeout: float = FRAME_TIMEOUT) -> VideoProbe:
    """Return the probe of a clip without counting its packets, which is what the player's facts are read from.

    Raises:
        RuntimeError: If ffprobe cannot read the file or runs past *timeout*.
        ValueError: If the file has no video stream.
    """
    return parse_probe(ffprobe_json(path, count_packets=False, timeout=timeout))


def locate_frame(probe: VideoProbe, seconds: float) -> FrameTarget:
    """Return the frame of a clip that *seconds* lies in, by the rule the player uses.

    Raises:
        ValueError: If the clip's frame rate or length is unknown.
    """
    frames = _frames_of(probe)
    start = probe.stream_offset
    frame = frame_at(seconds, start=start, fps=probe.fps, frames=frames)
    return FrameTarget(frame=frame, seek=frame_seek_seconds(frame, start=start, fps=probe.fps))


def locate_last_frame(probe: VideoProbe) -> FrameTarget:
    """Return the last frame of a clip.

    Raises:
        ValueError: If the clip's frame rate or length is unknown.
    """
    frame = _frames_of(probe) - 1
    return FrameTarget(frame=frame, seek=frame_seek_seconds(frame, start=probe.stream_offset, fps=probe.fps))


def _frames_of(probe: VideoProbe) -> int:
    frames = frame_total(probe)
    if probe.fps <= 0 or not frames:
        raise ValueError("The video's frame rate or length could not be read.")
    return frames


def frame_color_filter(probe: VideoProbe) -> str | None:
    """Return the filter that makes the frame's colours match the player's, or ``None`` when ffmpeg already does.

    A clip that names no colour matrix (the clips of the CUDA backends) is shown as BT.709 by a player, while ffmpeg converts
    it as BT.601; the filter converts it as BT.709 and leaves RGB behind. A tagged clip is converted by its own tag.
    """
    return _PLAYER_MATRIX_FILTER if probe.color_space in _UNTAGGED else None


def frame_png_command(path: Path, seek: float, color: str | None, *, edge: int | None = None) -> list[str]:
    """Return the ffmpeg command that prints the first frame at or after *seek* as a PNG on stdout.

    Args:
        path: The video.
        seek: The time to start decoding at, in seconds.
        color: The colour filter of ``frame_color_filter``, if any.
        edge: Shrinks the frame to fit this many pixels on each side; ``None`` keeps its full size.
    """
    command = ["ffmpeg", "-v", "error", "-ss", f"{seek:.6f}", "-i", str(path), "-map", "0:v:0", *_filter_arguments(color, edge)]
    return [*command, "-frames:v", "1", "-f", "image2pipe", "-c:v", "png", "-"]


def last_frame_png_command(path: Path, out: Path, color: str | None, *, edge: int | None = None) -> list[str]:
    """Return the ffmpeg command that writes the last frame of a video as a PNG to *out*, found by a time seek from the end.

    For a clip whose frame rate or length is unknown. ``-update 1`` rewrites *out* with every frame of the last second, so the file is what the last one left; that needs a file, not a pipe.

    Args:
        path: The video.
        out: The PNG to write.
        color: The colour filter of ``frame_color_filter``, if any.
        edge: Shrinks the frame to fit this many pixels on each side; ``None`` keeps its full size.
    """
    command = ["ffmpeg", "-v", "error", "-y", "-sseof", "-1", "-i", str(path), "-map", "0:v:0", *_filter_arguments(color, edge)]
    return [*command, "-update", "1", "-f", "image2", "-c:v", "png", str(out)]


def _filter_arguments(color: str | None, edge: int | None) -> list[str]:
    """Return the ``-vf`` arguments for the colour filter and the shrink to *edge*, or none when neither applies."""
    filters = [color] if color else []
    if edge is not None:
        filters.append(f"scale={edge}:{edge}:force_original_aspect_ratio=decrease")
    return ["-vf", ",".join(filters)] if filters else []


def extract_frame_png(path: Path, probe: VideoProbe, seek: float, *, edge: int | None = None, timeout: float = FRAME_TIMEOUT) -> bytes:
    """Return the PNG of the frame at *seek* seconds, the time a ``FrameTarget`` carries.

    Raises:
        RuntimeError: If ffmpeg fails, runs past *timeout* or prints no frame.
    """
    png = capture_ffmpeg(frame_png_command(path, seek, frame_color_filter(probe), edge=edge), timeout=timeout, failure="ffmpeg could not read the frame")
    if not png:
        raise RuntimeError("ffmpeg returned no frame")
    return png


def extract_last_frame_png(path: Path, probe: VideoProbe, *, edge: int | None = None, timeout: float = FRAME_TIMEOUT) -> bytes:
    """Return the PNG of the last frame of a clip by a time seek from its end, for a clip whose frame rate or length is unknown.

    A clip whose container holds no duration makes ffmpeg ignore ``-sseof``: it decodes the whole clip, and a long one runs past
    *timeout*, so the caller shows its fallback.

    Raises:
        RuntimeError: If ffmpeg fails, runs past *timeout* or writes no frame.
    """
    with tempfile.TemporaryDirectory() as scratch:
        out = Path(scratch) / "last.png"
        capture_ffmpeg(last_frame_png_command(path, out, frame_color_filter(probe), edge=edge), timeout=timeout, failure="ffmpeg could not read the frame")
        if not out.is_file() or out.stat().st_size == 0:
            raise RuntimeError("ffmpeg returned no frame")
        return out.read_bytes()


def _tenths(seconds: float) -> str:
    """Return *seconds* to one decimal, a tie rounding up: ``1.25`` is ``1.3``, as ``toFixed(1)`` in the Web UI's ``pointLabel`` writes it."""
    return str(Decimal(seconds).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def frame_file_name(clip_name: str, seconds: float) -> str:
    """Return the file name of a saved frame: the clip's name with unsafe characters replaced, then the time to one decimal (``fox_3.2s.png``).

    The time reads as the Web UI labels it, so a frame at 1.25 s is ``_1.3s``, not the ``_1.2s`` that Python's own rounding gives.
    """
    stem = _UNSAFE_NAME_CHARACTERS.sub("_", Path(clip_name).name).strip(" .")[:_NAME_LIMIT]
    return f"{stem or 'frame'}_{_tenths(seconds)}s.png"


def save_video_frame(path: Path, seconds: float, dest_dir: Path) -> Path:
    """Save the frame of a clip that *seconds* lies in as a full-size PNG at ``<dest_dir>/<sha256 of the PNG>/<clip>_<t>s.png``.

    A file already there (the same picture under the same name) is left as it is. The name is the clip's stem and the frame's
    own time (``frame / fps``) to one decimal, which is what the Web UI's time chip labels the frame.

    Returns:
        The path of the saved PNG.

    Args:
        path: The video.
        seconds: A time on the clip's clock, snapped to a frame.
        dest_dir: The folder the hash folders are made in.

    Raises:
        ValueError: If the clip has no video stream or its frame rate or length is unknown.
        RuntimeError: If ffprobe or ffmpeg fails, runs too long or finds no frame.
    """
    probe = probe_for_frames(path)
    target = locate_frame(probe, seconds)
    png = extract_frame_png(path, probe, target.seek)
    name = frame_file_name(path.stem, target.frame / probe.fps)
    saved = dest_dir / hashlib.sha256(png).hexdigest() / name
    if not saved.is_file():
        saved.parent.mkdir(parents=True, exist_ok=True)
        write_bytes_atomic(saved, png)
    return saved
