"""Read a video file's stream properties with ffprobe."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

from zvisiongenerator.utils.ffmpeg import tool_error

# The message of the ``ValueError`` for a file without a video stream.
NO_VIDEO_STREAM = "no video stream"

# ffprobe prints these for an unknown value.
_UNKNOWN = frozenset({"", "N/A", "unknown"})


@dataclass(frozen=True)
class VideoProbe:
    """What ffprobe reports about a clip's first video stream and, when present, its first audio stream."""

    # Stored size, and the size a player shows (swapped when the clip is rotated a quarter turn).
    coded_width: int
    coded_height: int
    width: int
    height: int
    codec: str | None
    profile: str | None
    pix_fmt: str | None
    bits_per_sample: int | None
    sample_aspect_ratio: str | None
    # Frame rates as ffprobe writes them (``30000/1001``); ``0/0`` when unknown.
    r_frame_rate: str
    avg_frame_rate: str
    time_base: str
    field_order: str | None
    color_space: str | None
    color_primaries: str | None
    color_trc: str | None
    # Quarter turns the player applies: 0, 90, 180 or 270 (other values mean an unsupported angle).
    rotation: int
    # Packets counted by ffprobe (``-count_packets``), else the container's frame count.
    frame_count: int | None
    # Seconds; ``video_start`` is the video stream's start time, ``format_start`` is the container's.
    duration: float | None
    video_start: float | None
    format_start: float | None
    has_audio: bool
    # The first audio stream's codec name (``aac``, ``opus``), when the clip has sound.
    audio_codec: str | None = None

    @property
    def fps(self) -> float:
        """Return the frame rate (``r_frame_rate``) in frames per second, or 0.0 when unknown."""
        rate = parse_fraction(self.r_frame_rate)
        return float(rate) if rate else 0.0

    @property
    def square_pixels(self) -> bool:
        """Return whether the pixels are square: no sample aspect ratio, ``1:1`` or ``0:1`` (unset), or one that is unreadable."""
        if self.sample_aspect_ratio is None:
            return True
        ratio = parse_fraction(self.sample_aspect_ratio.replace(":", "/"))
        return ratio is None or ratio in (Fraction(0), Fraction(1))


def parse_fraction(text: str | None) -> Fraction | None:
    """Parse an ffprobe fraction such as ``30000/1001`` or ``25``; return ``None`` when it is missing or has a zero denominator."""
    if text is None or text in _UNKNOWN:
        return None
    try:
        return Fraction(text)
    except ValueError, ZeroDivisionError:
        return None


def frame_rates_match(r_frame_rate: str, avg_frame_rate: str | None, tolerance: float = 0.001) -> bool:
    """Return whether a clip's real and average frame rates agree, which a constant-rate clip's do.

    Args:
        r_frame_rate: The stream's base frame rate; an unreadable one fails the check.
        avg_frame_rate: The frame rate averaged over the clip; a missing, ``0/0`` or zero one cannot disagree.
        tolerance: Relative difference allowed (0.001 is 0.1%), so ``30000/1001`` and ``1440000/48047`` agree.
    """
    rate = parse_fraction(r_frame_rate)
    if rate is None or rate <= 0:
        return False
    average = parse_fraction(avg_frame_rate)
    if average is None or average <= 0:
        return True
    return abs(rate - average) <= rate * Fraction(str(tolerance))


def parse_probe(data: dict[str, Any]) -> VideoProbe:
    """Build a ``VideoProbe`` from ffprobe's ``-show_streams -show_format`` JSON.

    Cover art (an ``attached_pic`` stream) is not a video stream.

    Raises:
        ValueError: If the file has no video stream, or its video stream has no size.
    """
    streams = [stream for stream in data.get("streams") or [] if isinstance(stream, dict)]
    video = next((s for s in streams if s.get("codec_type") == "video" and not (s.get("disposition") or {}).get("attached_pic")), None)
    if video is None:
        raise ValueError(NO_VIDEO_STREAM)
    coded_width, coded_height = _int(video.get("width")), _int(video.get("height"))
    if not coded_width or not coded_height:
        raise ValueError("video stream has no size")
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    fmt = data.get("format") or {}
    rotation = _rotation(video)
    sideways = rotation % 180 == 90
    return VideoProbe(
        coded_width=coded_width,
        coded_height=coded_height,
        width=coded_height if sideways else coded_width,
        height=coded_width if sideways else coded_height,
        codec=_text(video.get("codec_name")),
        profile=_text(video.get("profile")),
        pix_fmt=_text(video.get("pix_fmt")),
        bits_per_sample=_int(video.get("bits_per_raw_sample")),
        sample_aspect_ratio=_text(video.get("sample_aspect_ratio")),
        r_frame_rate=str(video.get("r_frame_rate") or "0/0"),
        avg_frame_rate=str(video.get("avg_frame_rate") or "0/0"),
        time_base=str(video.get("time_base") or "0/0"),
        field_order=_text(video.get("field_order")),
        color_space=_text(video.get("color_space")),
        color_primaries=_text(video.get("color_primaries")),
        color_trc=_text(video.get("color_transfer")),
        rotation=rotation,
        frame_count=_int(video.get("nb_read_packets")) or _int(video.get("nb_frames")),
        duration=_float(fmt.get("duration")),
        video_start=_float(video.get("start_time")),
        format_start=_float(fmt.get("start_time")),
        has_audio=audio is not None,
        audio_codec=_text(audio.get("codec_name")) if audio is not None else None,
    )


def ffprobe_json(path: str | Path, *, count_packets: bool = True) -> dict[str, Any]:
    """Return ffprobe's JSON for a file's streams and container (``-show_streams -show_format``).

    Args:
        path: The media file.
        count_packets: Whether ffprobe also counts the video packets, which reads the whole file (the container's frame
            count can be missing or wrong); off for a quick look at the metadata.

    Raises:
        RuntimeError: If ffprobe cannot read the file or prints something that is not JSON.
    """
    command = ["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", "-show_format", *(["-count_packets"] if count_packets else []), str(path)]
    try:
        result = subprocess.run(command, stdin=subprocess.DEVNULL, capture_output=True, text=True, check=True)
        return json.loads(result.stdout)
    except subprocess.CalledProcessError as exc:
        raise tool_error(f"ffprobe could not read {path}", exc.stderr, limit=500) from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"ffprobe returned unreadable output for {path}") from exc


def probe_video(path: str | Path) -> VideoProbe:
    """Probe a video file, counting its packets (the container's frame count can be missing or wrong).

    Raises:
        RuntimeError: If ffprobe cannot read the file.
        ValueError: If the file has no video stream.
    """
    return parse_probe(ffprobe_json(path))


def audio_decodes(path: str | Path, *, start: float, duration: float) -> bool:
    """Return whether ffmpeg decodes a stretch of the first audio stream into samples.

    Readability follows the exit code and the decoded output; warnings on stderr (common in phone AAC) do not fail it.

    Args:
        path: The video file.
        start: Seconds from the start of the file where the stretch begins.
        duration: Length of the stretch in seconds.
    """
    command = ["ffmpeg", "-v", "error", "-ss", f"{max(0.0, start):.6f}", "-t", f"{max(0.0, duration):.6f}", "-i", str(path), "-map", "0:a:0", "-vn", "-ac", "1", "-ar", "8000", "-f", "s16le", "-"]
    result = subprocess.run(command, stdin=subprocess.DEVNULL, capture_output=True, check=False)
    return result.returncode == 0 and bool(result.stdout)


def _rotation(stream: dict[str, Any]) -> int:
    """Return the quarter turns a player applies, from the display matrix (counter-clockwise, so negated) or the legacy ``rotate`` tag (clockwise)."""
    for side_data in stream.get("side_data_list") or []:
        angle = _float(side_data.get("rotation")) if isinstance(side_data, dict) else None
        if angle is not None:
            return round(-angle) % 360
    angle = _float((stream.get("tags") or {}).get("rotate"))
    return round(angle) % 360 if angle is not None else 0


def _text(value: Any) -> str | None:
    """Return a stripped string, or ``None`` for a missing or unknown value."""
    if value is None:
        return None
    text = str(value).strip()
    return None if text in _UNKNOWN else text


def _int(value: Any) -> int | None:
    try:
        return int(value)
    except TypeError, ValueError:
        return None


def _float(value: Any) -> float | None:
    try:
        return float(value)
    except TypeError, ValueError:
        return None
