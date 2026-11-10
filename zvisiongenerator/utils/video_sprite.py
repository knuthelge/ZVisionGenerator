"""Build a contact sheet of a clip: one JPEG holding a small picture of evenly spaced frames, for the player's hover picture and the Extend bar."""

from __future__ import annotations

from typing import TYPE_CHECKING

from zvisiongenerator.utils.ffmpeg import capture_ffmpeg
from zvisiongenerator.utils.video_frame import frame_color_filter, probe_for_frames
from zvisiongenerator.utils.video_point import frame_total

if TYPE_CHECKING:
    from pathlib import Path

    from zvisiongenerator.utils.video_probe import VideoProbe

__all__ = ["SPRITE_COLUMNS", "SPRITE_MAX_TILES", "SPRITE_TILE_EDGE", "SPRITE_TIMEOUT", "build_sprite", "sprite_command", "sprite_frames"]

SPRITE_COLUMNS = 11
"""Tiles across, and rows down: the sheet is an 11 x 11 grid, whatever the number of frames."""

SPRITE_MAX_TILES = SPRITE_COLUMNS * SPRITE_COLUMNS
"""The most frames a sheet holds."""

SPRITE_TILE_EDGE = 160
"""Pixels on the long side of one tile."""

SPRITE_TIMEOUT = 300.0
"""Seconds ffmpeg may take to decode a clip into its sheet."""

_JPEG_QUALITY = "5"


def sprite_frames(frames: int) -> list[int]:
    """Return the frame each tile shows, left to right and top to bottom.

    A clip of 121 frames or fewer has one tile per frame; a longer one has 121 tiles, tile ``i`` showing frame
    ``round(i / 120 * last)``. A one-frame clip is one tile.

    Raises:
        ValueError: If *frames* is less than 1.
    """
    if frames < 1:
        raise ValueError("A contact sheet needs at least one frame.")
    tiles = min(frames, SPRITE_MAX_TILES)
    if tiles == 1:
        return [0]
    last = frames - 1
    return [round(index / (tiles - 1) * last) for index in range(tiles)]


def _tile_scale(probe: VideoProbe) -> str:
    """Return the scale filter that fits a frame's long side to a tile, keeping its shape and an even short side."""
    return f"scale={SPRITE_TILE_EDGE}:-2" if probe.width >= probe.height else f"scale=-2:{SPRITE_TILE_EDGE}"


def sprite_command(path: Path, probe: VideoProbe, picks: list[int]) -> list[str]:
    """Return the ffmpeg command that prints the contact sheet of the frames *picks* (from ``sprite_frames``) as a JPEG on stdout."""
    select = "+".join(f"eq(n,{frame})" for frame in picks)
    filters = [f"select='{select}'", *([color] if (color := frame_color_filter(probe)) else []), _tile_scale(probe), f"tile={SPRITE_COLUMNS}x{SPRITE_COLUMNS}"]
    return ["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:v:0", "-an", "-vf", ",".join(filters), "-frames:v", "1", "-f", "image2pipe", "-c:v", "mjpeg", "-q:v", _JPEG_QUALITY, "-"]


def build_sprite(path: Path) -> bytes:
    """Return the JPEG contact sheet of a clip.

    The frame count is the one the player's facts report, so the server and the player agree on the number of tiles.

    Raises:
        ValueError: If the clip has no video stream or its length is unknown.
        RuntimeError: If ffprobe or ffmpeg fails, runs too long or prints no picture.
    """
    probe = probe_for_frames(path)
    frames = frame_total(probe)
    if not frames:
        raise ValueError("The video's length could not be read.")
    sheet = capture_ffmpeg(sprite_command(path, probe, sprite_frames(frames)), timeout=SPRITE_TIMEOUT, failure="ffmpeg could not build the contact sheet")
    if not sheet:
        raise RuntimeError("ffmpeg returned no contact sheet")
    return sheet
