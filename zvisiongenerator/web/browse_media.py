"""Serve previews, facts, contact sheets and playback of the media files the Web UI's file browser lists.

``media_file`` is the one check every browse route goes through; ``confined_media_file`` wraps a file the caller has already confined. ``serve_preview`` and ``serve_sprite``
answer from a disk cache under ``<data dir>/cache/browse``. Previews decode at most two files at a time and contact sheets one at a time, on a slot of their own so a long sheet never holds a thumbnail
back; ``serve_info`` reads the file on each request.
"""

from __future__ import annotations

import hashlib
import json
import mimetypes
import os
import stat
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, Response

from zvisiongenerator.utils.atomic_write import write_bytes_atomic
from zvisiongenerator.utils.audio_peaks import read_waveform
from zvisiongenerator.utils.lora_preview import make_preview
from zvisiongenerator.utils.video_frame import extract_frame_png, extract_last_frame_png, locate_frame, locate_last_frame, probe_for_frames
from zvisiongenerator.utils.video_point import frame_total
from zvisiongenerator.utils.video_probe import ffprobe_json
from zvisiongenerator.utils.video_sprite import build_sprite
from zvisiongenerator.web.path_browser import BrowsePurpose

if TYPE_CHECKING:
    from starlette.requests import Request

__all__ = [
    "CACHE_CAP_BYTES",
    "INFO_KINDS",
    "MEDIA_KINDS",
    "PREVIEW_KINDS",
    "SPRITE_KINDS",
    "TOOL_TIMEOUT",
    "MediaFile",
    "cache_directory",
    "confined_media_file",
    "media_file",
    "serve_info",
    "serve_media",
    "serve_preview",
    "serve_sprite",
    "trim_cache",
]

PREVIEW_KINDS: Final = ("image", "video")
"""The preview kinds the preview route draws a picture for."""

INFO_KINDS: Final = ("image", "audio", "video")
"""The preview kinds the info route describes."""

MEDIA_KINDS: Final = ("audio", "video", "image")
"""The preview kinds the media route serves: a player's audio and video, and an image at its full size (the painter draws it)."""

SPRITE_KINDS: Final = ("video",)
"""The preview kinds the sprite route draws a contact sheet for."""

TOOL_TIMEOUT: Final = 10.0
"""Seconds one ffmpeg or ffprobe call may run before it is stopped."""

CACHE_CAP_BYTES: int = 256 * 1024 * 1024
"""The most the cache keeps; the oldest entries go first when it grows past this."""

_CACHE_CONTROL: Final = "private, no-cache"
_CAPTURE_SECONDS_FRACTION: Final = 0.1
_CACHE_SUFFIX: Final = ".webp"
# How many files are decoded at the same time; a request past that waits on its worker thread.
_decode_slots = threading.BoundedSemaphore(2)
# A contact sheet is a long ffmpeg run, so sheets queue on their own slot and never take one of the previews'.
_sprite_slot = threading.BoundedSemaphore(1)
_MEDIA_TYPES: Final = {
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".m4a": "audio/mp4",
    ".flac": "audio/flac",
    ".ogg": "audio/ogg",
    ".mp4": "video/mp4",
    ".mov": "video/quicktime",
    ".m4v": "video/x-m4v",
    ".mkv": "video/x-matroska",
    ".webm": "video/webm",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}


@dataclass(frozen=True)
class MediaFile:
    """Describe a file a browse route may serve: where it really is, what kind it is and its state on disk."""

    path: Path
    kind: str
    stat: os.stat_result


# ── The file check ───────────────────────────────────────────────────────────


def media_file(purpose: BrowsePurpose, path_text: str, kinds: tuple[str, ...]) -> MediaFile:
    """Return the regular file *path_text* names when *purpose* may preview it in a way listed in *kinds*.

    The path is resolved through links, and the file it ends at must be a regular file with an extension the
    purpose accepts, so a ``photo.png`` link to some other kind of file is refused.

    Args:
        purpose: The purpose the browser is choosing for.
        path_text: The absolute path of the file (``~`` is expanded).
        kinds: The preview kinds the calling route serves.

    Raises:
        ValueError: If the purpose has no preview kind the route serves, or the path is not absolute.
        FileNotFoundError: If nothing usable is at the path: missing, a folder, not a regular file, or an extension the purpose does not accept.
    """
    kind = purpose.preview
    if kind is None or kind not in kinds:
        raise ValueError(f"{purpose.label} has nothing this route can show.")
    path = Path(path_text.strip()).expanduser()
    if not path.is_absolute():
        raise ValueError("Enter an absolute path, or one starting with ~.")
    if path.suffix.lower() not in purpose.accepted_extensions:
        raise FileNotFoundError(f"No {purpose.label.lower()} file at {path}.")
    try:
        resolved = path.resolve(strict=True)
        file_stat = resolved.stat()
    except (OSError, RuntimeError) as exc:
        raise FileNotFoundError(f"No file at {path}.") from exc
    if not stat.S_ISREG(file_stat.st_mode) or resolved.suffix.lower() not in purpose.accepted_extensions:
        raise FileNotFoundError(f"No {purpose.label.lower()} file at {path}.")
    return MediaFile(path=resolved, kind=kind, stat=file_stat)


def confined_media_file(path: Path, kind: str) -> MediaFile:
    """Return the ``MediaFile`` of *path*, a file the caller has already confined, as a *kind* (``image`` or ``video``) the preview route draws.

    Raises:
        FileNotFoundError: If *path* is not a regular file.
    """
    try:
        file_stat = path.stat()
    except OSError as exc:
        raise FileNotFoundError(f"No file at {path}.") from exc
    if not stat.S_ISREG(file_stat.st_mode):
        raise FileNotFoundError(f"No file at {path}.")
    return MediaFile(path=path, kind=kind, stat=file_stat)


# ── The cache ────────────────────────────────────────────────────────────────


def cache_directory(data_dir: Path) -> Path:
    """Return the folder under *data_dir* that keeps the browser's previews."""
    return data_dir / "cache" / "browse"


def _cache_key(variant: str, media: MediaFile) -> str:
    """Return the key of one answer: its variant (the video frame and the picture's edge, ``info-v2`` or ``sprite-v1``), the resolved path, the modification time and the size.

    A changed file has a new key, so it is read again; the two stills of a video have different keys.
    """
    parts = (variant, str(media.path), str(media.stat.st_mtime_ns), str(media.stat.st_size))
    return hashlib.sha256("\0".join(parts).encode("utf-8", errors="surrogateescape")).hexdigest()


def _read_cached(cache_dir: Path, key: str) -> bytes | None:
    try:
        return (cache_dir / f"{key}{_CACHE_SUFFIX}").read_bytes()
    except OSError:
        return None


def _write_cached(cache_dir: Path, key: str, body: bytes) -> None:
    """Keep *body* under *key*; a cache that can't be written to is ignored."""
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        write_bytes_atomic(cache_dir / f"{key}{_CACHE_SUFFIX}", body)
        _note_written(cache_dir, len(body))
    except OSError:
        return


# Bytes written per cache folder since it was last trimmed; the folder is walked only after a stretch of writes.
_written_since_trim: dict[Path, int] = {}


def _note_written(cache_dir: Path, size: int) -> None:
    written = _written_since_trim.get(cache_dir)
    total = size if written is None else written + size
    # The first write of a run always trims, as the folder may already be over the cap from an earlier run.
    if written is None or total >= CACHE_CAP_BYTES // 16:
        _written_since_trim[cache_dir] = 0
        trim_cache(cache_dir, CACHE_CAP_BYTES)
    else:
        _written_since_trim[cache_dir] = total


def trim_cache(cache_dir: Path, cap: int) -> None:
    """Delete the oldest cached files until the folder holds at most *cap* bytes."""
    entries: list[tuple[float, int, Path]] = []
    try:
        with os.scandir(cache_dir) as scan:
            for item in scan:
                try:
                    if item.is_file(follow_symlinks=False) and item.name.endswith(_CACHE_SUFFIX):
                        info = item.stat()
                        entries.append((info.st_mtime, info.st_size, Path(item.path)))
                except OSError:
                    continue
    except OSError:
        return
    total = sum(size for _, size, _ in entries)
    for _, size, path in sorted(entries):
        if total <= cap:
            break
        path.unlink(missing_ok=True)
        total -= size


# ── Producing the answers ────────────────────────────────────────────────────


def _picture(source: bytes | Path, edge: int) -> bytes:
    made = make_preview(source, edge)
    if made is None:
        raise ValueError("Not an image that can be previewed.")
    return made[1]


def _video_still(path: Path, frame: str, edge: int, at: float | None) -> bytes:
    """Return a PNG of the frame 10% into the video, of its last frame, or, given *at*, of the frame that second lies in.

    Every still is read the way a saved frame is, so a clip that names no colour matrix has the same colours whichever way it is asked for.
    Only *at* needs the clip's frame facts. A clip without a readable frame rate or length (a browser recording without a container
    duration) still gets its tenth and last stills, by a time seek: the tenth at 10% of its length, or at the start when that is unknown.

    Raises:
        ValueError: If *at* is given and the clip's frame rate or length is unknown.
    """
    probe = probe_for_frames(path, timeout=TOOL_TIMEOUT)
    if at is not None:
        return extract_frame_png(path, probe, locate_frame(probe, at).seek, edge=edge, timeout=TOOL_TIMEOUT)
    tenth = (probe.duration or 0.0) * _CAPTURE_SECONDS_FRACTION
    try:
        seek = locate_last_frame(probe).seek if frame == "last" else locate_frame(probe, tenth).seek
    except ValueError:
        if frame == "last":
            return extract_last_frame_png(path, probe, edge=edge, timeout=TOOL_TIMEOUT)
        seek = tenth
    return extract_frame_png(path, probe, seek, edge=edge, timeout=TOOL_TIMEOUT)


def _build_preview(media: MediaFile, frame: str, edge: int, at: float | None) -> bytes:
    """Return the WebP preview of *media* at most *edge* px on its longest side: the picture itself, or a still of the video at *frame* (``tenth`` or ``last``) or at second *at*.

    Raises:
        ValueError: If the picture is not an image that can be previewed.
        RuntimeError: If ffmpeg or ffprobe fails or times out.
    """
    if media.kind == "image":
        return _picture(media.path, edge)
    return _picture(_video_still(media.path, frame, edge, at), edge)


def _image_facts(path: Path) -> dict[str, Any]:
    from PIL import Image

    from zvisiongenerator.utils.input_images import upright_size

    try:
        width, height = upright_size(path)
    except Image.DecompressionBombError as exc:
        raise ValueError(f"{path.name} has too many pixels to read.") from exc
    return {"width": width, "height": height}


def _first_stream(data: dict[str, Any], codec_type: str) -> dict[str, Any]:
    stream = next((s for s in data.get("streams") or [] if isinstance(s, dict) and s.get("codec_type") == codec_type), None)
    if stream is None:
        raise ValueError(f"The file has no {codec_type} stream.")
    return stream


def _audio_facts(path: Path) -> dict[str, Any]:
    stream = _first_stream(ffprobe_json(path, count_packets=False, timeout=TOOL_TIMEOUT), "audio")
    peaks, seconds = read_waveform(path, timeout=TOOL_TIMEOUT)
    return {"seconds": seconds, "sample_rate": int(stream.get("sample_rate") or 0) or None, "channels": stream.get("channels"), "peaks": peaks}


def _video_facts(path: Path) -> dict[str, Any]:
    probe = probe_for_frames(path, timeout=TOOL_TIMEOUT)
    return {"width": probe.width, "height": probe.height, "fps": probe.fps or None, "frames": frame_total(probe), "seconds": probe.duration, "start": probe.stream_offset}


def _build_info(media: MediaFile) -> bytes:
    """Return the JSON facts of *media* by its kind: the size of an image, the length, rate and waveform of audio, the size, rate, length and first-frame time (``start``) of a video.

    Raises:
        ValueError: If the file has no stream of its kind or is not a readable image.
        RuntimeError: If ffmpeg or ffprobe fails or times out.
        OSError: If the file can't be read.
    """
    facts = {"image": _image_facts, "audio": _audio_facts, "video": _video_facts}[media.kind](media.path)
    return json.dumps(facts).encode("utf-8")


# ── Answering requests ───────────────────────────────────────────────────────


def _etag_matches(if_none_match: str | None, etag: str) -> bool:
    if not if_none_match:
        return False
    candidates = {part.strip().removeprefix("W/") for part in if_none_match.split(",")}
    return "*" in candidates or etag in candidates


def _headers(etag: str) -> dict[str, str]:
    return {"ETag": etag, "Cache-Control": _CACHE_CONTROL, "X-Content-Type-Options": "nosniff"}


def _not_modified(request: Request, headers: dict[str, str]) -> Response | None:
    """Return the 304 answer when the request's ``If-None-Match`` names the ETag in *headers*, else None."""
    if _etag_matches(request.headers.get("if-none-match"), headers["ETag"]):
        return Response(status_code=304, headers=headers)
    return None


def _attempt(produce: Callable[[], bytes]) -> bytes | None:
    """Run *produce* and return its bytes, or None when the file can't be read or decoded."""
    try:
        return produce()
    except OSError, RuntimeError, ValueError:
        return None


def _cached_or_decoded(cache_dir: Path, key: str, produce: Callable[[], bytes], slots: threading.BoundedSemaphore = _decode_slots) -> bytes | None:
    """Return the cached bytes under *key*, else decode with *produce* once a slot of *slots* is free and cache the result; None, caching nothing, when the decode fails.

    The cache is read again after the slot is taken, so a request that waited on another one for the same key reuses what that one stored.
    """
    body = _read_cached(cache_dir, key)
    if body is None:
        with slots:
            body = _read_cached(cache_dir, key)
            if body is None:
                body = _attempt(produce)
                if body is not None:
                    _write_cached(cache_dir, key, body)
    return body


def _answer(body: bytes | None, media_type: str, headers: dict[str, str]) -> Response:
    if body is None:
        return Response(content="This file can't be previewed.", status_code=404, media_type="text/plain")
    return Response(content=body, media_type=media_type, headers=headers)


async def serve_preview(request: Request, media: MediaFile, *, frame: str, cache_dir: Path, edge: int, at: float | None = None) -> Response:
    """Answer with the WebP preview of *media* at most *edge* px wide or tall; *frame* picks a video's still (``tenth`` or ``last``), or *at* picks the frame a second lies in.

    The second is part of the cache key and the ETag, so each frame is cached on its own. A decode that fails gives 404 and is not remembered, so the next request tries again.
    """
    key = _cache_key(f"{frame}:{edge}" if at is None else f"{frame}:{edge}:at={at!r}", media)
    headers = _headers(f'"{key}"')
    if (not_modified := _not_modified(request, headers)) is not None:
        return not_modified
    body = await run_in_threadpool(_cached_or_decoded, cache_dir, key, lambda: _build_preview(media, frame, edge, at))
    return _answer(body, "image/webp", headers)


async def serve_info(request: Request, media: MediaFile) -> Response:
    """Answer with the JSON facts of *media*, read from the file on each request."""
    headers = _headers(f'"{_cache_key("info-v2", media)}"')
    if (not_modified := _not_modified(request, headers)) is not None:
        return not_modified
    body = await run_in_threadpool(_attempt, lambda: _build_info(media))
    return _answer(body, "application/json", headers)


async def serve_sprite(request: Request, media: MediaFile, cache_dir: Path) -> Response:
    """Answer with the JPEG contact sheet of a video: 121 frames at most, in an 11 x 11 grid, cached by the file's path, time and size.

    A sheet that can't be built gives 404 and is not remembered, so the next request tries again. The JPEG is kept under the cache's own ``.webp`` file name, which the trim goes by.
    """
    key = _cache_key("sprite-v1", media)
    headers = _headers(f'"{key}"')
    if (not_modified := _not_modified(request, headers)) is not None:
        return not_modified
    body = await run_in_threadpool(_cached_or_decoded, cache_dir, key, lambda: build_sprite(media.path), _sprite_slot)
    return _answer(body, "image/jpeg", headers)


def serve_media(media: MediaFile) -> FileResponse:
    """Answer with the audio, video or image file itself, ready for range requests, typed by its extension and never sniffed as a page."""
    media_type = _MEDIA_TYPES.get(media.path.suffix.lower()) or mimetypes.guess_type(media.path.name)[0] or "application/octet-stream"
    return FileResponse(media.path, stat_result=media.stat, media_type=media_type, headers={"Cache-Control": _CACHE_CONTROL, "X-Content-Type-Options": "nosniff"})
