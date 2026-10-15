"""Build the Web UI's extend responses: check and describe a source video, fit its window to memory, and build the rewrite context of an extension."""

from __future__ import annotations

import functools
import sys
from pathlib import Path
from typing import Any

from zvisiongenerator.backends import accelerator_memory_budget_or_none as memory_budget, supports_video_audio, supports_video_extend
from zvisiongenerator.core.video_types import ExtendSource
from zvisiongenerator.utils.ffmpeg import require_ffmpeg
from zvisiongenerator.utils.model_memory import clip_memory_fit
from zvisiongenerator.utils.paths import resolve_model_path
from zvisiongenerator.utils.prompt_enhance import EnhanceContext
from zvisiongenerator.utils.video_extend import ExtendInspection, extend_enhance_context, inspect_extend_source, recorded_prompts, require_extend_source, resolve_extend_frames, window_frames
from zvisiongenerator.web.config import WebUiConfig, preferred_option
from zvisiongenerator.web.gallery import output_media_url
from zvisiongenerator.web.json_fields import json_field
from zvisiongenerator.web.path_browser import SOURCE_VIDEO_EXTENSIONS


EXTEND_UNAVAILABLE_MESSAGE = "Extending a video is not available on this platform."


def _source_video_path(value: Any, *, key: str) -> Path:
    """Return the host-local path of a source video, checked like a reference image plus its extension.

    Raises:
        ValueError: If the value is missing, a URL, or not a video file name.
        FileNotFoundError: If no file is at the path.
    """
    if value is not None and not isinstance(value, str):
        raise ValueError(f"Field '{key}' must be a path string.")
    text = (value or "").strip()
    if not text:
        raise ValueError(f"Field '{key}' is required.")
    if "://" in text:
        raise ValueError(f"Field '{key}' must be a host-local file path on the machine running the Web UI host.")
    path = Path(text).expanduser()
    if path.suffix.lower() not in SOURCE_VIDEO_EXTENSIONS:
        raise ValueError(f"Field '{key}' must be a video file ({', '.join(SOURCE_VIDEO_EXTENSIONS)}).")
    if not path.is_file():
        raise FileNotFoundError(f"Source video not found: {path}")
    # A symlink may point at a file of another type: the name that counts is the one the link resolves to.
    resolved = path.resolve()
    if resolved.suffix.lower() not in SOURCE_VIDEO_EXTENSIONS:
        raise ValueError(f"Field '{key}' must be a video file ({', '.join(SOURCE_VIDEO_EXTENSIONS)}).")
    return resolved


def load_source_video(value: Any, *, key: str) -> ExtendSource:
    """Return the checked source of an extension for a job; every refusal, a missing file included, is a ``ValueError``."""
    try:
        path = _source_video_path(value, key=key)
        return require_extend_source(_inspect_source(path), path)
    except FileNotFoundError as exc:
        raise ValueError(str(exc)) from exc


def _file_identity(path: Path) -> tuple[str, int, int]:
    """Return the cache key of a file: its resolved path, modification time in nanoseconds and size.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    resolved = path.resolve()
    stat = resolved.stat()
    return (str(resolved), stat.st_mtime_ns, stat.st_size)


# The inspect route, the rewrite context and the submit read the same clip in turn; an edited or replaced file has another key.
@functools.lru_cache(maxsize=16)
def _inspect_by_identity(key: tuple[str, int, int]) -> ExtendInspection:
    """Return the inspection of the file a ``_file_identity`` key names."""
    return inspect_extend_source(key[0])


def _inspect_source(path: Path) -> ExtendInspection:
    """Return the inspection of a source clip, reusing the last one while the file is unchanged.

    Raises:
        ValueError: If ffmpeg is not installed, so a missing ffprobe is not reported as a missing file.
        FileNotFoundError: If the file does not exist.
    """
    try:
        require_ffmpeg()
    except RuntimeError as exc:
        raise ValueError(str(exc)) from exc
    return _inspect_by_identity(_file_identity(path))


def requested_extend_frames(frames: int | None, seconds: float | None, fps: float) -> int:
    """Return the new frames of an extension: *frames* as given, else *seconds* at *fps* snapped to a multiple of 8, else the 2 s default.

    Raises:
        ValueError: If *frames* is not a multiple of 8 of at least 8, or *seconds* is not positive.
    """
    return resolve_extend_frames(frames=frames, seconds=seconds, fps=fps, strict=True)[0]


def extend_source_response(payload: Any, web_config: WebUiConfig, media_root: Path) -> dict[str, Any]:
    """Build the ``POST /api/extend/source`` response for a request body.

    Raises:
        ValueError: If the body, path, length, model or low-memory flag is invalid.
        FileNotFoundError: If no file is at the path.
    """
    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object.")
    path = _source_video_path(payload.get("path"), key="path")
    requested_frames = json_field(payload, "extend_frames", int, "a positive integer", minimum=1)
    model = json_field(payload, "model", str, "a string")
    low_memory = json_field(payload, "low_memory", bool, "a boolean")

    inspection = _inspect_source(path)
    probe = inspection.probe
    scene, prompt = recorded_prompts(inspection.recorded)
    recorded_model = inspection.recorded.model if inspection.recorded is not None else None
    fps = probe.fps if probe is not None else 0.0
    extend_frames = None if requested_frames is None and fps <= 0 else resolve_extend_frames(frames=requested_frames, seconds=None, fps=fps, strict=False)[0]
    window = window_frames(extend_frames) if extend_frames is not None else None
    memory_fit = None
    if inspection.source is not None and window is not None:
        memory_model = model or (recorded_model if recorded_model in web_config.video_model_options else None) or preferred_option(web_config.default_models.video, web_config.video_model_options)
        memory_fit = _extend_memory_fit(memory_model, inspection.source, window, low_memory is not False, web_config)
    frame_count = probe.frame_count if probe is not None else None
    return {
        "path": str(path),
        "filename": path.name,
        "media_url": output_media_url(media_root, path),
        "width": probe.width if probe is not None else None,
        "height": probe.height if probe is not None else None,
        "rotation": probe.rotation if probe is not None else None,
        "fps": fps if probe is not None and fps > 0 else None,
        "frame_count": frame_count,
        "duration_seconds": _source_duration(probe, fps, frame_count),
        "has_audio": probe.has_audio if probe is not None else None,
        "scene": scene,
        "prompt": prompt,
        "legacy": scene is None and prompt is not None,
        "model": recorded_model if recorded_model in web_config.video_model_options else None,
        "lora": inspection.recorded.lora if inspection.recorded is not None else None,
        "problems": [{"code": problem.code, "message": problem.message} for problem in inspection.problems],
        "extend_frames": extend_frames,
        "window_frames": window,
        "memory_fit": memory_fit,
    }


def _source_duration(probe: Any, fps: float, frame_count: int | None) -> float | None:
    """Return a clip's length in seconds: its frames at its frame rate, else the container's duration."""
    if frame_count is not None and fps > 0:
        return round(frame_count / fps, 3)
    return probe.duration if probe is not None else None


def _extend_memory_fit(model: str | None, source: ExtendSource, window: int, low_memory: bool, web_config: WebUiConfig) -> dict[str, Any] | None:
    """Return how an extension window fits this machine's memory (``status``, ``required_gb``, ``budget_gb``), or ``None`` when unknown."""
    budget = memory_budget()
    if model is None or budget is None:
        return None
    try:
        resolved = resolve_model_path(model, aliases=web_config.app_config.get("model_aliases", {}), platform_key=sys.platform)
    except KeyError, ValueError:
        return None
    return clip_memory_fit(resolved, width=source.width, height=source.height, frames=window, low_memory=low_memory, budget=budget)


def extend_context_from_body(raw: dict[str, Any], prompt: str) -> EnhanceContext:
    """Build the rewrite context of an extension from the source clip's own file and the request's length and audio.

    Raises:
        ValueError: If extending is unavailable, the source cannot be used or the length is invalid.
    """
    if not supports_video_extend():
        raise ValueError(EXTEND_UNAVAILABLE_MESSAGE)
    source = load_source_video(raw["source_video"], key="context.source_video")
    extend_frames = json_field(raw, "extend_frames", int, "an integer", name="context.extend_frames")
    seconds = json_field(raw, "extend_seconds", (int, float), "a number", name="context.extend_seconds")
    frames = requested_extend_frames(extend_frames, None if seconds is None else float(seconds), source.fps)
    return extend_enhance_context(source, frames, no_audio=not (raw.get("audio") is True and supports_video_audio()), prompt=prompt)
