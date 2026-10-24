"""Upscale one existing video file: check it, then stream its frames through SeedVR2 into a new file with progress events."""

from __future__ import annotations

import math
import os
import shutil
import tempfile
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

from zvisiongenerator.core.progress_events import ProgressCallback, emit_generation_finished, emit_progress, exception_message
from zvisiongenerator.core.types import StageOutcome
from zvisiongenerator.core.video_types import ExtendSource, VideoFileUpscaleRequest
from zvisiongenerator.utils.filename import unique_output_path
from zvisiongenerator.utils.interactive import SkipSignal
from zvisiongenerator.utils.provenance import UPSCALE_METHOD_SEEDVR2, build_video_upscale_payload, mp4_config_args
from zvisiongenerator.utils.seedvr2 import SEEDVR2_BITS
from zvisiongenerator.utils.upscale import VIDEO_UPSCALE_FACTOR, VIDEO_UPSCALE_SEED, WORK_DIR_PREFIX
from zvisiongenerator.utils.video_extend import build_extend_source, inspect_extend_source
from zvisiongenerator.utils.video_splice import build_audio_prep_command, build_final_mux_command, build_segment_command, run_ffmpeg, write_segment_frames
from zvisiongenerator.utils.video_tail import stream_frames

if TYPE_CHECKING:
    import numpy as np

# The problems that stop an upscale; every other one that Extend reports (a short, off-grid, small, 10-bit or already
# upscaled clip) is fine here.
_REFUSED_CODES = frozenset({"no_video_stream", "unreadable", "unreadable_audio", "hdr", "variable_frame_rate", "interlaced", "non_square_pixels"})
# The progress phase the job card shows as batches.
_PHASE = "video_file_upscale"
_STAGE = "upscale"
# Tells the client which workflow stage each batch belongs to, as the stage wrapper does for denoising steps.
_STAGE_FIELDS = {"workflow_stage_index": 1, "workflow_total_stages": 1, "workflow_stage_name": _STAGE}
_OUTPUT_SUFFIX = f"_{VIDEO_UPSCALE_FACTOR}x"
_FAILURE_MESSAGE = "Video upscale failed."


class _Cancelled(Exception):
    """A quit was requested while the upscale was running."""


def read_video_upscale_source(path: str | Path) -> tuple[ExtendSource, dict[str, Any] | None]:
    """Check a video file and return it as an upscale source with the raw settings recorded in it.

    The clip is checked as for Extend, but only the problems that make the upscale itself wrong stop it: no video
    stream, an unreadable file or sound, HDR, a variable frame rate, interlacing and non-square pixels.

    Returns:
        The checked clip (displayed size, frame rate, colour tags, sound) and its embedded config payload, ``None`` when it has none.

    Raises:
        FileNotFoundError: If *path* is not a file.
        ValueError: If the clip cannot be upscaled; the message names every refused problem.
    """
    inspection = inspect_extend_source(path)
    refused = [problem for problem in inspection.problems if problem.code in _REFUSED_CODES]
    if refused or inspection.probe is None:
        raise ValueError(f"{Path(path).name} can't be upscaled: " + " ".join(problem.message for problem in refused))
    return build_extend_source(path, inspection.probe, inspection.recorded), inspection.config


def run_video_file_upscale(
    upscaler: Any,
    model: Any,
    request: VideoFileUpscaleRequest,
    *,
    progress_callback: ProgressCallback | None = None,
    skip_signal: SkipSignal | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> StageOutcome:
    """Upscale ``request.source`` into ``<name>_2x.mp4`` beside it and emit the events of a one-video batch.

    The frames stream from the source through the engine's ``upscale_video`` into an encoder, so memory stays flat
    for any length. Everything is built in a hidden work folder and moved into place at the end, so a quit or a
    failure leaves no file. The source's first audio stream is prepared before the GPU starts and muxed back in at
    the end, and the clip's recorded settings are embedded in the result.

    Args:
        upscaler: The SeedVR2 engine; it needs ``upscale_video``.
        model: The engine's loaded model.
        request: The checked clip, its settings and the plan.
        progress_callback: Receives the progress events.
        skip_signal: A pending quit stops the upscale at the next batch.
        clock: Seconds, for the time left and the recorded time.

    Returns:
        ``success``, ``skipped`` when the job was stopped, or ``failed``.
    """
    skip = skip_signal or SkipSignal()
    stem = request.source.path.stem
    context = {"mode": "video", "run_index": 0, "total_runs": 1, "ran_iterations": 1, "total_iterations": 1, "set_name": stem, "prompt_index": 0, "total_prompts": 1}
    prompt = request.source.prompt or ""
    emit_progress(progress_callback, "batch_started", mode="video", total_iterations=1, total_runs=1)
    emit_progress(progress_callback, "prompt_started", **context, prompt=prompt, seed=VIDEO_UPSCALE_SEED)
    if _quit_pending(skip):
        emit_progress(progress_callback, "batch_cancelled", mode="video", completed_iterations=0, total_iterations=1)
        return StageOutcome.skipped

    event_context = {**context, "prompt": prompt, "seed": VIDEO_UPSCALE_SEED, "filename": stem}
    emit_progress(progress_callback, "generation_started", **event_context, retry=0)
    emit_progress(progress_callback, "workflow_started", total_stages=1, stage_name=_STAGE, stage_names=[_STAGE], **event_context)
    emit_progress(progress_callback, "workflow_stage_started", stage_index=1, total_stages=1, stage_name=_STAGE, **event_context)
    started = clock()
    output: Path | None = None
    message: str | None = None
    try:
        output = _upscale_to_output(upscaler, model, request, skip=skip, clock=clock, progress=lambda payload: emit_progress(progress_callback, "step_progress", **context, **_STAGE_FIELDS, **payload))
        outcome = StageOutcome.success
    except _Cancelled:
        outcome = StageOutcome.skipped
    except Exception as exc:  # noqa: BLE001 - any failure of the engine, ffmpeg or the files ends this job, never the worker
        outcome = StageOutcome.failed
        message = exception_message(str(exc) or type(exc).__name__)
    status = {StageOutcome.success: "success", StageOutcome.skipped: "skipped"}.get(outcome, "failed")
    emit_progress(progress_callback, "workflow_stage_completed", stage_index=1, total_stages=1, stage_name=_STAGE, outcome=status, **event_context)
    emit_progress(progress_callback, "workflow_finished", total_stages=1, completed_stages=1, stage_name=_STAGE, status=status, **event_context)
    emit_generation_finished(
        progress_callback,
        event_context=event_context,
        status=status,
        filename=output.name if output else None,
        generation_time=clock() - started,
        output_path=str(output) if output else None,
        message=(message or _FAILURE_MESSAGE) if status == "failed" else None,
    )
    if status == "success":
        emit_progress(progress_callback, "batch_completed", mode="video", completed_iterations=1, total_iterations=1)
    elif status == "skipped":
        emit_progress(progress_callback, "batch_cancelled", mode="video", completed_iterations=0, total_iterations=1)
    else:
        emit_progress(progress_callback, "batch_failed", mode="video", completed_iterations=1, total_iterations=1, message=message or _FAILURE_MESSAGE)
    return outcome


def _upscale_to_output(
    upscaler: Any,
    model: Any,
    request: VideoFileUpscaleRequest,
    *,
    skip: SkipSignal,
    clock: Callable[[], float],
    progress: Callable[[dict[str, Any]], None],
) -> Path:
    """Build the upscaled file in a work folder, move it beside the source and return its path.

    Raises:
        _Cancelled: If a quit was requested; nothing is left behind.
        RuntimeError: If the engine or ffmpeg fails.
    """
    source = request.source
    started = clock()
    work = Path(tempfile.mkdtemp(dir=request.output_dir, prefix=WORK_DIR_PREFIX))
    try:
        audio = _prepare_audio(source, work)
        total = math.ceil(source.frame_count / request.batch_size)
        progress({"phase": _PHASE, "current_step": 0, "total_steps": total})
        frames_started = clock()

        def report(done: int, batches: int) -> None:
            elapsed = clock() - frames_started
            progress({"phase": _PHASE, "current_step": done, "total_steps": batches, "eta_secs": elapsed / done * (batches - done) if done else None})

        def interrupt() -> None:
            if _quit_pending(skip):
                raise _Cancelled

        video = work / "video.mp4"
        written = _write_upscaled_video(upscaler, model, request, video, progress=report, interrupt=interrupt)
        payload = build_video_upscale_payload(
            request.recorded_config,
            source_path=str(source.path),
            source_width=source.width,
            source_height=source.height,
            width=request.width,
            height=request.height,
            frame_count=written,
            audio=audio is not None,
            upscale={
                "method": UPSCALE_METHOD_SEEDVR2,
                "factor": VIDEO_UPSCALE_FACTOR,
                "model": request.upscale_model,
                "quantize": SEEDVR2_BITS,
                "softness": request.softness,
                "seed": VIDEO_UPSCALE_SEED,
            },
            generation_time=clock() - started,
        )
        finished = work / "finished.mp4"
        run_ffmpeg(build_final_mux_command(video, audio, finished, video_offset=source.video_start, metadata=mp4_config_args(payload)), "ffmpeg could not finish the upscaled video")
        interrupt()
        target = unique_output_path(request.output_dir, f"{source.path.stem}{_OUTPUT_SUFFIX}", ".mp4")
        os.replace(finished, target)
        return target
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _prepare_audio(source: ExtendSource, work: Path) -> Path | None:
    """Write the source's first audio stream to the work folder as AAC (copied when it already is) and return it; ``None`` without sound."""
    if not source.has_audio:
        return None
    audio = work / "audio.m4a"
    copy = source.probe is not None and source.probe.audio_codec == "aac"
    run_ffmpeg(build_audio_prep_command(source.path, audio, copy=copy), "ffmpeg could not read the video's sound")
    return audio


def _write_upscaled_video(
    upscaler: Any,
    model: Any,
    request: VideoFileUpscaleRequest,
    video: Path,
    *,
    progress: Callable[[int, int], None],
    interrupt: Callable[[], None],
) -> int:
    """Stream the source's frames through the engine into the video-only file *video* and return the frames written.

    The engine's iterator and the reader are always closed here, because the writer stops quietly when its encoder
    exits early, and the engine frees its models only when its iterator ends or is closed.
    """
    source = request.source
    frames = stream_frames(source.path, width=source.width, height=source.height)
    upscaled = upscaler.upscale_video(
        model,
        frames,
        width=request.width,
        height=request.height,
        total_frames=source.frame_count,
        seed=VIDEO_UPSCALE_SEED,
        softness=request.softness,
        batch_size=request.batch_size,
        progress=progress,
        interrupt=interrupt,
    )
    written = 0

    def counted() -> Iterator[np.ndarray]:
        nonlocal written
        for frame in upscaled:
            written += 1
            yield frame

    try:
        command = build_segment_command(request.width, request.height, source.fps_text, video, color_space=source.color_space, color_primaries=source.color_primaries, color_trc=source.color_trc)
        write_segment_frames(counted(), command, description="the upscaled video")
    finally:
        _close(upscaled)
        _close(frames)
    return written


def _close(iterator: Any) -> None:
    """Close *iterator* when it can be closed (a generator is), so what it holds is released."""
    close = getattr(iterator, "close", None)
    if close is not None:
        close()


def _quit_pending(skip: SkipSignal) -> bool:
    """Return whether a quit is queued on *skip* (it stays queued)."""
    return skip.pending() == "quit"
