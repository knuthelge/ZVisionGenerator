"""Cut a clip at a frame so an extension can continue from there instead of from its end."""

from __future__ import annotations

import contextlib
import math
import os
import shutil
import tempfile
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING

import psutil

from zvisiongenerator.core.video_types import ExtendCut, ExtendSource
from zvisiongenerator.utils.ffmpeg import STAGING_FILE_PREFIX
from zvisiongenerator.utils.video_extend import AUDIO_SAMPLE_RATE, CARRY_FRAMES, build_extend_source
from zvisiongenerator.utils.video_point import frame_at
from zvisiongenerator.utils.video_probe import parse_fraction, probe_video
from zvisiongenerator.utils.video_splice import AUDIO_BITRATE, REENCODE_CRF, color_filter, run_ffmpeg, source_audio_chain

if TYPE_CHECKING:
    import argparse
    from collections.abc import Iterator

__all__ = [
    "CUT_DIR_PREFIX",
    "MIN_CUT_FRAMES",
    "applied_extend_cut",
    "build_cut_command",
    "cut_extend_source",
    "extend_cut_point",
    "moved_track_end",
    "sweep_stale_cuts",
]

# The fewest frames a cut may keep: the extension carries this many frames of the source's end into its window.
MIN_CUT_FRAMES = CARRY_FRAMES
# Prefix of the hidden temporary folder a job's cut is made in, beside the job's output. The process id follows it, so a later job can tell
# a folder in use from one a killed process left behind.
CUT_DIR_PREFIX = f"{STAGING_FILE_PREFIX}cut-"
_CUT_FILE_NAME = "cut.mp4"


def extend_cut_point(source: ExtendSource, seconds: float) -> ExtendCut | None:
    """Return the cut that continues *source* from the frame *seconds* lies in, or ``None`` when that is the clip's end.

    The time is snapped to a frame by the rule the player uses and that frame is the last one kept. A time at or past the
    last frame keeps the whole clip, so there is nothing to cut.

    Args:
        source: The checked clip.
        seconds: A time on the clip's clock.

    Raises:
        ValueError: If *seconds* is negative or not finite, or the cut would keep fewer than ``MIN_CUT_FRAMES`` frames.
    """
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError("The point to extend from must be a finite number of seconds, 0 or more.")
    frame = frame_at(seconds, start=source.video_start, fps=source.fps, frames=source.frame_count)
    keep = frame + 1
    if keep >= source.frame_count:
        return None
    if keep < MIN_CUT_FRAMES:
        raise ValueError(
            f"Continuing from {seconds:g} s keeps {keep} frames of {source.path.name}; at least {MIN_CUT_FRAMES} frames ({MIN_CUT_FRAMES / source.fps:.2f} s at {source.fps:g} fps) are needed."
        )
    return ExtendCut(keep_frames=keep, seconds=round(source.video_start + frame / source.fps, 6), original=source.path)


def moved_track_end(source: ExtendSource, cut: ExtendCut) -> float | None:
    """Return where in its audio track the cut clip's sound ends: the source's end moved back by the removed frames, never below 0.

    ``None`` when the source records no track end.
    """
    if source.audio_track_end is None:
        return None
    removed = source.frame_count - cut.keep_frames
    return round(max(0.0, source.audio_track_end - removed / source.fps), 6)


def build_cut_command(source: ExtendSource, cut: ExtendCut, output: str | Path) -> list[str]:
    """Build the ffmpeg command that writes the first ``cut.keep_frames`` frames of *source*, sound included, as a new clip.

    The picture is encoded again at the quality the join's own re-encode uses (crf 12) with the source's colour tags,
    because a stream copy can break the last frames before a cut, with libx264's default preset as the join's own
    re-encodes use. The join usually copies this stream into the extended clip, so it is encoded as the clip's own
    picture. The sound is moved to the first frame, as the join reads it, and cut to the kept length; a silent source
    gets no sound. The result carries no metadata.

    Args:
        source: The clip to cut.
        cut: The frames to keep.
        output: The cut clip.
    """
    colors = color_filter(source.color_space, source.color_primaries, source.color_trc)
    video = [
        *("-map", "0:v:0"),
        *("-vf", f"trim=end_frame={cut.keep_frames},setpts=PTS-STARTPTS,{colors}"),
        *("-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(REENCODE_CRF), "-r", source.fps_text),
    ]
    if source.has_audio:
        chain = source_audio_chain(samples=_kept_samples(source, cut), offset=source.video_start)
        audio = ["-map", "0:a:0", "-af", chain, *("-c:a", "aac", "-b:a", AUDIO_BITRATE)]
    else:
        audio = ["-an"]
    return ["ffmpeg", "-y", "-v", "error", "-i", str(source.path), *video, *audio, "-map_metadata", "-1", str(output)]


def cut_extend_source(source: ExtendSource, cut: ExtendCut, work_dir: Path) -> ExtendSource:
    """Cut *source* into *work_dir* and return the cut clip as the source to extend.

    The value is built from the cut file's own probe, so its timing is the cut's; it keeps the original clip's recorded
    scene, prompt, model, LoRAs and audio track, which the cut file does not carry.

    Raises:
        RuntimeError: If ffmpeg fails, or the cut does not have the kept frames and the source's displayed size.
    """
    output = work_dir / _CUT_FILE_NAME
    run_ffmpeg(build_cut_command(source, cut, output), "ffmpeg could not cut the video")
    probe = probe_video(output)
    if probe.frame_count != cut.keep_frames:
        raise RuntimeError(f"The cut video has {probe.frame_count} frames, expected {cut.keep_frames}.")
    if (probe.width, probe.height) != (source.width, source.height):
        raise RuntimeError(f"The cut video is {probe.width}x{probe.height}, expected {source.width}x{source.height}.")
    cut_source = build_extend_source(output, probe, None)
    return replace(
        cut_source,
        scene=source.scene,
        prompt=source.prompt,
        model=source.model,
        lora=source.lora,
        audio_track_path=source.audio_track_path,
        audio_track_end=moved_track_end(source, cut),
    )


@contextlib.contextmanager
def applied_extend_cut(args: argparse.Namespace) -> Iterator[None]:
    """Make the job's cut, run with ``args.extend_source`` set to it, and remove it afterwards.

    A job with no ``extend_cut`` is left as it is. Otherwise the cut is made once, in a hidden folder under the save
    folder, which is created first because the workflow only creates it after the model has loaded. The folder is removed
    when the block ends, whether it finished, failed or was stopped, and the original source is put back. Folders of
    earlier cuts that a killed process left behind are removed first (see :func:`sweep_stale_cuts`).

    Args:
        args: The job's arguments: ``extend_source`` (the original clip), ``extend_cut`` and ``output`` (the save folder).

    Raises:
        RuntimeError: If the cut cannot be made.
    """
    cut: ExtendCut | None = getattr(args, "extend_cut", None)
    if cut is None:
        yield
        return
    original = args.extend_source
    save_dir = Path(args.output)
    save_dir.mkdir(parents=True, exist_ok=True)
    sweep_stale_cuts(save_dir)
    work_dir = Path(tempfile.mkdtemp(dir=save_dir, prefix=f"{CUT_DIR_PREFIX}{os.getpid()}-"))
    try:
        args.extend_source = cut_extend_source(original, cut, work_dir)
        yield
    finally:
        args.extend_source = original
        shutil.rmtree(work_dir, ignore_errors=True)


def sweep_stale_cuts(save_dir: Path) -> None:
    """Remove the cut folders in *save_dir* that a process which is gone left behind (a kill skips the cleanup).

    A cut folder is named for the process that made it. It is removed only when that process no longer exists, or when a
    different process has since taken its id (it started after the folder was last written), so the folder of a job running
    in another process, the CLI beside the Web UI for one, is never touched. A folder of this process is left alone, and so
    is one whose name carries no process id. Whatever cannot be removed is skipped.

    Args:
        save_dir: The job's save folder.
    """
    try:
        folders = [entry for entry in save_dir.iterdir() if entry.name.startswith(CUT_DIR_PREFIX) and entry.is_dir() and not entry.is_symlink()]
    except OSError:
        return
    for folder in folders:
        pid_text = folder.name.removeprefix(CUT_DIR_PREFIX).partition("-")[0]
        if pid_text.isdigit() and int(pid_text) != os.getpid() and not _is_running_since(int(pid_text), folder):
            shutil.rmtree(folder, ignore_errors=True)


def _is_running_since(pid: int, folder: Path) -> bool:
    """Return whether process *pid* is the one that made *folder*: it exists and started before the folder was last written."""
    try:
        started = psutil.Process(pid).create_time()
        return started <= folder.stat().st_mtime
    except psutil.NoSuchProcess, FileNotFoundError:
        return False
    except psutil.Error, OSError:
        return True  # cannot tell (access denied): keep it


def _kept_samples(source: ExtendSource, cut: ExtendCut) -> int:
    """Return the length of the kept frames in samples at the joined audio's rate."""
    rate = parse_fraction(source.fps_text) or Fraction(source.fps).limit_denominator(1_000_000)
    return round(Fraction(cut.keep_frames * AUDIO_SAMPLE_RATE) / rate)
