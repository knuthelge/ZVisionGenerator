"""Check a clip that a video is extended from, and the frame and sample arithmetic of an extension window."""

from __future__ import annotations

import math
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from zvisiongenerator.core.video_types import ExtendSource
from zvisiongenerator.utils.prompt_compose import is_original_prompt
from zvisiongenerator.utils.prompt_enhance import EnhanceContext, enhance_context
from zvisiongenerator.utils.provenance import RecordedSettings, read_mp4_config, recorded_settings
from zvisiongenerator.utils.video_probe import NO_VIDEO_STREAM, VideoProbe, audio_decodes, frame_rates_match, probe_video

if TYPE_CHECKING:
    import argparse

    import numpy as np

    from zvisiongenerator.core.video_family import VideoFamily

# Pixel frames of the source's end that are pinned at the start of the generated window (4 latent frames).
CARRY_FRAMES = 25
# New frames come in steps of one latent frame.
EXTEND_FRAME_STEP = 8
DEFAULT_EXTEND_SECONDS = 2.0
# How far a source's frame rate may stray from a family's ``extend_fps`` (the backends' own check uses the same).
EXTEND_FPS_TOLERANCE = 1e-3
# Sample rate of the generated audio.
AUDIO_SAMPLE_RATE = 48000
# Sound of the carried frames kept before the new part, to fill a source track that ends short of its video.
AUDIO_LEAD_SECONDS = 0.25
# Length of the fades at the seams of the joined sound (8 ms at 48 kHz).
AUDIO_FADE_SAMPLES = 384

# The generation grid (sizes must be multiples of it) and the smallest usable edge.
_SIZE_GRID = 32
_MIN_EDGE = 64
_HDR_TRANSFERS = frozenset({"smpte2084", "arib-std-b67"})
_HDR_COLOR_SPACES = frozenset({"bt2020nc", "bt2020c"})
_PROGRESSIVE_FIELD_ORDERS = frozenset({None, "progressive", "unknown"})
# Bit depths above 8 in a pixel format name: ``yuv420p10le``, ``gray12le``, ``yuv444p16be``, ``p010le``.
_HIGH_DEPTH_PIX_FMT = re.compile(r"(?:9|1[0246])(?:le|be)$|^p[024]1[026]")


@dataclass(frozen=True)
class ExtendProblem:
    """One reason a clip cannot be extended: a stable code for callers and a message for people."""

    code: str
    message: str


@dataclass(frozen=True)
class ExtendInspection:
    """The result of checking a clip: what was read, the problems found, and the source when there are none."""

    probe: VideoProbe | None
    recorded: RecordedSettings | None
    problems: tuple[ExtendProblem, ...]
    source: ExtendSource | None


def window_frames(extend_frames: int, family: VideoFamily | None = None) -> int:
    """Return the frames generated for an extension: the carried frames plus the new ones (LTX: 25 + 8n).

    Args:
        extend_frames: The new frames.
        family: The model's family, whose ``carry_frames`` are carried; ``None`` carries the LTX frames.
    """
    return (family.carry_frames if family is not None else CARRY_FRAMES) + extend_frames


def extend_audio_output(source: ExtendSource, *, no_audio: bool) -> bool:
    """Return whether an extension has sound: the source has audio and the job did not ask for none.

    A silent source stays silent, so it gets no generated audio, no joined audio and no Sound rules in a rewrite.
    """
    return source.has_audio and not no_audio


def carry_latent_frames() -> int:
    """Return how many latent frames the carried pixel frames fill (the first covers one pixel frame, each next eight)."""
    return (CARRY_FRAMES - 1) // EXTEND_FRAME_STEP + 1


def extend_step(family: VideoFamily | None) -> int:
    """Return the step between the lengths *family* can extend by (the LTX step for a model without a description)."""
    return family.extend_frame_step if family is not None else EXTEND_FRAME_STEP


def snap_extend_frames(frames: float, step: int = EXTEND_FRAME_STEP) -> int:
    """Round a frame count to the nearest multiple of *step* (ties round to even), at least *step*.

    Raises:
        ValueError: If *frames* is not finite.
    """
    if not math.isfinite(frames):
        raise ValueError(f"The frame count must be a finite number, got {frames}.")
    return max(step, round(frames / step) * step)


def extend_frames_from_seconds(seconds: float, fps: float, step: int = EXTEND_FRAME_STEP) -> int:
    """Return the new frames for a length in seconds at the source's frame rate: a multiple of *step*, at least *step* (ties round to even).

    Raises:
        ValueError: If *seconds* or *fps* is not a finite number greater than 0.
    """
    _require_positive_finite(seconds, "The extension length")
    _require_positive_finite(fps, "The frame rate")
    return snap_extend_frames(seconds * fps, step)


def resolve_extend_frames(
    *, frames: int | None, seconds: float | None, fps: float, strict: bool, frames_name: str = "extend_frames", seconds_name: str = "extend_seconds", step: int = EXTEND_FRAME_STEP
) -> tuple[int, str | None]:
    """Return the new frames for a requested length and, when a lenient request was changed, a note saying how.

    A length given in frames wins over one in seconds; with neither, the default length applies.

    Args:
        frames: Requested frames.
        seconds: Requested seconds, converted at *fps* (always rounded to a multiple of *step*).
        fps: The source's frame rate.
        strict: Require *frames* to be a multiple of *step* of at least *step* and *seconds* to be positive, and raise otherwise.
            When false, *frames* are snapped to a multiple of *step* and a note is returned for any change.
        frames_name: How the caller's surface names the frames option in messages (a flag or a field).
        seconds_name: How the caller's surface names the seconds option in messages.
        step: The step between the lengths the model's family can add (its ``extend_frame_step``).

    Returns:
        The frame count and a message when a lenient request was rounded, else ``None`` (always ``None`` when *strict*).

    Raises:
        ValueError: If *strict* and the request is invalid, or if *seconds* or *fps* is not a finite number greater than 0.
    """
    if frames is not None:
        if strict:
            if frames < step or frames % step:
                raise ValueError(f"{frames_name} must be a multiple of {step}, at least {step}.")
            return frames, None
        snapped = snap_extend_frames(frames, step)
        return snapped, None if snapped == frames else f"{frames_name} {frames} rounded to {snapped} (a multiple of {step})."
    if seconds is None:
        return extend_frames_from_seconds(DEFAULT_EXTEND_SECONDS, fps, step), None
    if strict:
        if seconds <= 0:
            raise ValueError(f"{seconds_name} must be positive.")
        try:
            return extend_frames_from_seconds(seconds, fps, step), None
        except ValueError as exc:
            raise ValueError(f"{seconds_name}: {exc}") from exc
    resolved = extend_frames_from_seconds(seconds, fps, step)
    if abs(resolved - seconds * fps) < 1e-6:
        return resolved, None
    return resolved, f"{seconds_name} {seconds:g} is {seconds * fps:g} frames at {fps:g} fps; using {resolved} frames ({resolved / fps:.2f} s)."


def extend_enhance_context(source: ExtendSource, extend_frames: int, *, no_audio: bool, prompt: str | None = None, family: VideoFamily | None = None) -> EnhanceContext:
    """Return the rewrite context of an extension: the generation window, the source's prompt and the new frames.

    Args:
        source: The checked source clip.
        extend_frames: The new frames.
        no_audio: Whether the job asked for no sound; a silent source has none either way.
        prompt: The user's prompt template, to tell whether it is still a legacy source's own prompt. Left out, nothing is
            known about it, and callers set ``prompt_is_original`` for each prompt later.
        family: The model's family, which sets the carried frames of the window (``None``: the LTX frames).
    """
    return enhance_context(
        mode="video",
        width=source.width,
        height=source.height,
        frames=window_frames(extend_frames, family),
        fps=source.fps,
        audio=extend_audio_output(source, no_audio=no_audio),
        reference_image=False,
        extends=True,
        previous=source.prompt,
        extract_from=source.prompt if source.legacy else None,
        prompt_is_original=prompt is not None and source.legacy and is_original_prompt(prompt, source.prompt or ""),
        new_frames=extend_frames,
    )


def apply_extend_args(args: argparse.Namespace, source: ExtendSource, extend_frames: int, family: VideoFamily | None = None) -> None:
    """Set the arguments the extend workflow reads: the source, the new frames, the source's displayed size and frame rate, and the window length.

    The window is the family's carried frames plus the new ones (``None``: the LTX frames).
    """
    args.extend_source = source
    args.extend_frames = extend_frames
    args.width, args.height = source.width, source.height
    args.fps = source.fps
    args.num_frames = window_frames(extend_frames, family)


def _require_positive_finite(value: float, label: str) -> None:
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{label} must be a finite number greater than 0, got {value}.")


def window_audio_cut(extend_frames: int, fps: float) -> tuple[int, int]:
    """Return ``(skip, keep)`` in samples for the window's audio: drop the carried frames' sound, keep the new frames' sound."""
    return round(CARRY_FRAMES * AUDIO_SAMPLE_RATE / fps), round(extend_frames * AUDIO_SAMPLE_RATE / fps)


def audio_lead_samples(fps: float) -> int:
    """Return how many samples of the carried frames' sound are kept in front of the new part (at most all of it)."""
    return min(round(CARRY_FRAMES * AUDIO_SAMPLE_RATE / fps), round(AUDIO_LEAD_SECONDS * AUDIO_SAMPLE_RATE))


def cut_window_audio(samples: np.ndarray, skip: int, keep: int, lead: int = 0) -> np.ndarray:
    """Cut ``(channels, samples)`` window audio to exactly ``lead + keep`` samples, starting ``lead`` before ``skip``.

    The ``lead`` samples are the end of the carried frames' sound and the ``keep`` samples that follow are the new
    part. When the window's sound ends early, its last samples fade out and the rest is zero.
    """
    import numpy as np

    kept = np.array(samples[:, max(0, skip - lead) : skip + keep], dtype=np.float32)
    short = lead + keep - kept.shape[1]
    if short <= 0:
        return kept
    fade = min(AUDIO_FADE_SAMPLES, kept.shape[1])
    if fade:
        kept[:, kept.shape[1] - fade :] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
    return np.pad(kept, ((0, 0), (0, short)))


def recorded_prompts(recorded: RecordedSettings | None) -> tuple[str | None, str | None]:
    """Return the ``(scene, prompt)`` recorded in a clip; a missing or blank one is ``None``."""
    if recorded is None:
        return None, None
    return _non_blank(recorded.scene), _non_blank(recorded.prompt)


def extend_source_problems(probe: VideoProbe, recorded: RecordedSettings | None, *, audio_ok: bool = True) -> list[ExtendProblem]:
    """List why a probed clip cannot be extended; an empty list means it can.

    A clip stored sideways and rotated by its display matrix is fine: its displayed size is checked.

    Args:
        probe: The clip's stream properties.
        recorded: The settings embedded in the clip, if any.
        audio_ok: Whether the clip's audio decodes (only used when it has audio).
    """
    problems: list[ExtendProblem] = []
    if probe.rotation % 90 != 0:
        problems.append(ExtendProblem("unreadable", f"The video is rotated by an unsupported angle ({probe.rotation} degrees)."))
    if probe.frame_count is None or probe.fps <= 0:
        problems.append(ExtendProblem("unreadable", "The video's length or frame rate could not be read."))
    elif probe.frame_count < CARRY_FRAMES:
        problems.append(ExtendProblem("too_short", f"The video has {probe.frame_count} frames; it needs at least {CARRY_FRAMES} to continue from."))
    if probe.width % _SIZE_GRID or probe.height % _SIZE_GRID:
        problems.append(ExtendProblem("off_grid", f"The video is {probe.width}x{probe.height}; its width and height must be multiples of {_SIZE_GRID}."))
    if probe.width < _MIN_EDGE or probe.height < _MIN_EDGE:
        problems.append(ExtendProblem("too_small", f"The video is {probe.width}x{probe.height}; each side must be at least {_MIN_EDGE} pixels."))
    if not frame_rates_match(probe.r_frame_rate, probe.avg_frame_rate):
        problems.append(ExtendProblem("variable_frame_rate", f"The video's frame rate varies ({probe.r_frame_rate} base, {probe.avg_frame_rate} average)."))
    if not probe.square_pixels:
        problems.append(ExtendProblem("non_square_pixels", f"The video has non-square pixels ({probe.sample_aspect_ratio})."))
    if _high_bit_depth(probe):
        problems.append(ExtendProblem("high_bit_depth", "The video has more than 8 bits per channel."))
    if probe.color_trc in _HDR_TRANSFERS or probe.color_space in _HDR_COLOR_SPACES:
        problems.append(ExtendProblem("hdr", "The video is HDR."))
    if probe.field_order not in _PROGRESSIVE_FIELD_ORDERS:
        problems.append(ExtendProblem("interlaced", "The video is interlaced."))
    if recorded is not None and recorded.generation.get("upscale"):
        problems.append(ExtendProblem("upscaled", "The video was upscaled; extend the original instead, since the upscaled size is not one the model generates at."))
    if probe.has_audio and not audio_ok:
        problems.append(ExtendProblem("unreadable_audio", "The video's audio track could not be decoded."))
    return problems


def extend_family_problems(fps: float, family: VideoFamily | None) -> list[ExtendProblem]:
    """List why a clip at *fps* cannot be extended by *family*; an empty list means the family takes its frame rate.

    A family that pins the rate (``extend_fps``) takes only that rate, within the tolerance the backends apply; a clip whose
    rate could not be read is already reported by ``extend_source_problems``.
    """
    if family is None or family.extend_fps is None or fps <= 0 or abs(fps - family.extend_fps) <= EXTEND_FPS_TOLERANCE:
        return []
    return [ExtendProblem("frame_rate", f"The video runs at {fps:g} fps; {family.label} extends {family.extend_fps} fps clips only.")]


def with_family_problems(inspection: ExtendInspection, family: VideoFamily | None) -> ExtendInspection:
    """Return *inspection* with the problems *family* adds; a clip with any problem has no source."""
    if inspection.probe is None:
        return inspection
    added = extend_family_problems(inspection.probe.fps, family)
    if not added:
        return inspection
    return ExtendInspection(inspection.probe, inspection.recorded, (*inspection.problems, *added), None)


def require_extend_family(source: ExtendSource, family: VideoFamily | None) -> None:
    """Raise unless *family* extends a clip at the frame rate of *source*.

    Raises:
        ValueError: If the clip's frame rate is not the one the family extends; the message says which.
    """
    problems = extend_family_problems(source.fps, family)
    if problems:
        raise ValueError(f"{source.path.name} cannot be extended: " + " ".join(problem.message for problem in problems))


def build_extend_source(path: str | Path, probe: VideoProbe, recorded: RecordedSettings | None) -> ExtendSource:
    """Build the source value for a probed clip that passed its checks.

    Raises:
        ValueError: If the probe has no frame count.
    """
    if probe.frame_count is None:
        raise ValueError("The video's length could not be read.")
    scene, prompt = recorded_prompts(recorded)
    return ExtendSource(
        path=Path(path).expanduser().resolve(),
        width=probe.width,
        height=probe.height,
        fps=probe.fps,
        fps_text=probe.r_frame_rate,
        frame_count=probe.frame_count,
        has_audio=probe.has_audio,
        video_start=_stream_offset(probe),
        color_space=probe.color_space,
        color_primaries=probe.color_primaries,
        color_trc=probe.color_trc,
        scene=scene,
        prompt=prompt,
        model=recorded.model if recorded else None,
        lora=recorded.lora if recorded else None,
        probe=probe,
    )


def inspect_extend_source(path: str | Path) -> ExtendInspection:
    """Probe a clip and list the problems that keep it from being extended.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    source_path = Path(path)
    if not source_path.is_file():
        raise FileNotFoundError(f"Video not found: {source_path}")
    try:
        probe = probe_video(source_path)
    except ValueError as exc:
        problem = ExtendProblem("no_video_stream", "The file has no video stream.") if str(exc) == NO_VIDEO_STREAM else ExtendProblem("unreadable", str(exc))
        return ExtendInspection(None, None, (problem,), None)
    except RuntimeError as exc:
        return ExtendInspection(None, None, (ExtendProblem("unreadable", str(exc)),), None)
    recorded = _read_recorded(source_path)
    problems = tuple(extend_source_problems(probe, recorded, audio_ok=_tail_audio_decodes(source_path, probe)))
    source = None if problems else build_extend_source(source_path, probe, recorded)
    return ExtendInspection(probe, recorded, problems, source)


def load_extend_source(path: str | Path) -> ExtendSource:
    """Return the checked source for a clip, with the scene and prompt it was made from.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the clip cannot be extended; the message names every problem.
    """
    return require_extend_source(inspect_extend_source(path), path)


def require_extend_source(inspection: ExtendInspection, path: str | Path) -> ExtendSource:
    """Return the source of an inspection that found no problems.

    Raises:
        ValueError: If the clip cannot be extended; the message names every problem.
    """
    if inspection.source is None:
        raise ValueError(f"{Path(path).name} cannot be extended: " + " ".join(problem.message for problem in inspection.problems))
    return inspection.source


def _read_recorded(path: Path) -> RecordedSettings | None:
    """Read the settings embedded in a clip; a clip without readable settings gives ``None``."""
    try:
        config = read_mp4_config(path)
        return recorded_settings(config) if isinstance(config, dict) else None
    except subprocess.CalledProcessError, OSError, ValueError, TypeError, AttributeError:
        return None


def _tail_audio_decodes(path: Path, probe: VideoProbe) -> bool:
    """Check that the audio under the carried frames decodes; a clip without audio or a known tail passes."""
    if not probe.has_audio or probe.frame_count is None or probe.fps <= 0:
        return True
    start = _stream_offset(probe) + max(0, probe.frame_count - CARRY_FRAMES) / probe.fps
    return audio_decodes(path, start=start, duration=CARRY_FRAMES / probe.fps)


def _stream_offset(probe: VideoProbe) -> float:
    """Return how long after the container's start the video stream begins, in seconds (never negative)."""
    return max(0.0, (probe.video_start or 0.0) - (probe.format_start or 0.0))


def _high_bit_depth(probe: VideoProbe) -> bool:
    return bool(probe.bits_per_sample and probe.bits_per_sample > 8) or bool(probe.pix_fmt and _HIGH_DEPTH_PIX_FMT.search(probe.pix_fmt))


def _non_blank(text: str | None) -> str | None:
    return text if text and text.strip() else None
