"""Video workflow stage functions.

Each stage has signature:
    (request: VideoGenerationRequest, artifacts: VideoWorkingArtifacts) -> StageOutcome

Stages are pure functions operating on the immutable request and mutable artifacts.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import time
import warnings
from collections.abc import Callable
from pathlib import Path

from zvisiongenerator.core.types import StageOutcome
from zvisiongenerator.core.video_types import VideoGenerationRequest, VideoWorkingArtifacts
from zvisiongenerator.utils.ffmpeg import strip_audio
from zvisiongenerator.utils.provenance import build_video_config_payload, embed_mp4_config
from zvisiongenerator.utils.prompt_compose import compose_video_prompt, expand_random_choices, strip_quotes_while_original
from zvisiongenerator.utils.video_extend import extend_audio_output
from zvisiongenerator.utils.video_splice import splice_clips
from zvisiongenerator.workflows.enhance_stage import apply_planned_enhancement


def _upscale_kwargs(request: VideoGenerationRequest, artifacts: VideoWorkingArtifacts) -> dict[str, object]:
    """Return the backend keyword arguments of the upscale: ``stage1_steps``, and ``review`` when one is set.

    The review callable is wrapped so that the time it blocks is added to ``artifacts.review_wait``.

    Raises:
        ValueError: If upscale is on without ``upscale_steps``.
    """
    if not request.upscale:
        return {}
    if request.upscale_steps is None:
        msg = "upscale_steps must be set when upscale is enabled"
        raise ValueError(msg)
    kwargs: dict[str, object] = {"stage1_steps": request.upscale_steps}
    if request.review is not None:
        kwargs["review"] = _timed_review(request.review, artifacts)
    return kwargs


def _timed_review(review: Callable[[Path], None], artifacts: VideoWorkingArtifacts) -> Callable[[Path], None]:
    """Wrap *review* so the seconds it blocks, also when it raises, are added to ``artifacts.review_wait``."""

    def _timed(path: Path) -> None:
        started = time.monotonic()
        try:
            review(path)
        finally:
            artifacts.review_wait += time.monotonic() - started

    return _timed


def resolve_prompt_stage(request: VideoGenerationRequest, artifacts: VideoWorkingArtifacts) -> StageOutcome:
    """Resolve the video's scene and prompt to the text sent to the model; planned ``request.resolved_*`` values win.

    ``{a|b|c}`` choice blocks are expanded in each part, supporting nesting. ``artifacts.resolved_prompt`` is the
    scene and prompt composed; the scene alone is recorded as ``metadata["scene"]`` when there is one.

    When extending a clip recorded with a single prompt (no scene) and the prompt is still that original, its quoted
    sentences are dropped so the old dialogue is not repeated. The action sent is recorded as ``metadata["prompt_sent"]``.
    """
    prompt = request.resolved_prompt if request.resolved_prompt is not None else expand_random_choices(request.prompt)
    scene = request.resolved_scene if request.resolved_scene is not None else expand_random_choices(request.scene or "")
    if request.extend is not None:
        prompt = _extend_action(request, prompt)
        artifacts.metadata["prompt_sent"] = prompt
    artifacts.resolved_prompt = compose_video_prompt(scene, prompt)
    if scene.strip():
        artifacts.metadata["scene"] = scene
    return StageOutcome.success


def _extend_action(request: VideoGenerationRequest, resolved_prompt: str) -> str:
    """Return the action to send when extending: *resolved_prompt*, without quoted sentences while a legacy source's prompt is unchanged."""
    source = request.extend.source
    if not source.legacy:
        return resolved_prompt
    return strip_quotes_while_original(resolved_prompt, template=request.prompt, recorded_prompt=source.prompt)


def enhance_prompt_stage(request: VideoGenerationRequest, artifacts: VideoWorkingArtifacts) -> StageOutcome:
    """Apply the video scene and action rewrite planned during preflight; never fails the generation.

    The action alone is stored as ``metadata["enhanced_prompt"]`` and the scene alone as ``metadata["enhanced_scene"]``;
    only ``artifacts.resolved_prompt`` (and the ``on_prompt_enhanced`` text) holds them composed. No LLM runs here.
    """
    action = request.enhanced_prompt
    if action is None:
        return StageOutcome.success
    scene = request.enhanced_scene
    apply_planned_enhancement(
        request,
        artifacts,
        resolved_prompt=compose_video_prompt(scene, action),
        extra_metadata={"enhanced_scene": scene} if scene and scene.strip() else None,
    )
    return StageOutcome.success


def generate_filename_stage(request: VideoGenerationRequest, artifacts: VideoWorkingArtifacts) -> StageOutcome:
    """Populate artifacts.filename as ``{set_name}_{timestamp}.{format}``, with a counter when that name is taken."""
    from zvisiongenerator.utils.filename import generate_filename, unique_output_path

    base = generate_filename(request.filename_base)
    artifacts.filename = unique_output_path(request.output_dir, base, f".{request.output_format}").name
    return StageOutcome.success


def text_to_video_stage(request: VideoGenerationRequest, artifacts: VideoWorkingArtifacts) -> StageOutcome:
    """Call backend.text_to_video(), store path in artifacts."""
    prompt = artifacts.resolved_prompt or request.prompt
    prompt_one_line = " ".join(prompt.split())
    print(f"  Prompt: {prompt_one_line[:100]}{'...' if len(prompt_one_line) > 100 else ''}")
    output_path = str(Path(request.output_dir) / artifacts.filename)
    kwargs = _upscale_kwargs(request, artifacts)
    t0 = time.monotonic()
    result = request.backend.text_to_video(
        model=request.model,
        prompt=artifacts.resolved_prompt or request.prompt,
        width=request.width,
        height=request.height,
        num_frames=request.num_frames,
        seed=request.seed,
        steps=request.steps,
        output_path=output_path,
        step_callback=request.step_callback,
        **kwargs,
    )
    artifacts.generation_time = time.monotonic() - t0 - artifacts.review_wait
    if result is None:
        return StageOutcome.failed
    artifacts.video_path = result
    return StageOutcome.success


def image_to_video_stage(request: VideoGenerationRequest, artifacts: VideoWorkingArtifacts) -> StageOutcome:
    """Call backend.image_to_video(), store path in artifacts."""
    if request.image_path is None:
        msg = "image_to_video_stage requires image_path to be set"
        raise ValueError(msg)
    prompt = artifacts.resolved_prompt or request.prompt
    prompt_one_line = " ".join(prompt.split())
    print(f"  Prompt: {prompt_one_line[:100]}{'...' if len(prompt_one_line) > 100 else ''}")
    output_path = str(Path(request.output_dir) / artifacts.filename)
    kwargs = _upscale_kwargs(request, artifacts)
    t0 = time.monotonic()
    result = request.backend.image_to_video(
        model=request.model,
        image_path=request.image_path,
        prompt=artifacts.resolved_prompt or request.prompt,
        width=request.width,
        height=request.height,
        num_frames=request.num_frames,
        seed=request.seed,
        steps=request.steps,
        output_path=output_path,
        step_callback=request.step_callback,
        **kwargs,
    )
    artifacts.generation_time = time.monotonic() - t0 - artifacts.review_wait
    if result is None:
        return StageOutcome.failed
    artifacts.video_path = result
    return StageOutcome.success


def extend_video_stage(request: VideoGenerationRequest, artifacts: VideoWorkingArtifacts) -> StageOutcome:
    """Call backend.extend_video() to generate the new part of the clip, and keep it for ``join_clips_stage``.

    The segment is written to a directory in the system temp folder; ``artifacts.metadata`` holds the
    directory as ``extend_segment_dir`` and the result as ``extend_segment``. The directory is removed again when the
    backend fails or returns nothing; otherwise ``join_clips_stage`` removes it.

    Raises:
        ValueError: If the request has no ``extend``.
        RuntimeError: If the backend cannot extend videos.
    """
    extend = request.extend
    if extend is None:
        raise ValueError("extend_video_stage requires request.extend to be set")
    extender = getattr(request.backend, "extend_video", None)
    if not callable(extender):
        raise RuntimeError("This video backend cannot extend videos.")
    prompt = artifacts.resolved_prompt if artifacts.resolved_prompt is not None else request.prompt
    prompt_one_line = " ".join(prompt.split())
    print(f"  Prompt: {prompt_one_line[:100]}{'...' if len(prompt_one_line) > 100 else ''}")
    Path(request.output_dir).mkdir(parents=True, exist_ok=True)
    segment_dir = tempfile.mkdtemp(prefix="ziv-extend-")
    t0 = time.monotonic()
    try:
        segment = extender(
            model=request.model,
            source=extend.source,
            prompt=prompt,
            extend_frames=extend.frames,
            seed=request.seed,
            steps=request.steps,
            segment_dir=segment_dir,
            audio_output=_extend_has_audio(request),
            step_callback=request.step_callback,
        )
    except BaseException:
        shutil.rmtree(segment_dir, ignore_errors=True)
        raise
    artifacts.generation_time = time.monotonic() - t0
    if segment is None:
        shutil.rmtree(segment_dir, ignore_errors=True)
        return StageOutcome.failed
    artifacts.metadata["extend_segment"] = segment
    artifacts.metadata["extend_segment_dir"] = segment_dir
    return StageOutcome.success


def join_clips_stage(request: VideoGenerationRequest, artifacts: VideoWorkingArtifacts) -> StageOutcome:
    """Splice the generated segment onto the source clip into the output file; remove the segment directory afterwards.

    A failing splice warns and fails the generation.

    Raises:
        ValueError: If the request has no ``extend``, or no segment or output filename is set.
    """
    extend = request.extend
    segment = artifacts.metadata.pop("extend_segment", None)
    segment_dir = artifacts.metadata.pop("extend_segment_dir", None)
    try:
        if extend is None or segment is None or artifacts.filename is None:
            raise ValueError("join_clips_stage requires request.extend, a generated segment and an output filename")
        output = Path(request.output_dir) / artifacts.filename
        try:
            artifacts.video_path = splice_clips(extend.source, segment, output, extend_frames=extend.frames, include_audio=_extend_has_audio(request))
        except (subprocess.CalledProcessError, RuntimeError) as exc:
            detail = exc.stderr.decode("utf-8", "replace").strip() if isinstance(exc, subprocess.CalledProcessError) and exc.stderr else str(exc)
            warnings.warn(f"Joining the clips failed: {detail}", stacklevel=2)
            return StageOutcome.failed
    finally:
        if segment_dir is not None:
            shutil.rmtree(segment_dir, ignore_errors=True)
    return StageOutcome.success


def _extend_has_audio(request: VideoGenerationRequest) -> bool:
    """Return whether an extension carries sound: the source has audio and the job did not ask for none."""
    return extend_audio_output(request.extend.source, no_audio=request.no_audio)


def log_video_stage(request: VideoGenerationRequest, artifacts: VideoWorkingArtifacts) -> StageOutcome:
    """Embed config metadata in the video file and print generation info to console."""
    path = artifacts.video_path or "unknown"
    if artifacts.video_path is not None:
        payload = build_video_config_payload(request, artifacts)
        try:
            embed_mp4_config(artifacts.video_path, payload)
        except subprocess.CalledProcessError as e:
            err_msg = e.stderr.decode("utf-8", "replace").strip() if e.stderr else str(e)
            warnings.warn(f"ffmpeg metadata embed failed: {err_msg}", stacklevel=2)
    t = artifacts.generation_time
    print(f"  Video saved: {path}  ({t:.1f}s)")
    return StageOutcome.success


def strip_audio_stage(request: VideoGenerationRequest, artifacts: VideoWorkingArtifacts) -> StageOutcome:
    """Strip audio track from generated video when no_audio is set."""
    if not request.no_audio:
        return StageOutcome.success
    if artifacts.video_path is None:
        return StageOutcome.success
    try:
        strip_audio(artifacts.video_path)
    except subprocess.CalledProcessError as e:
        err_msg = e.stderr.decode("utf-8", "replace").strip() if e.stderr else str(e)
        warnings.warn(f"ffmpeg strip-audio failed: {err_msg}", stacklevel=2)
        return StageOutcome.failed
    return StageOutcome.success
