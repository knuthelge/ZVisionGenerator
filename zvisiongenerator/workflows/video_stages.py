"""Video workflow stage functions.

Each stage has signature:
    (request: VideoGenerationRequest, artifacts: VideoWorkingArtifacts) -> StageOutcome

Stages are pure functions operating on the immutable request and mutable artifacts.
"""

from __future__ import annotations

import subprocess
import time
import warnings
from pathlib import Path

from zvisiongenerator.core.types import StageOutcome
from zvisiongenerator.core.video_types import VideoGenerationRequest, VideoWorkingArtifacts
from zvisiongenerator.utils.ffmpeg import strip_audio
from zvisiongenerator.utils.prompt_compose import expand_random_choices


def resolve_prompt_stage(request: VideoGenerationRequest, artifacts: VideoWorkingArtifacts) -> StageOutcome:
    """Replace {a|b|c} random choice blocks in video prompt, supporting nesting."""
    artifacts.resolved_prompt = expand_random_choices(request.prompt)
    return StageOutcome.success


def generate_filename_stage(request: VideoGenerationRequest, artifacts: VideoWorkingArtifacts) -> StageOutcome:
    """Populate artifacts.filename using the shared filename generator.

    Produces: {set_name}_{timestamp}_{WxH}_{frames}f_{model}_{lora}_steps{N}_seed{S}.{format}
    """
    from zvisiongenerator.utils.filename import generate_filename

    base = generate_filename(
        set_name=request.filename_base,
        width=request.width,
        height=request.height,
        seed=request.seed,
        steps=request.steps,
        model=request.model_name,
        lora_paths=request.lora_paths or None,
        lora_weights=request.lora_weights or None,
        num_frames=request.num_frames,
    )
    artifacts.filename = f"{base}.{request.output_format}"
    return StageOutcome.success


def text_to_video_stage(request: VideoGenerationRequest, artifacts: VideoWorkingArtifacts) -> StageOutcome:
    """Call backend.text_to_video(), store path in artifacts."""
    prompt = artifacts.resolved_prompt or request.prompt
    prompt_one_line = " ".join(prompt.split())
    print(f"  Prompt: {prompt_one_line[:100]}{'...' if len(prompt_one_line) > 100 else ''}")
    output_path = str(Path(request.output_dir) / artifacts.filename)
    kwargs: dict[str, object] = {}
    if request.upscale:
        if request.upscale_steps is None:
            msg = "upscale_steps must be set when upscale is enabled"
            raise ValueError(msg)
        kwargs["stage1_steps"] = request.upscale_steps
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
    artifacts.generation_time = time.monotonic() - t0
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
    kwargs: dict[str, object] = {}
    if request.upscale:
        if request.upscale_steps is None:
            msg = "upscale_steps must be set when upscale is enabled"
            raise ValueError(msg)
        kwargs["stage1_steps"] = request.upscale_steps
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
    artifacts.generation_time = time.monotonic() - t0
    if result is None:
        return StageOutcome.failed
    artifacts.video_path = result
    return StageOutcome.success


def log_video_stage(request: VideoGenerationRequest, artifacts: VideoWorkingArtifacts) -> StageOutcome:
    """Print video generation info to console."""
    path = artifacts.video_path or "unknown"
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
