"""Video core types — VideoGenerationRequest, VideoWorkingArtifacts, VideoStage."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from zvisiongenerator.core.types import StageOutcome


@dataclass(frozen=True)
class VideoGenerationRequest:
    """Immutable — all user/config inputs for a single video generation."""

    # Backend + model
    backend: Any
    model: Any
    prompt: str

    # Model metadata
    model_name: str | None = None
    model_family: str = "unknown"

    # LoRA
    lora_paths: list[str] = field(default_factory=list)
    lora_weights: list[float] = field(default_factory=list)

    # Generation params
    width: int = 704
    height: int = 448
    num_frames: int = 49
    seed: int = 0
    steps: int = 8
    step_callback: Callable[[dict[str, Any]], None] | None = None

    # I2V params
    image_path: str | None = None

    # Upscale params
    upscale: int | None = None
    upscale_steps: int | None = None
    # Called with the path of a temporary video of the half-size render, between the two upscale stages; only with upscale.
    review: Callable[[Path], None] | None = None

    # Audio
    no_audio: bool = False

    # Output
    output_dir: str = "."
    output_format: str = "mp4"
    filename_base: str | None = None

    # Planned prompt text from preflight: ``{a|b}`` choices already made and the auto-enhanced rewrite, if any
    resolved_prompt: str | None = None
    enhanced_prompt: str | None = None
    on_prompt_enhanced: Callable[[str], None] | None = None

    # Scene (who and where, look, style, camera) sent before the prompt, as written / with choices made / as rewritten by the enhancer
    scene: str | None = None
    resolved_scene: str | None = None
    enhanced_scene: str | None = None
    # The user's own text, recorded when ``prompt``/``scene`` already hold an enhanced rewrite made before the job (e.g. in the Web UI)
    scene_input: str | None = None
    prompt_input: str | None = None


@dataclass
class VideoWorkingArtifacts:
    """Mutable — holds working state accumulated by video stages."""

    resolved_prompt: str | None = None
    video_path: Path | None = None
    generation_time: float = 0.0
    # Seconds spent waiting on a review; not part of ``generation_time``.
    review_wait: float = 0.0
    filename: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


VideoStage = Callable[[VideoGenerationRequest, VideoWorkingArtifacts], StageOutcome]
