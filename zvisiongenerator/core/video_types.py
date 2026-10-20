"""Video core types — VideoGenerationRequest, VideoWorkingArtifacts, VideoStage."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from zvisiongenerator.core.types import StageOutcome

if TYPE_CHECKING:
    from zvisiongenerator.utils.video_probe import VideoProbe


@dataclass(frozen=True)
class ExtendSource:
    """A checked clip to continue: its displayed geometry, timing, colour tags and the settings recorded in it."""

    path: Path
    # Displayed size: a clip stored sideways and rotated by its display matrix reports the size it is shown at.
    width: int
    height: int
    fps: float
    # The frame rate as ffprobe writes it (e.g. ``30000/1001``), passed to ffmpeg unchanged.
    fps_text: str
    frame_count: int
    has_audio: bool
    # Seconds from the start of the file to the first video frame (an input ``-ss`` is relative to the file start).
    video_start: float
    color_space: str | None = None
    color_primaries: str | None = None
    color_trc: str | None = None
    # Scene and prompt recorded in the file (the action only when a scene was recorded).
    scene: str | None = None
    prompt: str | None = None
    model: str | None = None
    lora: str | None = None
    # The probe this value was built from. The source file is never modified, so it stays valid for the whole job;
    # it is not part of the value's identity.
    probe: VideoProbe | None = field(default=None, compare=False, repr=False)

    @property
    def legacy(self) -> bool:
        """Whether the file predates scenes: a single recorded prompt that holds the scene and the action together."""
        return self.scene is None and bool(self.prompt)


@dataclass(frozen=True)
class ExtendSpec:
    """What to append to a source clip: the clip and the number of new frames (a multiple of 8)."""

    source: ExtendSource
    frames: int


@dataclass(frozen=True)
class ExtendSegment:
    """The generated part of an extension: its video and, when the source has sound, its audio as a WAV file.

    The WAV starts ``lead_samples`` before the new part: that lead-in is the model's reconstruction of the source's
    last sound, which fills the end of a source track that is shorter than its video. The new part follows it.
    """

    video_path: Path
    audio_path: Path | None = None
    lead_samples: int = 0


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
    # How the text was enhanced (options, model, tier, format), recorded in the saved video; None when it was not
    enhance_record: dict[str, Any] | None = None

    # Extend: the clip to continue and the new frames to add; ``None`` for a plain generation
    extend: ExtendSpec | None = None


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
    # One-line reason set by a stage (or the runner) when the generation fails.
    failure_message: str | None = None


VideoStage = Callable[[VideoGenerationRequest, VideoWorkingArtifacts], StageOutcome]
