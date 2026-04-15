"""Image core types — ImageGenerationRequest, ImageWorkingArtifacts, ImageStage."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from PIL import Image

from zvisiongenerator.core.types import StageOutcome


@dataclass(frozen=True)
class ImageGenerationRequest:
    """Immutable — all user/config inputs for a single image generation."""

    # Backend + model
    backend: Any
    model: Any
    prompt: str

    # Model metadata
    model_name: str | None = None
    model_family: str = "unknown"
    supports_negative_prompt: bool = False

    # LoRA
    lora_paths: list[str] | None = None
    lora_weights: list[float] | None = None

    # Generation params
    negative_prompt: str | None = None
    ratio: str | None = None
    size: str | None = None
    width: int = 832
    height: int = 1216
    seed: int = 0
    steps: int = 10
    guidance: float = 0.5
    scheduler: str | None = None
    skip_signal: Any | None = None

    # Upscale params
    upscale_factor: int | None = None
    upscale_denoise: float | None = None
    upscale_steps: int | None = None
    upscale_guidance: float | None = None
    upscale_sharpen: bool = True
    upscale_save_pre: bool = False

    # Reference image params
    image_path: str | None = None
    image_strength: float = 0.5

    # Sharpening (from config)
    sharpen_amount_normal: float = 0.8
    sharpen_amount_upscaled: float = 1.2
    sharpen_amount_pre_upscale: float = 0.4

    # Post-processing flags
    sharpen: bool = True
    sharpen_amount_override: float | None = None
    contrast: bool = False
    contrast_amount: float = 1.0
    saturation: bool = False
    saturation_amount: float = 1.0

    # Output
    output_dir: str = "."
    filename_base: str | None = None


@dataclass
class ImageWorkingArtifacts:
    """Mutable — holds working state accumulated by image stages."""

    image: Image.Image | None = None
    resolved_prompt: str | None = None
    generation_time: float = 0.0
    filename: str | None = None
    filepath: str | None = None
    was_upscaled: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


ImageStage = Callable[[ImageGenerationRequest, ImageWorkingArtifacts], StageOutcome]
