"""Image core types — JobPrompt, ImageGenerationRequest, ImageWorkingArtifacts, ImageStage."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

from PIL import Image

from zvisiongenerator.core.types import StageOutcome

if TYPE_CHECKING:
    import numpy as np

    from zvisiongenerator.core.matte import MatteChain
    from zvisiongenerator.processing.edit_region import Plan


@dataclass(frozen=True)
class JobPrompt:
    """A prompt a job's generations condition on.

    Attributes:
        prompt: The text the generation sends.
        negative_prompt: The negative prompt, or ``None`` when the model takes none.
    """

    prompt: str
    negative_prompt: str | None


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
    # How many references the model takes; 0 means none.
    max_reference_images: int = 0

    # LoRA
    lora_paths: list[str] | None = None
    lora_weights: list[float] | None = None

    # Generation params
    negative_prompt: str | None = None
    json_prompt: bool = False
    ratio: str | None = None
    size: str | None = None
    width: int = 832
    height: int = 1216
    seed: int = 0
    steps: int = 10
    guidance: float = 0.5
    scheduler: str | None = None
    # The scheduler's config name (e.g. "beta"); ``scheduler`` may hold the backend class path it resolves to.
    scheduler_name: str | None = None
    steps_explicit: bool = False
    guidance_explicit: bool = False
    first_sigma: float | None = None
    # Quantization level the model was loaded at (4 or 8), recorded with the image.
    quantize: int | None = None
    skip_signal: Any | None = None
    step_callback: Callable[[dict[str, Any]], None] | None = None

    # Upscale params
    upscale_factor: int | None = None
    upscale_denoise: float | None = None
    upscale_steps: int | None = None
    upscale_guidance: float | None = None
    upscale_sharpen: bool = True
    upscale_save_pre: bool = False
    # Existing image an upscale job refines; set only by the upscale workflow.
    upscale_source: str | None = None
    # "creative" refines the upscaled image with the image model; "seedvr2" runs the SeedVR2 upscaler instead.
    upscale_method: str = "creative"
    # SeedVR2 only: the model ("seedvr2-3b" or "seedvr2-7b"), the bits it was loaded at, and its softness.
    upscale_model: str | None = None
    upscale_quantize: int | None = None
    upscale_softness: float | None = None
    # False when a SeedVR2 upscale's source records no seed, steps or guidance, so ``seed``/``steps``/``guidance``
    # hold class defaults that must not be saved as if they were the original's settings.
    source_settings_known: bool = True
    # Shows the small render before the upscale and returns whether to go on; set only when review is on.
    review: Callable[[Image.Image], bool] | None = None

    # Start image params (img2img)
    image_path: str | None = None
    image_strength: float = 0.5
    # References: images whose content the result reuses, in prompt order; independent of the start image.
    reference_paths: tuple[str, ...] = ()

    # Edit params: the image to edit, the painted area (bool, the picture's upright size; set only with ``edit_source``), and the
    # ``(side, amount)`` to extend by
    edit_source: str | None = None
    edit_paint: np.ndarray | None = field(default=None, compare=False, repr=False)
    edit_extend: tuple[str, int] | None = None

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
    transparent: bool = False
    # The job's background removal chain, and the hook the matte stage calls before the chain loads to free memory for
    # it; the hook gets the pixel count of the image about to be cut out
    matte: MatteChain | None = None
    make_room: Callable[[int], None] | None = None

    # Output
    output_dir: str = "."
    filename_base: str | None = None

    # Planned prompt text from preflight: ``{a|b}`` choices already made and the auto-enhanced rewrite, if any
    resolved_prompt: str | None = None
    enhanced_prompt: str | None = None
    on_prompt_enhanced: Callable[[str], None] | None = None
    # The user's own text, recorded when ``prompt`` already holds an enhanced rewrite made before the job (e.g. in the Web UI)
    prompt_input: str | None = None
    # How the prompt was enhanced (options, model, tier, format), recorded in the saved image; None when it was not
    enhance_record: dict[str, Any] | None = None


@dataclass
class EditWork:
    """Mutable — what the edit stages hand one another for a single edit.

    Attributes:
        canvas: The full-size RGB picture the result is pasted onto (the source, or the source on its extended canvas).
        cutout: The source as RGBA with its own colours (not flattened on the backdrop), or ``None`` when the source is opaque.
        paint: The area to redraw (bool, canvas size), or ``None`` for a whole-picture edit or an Extend.
        crop: The part of the canvas the model works on, at ``target`` size. For an Extend, the source scaled to about one megapixel with the new strip beside it, in whole 16 px cells.
        crop_paint: The paint of that crop (bool, ``target`` size).
        reference: The picture the model sees while it draws: the crop, or for an Extend the drawn canvas with a blurred strip.
        plan: Where the crop sits on the canvas, and the size the model draws it at; ``None`` for an Extend.
        drawn_source_box: Where the source sits on the Extend's crop ``(left, top, right, bottom)``, or ``None`` without an Extend.
        source_box: Where the source sits on an extended canvas ``(left, top, right, bottom)``, or ``None`` without an Extend.
    """

    canvas: Image.Image
    cutout: Image.Image | None = None
    paint: np.ndarray | None = None
    crop: Image.Image | None = None
    crop_paint: np.ndarray | None = None
    reference: Image.Image | None = None
    plan: Plan | None = None
    source_box: tuple[int, int, int, int] | None = None
    drawn_source_box: tuple[int, int, int, int] | None = None


@dataclass
class ImageWorkingArtifacts:
    """Mutable — holds working state accumulated by image stages."""

    image: Image.Image | None = None
    # The loaded references, each at its own size; ``image`` stays the start image.
    reference_images: list[Image.Image] = field(default_factory=list)
    resolved_prompt: str | None = None
    generation_time: float = 0.0
    filename: str | None = None
    filepath: str | None = None
    was_upscaled: bool = False
    # The working state of an edit job; ``None`` for every other workflow.
    edit: EditWork | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


ImageStage = Callable[[ImageGenerationRequest, ImageWorkingArtifacts], StageOutcome]
