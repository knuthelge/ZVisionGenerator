"""Upscale sizing, limits and defaults shared by in-run upscaling and upscaling existing images."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from zvisiongenerator.utils.alignment import round_to_alignment
from zvisiongenerator.utils.seedvr2 import SeedVR2Tier, estimate_seedvr2_bytes, seedvr2_over_budget

if TYPE_CHECKING:
    from PIL import Image

UPSCALE_FACTORS: tuple[int, ...] = (2, 4)
DEFAULT_MAX_MEGAPIXELS = 20.0
_DEFAULT_DENOISE = {2: 0.3, 4: 0.4}
_DEFAULT_EXISTING_DENOISE_SMALL = 0.4
_DEFAULT_EXISTING_DENOISE_LARGE = 0.2
_DEFAULT_EXISTING_LARGE_MEGAPIXELS = 2.0
SEEDVR2_MEMORY_WARNING = "May need more memory than this computer has: it may be slow or fail."
VIDEO_ONE_FRAME_WARNING = "This size is upscaled one frame at a time on this computer, so it is very slow and the motion is less steady than at smaller sizes."
# The ``UpscaleOption.warning_code`` values: what the warning is about.
WARNING_MEMORY = "memory"
WARNING_ONE_FRAME_BATCH = "one_frame_batch"
VIDEO_UPSCALE_FACTOR = 2
VIDEO_UPSCALE_SEED = 42
WORK_DIR_PREFIX = ".ziv-tmp-upscale-"


# The frames per batch a video upscale may use are 4n+1 (the model packs 4 frames into one latent frame) up to this many. A card
# whose memory is unknown gets the default.
_VIDEO_MAX_BATCH = 29
_VIDEO_DEFAULT_BATCH = 5
_VIDEO_BATCH_STEP = 4
# The torch peak of one batch is the larger of two terms, fitted to what was measured on a 10 GB card (7B FP8, every block
# streamed from system memory, the VAE in 512-pixel tiles); the DiT's activations grow with the output megapixels times the
# latent frames in a batch, while the tiled VAE does not depend on the size:
#   DiT  = 0.6 + 0.535 GiB per megapixel per latent frame (a batch of 1 at 3.0, 8.3 and 14.8 MP peaked at 3.45, 4.92 and 8.33 GiB,
#          a batch of 5 at 4.2, 5.5 and 6.9 MP at 5.01, 6.39 and 7.85, a batch of 13 at 3.6 MP at 8.25)
#   VAE  = 3.45 GiB for 1 latent frame at most, 4.97 for 2 (a batch of 5), 6.4 for 3 or more (batches of 9 to 29 at 1.4 to 2.4 MP:
#          6.37 to 6.40); 3B peaks alike. Batches that ran out of memory: 2, 3 and 5 at 8.3 MP, 29 at 2.4 MP.
# The card then holds the CUDA context and other programs on top, up to 2.4 GB in these runs, which ``weights_gb`` plus the headroom
# below cover. The estimate is at or above the card peak of every run that finished. The VAE tile size (256 to 1024 pixels)
# does not change the peak; smaller tiles only run slower.
_DIT_BASE_GIB = 0.6
_DIT_GIB_PER_MP_LATENT = 0.535
_VAE_PEAK_GIB = {1: 3.45, 2: 4.97}
_VAE_PEAK_GIB_LARGE = 6.4
_VIDEO_HEADROOM_GB = 1.0
_GIB_TO_GB = 1.073741824


@dataclass(frozen=True)
class UpscaleOption:
    """One upscale factor for a given image: its output size, whether it may run and any caution to show."""

    factor: int
    width: int
    height: int
    allowed: bool
    reason: str | None = None
    warning: str | None = None
    # Why there is a warning (``WARNING_MEMORY`` or ``WARNING_ONE_FRAME_BATCH``), for clients that act on it; ``None`` without one.
    warning_code: str | None = None


def upscale_output_size(width: int, height: int, factor: int) -> tuple[int, int]:
    """Return the size ``upscale_stage`` produces: each side times *factor*, rounded to 16-pixel alignment."""
    return round_to_alignment(width * factor), round_to_alignment(height * factor)


def seedvr2_output_size(width: int, height: int, factor: int) -> tuple[int, int]:
    """Return the size SeedVR2 produces: each side times *factor*, rounded down to an even number (as mflux does)."""
    return int(width * factor) // 2 * 2, int(height * factor) // 2 * 2


def seedvr2_input_image(image: Image.Image, width: int, height: int, softness: float) -> Image.Image:
    """Return *image* as RGB at *width* × *height* for SeedVR2, pre-blurred by *softness* (0-1) the way mflux does it.

    Softness maps to a factor from 1 to 8: above 1 the image is first shrunk by that factor, then enlarged to the
    target, both bicubic; at 0 it is only enlarged.
    """
    from PIL import Image

    rgb = image.convert("RGB")
    factor = 1.0 + max(0.0, min(1.0, softness)) * 7.0
    if factor > 1.0:
        rgb = rgb.resize((max(2, int(width / factor)), max(2, int(height / factor))), Image.Resampling.BICUBIC)
    return rgb.resize((width, height), Image.Resampling.BICUBIC)


def seedvr2_options(width: int, height: int, tier: SeedVR2Tier, *, budget_bytes: int | None) -> list[UpscaleOption]:
    """Return every upscale factor for a *width* × *height* image under the SeedVR2 plain upscale.

    No factor is ever disabled; one that needs more memory than *budget_bytes* carries a warning instead.

    Args:
        width: Source image width in pixels.
        height: Source image height in pixels.
        tier: The SeedVR2 tier whose memory cost applies.
        budget_bytes: Accelerator memory this machine may use, or ``None`` when unknown (no warning then).
    """
    options: list[UpscaleOption] = []
    for factor in UPSCALE_FACTORS:
        out_width, out_height = seedvr2_output_size(width, height, factor)
        over_budget = seedvr2_over_budget(estimate_seedvr2_bytes(tier, out_width * out_height / 1e6), budget_bytes)
        options.append(
            UpscaleOption(
                factor=factor, width=out_width, height=out_height, allowed=True, warning=SEEDVR2_MEMORY_WARNING if over_budget else None, warning_code=WARNING_MEMORY if over_budget else None
            )
        )
    return options


def estimate_seedvr2_video_bytes(tier: SeedVR2Tier, megapixels: float, batch_size: int) -> int:
    """Return the accelerator memory the streamed video upscale needs for an output of *megapixels* at *batch_size* frames per batch, in bytes.

    Clip length never changes it, because the frames stream.
    """
    latent_frames = -(-(batch_size - 1) // _VIDEO_BATCH_STEP) + 1  # the core pads a batch up to 4n+1 frames
    vae_gib = _VAE_PEAK_GIB.get(latent_frames, _VAE_PEAK_GIB_LARGE)
    dit_gib = _DIT_BASE_GIB + _DIT_GIB_PER_MP_LATENT * megapixels * latent_frames
    return int((tier.weights_gb + _VIDEO_HEADROOM_GB + max(vae_gib, dit_gib) * _GIB_TO_GB) * 1e9)


def video_upscale_batch_size(width: int, height: int, tier: SeedVR2Tier, *, budget_bytes: int | None) -> int:
    """Return the frames per SeedVR2 batch for a video upscaled to *width* × *height* (the output size) on a card with *budget_bytes*.

    It is the largest batch whose estimated memory fits the budget, the same test that marks a size as tight; larger batches
    mean fewer seams between batches and less time per frame. One frame is the smallest batch, used even when it does not fit.
    With an unknown budget the batch is 5.
    """
    if budget_bytes is None:
        return _VIDEO_DEFAULT_BATCH
    megapixels = width * height / 1e6
    for batch_size in range(_VIDEO_MAX_BATCH, 1, -_VIDEO_BATCH_STEP):
        if not seedvr2_over_budget(estimate_seedvr2_video_bytes(tier, megapixels, batch_size), budget_bytes):
            return batch_size
    return 1


def seedvr2_video_option(width: int, height: int, tier: SeedVR2Tier, *, budget_bytes: int | None) -> UpscaleOption:
    """Return the one 2× option for a *width* × *height* video under the SeedVR2 video upscale.

    It is never disabled. At the batch ``video_upscale_batch_size`` picks, it carries a memory warning when the estimate is
    over *budget_bytes*, and otherwise a slow-and-less-steady warning when that batch is a single frame.

    Args:
        width: Source video width in pixels, as displayed.
        height: Source video height in pixels, as displayed.
        tier: The SeedVR2 tier whose memory cost applies.
        budget_bytes: Accelerator memory this machine may use, or ``None`` when unknown (no warning then).
    """
    out_width, out_height = seedvr2_output_size(width, height, VIDEO_UPSCALE_FACTOR)
    option = UpscaleOption(factor=VIDEO_UPSCALE_FACTOR, width=out_width, height=out_height, allowed=True)
    if budget_bytes is None:
        return option
    batch_size = video_upscale_batch_size(out_width, out_height, tier, budget_bytes=budget_bytes)
    if seedvr2_over_budget(estimate_seedvr2_video_bytes(tier, out_width * out_height / 1e6, batch_size), budget_bytes):
        return replace(option, warning=SEEDVR2_MEMORY_WARNING, warning_code=WARNING_MEMORY)
    if batch_size == 1:
        return replace(option, warning=VIDEO_ONE_FRAME_WARNING, warning_code=WARNING_ONE_FRAME_BATCH)
    return option


def upscale_options(width: int, height: int, capabilities: dict[str, Any], config: dict[str, Any]) -> list[UpscaleOption]:
    """Return every upscale factor for a *width* × *height* image, with the reason when one is not allowed.

    Args:
        width: Source image width in pixels.
        height: Source image height in pixels.
        capabilities: Resolved model defaults (``supports_upscale``, ``supports_img2img``, ``dimension_max``).
        config: Loaded config dict; ``upscale.max_megapixels`` caps the output size.
    """
    unsupported = _unsupported_reason(capabilities)
    dimension_max = capabilities.get("dimension_max")
    megapixels = max_megapixels(config)
    options: list[UpscaleOption] = []
    for factor in UPSCALE_FACTORS:
        out_width, out_height = upscale_output_size(width, height, factor)
        reason = unsupported
        if reason is None and dimension_max is not None and max(out_width, out_height) > dimension_max:
            reason = f"{out_width}×{out_height} is larger than this model's {dimension_max}-pixel limit."
        if reason is None and out_width * out_height > megapixels * 1_000_000:
            reason = f"{out_width}×{out_height} is over the {megapixels:g}-megapixel upscale limit."
        options.append(UpscaleOption(factor=factor, width=out_width, height=out_height, allowed=reason is None, reason=reason))
    return options


def max_megapixels(config: dict[str, Any]) -> float:
    """Return ``upscale.max_megapixels`` from *config*, or the default of 20."""
    value = config.get("upscale", {}).get("max_megapixels", DEFAULT_MAX_MEGAPIXELS)
    if isinstance(value, bool) or not isinstance(value, int | float) or value <= 0:
        raise ValueError(f"config 'upscale.max_megapixels' must be a positive number, got {value!r}.")
    return float(value)


def default_upscale_denoise(config: dict[str, Any], factor: int) -> float:
    """Return the configured refinement denoise for *factor* (``upscale.default_denoise_2x`` / ``_4x``)."""
    return config.get("upscale", {}).get(f"default_denoise_{factor}x", _DEFAULT_DENOISE.get(factor, 0.3))


def existing_image_denoise(config: dict[str, Any], width: int, height: int) -> float:
    """Return the refinement denoise for upscaling a finished image to *width* × *height*.

    Chosen by output size, not factor: models are trained around 1–2 megapixels, and above that a high denoise
    invents new detail instead of sharpening what is there. Outputs up to ``upscale.existing_large_megapixels``
    use ``existing_denoise_small``, larger ones ``existing_denoise_large``. So a 2× of a 2× gets the same light
    touch as a direct 4× to the same size. Kept apart from the in-run defaults, which refine a small render.
    """
    upscale = config.get("upscale", {})
    threshold = upscale.get("existing_large_megapixels", _DEFAULT_EXISTING_LARGE_MEGAPIXELS) * 1_000_000
    if width * height > threshold:
        return upscale.get("existing_denoise_large", _DEFAULT_EXISTING_DENOISE_LARGE)
    return upscale.get("existing_denoise_small", _DEFAULT_EXISTING_DENOISE_SMALL)


def _unsupported_reason(capabilities: dict[str, Any]) -> str | None:
    if not capabilities.get("supports_upscale", True):
        return "This model does not support upscaling."
    if not capabilities.get("supports_img2img", True):
        return "This model cannot refine an existing image."
    return None
