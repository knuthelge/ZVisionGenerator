"""Upscale sizing, limits and defaults shared by in-run upscaling and upscaling existing images."""

from __future__ import annotations

from dataclasses import dataclass
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


@dataclass(frozen=True)
class UpscaleOption:
    """One upscale factor for a given image: its output size, whether it may run and any caution to show."""

    factor: int
    width: int
    height: int
    allowed: bool
    reason: str | None = None
    warning: str | None = None


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
        options.append(UpscaleOption(factor=factor, width=out_width, height=out_height, allowed=True, warning=SEEDVR2_MEMORY_WARNING if over_budget else None))
    return options


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
