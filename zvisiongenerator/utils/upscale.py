"""Upscale sizing, limits and defaults shared by in-run upscaling and upscaling existing images."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from zvisiongenerator.utils.alignment import round_to_alignment

UPSCALE_FACTORS: tuple[int, ...] = (2, 4)
DEFAULT_MAX_MEGAPIXELS = 20.0
_DEFAULT_DENOISE = {2: 0.3, 4: 0.4}
_DEFAULT_EXISTING_DENOISE_SMALL = 0.4
_DEFAULT_EXISTING_DENOISE_LARGE = 0.2
_DEFAULT_EXISTING_LARGE_MEGAPIXELS = 2.0


@dataclass(frozen=True)
class UpscaleOption:
    """One upscale factor for a given image: its output size and whether it may run."""

    factor: int
    width: int
    height: int
    allowed: bool
    reason: str | None = None


def upscale_output_size(width: int, height: int, factor: int) -> tuple[int, int]:
    """Return the size ``upscale_stage`` produces: each side times *factor*, rounded to 16-pixel alignment."""
    return round_to_alignment(width * factor), round_to_alignment(height * factor)


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
