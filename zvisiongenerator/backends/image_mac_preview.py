"""Render cheap, low-resolution previews of in-progress mflux latents without running the VAE."""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import numpy as np
from PIL import Image
from mflux.models.flux2.latent_creator.flux2_latent_creator import Flux2LatentCreator
from mflux.models.ideogram4.latent_creator.ideogram4_latent_creator import Ideogram4LatentCreator
from mflux.models.z_image.latent_creator.z_image_latent_creator import ZImageLatentCreator

from zvisiongenerator.core.latent_preview import FLUX2_RGB_BIAS, FLUX2_RGB_FACTORS, QWEN_IMAGE_RGB_BIAS, QWEN_IMAGE_RGB_FACTORS, ZIMAGE_RGB_BIAS, ZIMAGE_RGB_FACTORS

# The projections as ``(factors, bias)`` arrays, built once rather than per preview.
_ZIMAGE_PROJECTION = (mx.array(ZIMAGE_RGB_FACTORS, dtype=mx.float32), mx.array(ZIMAGE_RGB_BIAS, dtype=mx.float32))
_FLUX2_PROJECTION = (mx.array(FLUX2_RGB_FACTORS, dtype=mx.float32), mx.array(FLUX2_RGB_BIAS, dtype=mx.float32))
_QWEN_IMAGE_PROJECTION = (mx.array(QWEN_IMAGE_RGB_FACTORS, dtype=mx.float32), mx.array(QWEN_IMAGE_RGB_BIAS, dtype=mx.float32))


def estimate_clean_latents(previous: mx.array, current: mx.array, noise_previous: float, noise_current: float) -> mx.array:
    """Estimate the fully denoised latents from one flow-matching Euler step.

    With ``x = (1 - n) * x0 + n * noise`` and an Euler step ``current = previous + u * (n_current - n_previous)``,
    the velocity is ``u = noise - x0``, so ``x0 = current - n_current * u``. Early, noisy steps then preview the
    model's predicted image instead of mostly noise.

    Args:
        previous: Latents before the step.
        current: Latents after the step.
        noise_previous: Noise level (sigma) of ``previous``.
        noise_current: Noise level (sigma) of ``current``.

    Returns:
        The clean-latent estimate, or ``current`` when the step did not change the noise level.
    """
    step = noise_current - noise_previous
    if step == 0:
        return current
    velocity = (current.astype(mx.float32) - previous.astype(mx.float32)) / step
    return current.astype(mx.float32) - noise_current * velocity


def render_latent_preview(model: Any, family: str, latents: mx.array, height: int, width: int) -> Image.Image | None:
    """Project one family's in-loop latents to an RGB preview at 1/8 of the output resolution.

    Args:
        model: The loaded mflux model (FLUX.2 Klein needs its VAE batch-norm statistics).
        family: Image model family (``zimage``, ``flux2_klein``, ``ideogram4`` or ``krea2``).
        latents: Latents exactly as the denoising loop passes them to in-loop callbacks.
        height: Output image height in pixels.
        width: Output image width in pixels.

    Returns:
        The preview image, or ``None`` for families without a projection.
    """
    if family == "zimage":
        spatial = ZImageLatentCreator.unpack_latents(latents, height, width)[0]
        return _project_to_image(spatial, _ZIMAGE_PROJECTION)
    if family == "flux2_klein":
        return _project_to_image(_flux2_klein_spatial(model, latents, height, width), _FLUX2_PROJECTION)
    if family == "ideogram4":
        spatial = Ideogram4LatentCreator.unpack_latents(latents, height, width)[0]
        return _project_to_image(spatial, _FLUX2_PROJECTION)
    if family == "krea2":
        # Krea 2 denoises unpacked (1, 16, h, w) latents.
        return _project_to_image(latents[0], _QWEN_IMAGE_PROJECTION)
    return None


def _flux2_klein_spatial(model: Any, latents: mx.array, height: int, width: int) -> mx.array:
    """Undo FLUX.2 Klein's batch-norm and 2x2 patchify to recover ``(32, h, w)`` VAE latents."""
    packed = Flux2LatentCreator.unpack_latents(latents, height, width)[0]
    bn = model.vae.bn
    packed = packed * mx.sqrt(bn.running_var.reshape(-1, 1, 1) + bn.eps) + bn.running_mean.reshape(-1, 1, 1)
    channels, patch_h, patch_w = packed.shape
    spatial = packed.reshape(channels // 4, 2, 2, patch_h, patch_w).transpose(0, 3, 1, 4, 2)
    return spatial.reshape(channels // 4, patch_h * 2, patch_w * 2)


def _project_to_image(spatial: mx.array, projection: tuple[mx.array, mx.array]) -> Image.Image:
    """Map ``(channels, h, w)`` latents to RGB with a linear ``(factors, bias)`` projection."""
    factors, bias = projection
    rgb = mx.einsum("chw,cr->hwr", spatial.astype(mx.float32), factors) + bias
    pixels = np.array(mx.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    return Image.fromarray(pixels)
