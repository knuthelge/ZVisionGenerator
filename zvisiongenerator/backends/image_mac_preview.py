"""Render cheap, low-resolution previews of in-progress mflux latents without running the VAE."""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import numpy as np
from PIL import Image
from mflux.models.flux2.latent_creator.flux2_latent_creator import Flux2LatentCreator
from mflux.models.ideogram4.latent_creator.ideogram4_latent_creator import Ideogram4LatentCreator
from mflux.models.z_image.latent_creator.z_image_latent_creator import ZImageLatentCreator

# Linear latent -> RGB projections (rows: latent channels, columns: R, G, B in [0, 1]).
# Least-squares fits against each VAE's own encodings of a set of sample images, so a
# preview costs one small matrix multiply instead of a VAE decode.
_ZIMAGE_RGB_FACTORS = (
    (-0.0132, 0.0213, 0.0386),
    (0.0131, 0.0276, 0.0478),
    (0.0276, -0.0267, -0.0111),
    (-0.0132, 0.0016, 0.0221),
    (0.0462, 0.0395, 0.0195),
    (-0.0167, 0.0044, -0.0024),
    (0.0275, 0.0523, 0.0476),
    (-0.0332, -0.0279, -0.0254),
    (-0.0256, -0.0039, 0.0492),
    (0.0451, 0.0385, -0.0184),
    (0.0061, 0.0421, 0.0299),
    (0.0569, 0.0285, 0.0249),
    (0.0391, 0.0274, 0.0339),
    (-0.0620, -0.0148, -0.0596),
    (-0.0067, -0.0412, -0.0152),
    (-0.0711, -0.0506, -0.0394),
)
_ZIMAGE_RGB_BIAS = (0.4974, 0.4753, 0.4546)

# The FLUX.2 VAE is shared by FLUX.2 Klein and Ideogram 4 (identical fits for both).
_FLUX2_RGB_FACTORS = (
    (-0.0003, 0.0022, -0.0005),
    (0.0020, -0.0089, 0.0104),
    (-0.0019, 0.0084, 0.0218),
    (0.0469, 0.0748, 0.0993),
    (0.0146, -0.0051, -0.0030),
    (-0.0001, -0.0042, -0.0101),
    (-0.0238, 0.0234, -0.0181),
    (0.0032, 0.0011, -0.0020),
    (-0.0855, -0.0638, -0.0301),
    (0.0040, -0.0029, 0.0058),
    (-0.0011, 0.0071, 0.0038),
    (0.0171, 0.0059, -0.0254),
    (0.0002, 0.0034, 0.0038),
    (0.0001, -0.0024, -0.0027),
    (0.0021, 0.0099, 0.0034),
    (-0.0365, 0.0109, -0.0012),
    (0.0062, 0.0122, 0.0033),
    (0.0109, 0.0008, -0.0010),
    (0.0021, -0.0009, -0.0017),
    (0.0304, -0.0281, -0.0016),
    (0.0003, 0.0032, 0.0045),
    (0.0002, -0.0087, -0.0056),
    (-0.0053, 0.0012, -0.0125),
    (0.0036, 0.0032, 0.0058),
    (0.0036, 0.0134, 0.0105),
    (-0.0041, -0.0009, -0.0036),
    (0.0062, -0.0078, 0.0019),
    (0.0011, -0.0037, -0.0062),
    (-0.0043, -0.0101, -0.0086),
    (-0.0086, 0.0006, 0.0151),
    (0.0057, 0.0062, 0.0037),
    (0.0047, -0.0024, -0.0131),
)
_FLUX2_RGB_BIAS = (0.5027, 0.4707, 0.4311)

# The projections as ``(factors, bias)`` arrays, built once rather than per preview.
_ZIMAGE_PROJECTION = (mx.array(_ZIMAGE_RGB_FACTORS, dtype=mx.float32), mx.array(_ZIMAGE_RGB_BIAS, dtype=mx.float32))
_FLUX2_PROJECTION = (mx.array(_FLUX2_RGB_FACTORS, dtype=mx.float32), mx.array(_FLUX2_RGB_BIAS, dtype=mx.float32))


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
        family: Image model family (``zimage``, ``flux2_klein`` or ``ideogram4``).
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
