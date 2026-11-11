"""Region editing on the diffusers pipelines: per-family latent layouts and the step-end blend.

The pipeline redraws a crop from noise at strength 1.0. After step ``i`` its latents sit at noise level ``sigma[i + 1]``;
outside the painted area they are replaced by the source's clean latents noised to that same level with the run's own
starting noise: ``latents = (1 - mask) * noised_clean + mask * latents``. The mask is 1 where the model may draw.
Sigma 0 at the last step pins the exact clean latents outside the paint.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from zvisiongenerator.processing.edit_region import EDIT_CELL_PIXELS, soft_cell_mask

if TYPE_CHECKING:
    from PIL import Image

__all__ = ["Prepared", "blend_callback", "prepare"]


@dataclass(frozen=True)
class Prepared:
    """What one edit feeds a pipeline.

    Attributes:
        noise: The run's starting noise, in the layout the pipeline takes as ``latents``.
        clean: The source crop's latents, in the layout the step-end callback sees.
        mask: 1 where the model may draw, in the layout of *clean*, broadcastable over its channels.
    """

    noise: torch.Tensor
    clean: torch.Tensor
    mask: torch.Tensor


def _randn(shape: tuple[int, ...], seed: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    generator = torch.Generator("cpu").manual_seed(seed)
    return torch.randn(shape, generator=generator, dtype=torch.float32).to(device=device, dtype=dtype)


def _pixels(pipe: Any, image: Image.Image) -> torch.Tensor:
    return pipe.image_processor.preprocess(image, height=image.height, width=image.width).to(device=pipe._execution_device, dtype=pipe.vae.dtype)


@torch.inference_mode()
def prepare(family: str, pipe: Any, crop: Image.Image, paint: np.ndarray, seed: int) -> Prepared:
    """Encode *crop* (already at the model's size) and draw the run's noise for *family*; *paint* is the crop-size bool mask.

    Raises:
        ValueError: *paint*'s size is not a whole number of latent cells.
    """
    device = pipe._execution_device
    grid = torch.from_numpy(soft_cell_mask(paint, EDIT_CELL_PIXELS[family])).to(device)
    if family == "zimage":
        vae = pipe.vae
        encoded = vae.encode(_pixels(pipe, crop)).latent_dist.mode()
        clean = ((encoded - vae.config.shift_factor) * vae.config.scaling_factor).float()
        return Prepared(_randn(tuple(clean.shape), seed, device, torch.float32), clean, grid[None, None])
    if family == "flux2_klein":
        clean = pipe._encode_vae_image(_pixels(pipe, crop), None)  # (1, 128, h/16, w/16), normalised
        return Prepared(_randn(tuple(clean.shape), seed, device, clean.dtype), pipe._pack_latents(clean), pipe._pack_latents(grid[None, None]))
    from zvisiongenerator.backends.cuda.image.backend import _krea2_encode_image

    clean = _krea2_encode_image(pipe, crop, crop.width, crop.height)  # packed (1, seq, 64)
    return Prepared(_randn(tuple(clean.shape), seed, device, clean.dtype), clean, grid.reshape(1, -1, 1))


def blend_callback(prepared: Prepared) -> Callable[[Any, int, Any, dict], dict]:
    """Return a ``callback_on_step_end`` that blends the source's latents back in outside the paint at every step.

    Z-Image's callback sees the same layout it takes as ``latents``; Klein's sees packed tokens while it takes
    patchified ones, so its noise is packed to match here. Krea 2 takes and sees packed tokens.
    """

    def on_step_end(pipe: Any, step: int, timestep: Any, kwargs: dict) -> dict:
        del timestep
        sigma = float(pipe.scheduler.sigmas[step + 1])
        latents = kwargs["latents"]
        noise = prepared.noise
        if noise.ndim == 4 and latents.ndim == 3:
            noise = noise.reshape(noise.shape[0], noise.shape[1], -1).permute(0, 2, 1)
        noised = (1.0 - sigma) * prepared.clean.float() + sigma * noise.float()
        kwargs["latents"] = ((1.0 - prepared.mask) * noised + prepared.mask * latents.float()).to(latents.dtype)
        return kwargs

    return on_step_end
