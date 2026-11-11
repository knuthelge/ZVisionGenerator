"""Region editing on the FLUX.2 Klein diffusers pipeline: the clean crop as a reference and the step-end blend.

The pipeline redraws a crop from noise at strength 1.0, with a reference image so the model sees the picture around
the paint. After step ``i`` its latents sit at noise level ``sigma[i + 1]``; outside the painted area they
are replaced by the source's clean latents noised to that same level with the run's own starting noise:
``latents = (1 - mask) * noised_clean + mask * latents``. The mask is 1 where the model may draw. Sigma 0 at the last step
pins the exact clean latents outside the paint. The reference is passed by the caller, as the pipeline's ``image`` input.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from zvisiongenerator.processing.edit_region import match_join, soft_cell_mask

if TYPE_CHECKING:
    from PIL import Image

__all__ = ["Prepared", "blend_callback", "match_strip", "prepare"]


@dataclass(frozen=True)
class Prepared:
    """What one edit feeds a pipeline.

    Attributes:
        noise: The run's starting noise, in the layout the pipeline takes as ``latents``.
        packed_noise: The same noise packed into tokens, in the layout the step-end callback sees.
        clean: The source crop's latents, in the layout the step-end callback sees.
        mask: 1 where the model may draw, in the layout of *clean*, broadcastable over its channels.
    """

    noise: torch.Tensor
    packed_noise: torch.Tensor
    clean: torch.Tensor
    mask: torch.Tensor


def _randn(shape: tuple[int, ...], seed: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    generator = torch.Generator("cpu").manual_seed(seed)
    return torch.randn(shape, generator=generator, dtype=torch.float32).to(device=device, dtype=dtype)


def _pixels(pipe: Any, image: Image.Image) -> torch.Tensor:
    return pipe.image_processor.preprocess(image, height=image.height, width=image.width).to(device=pipe._execution_device, dtype=pipe.vae.dtype)


@torch.inference_mode()
def prepare(pipe: Any, crop: Image.Image, paint: np.ndarray, seed: int) -> Prepared:
    """Encode *crop* (already at the model's size) and draw the run's noise; *paint* is the crop-size bool mask.

    Raises:
        ValueError: *paint*'s size is not a whole number of latent cells.
    """
    device = pipe._execution_device
    grid = torch.from_numpy(soft_cell_mask(paint)).to(device)
    clean = pipe._encode_vae_image(_pixels(pipe, crop), None)  # (1, 128, h/16, w/16), normalised
    noise = _randn(tuple(clean.shape), seed, device, clean.dtype)
    return Prepared(noise, pipe._pack_latents(noise), pipe._pack_latents(clean), pipe._pack_latents(grid[None, None]))


@torch.inference_mode()
def decode_packed(pipe: Any, packed: torch.Tensor, height: int, width: int) -> np.ndarray:
    """Decode packed latents (the layout the step-end callback sees) to an RGB array of *height* x *width*."""
    device, dtype = pipe._execution_device, pipe.vae.dtype
    latents = packed.to(device, dtype).permute(0, 2, 1).reshape(1, 128, height // 16, width // 16)
    mean = pipe.vae.bn.running_mean.view(1, -1, 1, 1).to(device, dtype)
    std = torch.sqrt(pipe.vae.bn.running_var.view(1, -1, 1, 1) + pipe.vae.config.batch_norm_eps).to(device, dtype)
    decoded = pipe.vae.decode(pipe._unpatchify_latents(latents * std + mean), return_dict=False)[0]
    return np.asarray(pipe.image_processor.postprocess(decoded, output_type="pil")[0].convert("RGB"))


def match_strip(pipe: Any, generated: Image.Image, final: torch.Tensor, crop: Image.Image, paint: np.ndarray) -> Image.Image:
    """Return *generated* with its *paint* strip colour-matched to *crop* across the join (:func:`match_join`), from the model's own *final* latents."""
    from PIL import Image

    view = decode_packed(pipe, final, crop.height, crop.width)
    return Image.fromarray(match_join(np.asarray(generated.convert("RGB")), view, np.asarray(crop.convert("RGB")), paint))


def blend_callback(prepared: Prepared, final: list[torch.Tensor] | None = None) -> Callable[[Any, int, Any, dict], dict]:
    """Return a ``callback_on_step_end`` that blends the source's latents back in outside the paint at every step.

    The callback sees packed tokens while the pipeline takes patchified latents, so it blends with the packed noise. With
    *final*, the model's own latents at the last step, before the pin to the source, are appended to it (on the CPU).
    """

    def on_step_end(pipe: Any, step: int, timestep: Any, kwargs: dict) -> dict:
        del timestep
        sigma = float(pipe.scheduler.sigmas[step + 1])
        latents = kwargs["latents"]
        if final is not None and sigma == 0.0:
            final.append(latents.detach().float().cpu().clone())
        noised = (1.0 - sigma) * prepared.clean.float() + sigma * prepared.packed_noise.float()
        kwargs["latents"] = ((1.0 - prepared.mask) * noised + prepared.mask * latents.float()).to(latents.dtype)
        return kwargs

    return on_step_end
