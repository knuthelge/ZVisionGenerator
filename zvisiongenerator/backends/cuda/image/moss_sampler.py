"""Sampling loop, prompt encoding and latent helpers for MOSS-Image.

The vendored ``MOSSImagePipeline.__call__`` has no step callback, interrupt, start index or image input, so this module
owns the denoising loop. It mirrors the vendored loop (``pipeline_moss_image.py``: schedule, CFG on FP32 predictions,
unconditional-branch layer skipping, Euler step) and the vendored VAE decode, with the device passed explicitly. Prompt
encoding calls the vendored ``encode_prompt`` itself. A golden test pins the loop and the decode against the vendored
pipeline.

MOSS latents are FLUX.2-VAE latents: patchified to 128 channels at 1/16 resolution and normalised with the VAE's
batch-norm statistics.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

from zvisiongenerator.core.moss import img2img_run_steps

if TYPE_CHECKING:
    import torch

__all__ = [
    "Conditioning",
    "decode",
    "denoise",
    "encode_image",
    "encode_prompt",
    "initial_noise",
    "noised",
    "set_schedule",
    "to_vae_latents",
]


@dataclass(frozen=True)
class Conditioning:
    """Text embeddings and their attention mask for one prompt."""

    embeds: torch.Tensor
    mask: torch.Tensor


def encode_prompt(tokenizer: Any, text_encoder: Any, prompt: str, *, device: torch.device | str, max_length: int) -> Conditioning:
    """Encode *prompt* with the vendored pipeline's ``encode_prompt`` on an explicit device."""
    from moss_image.pipeline_moss_image import MOSSImagePipeline

    holder = SimpleNamespace(tokenizer=tokenizer, text_encoder=text_encoder, _execution_device=device)
    embeds, mask = MOSSImagePipeline.encode_prompt(holder, prompt, max_length)
    return Conditioning(embeds=embeds, mask=mask)


def set_schedule(scheduler: Any, steps: int, strength: float, device: torch.device | str) -> int:
    """Set the scheduler's timesteps and begin index for *strength*, and return the first step index to run.

    Text-to-image uses ``strength`` 1.0 and starts at 0. The begin index is set after ``set_timesteps``, which resets it.
    """
    scheduler.set_timesteps(steps, device=device)
    total = len(scheduler.timesteps)
    start = total - img2img_run_steps(total, strength)
    scheduler.set_begin_index(start)
    return start


def initial_noise(shape: tuple[int, ...], generator: torch.Generator | None, device: torch.device | str) -> torch.Tensor:
    """Return FP32 Gaussian noise of *shape*, sampled as the vendored pipeline does."""
    import torch
    from diffusers.utils.torch_utils import randn_tensor

    return randn_tensor(shape, generator=generator, device=device, dtype=torch.float32)


def noised(clean: torch.Tensor, noise: torch.Tensor, sigma: float) -> torch.Tensor:
    """Blend *clean* latents with *noise* at flow-matching noise level *sigma*."""
    return (1 - sigma) * clean + sigma * noise


def denoise(
    transformer: Any,
    scheduler: Any,
    latents: torch.Tensor,
    *,
    cond: Conditioning,
    uncond: Conditioning | None,
    guidance: float,
    skip_layers: tuple[int, ...],
    start: int,
    on_step: Callable[[int, torch.Tensor, torch.Tensor], bool],
) -> torch.Tensor | None:
    """Run the Euler steps from *start* and return the latents, or None when *on_step* returns True.

    *uncond* is None for guidance 1.0 (one forward per step). Otherwise the unconditional branch skips *skip_layers* and
    the two FP32 predictions are combined as classifier-free guidance. *on_step* receives the step number (from 0), the
    timestep and the new latents.
    """
    for i, t in enumerate(scheduler.timesteps[start:]):
        pred = transformer(latents, t, cond.embeds, cond.mask).sample.float()
        if uncond is not None:
            skip_kwargs = {"skip_layers": skip_layers} if skip_layers else {}
            u = transformer(latents, t, uncond.embeds, uncond.mask, **skip_kwargs).sample.float()
            pred = u + guidance * (pred - u)
        latents = scheduler.step(pred, t, latents, return_dict=False)[0]
        if on_step(i, t, latents):
            return None
    return latents


def encode_image(vae: Any, image_processor: Any, image: Any, width: int, height: int) -> torch.Tensor:
    """Return the normalised, patchified FP32 latents of *image* resized to *width* x *height*."""
    import torch
    from diffusers import Flux2Pipeline

    device = next(vae.parameters()).device
    pixels = image_processor.preprocess(image, height=height, width=width).to(device=device, dtype=torch.float32)
    z = vae.encode(pixels).latent_dist.mode()
    z = Flux2Pipeline._patchify_latents(z)
    bn = vae.bn
    mean = bn.running_mean.view(1, -1, 1, 1).to(z)
    std = torch.sqrt(bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps).to(z)
    return (z - mean) / std


def to_vae_latents(vae: Any, latents: torch.Tensor) -> torch.Tensor:
    """Return *latents* de-normalised with the VAE's batch-norm statistics and unpatchified for decoding."""
    import torch
    from diffusers import Flux2Pipeline

    bn = vae.bn
    std = torch.sqrt(bn.running_var + vae.config.batch_norm_eps).to(latents).view(1, -1, 1, 1)
    z = latents * std + bn.running_mean.to(latents).view(1, -1, 1, 1)
    return Flux2Pipeline._unpatchify_latents(z)


def decode(vae: Any, image_processor: Any, latents: torch.Tensor, output_type: str = "pil") -> Any:
    """Decode *latents* to an image ("pil" returns the first image; "pt" and "np" return the batch)."""
    z = to_vae_latents(vae, latents.to(dtype=vae.dtype))
    images = vae.decode(z, return_dict=False)[0].clamp(-1, 1)
    images = image_processor.postprocess(images, output_type=output_type)
    return images[0] if output_type == "pil" else images
