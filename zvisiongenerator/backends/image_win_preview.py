"""Render cheap, low-resolution previews of in-progress diffusers latents without running the VAE."""

from __future__ import annotations

import warnings
from typing import Any

import torch
from PIL import Image
from diffusers import FlowMatchEulerDiscreteScheduler

from zvisiongenerator.core.latent_preview import FLUX2_RGB_BIAS, FLUX2_RGB_FACTORS, QWEN_IMAGE_RGB_BIAS, QWEN_IMAGE_RGB_FACTORS, ZIMAGE_RGB_BIAS, ZIMAGE_RGB_FACTORS
from zvisiongenerator.core.progress_events import preview_milestone_steps

__all__ = ["LivePreview", "create_live_preview", "estimate_clean_latents", "render_latent_preview"]

# FLUX.2 and FLUX.2 Klein pipelines share the FLUX.2 VAE and its packed latent layout.
_FLUX2_FAMILIES = frozenset({"flux2", "flux2_klein"})
_PREVIEW_FAMILIES = _FLUX2_FAMILIES | {"zimage", "krea2"}


def estimate_clean_latents(previous: torch.Tensor, current: torch.Tensor, noise_previous: float, noise_current: float) -> torch.Tensor:
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
    velocity = (current.float() - previous.float()) / step
    return current.float() - noise_current * velocity


def render_latent_preview(pipe: Any, family: str, latents: torch.Tensor, height: int, width: int) -> Image.Image | None:
    """Project one family's step-end latents to an RGB preview at 1/8 of the output resolution.

    Args:
        pipe: The running diffusers pipeline (FLUX.2 needs its VAE batch-norm statistics).
        family: Image model family (``zimage``, ``flux2``, ``flux2_klein`` or ``krea2``).
        latents: Latents exactly as the pipeline passes them to ``callback_on_step_end``.
        height: Output image height in pixels.
        width: Output image width in pixels.

    Returns:
        The preview image, or ``None`` for families without a projection.
    """
    if family == "zimage":
        return _project_to_image(latents[0], ZIMAGE_RGB_FACTORS, ZIMAGE_RGB_BIAS)
    if family in _FLUX2_FAMILIES:
        return _project_to_image(_flux2_spatial(pipe, latents, height, width), FLUX2_RGB_FACTORS, FLUX2_RGB_BIAS)
    if family == "krea2":
        return _project_to_image(_krea2_spatial(pipe, latents, height, width), QWEN_IMAGE_RGB_FACTORS, QWEN_IMAGE_RGB_BIAS)
    return None


def create_live_preview(family: str, total_steps: int, height: int, width: int) -> LivePreview | None:
    """Return a live-preview tracker for ``family``, or ``None`` when it has no latent projection."""
    if family not in _PREVIEW_FAMILIES:
        return None
    return LivePreview(family, total_steps, height, width)


class LivePreview:
    """Turn a pipeline's step-end latents into previews at milestone steps.

    Previews are best-effort: the first failure warns once and disables them for the rest of the run.
    """

    def __init__(self, family: str, total_steps: int, height: int, width: int):
        self._family = family
        self._total_steps = total_steps
        self._height = height
        self._width = width
        self._enabled = True
        self._milestones: frozenset[int] | None = None
        self._previous: torch.Tensor | None = None

    def observe(self, pipe: Any, step: int, latents: torch.Tensor | None) -> Image.Image | None:
        """Record the latents after 0-based ``step`` and return a preview when it is a milestone."""
        if not self._enabled or latents is None:
            return None
        try:
            if self._milestones is None:
                # img2img runs only part of the schedule; the pipeline knows how many steps it actually runs.
                run_steps = getattr(pipe, "num_timesteps", None)
                self._milestones = preview_milestone_steps(run_steps if isinstance(run_steps, int) else self._total_steps)
            preview = self._render(pipe, latents) if step + 1 in self._milestones else None
            # Keep these latents only when the next step renders a preview from them.
            self._previous = latents if step + 2 in self._milestones else None
            return preview
        except Exception as exc:  # noqa: BLE001 - previews are best-effort
            warnings.warn(f"Live preview failed for {self._family}: {exc}", stacklevel=2)
            self._enabled = False
            self._previous = None
            return None

    def _render(self, pipe: Any, latents: torch.Tensor) -> Image.Image | None:
        """Render the predicted final image when the step's noise levels are known, else the raw latents."""
        levels = _step_noise_levels(pipe.scheduler)
        if levels is not None and self._previous is not None:
            latents = estimate_clean_latents(self._previous, latents, *levels)
        return render_latent_preview(pipe, self._family, latents, self._height, self._width)


def _step_noise_levels(scheduler: Any) -> tuple[float, float] | None:
    """Return the noise levels before and after the step a flow-matching Euler scheduler just took."""
    if not isinstance(scheduler, FlowMatchEulerDiscreteScheduler):
        return None
    index = scheduler.step_index
    if index is None or not 1 <= index < len(scheduler.sigmas):
        return None
    return float(scheduler.sigmas[index - 1]), float(scheduler.sigmas[index])


def _flux2_spatial(pipe: Any, latents: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Undo FLUX.2's token packing, batch-norm and 2x2 patchify to recover ``(32, h, w)`` VAE latents."""
    patch = pipe.vae_scale_factor * 2
    patch_h, patch_w = height // patch, width // patch
    tokens, channels = latents.shape[1:]
    if tokens != patch_h * patch_w:
        raise ValueError(f"Expected {patch_h * patch_w} latent tokens for {width}x{height}, got {tokens}.")
    packed = latents[0].float().permute(1, 0).reshape(channels, patch_h, patch_w)
    bn = pipe.vae.bn
    std = torch.sqrt(bn.running_var.float() + pipe.vae.config.batch_norm_eps).to(packed.device)
    packed = packed * std.view(-1, 1, 1) + bn.running_mean.float().to(packed.device).view(-1, 1, 1)
    spatial = packed.reshape(channels // 4, 2, 2, patch_h, patch_w).permute(0, 3, 1, 4, 2)
    return spatial.reshape(channels // 4, patch_h * 2, patch_w * 2)


def _krea2_spatial(pipe: Any, latents: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Undo Krea 2's token packing of ``patch_size``-square patches to recover ``(16, h, w)`` VAE latents."""
    patch = pipe.patch_size
    tokens_h, tokens_w = height // (pipe.vae_scale_factor * patch), width // (pipe.vae_scale_factor * patch)
    tokens, channels = latents.shape[1:]
    if tokens != tokens_h * tokens_w:
        raise ValueError(f"Expected {tokens_h * tokens_w} latent tokens for {width}x{height}, got {tokens}.")
    # Each token holds (channel, row, column) of its patch, as Krea2Pipeline._pack_latents lays it out.
    packed = latents[0].float().reshape(tokens_h, tokens_w, channels // (patch * patch), patch, patch)
    return packed.permute(2, 0, 3, 1, 4).reshape(channels // (patch * patch), tokens_h * patch, tokens_w * patch)


def _project_to_image(spatial: torch.Tensor, factors: tuple[tuple[float, ...], ...], bias: tuple[float, ...]) -> Image.Image:
    """Map ``(channels, h, w)`` latents to RGB with a linear ``factors`` and ``bias`` projection."""
    weight = torch.tensor(factors, dtype=torch.float32, device=spatial.device)
    offset = torch.tensor(bias, dtype=torch.float32, device=spatial.device)
    rgb = torch.einsum("chw,cr->hwr", spatial.float(), weight) + offset
    pixels = (rgb.clamp(0.0, 1.0) * 255.0).to(torch.uint8).cpu().numpy()
    return Image.fromarray(pixels)
