"""Beta-distribution sigma scheduler for flow-matching models."""

from __future__ import annotations

from typing import TYPE_CHECKING

import mlx.core as mx
import numpy as np

if TYPE_CHECKING:
    from mflux.models.common.config.config import Config

from mflux.models.common.schedulers.base_scheduler import BaseScheduler
from mflux.models.common.schedulers.flow_match_euler_discrete_scheduler import FlowMatchEulerDiscreteScheduler

from zvisiongenerator.schedulers.beta_sigmas import beta_sigmas


class BetaScheduler(BaseScheduler):
    """Flow-matching scheduler using beta-distribution sigma spacing.

    Equivalent to ComfyUI's "beta" scheduler — concentrates timesteps
    at both ends of the schedule for better detail at start and end.
    """

    def __init__(self, config: "Config"):
        self.config = config
        self._sigmas = self._get_sigmas()

    @property
    def sigmas(self) -> mx.array:
        return self._sigmas

    @property
    def timesteps(self) -> mx.array:
        """Return the model timestep of each step (sigma x 1000), which FLUX.2 Klein's loop reads."""
        return self._sigmas[:-1] * 1000

    def _get_sigmas(self) -> mx.array:
        model_config = self.config.model_config
        num_steps = self.config.num_inference_steps

        sigmas_np = np.array(beta_sigmas(num_steps), dtype=np.float32)
        sigmas_np = np.append(sigmas_np, 0.0)

        # Apply sigma shift if required (same logic as LinearScheduler)
        if model_config.requires_sigma_shift:
            m = (model_config.sigma_max_shift - model_config.sigma_base_shift) / (model_config.sigma_max_seq_len - model_config.sigma_base_seq_len)
            b = model_config.sigma_base_shift - m * model_config.sigma_base_seq_len
            mu = m * self.config.width * self.config.height / 256 + b

            # Exponential shift on all sigmas except the trailing zero
            s = sigmas_np[:-1]
            sigmas_np[:-1] = np.exp(mu) / (np.exp(mu) + (1.0 / s - 1.0))

            if model_config.sigma_shift_terminal is not None:
                shifted = sigmas_np[:-1]
                one_minus = 1.0 - shifted
                scale = one_minus[-1] / (1.0 - model_config.sigma_shift_terminal)
                sigmas_np[:-1] = 1.0 - (one_minus / scale)

        return mx.array(sigmas_np, dtype=mx.float32)

    def step(self, noise: mx.array, timestep: int, latents: mx.array, **kwargs) -> mx.array:
        dt = (self._sigmas[timestep + 1] - self._sigmas[timestep]).astype(latents.dtype)
        return latents + noise.astype(latents.dtype) * dt


class KleinBetaScheduler(BetaScheduler):
    """BetaScheduler with FLUX.2 Klein's own shift: the step-dependent ``mu`` mflux and diffusers compute for Klein."""

    def set_image_seq_len(self, image_seq_len: int) -> None:
        """Shift the beta grid by Klein's ``mu`` for *image_seq_len*; mflux calls this once the image size is known."""
        mu = FlowMatchEulerDiscreteScheduler._compute_empirical_mu(image_seq_len=image_seq_len, num_steps=self.config.num_inference_steps)
        grid = mx.array(beta_sigmas(self.config.num_inference_steps), dtype=mx.float32)
        shifted = FlowMatchEulerDiscreteScheduler._time_shift_exponential_array(mu, 1.0, grid)
        self._sigmas = mx.concatenate([shifted, mx.zeros((1,), dtype=shifted.dtype)])
