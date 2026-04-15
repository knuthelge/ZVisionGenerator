"""Beta-distribution sigma scheduler for flow-matching models."""

from __future__ import annotations

from typing import TYPE_CHECKING

import mlx.core as mx
import numpy as np
from scipy.stats import beta as beta_dist

if TYPE_CHECKING:
    from mflux.models.common.config.config import Config

from mflux.models.common.schedulers.base_scheduler import BaseScheduler


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

    def _get_sigmas(self) -> mx.array:
        model_config = self.config.model_config
        num_steps = self.config.num_inference_steps

        # Beta PPF (inverse CDF) concentrates timesteps at both ends of the
        # schedule, matching ComfyUI's beta scheduler.
        # Reference: https://arxiv.org/abs/2407.12173
        ts = 1 - np.linspace(0, 1, num_steps, endpoint=False)
        sigmas_np = beta_dist.ppf(ts, 0.6, 0.6).astype(np.float32)
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
