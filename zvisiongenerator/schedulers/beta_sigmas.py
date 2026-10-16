"""Beta-distribution sigma grid shared by the mflux and diffusers backends."""

from __future__ import annotations

import numpy as np
from scipy.stats import beta as beta_dist


def beta_sigmas(num_steps: int) -> list[float]:
    """Return *num_steps* unshifted sigmas spaced by the beta(0.6, 0.6) inverse CDF.

    Matches ComfyUI's "beta" scheduler: timesteps concentrate at both ends of the schedule
    (https://arxiv.org/abs/2407.12173). The grid runs from 1.0 down to the CDF's value at ``1 / num_steps``;
    the model's own timestep shift is applied afterwards, as for a linear grid.
    """
    ts = 1 - np.linspace(0, 1, num_steps, endpoint=False)
    return beta_dist.ppf(ts, 0.6, 0.6).astype(np.float32).tolist()
