"""Custom schedulers for image generation, and the schedulers a model can be asked to run."""

from __future__ import annotations

from typing import Final

__all__ = ["SCHEDULER_CLASSES"]

SCHEDULER_CLASSES: Final[dict[str, dict[str, str | None]]] = {
    # ComfyUI's "beta": the beta(0.6, 0.6) grid, then the model's own timestep shift. diffusers gets the grid as custom sigmas.
    "beta": {"mflux": "zvisiongenerator.schedulers.beta_scheduler.BetaScheduler"},
    # The model's own evenly spaced schedule, for models that default to beta.
    "simple": {"mflux": None, "diffusers": None},
}
"""The schedulers a generation can pick, keyed by name, each with the class path per backend name.

A backend name the scheduler lists maps to its class path, or to ``None`` for the backend's own schedule; a backend it leaves
out receives the scheduler name itself.
"""
