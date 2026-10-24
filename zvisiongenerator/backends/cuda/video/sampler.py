"""Euler-ancestral sampling for the distilled LTX-2.5 rows: seed and sigma maths, a shared noise holder and a scheduler swap.

LTX-2.5's distilled schedule is sampled with diffusers' ``LTXEulerAncestralRFScheduler`` (eta 1, s_noise 1) instead of the
pipeline's stock deterministic Euler scheduler. Three things make that scheduler fit the LTX-2 pipelines unchanged:

* ``set_timesteps`` appends the terminal sigma 0 and drops the last timestep, so the pipeline runs exactly one forward
  per distilled sigma (8 in stage 1, 3 in stage 2) while the final step still lands on the clean sample.
* The pipeline calls ``scheduler.step`` without a generator, so the scheduler draws its noise from a
  :class:`SharedNoise` bound at construction. The pipeline deep-copies the scheduler for the audio latents;
  ``SharedNoise.__deepcopy__`` returns itself, so the video and audio schedulers consume one generator, video first.
* Each denoising pass gets a fresh scheduler and a generator seeded per stage (:func:`stage_seed`), installed with
  :func:`swap_scheduler`, which restores the original afterwards, even on error.

The caller decides when to use it. An Extend pass (one that carries conditions) must keep the stock scheduler whatever
sampler is configured, because the ancestral ``step`` would re-noise the pinned tail; the backend's pass-scheduler hook
makes that choice and only calls :func:`ancestral_scheduler` for unconditioned t2v and i2v passes.

Importing this module imports neither torch nor diffusers; both load inside :func:`ancestral_scheduler`.
"""

from __future__ import annotations

import functools
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from zvisiongenerator.utils.config import EULER_ANCESTRAL, LTX_SAMPLERS

__all__ = [
    "EULER_ANCESTRAL",
    "LTX_SAMPLERS",
    "STAGE_SEED_OFFSETS",
    "SharedNoise",
    "ancestral_scheduler",
    "stage_seed",
    "swap_scheduler",
    "with_terminal_sigma",
]

STAGE_SEED_OFFSETS = MappingProxyType({1: 10_000, 2: 20_000})
"""Offsets added to the user's seed for the step noise of each denoising stage (one-stage runs use stage 1)."""


def stage_seed(seed: int | None, stage: int) -> int | None:
    """Return the step-noise seed for *stage*, or ``None`` when the run is unseeded.

    The offset keeps the step noise independent of the initial noise, which is drawn from the plain seed.

    Args:
        seed: The user's seed, or ``None`` for a non-deterministic run.
        stage: 1 for the first (or only) denoising stage, 2 for the refinement stage.

    Returns:
        ``seed`` plus the stage offset (not wrapped), or ``None`` when *seed* is ``None``.

    Raises:
        ValueError: If *stage* is not 1 or 2.
    """
    if stage not in STAGE_SEED_OFFSETS:
        raise ValueError(f"stage must be one of {sorted(STAGE_SEED_OFFSETS)}, got {stage!r}.")
    if seed is None:
        return None
    return seed + STAGE_SEED_OFFSETS[stage]


def with_terminal_sigma(sigmas: Sequence[float]) -> list[float]:
    """Return *sigmas* as floats ending in the terminal sigma 0, appending it when missing.

    Args:
        sigmas: A descending sigma table, with or without the terminal 0.

    Raises:
        ValueError: If *sigmas* is empty.
    """
    values = [float(value) for value in sigmas]
    if not values:
        raise ValueError("sigmas must not be empty.")
    if values[-1] != 0.0:
        values.append(0.0)
    return values


@dataclass(frozen=True)
class SharedNoise:
    """The step-noise generator of one denoising stage, shared by a scheduler and its deep copies.

    Attributes:
        generator: A ``torch.Generator`` (CPU), or ``None`` to draw from torch's global state.
    """

    generator: Any

    def __deepcopy__(self, memo: dict[int, Any]) -> SharedNoise:
        """Return this holder itself, so a deep-copied scheduler keeps drawing from the same generator."""
        return self


@functools.cache
def _scheduler_class() -> type[Any]:
    """Build the ancestral scheduler subclass once; the diffusers import happens here, not at module import."""
    from diffusers import LTXEulerAncestralRFScheduler

    class AncestralDistilledScheduler(LTXEulerAncestralRFScheduler):
        """Euler-ancestral RF scheduler with a terminal sigma 0, no forward at ``t = 0`` and noise from :class:`SharedNoise`."""

        noise: SharedNoise | None = None

        def set_timesteps(self, num_inference_steps: int | None = None, device: Any = None, sigmas: Any = None, mu: float | None = None, **kwargs: object) -> None:
            if sigmas is None:
                raise ValueError("AncestralDistilledScheduler.set_timesteps requires explicit sigmas.")
            super().set_timesteps(num_inference_steps=num_inference_steps, device=device, sigmas=with_terminal_sigma(sigmas), mu=mu, **kwargs)
            self.timesteps = self.timesteps[:-1]

        def step(self, model_output: Any, timestep: Any, sample: Any, generator: Any = None, return_dict: bool = True) -> Any:
            if generator is None and self.noise is not None:
                generator = self.noise.generator
            return super().step(model_output, timestep, sample, generator=generator, return_dict=return_dict)

    return AncestralDistilledScheduler


def ancestral_scheduler(generator: Any, *, num_train_timesteps: int = 1000) -> Any:
    """Return a fresh Euler-ancestral scheduler (eta 1, s_noise 1) that draws its step noise from *generator*.

    ``set_timesteps(sigmas=...)`` appends the terminal 0 and trims the last timestep, so a table of N distilled sigmas
    gives N timesteps and N ``step`` calls. ``step`` uses the shared generator when the pipeline passes none. Deep copies
    share the generator (see :class:`SharedNoise`).

    Args:
        generator: The stage's ``torch.Generator``, from :func:`stage_seed`.
        num_train_timesteps: Scale of the timesteps handed to the transformer (sigma times this value).
    """
    scheduler = _scheduler_class()(num_train_timesteps=num_train_timesteps, eta=1.0, s_noise=1.0)
    scheduler.noise = SharedNoise(generator)
    return scheduler


@contextmanager
def swap_scheduler(pipeline: Any, scheduler: Any) -> Iterator[None]:
    """Install *scheduler* on *pipeline* for the ``with`` body and restore the original on exit, even on error."""
    original = pipeline.scheduler
    pipeline.scheduler = scheduler
    try:
        yield
    finally:
        pipeline.scheduler = original
