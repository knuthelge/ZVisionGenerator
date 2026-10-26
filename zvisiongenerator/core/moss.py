"""MOSS-Image settings every image backend shares: the sampler, the img2img step count and the text length."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

__all__ = [
    "MAX_SEQUENCE_LENGTH",
    "TURBO_GUIDANCE",
    "UNCOND_SKIP_LAYERS",
    "MossSampler",
    "img2img_run_steps",
    "sampler_settings",
]

# DiT layers the unconditional (negative) pass skips under classifier-free guidance.
UNCOND_SKIP_LAYERS: tuple[int, ...] = (22,)
# MOSS-Image Turbo has its guidance baked in: one conditional pass per step, which is guidance 1.0.
TURBO_GUIDANCE = 1.0
MAX_SEQUENCE_LENGTH = 4096


@dataclass(frozen=True)
class MossSampler:
    """The sampler arguments of a MOSS-Image run.

    Attributes:
        steps: Number of denoising steps.
        guidance: Classifier-free guidance scale; 1.0 runs the conditional pass only.
        negative_prompt: Text for the unconditional pass, or None when there is no guided pass.
        skip_layers: DiT layers the unconditional pass skips.
    """

    steps: int
    guidance: float
    negative_prompt: str | None
    skip_layers: tuple[int, ...]

    @property
    def do_cfg(self) -> bool:
        """Whether the run takes an unconditional pass per step."""
        return _uses_cfg(self.guidance)


def _uses_cfg(guidance: float) -> bool:
    """Return whether *guidance* turns on the unconditional pass."""
    return guidance != 1.0


def sampler_settings(*, turbo: bool, steps: int, guidance: float, negative_prompt: str | None) -> MossSampler:
    """Return the sampler arguments for a run; Turbo ignores *guidance* with a warning when it is not 1.0.

    Args:
        turbo: Whether the model is the distilled Turbo variant.
        steps: The step count from the config or the command line.
        guidance: The guidance from the config or the command line.
        negative_prompt: The negative prompt, if any; used only when guidance is on.
    """
    if turbo:
        if guidance != TURBO_GUIDANCE:
            warnings.warn(f"MOSS-Image Turbo runs without guidance; guidance {guidance:g} is ignored.", stacklevel=2)
        return MossSampler(steps=steps, guidance=TURBO_GUIDANCE, negative_prompt=None, skip_layers=())
    return MossSampler(
        steps=steps,
        guidance=guidance,
        negative_prompt=(negative_prompt or "") if _uses_cfg(guidance) else None,
        skip_layers=UNCOND_SKIP_LAYERS,
    )


def img2img_run_steps(total: int, strength: float) -> int:
    """Return how many of *total* steps an img2img run at *strength* denoises over.

    Uses diffusers' img2img rounding, which the CUDA backend's ``_img2img_sigmas`` shares: the first ``int(total - total * strength)``
    steps are skipped and at least one always runs.
    """
    return min(max(total - int(max(total - total * strength, 0.0)), 1), total)
