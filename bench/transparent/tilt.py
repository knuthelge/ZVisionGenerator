"""Run the image CLI with klein's starting noise tilted towards a flat background colour (a TKG-DM-style experiment).

TKG-DM (arXiv 2411.15580) shifts the initial noise outside a soft centre mask, ``z + (1 - A) * delta``, so the
background drifts to a colour without naming it in the prompt while the centre keeps the seed's own noise. Here
``delta`` is *strength* times the latent of a flat white or black image, read through klein's own VAE. Rendering the
same seed with a white and a black tilt gives a pair whose subject starts from identical noise.

Usage (app environment)::

    ZIV_BENCH_TILT=white:1.0 python -m bench.transparent.tilt <ziv-image arguments>

Only ``Flux2KleinPipeline`` text-to-image is patched; anything that passes its own latents is left alone.
"""

from __future__ import annotations

import os
import sys

import numpy as np

_TARGETS = {"white": 1.0, "black": -1.0}
_SIGMA = 0.5


def centre_mask(height: int, width: int, sigma: float = _SIGMA) -> np.ndarray:
    """Return a Gaussian centre weight in [0, 1] over a ``(height, width)`` grid spanning [-1, 1] on both axes."""
    ys, xs = np.linspace(-1.0, 1.0, height), np.linspace(-1.0, 1.0, width)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    return np.exp(-(xx**2 + yy**2) / (2 * sigma**2))


def parse_tilt(value: str) -> tuple[str, float]:
    """Return ``(target, strength)`` from ``white:1.0`` or ``black:0.5``.

    Raises:
        ValueError: When the target is not white or black, or the strength is not a number.
    """
    target, _, strength = value.partition(":")
    if target not in _TARGETS:
        raise ValueError(f"Tilt target must be white or black, not {target!r}.")
    return target, float(strength or 1.0)


def _install(target: str, strength: float) -> None:
    import torch
    from diffusers.pipelines.flux2 import pipeline_flux2_klein as klein
    from diffusers.utils.torch_utils import randn_tensor

    original = klein.Flux2KleinPipeline.prepare_latents

    def prepare_latents(self, batch_size, num_latents_channels, height, width, dtype, device, generator, latents=None):  # noqa: ANN001, ANN202
        if latents is not None:
            return original(self, batch_size, num_latents_channels, height, width, dtype, device, generator, latents)
        rows, cols = int(height) // (self.vae_scale_factor * 2), int(width) // (self.vae_scale_factor * 2)
        noise = randn_tensor((batch_size, num_latents_channels * 4, rows, cols), generator=generator, device=device, dtype=dtype)
        vae_device = klein.get_module_execution_device(self.vae)
        flat = torch.full((1, 3, 256, 256), _TARGETS[target], device=vae_device, dtype=self.vae.dtype)
        with torch.no_grad():
            colour = self._encode_vae_image(flat, generator=None).float().mean(dim=(2, 3))
        weight = torch.from_numpy(centre_mask(rows, cols)).to(device=device, dtype=torch.float32)
        shift = (1.0 - weight)[None, None] * (strength * colour.to(device))[:, :, None, None]
        print(f"[tilt] {target} x{strength}: shifting starting noise outside the centre", file=sys.stderr)
        return original(self, batch_size, num_latents_channels, height, width, dtype, device, generator, (noise.float() + shift).to(dtype))

    klein.Flux2KleinPipeline.prepare_latents = prepare_latents


def main() -> None:
    """Patch klein's starting noise from ``ZIV_BENCH_TILT``, then run the image CLI with the remaining arguments."""
    from zvisiongenerator.image_cli import main as image_main

    _install(*parse_tilt(os.environ["ZIV_BENCH_TILT"]))
    sys.argv = ["ziv-image", *sys.argv[1:]]
    image_main()


if __name__ == "__main__":
    main()
