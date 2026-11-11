"""Latent blending on mflux: redraw a painted area at strength 1.0 and keep the rest as the source's own latents.

After denoising step ``t`` the latents sit at noise level ``sigmas[t + 1]``. Outside the painted area they are replaced
by the source's clean latents noised to that same level with the run's own starting noise:
``latents = (1 - mask) * noised_clean + mask * latents``. The mask is 1 where the model may draw.

mflux's loops cannot replace latents from a callback, so the blend hooks the step itself: Z-Image and FLUX.2 Klein wrap
the run's ``config.scheduler.step`` from a before-loop callback; Krea 2 keeps its stepper in a local, so a scoped patch
of ``Krea2Sampler.make_stepper`` wraps the stepper it builds. At strength 1.0 mflux never encodes the source and draws
its starting noise from the seed alone, so :func:`prepare` draws the same noise again and encodes the source itself.

``mlx`` and ``mflux`` are imported inside the functions that need them, so this module imports anywhere.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from zvisiongenerator.processing.edit_region import EDIT_CELL_PIXELS, soft_cell_mask

if TYPE_CHECKING:
    from PIL import Image

__all__ = ["Prepared", "blend", "blended_stepper", "edit_blend", "latent_mask", "prepare"]


@dataclass(frozen=True)
class Prepared:
    """What one run blends with, all float32 and in the layout the family's loop holds its latents in.

    Attributes:
        clean: The source's clean latents.
        noise: The run's starting noise, the array mflux draws from the seed.
        mask: 1 where the model may draw, broadcastable against *clean*.
    """

    clean: Any
    noise: Any
    mask: Any


def latent_mask(family: str, paint: np.ndarray) -> np.ndarray:
    """Return *paint* (bool, height x width, 16 px aligned) as the family's float32 latent mask.

    Z-Image: ``(1, 1, H/8, W/8)``. Klein: ``(1, tokens, 1)``, one token per 16 px cell in row-major order. Krea 2:
    ``(1, 1, H/8, W/8)``, each 16 px cell repeated over its 2 x 2 latent pixels.

    The cell is the family's entry in ``EDIT_CELL_PIXELS``: one latent cell for Z-Image, one patch token for Klein, one
    2x2 patch for Krea 2.

    Raises:
        ValueError: *paint*'s sides are not multiples of the family's cell.
    """
    cells = soft_cell_mask(paint, EDIT_CELL_PIXELS[family])
    if family == "flux2_klein":
        return cells.reshape(1, -1, 1)
    if family == "krea2":
        cells = cells.repeat(2, axis=0).repeat(2, axis=1)
    return cells[None, None]


def prepare(family: str, model: Any, crop: Image.Image, paint: np.ndarray, seed: int) -> Prepared:
    """Encode *crop* (the size the model draws at) and draw the run's noise for *family*; *paint* is the crop-size bool mask.

    The arrays are evaluated before returning, so the source's encode is finished before the loop starts.

    Raises:
        ValueError: *paint*'s sides are not multiples of the family's cell.
    """
    import mlx.core as mx
    from mflux.models.common.vae.vae_util import VAEUtil
    from mflux.utils.image_util import ImageUtil

    mask = latent_mask(family, paint)
    width, height = crop.size
    pixels = ImageUtil.to_array(crop.convert("RGB"))
    if family == "zimage":
        from mflux.models.z_image.latent_creator import ZImageLatentCreator

        noise = ZImageLatentCreator.create_noise(seed, height, width)
        clean = ZImageLatentCreator.pack_latents(VAEUtil.encode(vae=model.vae, image=pixels, tiling_config=model.tiling_config), height, width)
    elif family == "flux2_klein":
        noise, clean = _klein_noise_and_clean(model, pixels, seed, height, width)
    else:  # krea2
        from mflux.models.krea2.latent_creator import Krea2LatentCreator

        noise = Krea2LatentCreator.create_noise(seed, height, width)
        clean = Krea2LatentCreator.pack_latents(VAEUtil.encode(vae=model.vae, image=pixels, tiling_config=model.tiling_config), height, width)
    prepared = Prepared(clean=clean.astype(mx.float32), noise=noise.astype(mx.float32), mask=mx.array(mask))
    mx.eval(prepared.clean, prepared.noise, prepared.mask)
    return prepared


def _klein_noise_and_clean(model: Any, pixels: Any, seed: int, height: int, width: int) -> tuple[Any, Any]:
    """Return Klein's packed starting noise and the packed clean latents, built as ``Flux2Klein._prepare_img2img_latents`` builds them."""
    from mflux.models.common.vae.vae_util import VAEUtil
    from mflux.models.flux2 import Flux2Klein
    from mflux.models.flux2.latent_creator.flux2_latent_creator import Flux2LatentCreator
    from mflux.models.flux2.variants.edit.flux2_klein_edit_helpers import _Flux2KleinEditHelpers

    noise, _ids, latent_height, latent_width = Flux2LatentCreator.prepare_packed_latents(seed=seed, height=height, width=width, batch_size=1)
    encoded = VAEUtil.encode(vae=model.vae, image=pixels, tiling_config=model.tiling_config)
    encoded = _Flux2KleinEditHelpers.ensure_4d_latents(encoded)
    encoded = _Flux2KleinEditHelpers.crop_to_even_spatial(encoded)
    encoded = Flux2Klein._match_latent_spatial_size(encoded=encoded, target_height=latent_height * 2, target_width=latent_width * 2)
    encoded = Flux2LatentCreator.patchify_latents(encoded)
    encoded = _Flux2KleinEditHelpers.bn_normalize_vae_encoded_latents(encoded, vae=model.vae)
    return noise, Flux2LatentCreator.pack_latents(encoded)


def blend(latents: Any, prepared: Prepared, sigma: float) -> Any:
    """Return *latents* with everything outside the paint replaced by the source at noise level *sigma*, in *latents*' dtype."""
    noised = (1.0 - sigma) * prepared.clean + sigma * prepared.noise
    return ((1.0 - prepared.mask) * noised + prepared.mask * latents).astype(latents.dtype)


class _BlendInstaller:
    """Before-loop callback that wraps the run's ``config.scheduler.step`` with the blend.

    The scheduler belongs to one run's ``Config``, so the wrap goes on that instance and never outlives the run.
    """

    def __init__(self, prepared: Prepared) -> None:
        self._prepared = prepared

    def call_before_loop(self, seed: int, prompt: str, latents: Any, config: Any, **_: Any) -> None:
        del seed, prompt, latents
        scheduler = config.scheduler
        original = scheduler.step
        prepared = self._prepared

        def step(*, noise: Any, timestep: int, latents: Any, **kwargs: Any) -> Any:
            """Take the scheduler's own step (Klein also passes ``sigmas=``), then blend at ``sigmas[timestep + 1]``."""
            stepped = original(noise=noise, timestep=timestep, latents=latents, **kwargs)
            return blend(stepped, prepared, float(scheduler.sigmas[timestep + 1]))

        scheduler.step = step


def _krea2_sampler() -> Any:
    from mflux.models.krea2.model.krea2_sampler import Krea2Sampler

    return Krea2Sampler


@contextlib.contextmanager
def blended_stepper(prepared: Prepared) -> Iterator[None]:
    """Make every Krea 2 stepper built inside the block blend after each step; ``make_stepper`` is restored however the block ends."""
    sampler = _krea2_sampler()
    saved = vars(sampler)["make_stepper"]
    make_stepper: Callable[..., Any] = sampler.make_stepper

    def blended_make_stepper(name: str, sigmas: Any, seed: int) -> Any:
        stepper = make_stepper(name, sigmas, seed)
        stepped = stepper.step

        def step(i: int, x: Any, v: Any, denoised: Any) -> Any:
            """Take the stepper's own step, then blend at ``sigmas[i + 1]``."""
            return blend(stepped(i, x, v, denoised), prepared, float(sigmas[i + 1]))

        stepper.step = step
        return stepper

    sampler.make_stepper = staticmethod(blended_make_stepper)
    try:
        yield
    finally:
        sampler.make_stepper = saved


@contextlib.contextmanager
def edit_blend(model: Any, family: str, prepared: Prepared) -> Iterator[None]:
    """Blend every generation run inside the block, for *family*; the hook is gone however the block ends.

    Z-Image and Klein register a before-loop callback on *model*; Krea 2 patches ``Krea2Sampler.make_stepper``.
    """
    if family == "krea2":
        with blended_stepper(prepared):
            yield
        return
    installer = _BlendInstaller(prepared)
    model.callbacks.register(installer)
    try:
        yield
    finally:
        with contextlib.suppress(ValueError):
            model.callbacks.before_loop.remove(installer)
