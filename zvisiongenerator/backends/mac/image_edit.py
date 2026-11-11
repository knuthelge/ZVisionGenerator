"""Latent blending on mflux for FLUX.2 Klein: redraw a painted area at strength 1.0 and keep the rest as the source's own latents.

After denoising step ``t`` the latents sit at noise level ``sigmas[t + 1]``. Outside the painted area they are replaced
by the source's clean latents noised to that same level with the run's own starting noise:
``latents = (1 - mask) * noised_clean + mask * latents``. The mask is 1 where the model may draw.

mflux's loop cannot replace latents from a callback, so the blend hooks the step itself: it wraps the run's
``config.scheduler.step`` from a before-loop callback. The run also gets the clean crop as a reference image
(``generate_with_references``), so the model sees the picture around the paint; the reference tokens are not part of
the latents the step sees. At strength 1.0 mflux never encodes the source and draws its starting noise from the seed
alone, so :func:`prepare` draws the same noise again and encodes the source itself.

``mlx`` and ``mflux`` are imported inside the functions that need them, so this module imports anywhere.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from zvisiongenerator.processing.edit_region import match_join, soft_cell_mask

if TYPE_CHECKING:
    from PIL import Image

__all__ = ["Prepared", "edit_blend", "match_strip", "prepare"]


@dataclass(frozen=True)
class Prepared:
    """What one run blends with, all float32 and in the layout Klein's loop holds its latents in.

    Attributes:
        clean: The source's clean latents.
        noise: The run's starting noise, the array mflux draws from the seed.
        mask: 1 where the model may draw, broadcastable against *clean*.
    """

    clean: Any
    noise: Any
    mask: Any


def prepare(model: Any, crop: Image.Image, paint: np.ndarray, seed: int) -> Prepared:
    """Encode *crop* (the size the model draws at) and draw the run's noise; *paint* is the crop-size bool mask.

    The arrays are evaluated before returning, so the source's encode is finished before the loop starts.

    Raises:
        ValueError: *paint*'s sides are not multiples of 16.
    """
    import mlx.core as mx
    from mflux.utils.image_util import ImageUtil

    mask = soft_cell_mask(paint).reshape(1, -1, 1)  # one token per 16 px cell, in row-major order
    width, height = crop.size
    noise, clean = _klein_noise_and_clean(model, ImageUtil.to_array(crop.convert("RGB")), seed, height, width)
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

    def __init__(self, prepared: Prepared, final: list[Any] | None = None) -> None:
        self._prepared = prepared
        self._final = final

    def call_before_loop(self, seed: int, prompt: str, latents: Any, config: Any, **_: Any) -> None:
        del seed, prompt, latents
        scheduler = config.scheduler
        original = scheduler.step
        prepared, final = self._prepared, self._final

        def step(*, noise: Any, timestep: int, latents: Any, **kwargs: Any) -> Any:
            """Take the scheduler's own step (Klein also passes ``sigmas=``), then blend at ``sigmas[timestep + 1]``."""
            stepped = original(noise=noise, timestep=timestep, latents=latents, **kwargs)
            sigma = float(scheduler.sigmas[timestep + 1])
            if final is not None and sigma == 0.0:
                final.append(stepped)  # the model's own latents at the last step, before the pin
            return blend(stepped, prepared, sigma)

        scheduler.step = step


@contextlib.contextmanager
def edit_blend(model: Any, prepared: Prepared, final: list[Any] | None = None) -> Iterator[None]:
    """Blend every generation run inside the block; the hook is gone however the block ends.

    The installer is a before-loop callback on *model*, which the reference view of the model shares. With *final*, the
    model's latents at the last step, before the pin to the source, are appended to it.
    """
    installer = _BlendInstaller(prepared, final)
    model.callbacks.register(installer)
    try:
        yield
    finally:
        model.callbacks.before_loop.remove(installer)


def match_strip(model: Any, generated: Image.Image, final: Any, crop: Image.Image, paint: np.ndarray) -> Image.Image:
    """Return *generated* with its *paint* strip colour-matched to *crop* across the join (:func:`match_join`), from the model's own *final* latents."""
    from mflux.utils.image_util import ImageUtil
    from PIL import Image

    latents = final.reshape(1, crop.height // 16, crop.width // 16, 128).transpose(0, 3, 1, 2)
    view = np.asarray(ImageUtil.to_pil(model.vae.decode_packed_latents(latents, tiling_config=model.tiling_config)).convert("RGB"))
    return Image.fromarray(match_join(np.asarray(generated.convert("RGB")), view, np.asarray(crop.convert("RGB")), paint))
