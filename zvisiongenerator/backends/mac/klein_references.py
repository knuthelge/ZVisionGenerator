"""FLUX.2 Klein references on macOS (mflux): generate from reference images without loading the model again.

mflux's ``Flux2KleinEdit`` builds its own model, so using it as documented would load Klein a second time. Instead
:func:`reference_view` builds a ``Flux2KleinEdit`` that shares the components of the already loaded ``Flux2Klein`` by
reference. Weights, baked LoRAs and the callback registry (progress, previews, Stop) are the loaded model's own.

A start image and references together need one more step: the edit path starts from pure noise, so
:func:`_start_image_latents` scopes a hook that builds the first latents from the start image with
``Flux2Klein._prepare_img2img_latents`` instead.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from typing import Any

from PIL import Image
from mflux.models.common.config.config import Config
from mflux.models.flux2.variants import Flux2KleinEdit
from mflux.models.flux2.variants.edit.flux2_klein_edit_helpers import _Flux2KleinEditHelpers
from mflux.utils.generated_image import GeneratedImage
from mlx import nn

from zvisiongenerator.backends.mac._temp_images import remove_file, save_temp_png

# What ``Flux2Initializer.init`` sets on a model; the view shares each by reference. A contract test pins this list.
SHARED_ATTRIBUTES = (
    "prompt_cache",
    "model_config",
    "callbacks",
    "tiling_config",
    "tokenizers",
    "vae",
    "transformer",
    "text_encoder",
    "bits",
    "small_text_encoder",
    "small_decoder",
    "lora_paths",
    "lora_scales",
    "compute_precision",
)

# ``Flux2KleinEdit.generate_image``'s default scheduler; the start-image hook builds its own config with it.
DEFAULT_SCHEDULER = "flow_match_euler_discrete"


def reference_view(model: Any) -> Flux2KleinEdit:
    """Return a ``Flux2KleinEdit`` over *model*'s components, built without loading anything.

    Args:
        model: A loaded ``Flux2Klein``.

    Returns:
        An edit model whose weights, tokenizers and callback registry are *model*'s own objects.
    """
    view = Flux2KleinEdit.__new__(Flux2KleinEdit)
    nn.Module.__init__(view)
    for name in SHARED_ATTRIBUTES:
        setattr(view, name, getattr(model, name))
    return view


@contextlib.contextmanager
def _start_image_latents(model: Any, config: Config) -> Iterator[None]:
    """Make the edit path's first latents the start image noised to *config*'s first sigma, for the duration of the block.

    ``Flux2KleinEdit.generate_image`` calls ``_Flux2KleinEditHelpers.prepare_generation_latents`` for its latents,
    which are pure noise. Inside the block that helper returns what ``Flux2Klein`` builds for img2img from
    ``config.image_path``, so the run denoises the start image's tail of the schedule while the references condition
    it. The helper is put back on exit.

    Args:
        model: The loaded ``Flux2Klein``, which builds the img2img latents.
        config: An img2img config with the start image as ``image_path``, the run's steps, size and scheduler.
    """
    original = _Flux2KleinEditHelpers.prepare_generation_latents

    def prepare_generation_latents(*, seed: int, height: int, width: int) -> Any:
        return model._prepare_img2img_latents(seed=seed, config=config)

    _Flux2KleinEditHelpers.prepare_generation_latents = staticmethod(prepare_generation_latents)
    try:
        yield
    finally:
        _Flux2KleinEditHelpers.prepare_generation_latents = staticmethod(original)


def generate_with_references(
    model: Any,
    reference_images: list[Image.Image],
    *,
    start_path: str | None = None,
    image_strength: float | None = None,
    **generate_kwargs: Any,
) -> GeneratedImage:
    """Generate with *reference_images* conditioning the result, from noise or from a start image.

    Args:
        model: A loaded ``Flux2Klein``.
        reference_images: The references, in prompt order; each keeps its own size.
        start_path: PNG of the start image (img2img), or ``None`` for a run from noise.
        image_strength: With *start_path*, mflux's strength: the share of the schedule skipped. ``None`` or ``0``
            starts from noise, as mflux does.
        **generate_kwargs: ``seed``, ``prompt``, ``num_inference_steps``, ``height``, ``width``, ``guidance`` and
            optionally ``scheduler``, as for ``Flux2Klein.generate_image``.

    Returns:
        The generated image. ``image_strength`` is never passed to the edit path on a references-only run: mflux
        would start a partial schedule from the first reference.
    """
    generate_kwargs.setdefault("scheduler", DEFAULT_SCHEDULER)
    view = reference_view(model)
    reference_paths = [save_temp_png(image) for image in reference_images]
    try:
        if start_path is None or not image_strength or image_strength <= 0.0:
            return view.generate_image(**generate_kwargs, image_paths=reference_paths)
        start_config = Config(
            model_config=model.model_config,
            num_inference_steps=generate_kwargs["num_inference_steps"],
            height=generate_kwargs["height"],
            width=generate_kwargs["width"],
            guidance=generate_kwargs["guidance"],
            image_path=start_path,
            image_strength=image_strength,
            scheduler=generate_kwargs["scheduler"],
        )
        with _start_image_latents(model, start_config):
            return view.generate_image(**generate_kwargs, image_paths=reference_paths, image_strength=image_strength)
    finally:
        for path in reference_paths:
            remove_file(path)
