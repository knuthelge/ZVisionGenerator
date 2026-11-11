"""Pin the diffusers 0.42 internals that FLUX.2 Klein region editing on Windows and Linux relies on.

Introspection, plus tiny-tensor checks of static helpers: no pipeline is constructed and no weights are loaded. A failure
here after a diffusers bump means a name, signature or layout the latent blend in ``backends/cuda/image/edit.py`` uses has
changed.
"""

from __future__ import annotations

import inspect

import torch
from diffusers.pipelines.flux2.pipeline_flux2_klein import Flux2KleinPipeline


def _parameters(callable_obj) -> list[str]:
    return list(inspect.signature(callable_obj).parameters)


def test_klein_encodes_an_image_to_patchified_normalised_latents():
    assert _parameters(Flux2KleinPipeline._encode_vae_image) == ["self", "image", "generator"]
    pipeline = object.__new__(Flux2KleinPipeline)
    raw = torch.randn(1, 32, 8, 8)
    pipeline.vae = type(
        "Vae",
        (),
        {
            "encode": lambda self, image: type("Out", (), {"latent_dist": type("Dist", (), {"mode": lambda self: raw})()})(),
            "bn": type("Bn", (), {"running_mean": torch.zeros(128), "running_var": torch.ones(128)})(),
            "config": type("Config", (), {"batch_norm_eps": 0.0})(),
        },
    )()

    encoded = pipeline._encode_vae_image(torch.zeros(1, 3, 64, 64), None)

    assert encoded.shape == (1, 128, 4, 4)


def test_klein_packs_a_grid_to_one_token_per_cell_in_row_major_order():
    grid = torch.arange(12, dtype=torch.float32).reshape(1, 1, 3, 4)

    packed = Flux2KleinPipeline._pack_latents(grid)

    assert packed.shape == (1, 12, 1)
    assert packed.flatten().tolist() == list(range(12))


def test_klein_takes_patchified_latents_and_packs_them_itself():
    assert "latents" in _parameters(Flux2KleinPipeline.prepare_latents)
    pipeline = object.__new__(Flux2KleinPipeline)
    pipeline.vae_scale_factor = 8
    patchified = torch.zeros(1, 128, 4, 4)

    packed, ids = pipeline.prepare_latents(1, 32, 64, 64, torch.float32, torch.device("cpu"), None, latents=patchified)

    assert packed.shape == (1, 16, 128)
    assert ids.shape[1] == 16


def test_klein_takes_the_reference_images_as_its_image_input():
    assert {"image", "latents"} <= set(_parameters(Flux2KleinPipeline.__call__))


def test_klein_unpatchifies_the_blend_layout_back_to_the_vae_layout_for_the_join_fix():
    unpatchified = Flux2KleinPipeline._unpatchify_latents(torch.zeros(1, 128, 3, 4))

    assert unpatchified.shape == (1, 32, 6, 8)


def test_klein_takes_the_step_end_callback_and_its_latents_back():
    parameters = _parameters(Flux2KleinPipeline.__call__)
    source = inspect.getsource(Flux2KleinPipeline.__call__)

    assert {"latents", "callback_on_step_end", "callback_on_step_end_tensor_inputs"} <= set(parameters)
    # The loop hands the callback (pipe, step, timestep, kwargs) and keeps the "latents" it returns.
    assert "callback_on_step_end(self, i, t, callback_kwargs)" in source
    assert 'callback_outputs.pop("latents", latents)' in source
