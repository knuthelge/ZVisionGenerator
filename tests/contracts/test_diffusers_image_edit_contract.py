"""Pin the diffusers 0.42 internals that region editing on Windows and Linux relies on.

Introspection, plus tiny-tensor checks of static helpers: no pipeline is constructed and no weights are loaded. A failure
here after a diffusers bump means a name, signature or layout the latent blend in ``backends/cuda/image/edit.py`` uses has
changed.
"""

from __future__ import annotations

import inspect

import torch
from diffusers.pipelines.flux2.pipeline_flux2_klein import Flux2KleinPipeline
from diffusers.pipelines.krea2.pipeline_krea2 import Krea2Pipeline
from diffusers.pipelines.z_image.pipeline_z_image import ZImagePipeline


def _parameters(callable_obj) -> list[str]:
    return list(inspect.signature(callable_obj).parameters)


def test_zimage_takes_starting_latents_at_four_dimensional_eighth_resolution():
    assert "latents" in _parameters(ZImagePipeline.prepare_latents)
    pipeline = object.__new__(ZImagePipeline)
    pipeline.vae_scale_factor = 8
    shape = (1, 16, 8, 8)

    taken = pipeline.prepare_latents(1, 16, 64, 64, torch.float32, torch.device("cpu"), None, latents=torch.zeros(shape))

    assert taken.shape == shape


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


def test_krea2_packs_latents_into_two_by_two_patch_tokens():
    assert _parameters(Krea2Pipeline._pack_latents) == ["self", "latents", "batch_size", "num_channels_latents", "height", "width"]
    pipeline = object.__new__(Krea2Pipeline)
    pipeline.patch_size = 2

    packed = pipeline._pack_latents(torch.zeros(1, 16, 8, 8), 1, 16, 8, 8)

    assert packed.shape == (1, 16, 64)


def test_krea2_takes_packed_starting_latents():
    assert "latents" in _parameters(Krea2Pipeline.prepare_latents)
    pipeline = object.__new__(Krea2Pipeline)
    packed = torch.zeros(1, 16, 64)

    assert pipeline.prepare_latents(1, 16, 64, 64, torch.float32, torch.device("cpu"), None, latents=packed).shape == packed.shape


def test_every_pipeline_takes_the_step_end_callback_and_its_latents_back():
    for pipeline in (ZImagePipeline, Flux2KleinPipeline, Krea2Pipeline):
        parameters = _parameters(pipeline.__call__)
        source = inspect.getsource(pipeline.__call__)

        assert {"latents", "callback_on_step_end", "callback_on_step_end_tensor_inputs"} <= set(parameters)
        # The loop hands the callback (pipe, step, timestep, kwargs) and keeps the "latents" it returns.
        assert "callback_on_step_end(self, i, t, callback_kwargs)" in source
        assert 'callback_outputs.pop("latents", latents)' in source
