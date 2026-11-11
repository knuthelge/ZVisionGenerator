"""Tests for region editing on the diffusers pipelines: latent layouts, the step-end blend and the backend's edit_region.

Real tensors on the CPU and fake pipelines; no model is loaded. Z-Image, FLUX.2 Klein and Krea 2 each get the latent
layout their diffusers pipeline documents, and the fake pipeline runs a short loop that calls the step-end callback the
way the real ones do.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")
diffusers = pytest.importorskip("diffusers")

from diffusers.pipelines.flux2.pipeline_flux2_klein import Flux2KleinPipeline  # noqa: E402
from diffusers.pipelines.krea2.pipeline_krea2 import Krea2Pipeline  # noqa: E402

from zvisiongenerator.utils.image_model_detect import ImageModelInfo  # noqa: E402

SIZE = 64
STEPS = 4
FAMILIES = ["zimage", "flux2_klein", "krea2"]


@pytest.fixture()
def edit_module():
    import zvisiongenerator.backends.cuda.image.edit as module

    return module


@pytest.fixture()
def backend_module():
    import zvisiongenerator.backends.cuda.image.backend as module

    return module


def _crop() -> Image.Image:
    return Image.new("RGB", (SIZE, SIZE), (90, 120, 150))


def _paint() -> np.ndarray:
    paint = np.zeros((SIZE, SIZE), dtype=bool)
    paint[16:48, 16:48] = True
    return paint


def _encoded(*shape: int) -> torch.Tensor:
    return torch.randn(shape, generator=torch.Generator("cpu").manual_seed(1))


class _Vae:
    """A VAE stand-in: encodes any pixels to fixed random latents of the family's shape."""

    dtype = torch.float32

    def __init__(self, family: str):
        self.family = family
        self.config = SimpleNamespace(shift_factor=0.1, scaling_factor=0.5, latents_mean=[0.2] * 16, latents_std=[2.0] * 16, batch_norm_eps=1e-4)
        self.bn = SimpleNamespace(running_mean=torch.full((128,), 0.3), running_var=torch.full((128,), 4.0))
        self.raw: torch.Tensor | None = None

    def encode(self, pixels: torch.Tensor) -> SimpleNamespace:
        height, width = pixels.shape[-2:]
        if self.family == "zimage":
            self.raw = _encoded(1, 16, height // 8, width // 8)
        elif self.family == "flux2_klein":
            self.raw = _encoded(1, 32, height // 8, width // 8)
        else:
            self.raw = _encoded(1, 16, 1, height // 8, width // 8)
        return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda: self.raw))


class _FakePipe:
    """A diffusers pipeline stand-in: real latent helpers where the pipelines have them, a short denoising loop for ``__call__``."""

    patch_size = 2
    vae_scale_factor = 8
    _execution_device = torch.device("cpu")

    def __init__(self, family: str, *, raises: bool = False):
        self.family = family
        self.vae = _Vae(family)
        self.scheduler = diffusers.FlowMatchEulerDiscreteScheduler()
        self.image_processor = SimpleNamespace(preprocess=lambda image, height, width: torch.zeros(1, 3, height, width))
        self.calls: list[dict] = []
        self.schedulers_seen: list[object] = []
        self.latents_seen: list[torch.Tensor] = []
        self.raises = raises
        self._interrupt = False

    # The real pipelines' helpers, bound to this stand-in.
    def _pack_latents(self, *args):
        if self.family == "krea2":
            return Krea2Pipeline._pack_latents(self, *args)
        return Flux2KleinPipeline._pack_latents(*args)

    def _encode_vae_image(self, image, generator):
        return Flux2KleinPipeline._encode_vae_image(self, image, generator)

    _patchify_latents = staticmethod(Flux2KleinPipeline._patchify_latents)

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        self.schedulers_seen.append(self.scheduler)
        if self.raises:
            raise RuntimeError("pipeline failed")
        self._interrupt = False
        steps = kwargs["num_inference_steps"]
        self.scheduler.set_timesteps(steps, mu=0.5)
        latents = kwargs["latents"]
        if self.family == "flux2_klein":
            latents = Flux2KleinPipeline._pack_latents(latents)  # what the real pipeline does with patchified input
        callback = kwargs.get("callback_on_step_end")
        for step, timestep in enumerate(self.scheduler.timesteps):
            if self._interrupt:
                continue
            latents = torch.full_like(latents, 7.0)  # the model's own prediction
            if callback is not None:
                inputs = {name: latents for name in kwargs["callback_on_step_end_tensor_inputs"]}
                latents = callback(self, step, timestep, inputs).pop("latents", latents)
            self.latents_seen.append(latents)
        return SimpleNamespace(images=[Image.new("RGB", (kwargs["width"], kwargs["height"]))])


def _backend(backend_module, family: str):
    backend = backend_module.DiffusersBackend()
    backend._model_info = ImageModelInfo(family=family, is_distilled=False, size=None)
    return backend


def _edit(backend, pipe, **overrides):
    arguments = {"image": _crop(), "paint": _paint(), "prompt": "a cat", "seed": 7, "steps": STEPS, "guidance": 3.0, **overrides}
    return backend.edit_region(model=pipe, **arguments)


class _Stop:
    """A skip signal that fires on its first check."""

    def check(self) -> bool:
        return True


class TestBlendCallback:
    """R5.1: the callback produces ``(1 - m) * noised + m * latents`` for all three layouts, with sigma from the scheduler."""

    @pytest.mark.parametrize("layout", ["zimage", "flux2_klein", "krea2"])
    @pytest.mark.parametrize("step", [0, 1, STEPS - 1])
    def test_outside_the_paint_the_clean_latents_are_noised_to_the_next_sigma(self, edit_module, layout, step):
        if layout == "zimage":
            noise, clean, mask = torch.randn(1, 16, 8, 8), torch.randn(1, 16, 8, 8), torch.rand(1, 1, 8, 8)
            latents = torch.randn(1, 16, 8, 8)
            expected_noise = noise
        elif layout == "flux2_klein":
            noise, clean, mask = torch.randn(1, 128, 4, 4), torch.randn(1, 16, 128), torch.rand(1, 16, 1)
            latents = torch.randn(1, 16, 128)
            expected_noise = Flux2KleinPipeline._pack_latents(noise)
        else:
            noise, clean, mask = torch.randn(1, 16, 64), torch.randn(1, 16, 64), torch.rand(1, 16, 1)
            latents = torch.randn(1, 16, 64)
            expected_noise = noise
        scheduler = diffusers.FlowMatchEulerDiscreteScheduler()
        scheduler.set_timesteps(STEPS, mu=0.5)
        sigma = float(scheduler.sigmas[step + 1])
        callback = edit_module.blend_callback(edit_module.Prepared(noise=noise, clean=clean, mask=mask))

        result = callback(SimpleNamespace(scheduler=scheduler), step, None, {"latents": latents})

        expected = (1 - mask) * ((1 - sigma) * clean + sigma * expected_noise) + mask * latents
        assert torch.allclose(result["latents"], expected, atol=1e-6)

    def test_the_last_step_pins_the_clean_latents_outside_the_paint(self, edit_module):
        clean, latents = torch.randn(1, 16, 8, 8), torch.randn(1, 16, 8, 8)
        mask = torch.zeros(1, 1, 8, 8)
        mask[..., 2:4, 2:4] = 1.0
        scheduler = diffusers.FlowMatchEulerDiscreteScheduler()
        scheduler.set_timesteps(STEPS, mu=0.5)
        callback = edit_module.blend_callback(edit_module.Prepared(noise=torch.randn(1, 16, 8, 8), clean=clean, mask=mask))

        result = callback(SimpleNamespace(scheduler=scheduler), STEPS - 1, None, {"latents": latents})["latents"]

        assert float(scheduler.sigmas[STEPS]) == 0.0
        assert torch.equal(result[..., 4:, :], clean[..., 4:, :])
        assert torch.equal(result[..., 2:4, 2:4], latents[..., 2:4, 2:4])

    def test_the_result_keeps_the_latents_dtype(self, edit_module):
        scheduler = diffusers.FlowMatchEulerDiscreteScheduler()
        scheduler.set_timesteps(STEPS, mu=0.5)
        prepared = edit_module.Prepared(noise=torch.randn(1, 16, 8, 8), clean=torch.randn(1, 16, 8, 8), mask=torch.ones(1, 1, 8, 8))

        result = edit_module.blend_callback(prepared)(SimpleNamespace(scheduler=scheduler), 0, None, {"latents": torch.randn(1, 16, 8, 8, dtype=torch.bfloat16)})

        assert result["latents"].dtype == torch.bfloat16

    def test_the_blend_runs_before_progress_and_the_preview(self, backend_module):
        seen: list[torch.Tensor] = []
        preview = SimpleNamespace(observe=lambda pipe, step, latents: seen.append(latents))
        marker = torch.full((1, 2), 5.0)
        callback = backend_module._make_step_callback(None, total_steps=1, step_callback=lambda payload: None, live_preview=preview, blend=lambda pipe, step, timestep, kwargs: {"latents": marker})

        callback(SimpleNamespace(scheduler=None, num_timesteps=1), 0, None, {"latents": torch.zeros(1, 2)})

        assert seen == [marker]


class TestPrepare:
    """R5.2: each family's noise has the shape its pipeline documents for ``latents``; clean and mask are in the callback's layout."""

    def test_zimage_takes_four_dimensional_latents_at_an_eighth_of_the_pixels(self, edit_module):
        pipe = _FakePipe("zimage")

        prepared = edit_module.prepare("zimage", pipe, _crop(), _paint(), 7)

        assert prepared.noise.shape == (1, 16, SIZE // 8, SIZE // 8)
        assert torch.allclose(prepared.clean, (pipe.vae.raw - 0.1) * 0.5)
        assert prepared.mask.shape == (1, 1, SIZE // 8, SIZE // 8)

    def test_zimage_cells_are_eight_pixels(self, edit_module):
        paint = np.zeros((SIZE, SIZE), dtype=bool)
        paint[:8, :4] = True  # half of the first 8 px cell

        prepared = edit_module.prepare("zimage", _FakePipe("zimage"), _crop(), paint, 7)

        assert float(prepared.mask[0, 0, 0, 0]) == 0.5
        assert float(prepared.mask.sum()) == 0.5

    def test_klein_takes_patchified_latents_and_sees_packed_ones(self, edit_module):
        pipe = _FakePipe("flux2_klein")

        prepared = edit_module.prepare("flux2_klein", pipe, _crop(), _paint(), 7)

        cells = SIZE // 16
        assert prepared.noise.shape == (1, 128, cells, cells)
        assert prepared.clean.shape == (1, cells * cells, 128)
        assert prepared.mask.shape == (1, cells * cells, 1)
        assert torch.equal(prepared.clean, Flux2KleinPipeline._pack_latents(Flux2KleinPipeline._encode_vae_image(pipe, torch.zeros(1, 3, SIZE, SIZE), None)))

    def test_klein_mask_cells_are_sixteen_pixels_in_token_order(self, edit_module):
        paint = np.zeros((SIZE, SIZE), dtype=bool)
        paint[:16, 16:32] = True  # the second cell of the first row

        mask = edit_module.prepare("flux2_klein", _FakePipe("flux2_klein"), _crop(), paint, 7).mask

        assert mask.flatten().tolist() == [0.0, 1.0] + [0.0] * 14

    def test_krea2_takes_and_sees_packed_tokens(self, edit_module):
        pipe = _FakePipe("krea2")

        prepared = edit_module.prepare("krea2", pipe, _crop(), _paint(), 7)

        tokens = (SIZE // 16) ** 2
        assert prepared.noise.shape == (1, tokens, 64)
        assert prepared.clean.shape == (1, tokens, 64)
        assert prepared.mask.shape == (1, tokens, 1)

    @pytest.mark.parametrize("family", FAMILIES)
    def test_the_noise_follows_the_seed_and_is_not_the_clean_latents(self, edit_module, family):
        first = edit_module.prepare(family, _FakePipe(family), _crop(), _paint(), 7)
        again = edit_module.prepare(family, _FakePipe(family), _crop(), _paint(), 7)
        other = edit_module.prepare(family, _FakePipe(family), _crop(), _paint(), 8)

        assert torch.equal(first.noise, again.noise)
        assert not torch.equal(first.noise, other.noise)

    @pytest.mark.parametrize("family", FAMILIES)
    def test_the_encode_runs_in_inference_mode(self, edit_module, family):
        prepared = edit_module.prepare(family, _FakePipe(family), _crop(), _paint(), 7)

        assert prepared.clean.is_inference()


class TestEditRegion:
    @pytest.mark.parametrize(
        ("family", "shape"),
        [("zimage", (1, 16, SIZE // 8, SIZE // 8)), ("flux2_klein", (1, 128, SIZE // 16, SIZE // 16)), ("krea2", (1, (SIZE // 16) ** 2, 64))],
    )
    def test_the_pipeline_gets_latents_of_its_documented_shape_at_the_crop_size(self, backend_module, family, shape):
        pipe = _FakePipe(family)

        result = _edit(_backend(backend_module, family), pipe)

        call = pipe.calls[0]
        assert call["latents"].shape == shape
        assert (call["width"], call["height"]) == (SIZE, SIZE)
        assert call["num_inference_steps"] == STEPS
        assert result.size == (SIZE, SIZE)

    @pytest.mark.parametrize("family", FAMILIES)
    def test_the_step_end_callback_always_carries_the_latents(self, backend_module, family):
        pipe = _FakePipe(family)

        _edit(_backend(backend_module, family), pipe)  # no step callback and no skip signal

        assert pipe.calls[0]["callback_on_step_end_tensor_inputs"] == ["latents"]
        assert callable(pipe.calls[0]["callback_on_step_end"])

    @pytest.mark.parametrize("family", FAMILIES)
    def test_every_step_is_blended_and_the_last_pins_the_clean_latents_outside_the_paint(self, backend_module, edit_module, family):
        pipe = _FakePipe(family)

        _edit(_backend(backend_module, family), pipe)

        expected = edit_module.prepare(family, _FakePipe(family), _crop(), _paint(), 7)
        final = pipe.latents_seen[-1]
        mask = expected.mask.expand_as(final)
        assert len(pipe.latents_seen) == STEPS
        assert (mask == 0).any()
        assert (mask == 1).any()
        # Outside the paint the final latents are the source's; inside it they are the model's own (7.0).
        assert torch.equal(final[mask == 0], expected.clean.expand_as(final)[mask == 0])
        assert torch.all(final[mask == 1] == 7.0)

    def test_the_guidance_of_krea2_is_converted_to_its_own_scale(self, backend_module):
        pipe = _FakePipe("krea2")

        _edit(_backend(backend_module, "krea2"), pipe, guidance=3.0)

        assert pipe.calls[0]["guidance_scale"] == 2.0

    def test_progress_is_reported_for_every_step(self, backend_module):
        pipe = _FakePipe("zimage")
        payloads: list[dict] = []

        _edit(_backend(backend_module, "zimage"), pipe, step_callback=payloads.append)

        assert [payload["current_step"] for payload in payloads] == list(range(1, STEPS + 1))

    @pytest.mark.parametrize("family", ["flux1", "flux2", "ideogram4", "moss"])
    def test_a_family_without_region_editing_raises(self, backend_module, family):
        with pytest.raises(ValueError, match=family):
            _edit(_backend(backend_module, family), _FakePipe("zimage"))

    def test_an_unloaded_backend_raises(self, backend_module):
        with pytest.raises(RuntimeError, match="load_model"):
            _edit(backend_module.DiffusersBackend(), _FakePipe("zimage"))

    def test_the_vae_encode_runs_in_inference_mode(self, backend_module):
        pipe = _FakePipe("zimage")

        _edit(_backend(backend_module, "zimage"), pipe)

        assert pipe.calls[0]["latents"].is_inference()


class TestSchedulerAndStop:
    """R5.3: the scheduler is put back after an edit; Stop returns None and clears the interrupt."""

    def test_a_beta_edit_swaps_the_scheduler_for_the_run_and_restores_it(self, backend_module):
        pipe = _FakePipe("zimage")
        original = pipe.scheduler

        _edit(_backend(backend_module, "zimage"), pipe, scheduler="beta")

        assert pipe.schedulers_seen[0] is not original
        assert pipe.calls[0]["sigmas"]
        assert pipe.scheduler is original

    def test_the_scheduler_is_restored_when_the_pipeline_raises(self, backend_module):
        pipe = _FakePipe("zimage", raises=True)
        original = pipe.scheduler

        with pytest.raises(RuntimeError, match="pipeline failed"):
            _edit(_backend(backend_module, "zimage"), pipe, scheduler="beta")

        assert pipe.scheduler is original

    def test_a_default_scheduler_edit_leaves_the_scheduler_alone(self, backend_module):
        pipe = _FakePipe("krea2")
        original = pipe.scheduler

        _edit(_backend(backend_module, "krea2"), pipe)

        assert pipe.schedulers_seen == [original]
        assert "sigmas" not in pipe.calls[0]
        assert pipe.scheduler is original

    @pytest.mark.parametrize("family", FAMILIES)
    def test_stop_returns_none_and_clears_the_interrupt(self, backend_module, family):
        pipe = _FakePipe(family)

        result = _edit(_backend(backend_module, family), pipe, skip_signal=_Stop())

        assert result is None
        assert pipe._interrupt is False

    def test_stop_ends_the_blend_with_the_interrupted_step(self, backend_module):
        pipe = _FakePipe("zimage")

        _edit(_backend(backend_module, "zimage"), pipe, skip_signal=_Stop())

        assert len(pipe.latents_seen) == 1  # the fake loop skips the steps after an interrupt, as the real ones do

    def test_the_scheduler_is_restored_after_a_stop(self, backend_module):
        pipe = _FakePipe("zimage")
        original = pipe.scheduler

        _edit(_backend(backend_module, "zimage"), pipe, scheduler="beta", skip_signal=_Stop())

        assert pipe.scheduler is original
