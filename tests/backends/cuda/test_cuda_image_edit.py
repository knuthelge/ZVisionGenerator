"""Tests for region editing on the diffusers pipelines: latent layouts, the step-end blend and the backend's edit_region.

Real tensors on the CPU and a fake pipeline; no model is loaded. FLUX.2 Klein gets the latent layout its diffusers
pipeline documents and the clean crop as its reference image, and the fake pipeline runs a short loop that calls the
step-end callback the way the real one does.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from tests._images import painted_box

torch = pytest.importorskip("torch")
diffusers = pytest.importorskip("diffusers")

from diffusers.pipelines.flux2.pipeline_flux2_klein import Flux2KleinPipeline  # noqa: E402

from zvisiongenerator.utils.image_model_detect import ImageModelInfo  # noqa: E402

SIZE = 64
STEPS = 4
FAMILY = "flux2_klein"


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


_PAINT_BOX = (16, 16, 48, 48)  # whole 16 px cells


def _encoded(*shape: int) -> torch.Tensor:
    return torch.randn(shape, generator=torch.Generator("cpu").manual_seed(1))


class _Vae:
    """A VAE stand-in: encodes any pixels to fixed random Klein latents."""

    dtype = torch.float32

    def __init__(self):
        self.config = SimpleNamespace(batch_norm_eps=1e-4)
        self.bn = SimpleNamespace(running_mean=torch.full((128,), 0.3), running_var=torch.full((128,), 4.0))
        self.raw: torch.Tensor | None = None
        self.decoded: list[torch.Tensor] = []

    def decode(self, latents: torch.Tensor, return_dict: bool = False) -> tuple[torch.Tensor]:
        self.decoded.append(latents)
        return (torch.zeros(1, 3, latents.shape[-2] * 8, latents.shape[-1] * 8),)

    def encode(self, pixels: torch.Tensor) -> SimpleNamespace:
        height, width = pixels.shape[-2:]
        self.raw = _encoded(1, 32, height // 8, width // 8)
        return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda: self.raw))


class _FakePipe:
    """A diffusers pipeline stand-in: real latent helpers where the pipeline has them, a short denoising loop for ``__call__``."""

    patch_size = 2
    vae_scale_factor = 8
    _execution_device = torch.device("cpu")

    def __init__(self, *, raises: bool = False):
        self.vae = _Vae()
        self.scheduler = diffusers.FlowMatchEulerDiscreteScheduler()
        self.image_processor = SimpleNamespace(
            preprocess=lambda image, height, width: torch.zeros(1, 3, height, width),
            postprocess=lambda decoded, output_type: [Image.new("RGB", (decoded.shape[-1], decoded.shape[-2]), (30, 60, 90))],
        )
        self.calls: list[dict] = []
        self.schedulers_seen: list[object] = []
        self.latents_seen: list[torch.Tensor] = []
        self.raises = raises
        self._interrupt = False

    # The real pipeline's helpers, bound to this stand-in.
    def _pack_latents(self, *args):
        return Flux2KleinPipeline._pack_latents(*args)

    def _encode_vae_image(self, image, generator):
        return Flux2KleinPipeline._encode_vae_image(self, image, generator)

    _patchify_latents = staticmethod(Flux2KleinPipeline._patchify_latents)
    _unpatchify_latents = staticmethod(Flux2KleinPipeline._unpatchify_latents)

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        self.schedulers_seen.append(self.scheduler)
        if self.raises:
            raise RuntimeError("pipeline failed")
        self._interrupt = False
        steps = kwargs["num_inference_steps"]
        self.scheduler.set_timesteps(steps, mu=0.5)
        latents = Flux2KleinPipeline._pack_latents(kwargs["latents"])  # what the real pipeline does with patchified input
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


def _prepared(clean: torch.Tensor, mask: torch.Tensor):
    from zvisiongenerator.backends.cuda.image.edit import Prepared

    noise = torch.randn(1, 128, 4, 4)
    return Prepared(noise=noise, packed_noise=Flux2KleinPipeline._pack_latents(noise), clean=clean, mask=mask)


def _backend(backend_module, family: str = FAMILY):
    backend = backend_module.DiffusersBackend()
    backend._model_info = ImageModelInfo(family=family, is_distilled=False, size=None)
    return backend


def _edit(backend, pipe, **overrides):
    arguments = {"image": _crop(), "reference": _crop(), "paint": painted_box(SIZE, SIZE, _PAINT_BOX), "prompt": "a cat", "seed": 7, "steps": STEPS, "guidance": 3.0, **overrides}
    return backend.edit_region(model=pipe, **arguments)


class _Stop:
    """A skip signal that fires on its first check."""

    def check(self) -> bool:
        return True


class TestBlendCallback:
    """The callback produces ``(1 - m) * noised + m * latents`` with sigma from the scheduler, in Klein's packed layout."""

    @pytest.mark.parametrize("step", [0, 1, STEPS - 1])
    def test_outside_the_paint_the_clean_latents_are_noised_to_the_next_sigma(self, edit_module, step):
        noise, clean, mask = torch.randn(1, 128, 4, 4), torch.randn(1, 16, 128), torch.rand(1, 16, 1)
        latents = torch.randn(1, 16, 128)
        scheduler = diffusers.FlowMatchEulerDiscreteScheduler()
        scheduler.set_timesteps(STEPS, mu=0.5)
        sigma = float(scheduler.sigmas[step + 1])
        callback = edit_module.blend_callback(edit_module.Prepared(noise=noise, packed_noise=Flux2KleinPipeline._pack_latents(noise), clean=clean, mask=mask))

        result = callback(SimpleNamespace(scheduler=scheduler), step, None, {"latents": latents})

        expected = (1 - mask) * ((1 - sigma) * clean + sigma * Flux2KleinPipeline._pack_latents(noise)) + mask * latents
        assert torch.allclose(result["latents"], expected, atol=1e-6)

    def test_the_last_steps_latents_are_kept_before_the_pin(self, edit_module):
        scheduler = diffusers.FlowMatchEulerDiscreteScheduler()
        scheduler.set_timesteps(STEPS, mu=0.5)
        final: list[torch.Tensor] = []
        callback = edit_module.blend_callback(_prepared(clean=torch.zeros(1, 16, 128), mask=torch.zeros(1, 16, 1)), final)
        latents = torch.randn(1, 16, 128)

        for step in range(STEPS):
            result = callback(SimpleNamespace(scheduler=scheduler), step, None, {"latents": latents})

        assert len(final) == 1 and torch.equal(final[0], latents)  # the model's own, not the pinned (zero) clean latents
        assert torch.equal(result["latents"], torch.zeros(1, 16, 128))

    def test_the_result_keeps_the_latents_dtype(self, edit_module):
        scheduler = diffusers.FlowMatchEulerDiscreteScheduler()
        scheduler.set_timesteps(STEPS, mu=0.5)
        prepared = _prepared(clean=torch.randn(1, 16, 128), mask=torch.ones(1, 16, 1))

        result = edit_module.blend_callback(prepared)(SimpleNamespace(scheduler=scheduler), 0, None, {"latents": torch.randn(1, 16, 128, dtype=torch.bfloat16)})

        assert result["latents"].dtype == torch.bfloat16


class TestPrepare:
    """The noise has the shape the pipeline documents for ``latents``; clean and mask are in the callback's layout."""

    def test_klein_takes_patchified_latents_and_sees_packed_ones(self, edit_module):
        pipe = _FakePipe()

        prepared = edit_module.prepare(pipe, _crop(), painted_box(SIZE, SIZE, _PAINT_BOX), 7)

        cells = SIZE // 16
        assert prepared.noise.shape == (1, 128, cells, cells)
        assert prepared.clean.shape == (1, cells * cells, 128)
        assert prepared.mask.shape == (1, cells * cells, 1)

    def test_klein_mask_cells_are_sixteen_pixels_in_token_order(self, edit_module):
        paint = np.zeros((SIZE, SIZE), dtype=bool)
        paint[:16, 16:32] = True  # the second cell of the first row

        mask = edit_module.prepare(_FakePipe(), _crop(), paint, 7).mask

        assert mask.flatten().tolist() == [0.0, 1.0] + [0.0] * 14

    def test_the_noise_follows_the_seed_and_is_not_the_clean_latents(self, edit_module):
        first = edit_module.prepare(_FakePipe(), _crop(), painted_box(SIZE, SIZE, _PAINT_BOX), 7)
        again = edit_module.prepare(_FakePipe(), _crop(), painted_box(SIZE, SIZE, _PAINT_BOX), 7)
        other = edit_module.prepare(_FakePipe(), _crop(), painted_box(SIZE, SIZE, _PAINT_BOX), 8)

        assert torch.equal(first.noise, again.noise)
        assert not torch.equal(first.noise, other.noise)

    def test_the_encode_runs_in_inference_mode(self, edit_module):
        prepared = edit_module.prepare(_FakePipe(), _crop(), painted_box(SIZE, SIZE, _PAINT_BOX), 7)

        assert prepared.clean.is_inference() and prepared.noise.is_inference()


class TestEditRegion:
    def test_the_pipeline_gets_latents_of_its_documented_shape_at_the_crop_size(self, backend_module):
        pipe = _FakePipe()

        result = _edit(_backend(backend_module), pipe)

        call = pipe.calls[0]
        assert call["latents"].shape == (1, 128, SIZE // 16, SIZE // 16)
        assert (call["width"], call["height"]) == (SIZE, SIZE)
        assert call["num_inference_steps"] == STEPS
        assert result.size == (SIZE, SIZE)

    def test_the_reference_the_caller_gives_is_the_pipelines_reference_image_not_the_crop(self, backend_module):
        pipe, crop, reference = _FakePipe(), _crop(), Image.new("RGB", (48, 32), (200, 10, 10))

        _edit(_backend(backend_module), pipe, image=crop, reference=reference)

        assert pipe.calls[0]["image"] == [reference]

    def test_the_join_fix_decodes_the_last_steps_latents_before_the_pin_and_matches_the_strip(self, backend_module):
        pipe = _FakePipe()
        paint = np.zeros((SIZE, SIZE), dtype=bool)
        paint[:, :16] = True  # a strip on the left

        result = _edit(_backend(backend_module), pipe, paint=paint, fix_join=True)

        assert len(pipe.vae.decoded) == 1  # one extra decode
        assert pipe.vae.decoded[0].shape == (1, 32, SIZE // 8, SIZE // 8)
        assert result.size == (SIZE, SIZE)

    def test_without_the_join_fix_nothing_is_decoded(self, backend_module):
        pipe = _FakePipe()

        _edit(_backend(backend_module), pipe)

        assert pipe.vae.decoded == []

    def test_the_step_end_callback_always_carries_the_latents(self, backend_module):
        pipe = _FakePipe()

        _edit(_backend(backend_module), pipe)  # no step callback and no skip signal

        assert pipe.calls[0]["callback_on_step_end_tensor_inputs"] == ["latents"]
        assert callable(pipe.calls[0]["callback_on_step_end"])

    def test_every_step_is_blended_and_the_last_pins_the_clean_latents_outside_the_paint(self, backend_module, edit_module):
        pipe = _FakePipe()

        _edit(_backend(backend_module), pipe)

        expected = edit_module.prepare(_FakePipe(), _crop(), painted_box(SIZE, SIZE, _PAINT_BOX), 7)
        final = pipe.latents_seen[-1]
        mask = expected.mask.expand_as(final)
        assert len(pipe.latents_seen) == STEPS
        assert (mask == 0).any()
        assert (mask == 1).any()
        # Outside the paint the final latents are the source's; inside it they are the model's own (7.0).
        assert torch.equal(final[mask == 0], expected.clean.expand_as(final)[mask == 0])
        assert torch.all(final[mask == 1] == 7.0)

    def test_progress_is_reported_for_every_step(self, backend_module):
        pipe = _FakePipe()
        payloads: list[dict] = []

        _edit(_backend(backend_module), pipe, step_callback=payloads.append)

        assert [payload["current_step"] for payload in payloads] == list(range(1, STEPS + 1))

    def test_an_unloaded_backend_raises(self, backend_module):
        with pytest.raises(RuntimeError, match="load_model"):
            _edit(backend_module.DiffusersBackend(), _FakePipe())


class TestSchedulerAndStop:
    """The scheduler is put back after an edit; Stop returns None and clears the interrupt."""

    def test_a_beta_edit_swaps_the_scheduler_for_the_run_and_restores_it(self, backend_module):
        pipe = _FakePipe()
        original = pipe.scheduler

        _edit(_backend(backend_module), pipe, scheduler="beta")

        assert pipe.schedulers_seen[0] is not original
        assert pipe.calls[0]["sigmas"]
        assert pipe.scheduler is original

    def test_the_scheduler_is_restored_when_the_pipeline_raises(self, backend_module):
        pipe = _FakePipe(raises=True)
        original = pipe.scheduler

        with pytest.raises(RuntimeError, match="pipeline failed"):
            _edit(_backend(backend_module), pipe, scheduler="beta")

        assert pipe.scheduler is original

    def test_a_default_scheduler_edit_leaves_the_scheduler_alone(self, backend_module):
        pipe = _FakePipe()
        original = pipe.scheduler

        _edit(_backend(backend_module), pipe)

        assert pipe.schedulers_seen == [original]
        assert "sigmas" not in pipe.calls[0]
        assert pipe.scheduler is original

    def test_stop_returns_none_and_clears_the_interrupt(self, backend_module):
        pipe = _FakePipe()

        result = _edit(_backend(backend_module), pipe, skip_signal=_Stop())

        assert result is None
        assert pipe._interrupt is False

    def test_stop_ends_the_blend_with_the_interrupted_step(self, backend_module):
        pipe = _FakePipe()

        _edit(_backend(backend_module), pipe, skip_signal=_Stop())

        assert len(pipe.latents_seen) == 1  # the fake loop skips the steps after an interrupt, as the real one does

    def test_stop_during_an_extend_returns_none_without_running_the_join_fix(self, backend_module):
        pipe = _FakePipe()
        strip = painted_box(SIZE, SIZE, (0, 0, 16, SIZE))  # a strip on the left

        result = _edit(_backend(backend_module), pipe, paint=strip, fix_join=True, skip_signal=_Stop())

        assert result is None
        assert pipe.vae.decoded == []
        assert pipe._interrupt is False
