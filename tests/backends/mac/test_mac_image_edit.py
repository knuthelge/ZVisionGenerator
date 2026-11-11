"""Tests for the mflux latent blend with fakes: mask layouts, the formula, and how the hooks install and come off.

NumPy arrays stand in for MLX arrays and small classes for mflux's scheduler and Krea 2 sampler, so this runs
everywhere. ``test_mac_image_edit_real.py`` runs the same blend through mflux's real loops.
"""

from __future__ import annotations

import importlib.util
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from tests._paths import REPO_ROOT
from zvisiongenerator.backends.mac import image_edit
from zvisiongenerator.backends.mac.image_edit import Prepared
from zvisiongenerator.processing.edit_region import EDIT_CELL_PIXELS

HEIGHT, WIDTH = 32, 48
LAYOUTS = {
    # family: (latents shape, mask shape)
    "zimage": ((16, 1, 4, 6), (1, 1, 4, 6)),
    "flux2_klein": ((1, 6, 128), (1, 6, 1)),
    "krea2": ((1, 16, 4, 6), (1, 1, 4, 6)),
}


def _paint() -> np.ndarray:
    paint = np.zeros((HEIGHT, WIDTH), dtype=bool)
    paint[0:16, 16:32] = True  # one whole 16 px cell: cell (0, 1)
    paint[16:24, 0:8] = True  # a quarter of 16 px cell (1, 0), a whole 8 px cell
    paint[28:32, 24:32] = True  # an eighth of 16 px cell (1, 1), half of 8 px cell (3, 3)
    return paint


class _Stop(Exception):
    """Stands in for mflux's ``StopImageGenerationException``."""


class TestLatentMask:
    def test_zimage_mask_is_the_painted_fraction_of_each_8px_cell(self):
        mask = image_edit.latent_mask("zimage", _paint())

        assert mask.shape == (1, 1, 4, 6)
        assert mask[0, 0, 0, 2] == 1.0 and mask[0, 0, 0, 3] == 1.0 and mask[0, 0, 1, 2] == 1.0  # the 16 px block is four 8 px cells
        assert mask[0, 0, 2, 0] == 1.0 and mask[0, 0, 3, 0] == 0.0 and mask[0, 0, 3, 3] == 0.5
        assert mask.dtype == np.float32

    def test_klein_mask_has_one_row_per_token_in_row_major_order(self):
        mask = image_edit.latent_mask("flux2_klein", _paint())

        assert mask.shape == (1, 6, 1)
        assert mask[0, :, 0].tolist() == [0.0, 1.0, 0.0, 0.25, 0.125, 0.0]  # grid (2 rows x 3 columns) read left to right, top to bottom

    def test_krea2_mask_repeats_each_16px_cell_over_its_2x2_latent_pixels(self):
        mask = image_edit.latent_mask("krea2", _paint())

        assert mask.shape == (1, 1, 4, 6)
        assert mask[0, 0].tolist() == [
            [0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
            [0.25, 0.25, 0.125, 0.125, 0.0, 0.0],
            [0.25, 0.25, 0.125, 0.125, 0.0, 0.0],
        ]

    def test_a_mask_that_is_not_whole_cells_is_refused(self):
        with pytest.raises(ValueError, match="whole number"):
            image_edit.latent_mask("krea2", np.zeros((24, 48), dtype=bool))

    def test_every_edit_family_has_a_layout_here(self):
        assert set(EDIT_CELL_PIXELS) == set(LAYOUTS)


def _prepared(family: str, seed: int = 0) -> Prepared:
    latents_shape, _ = LAYOUTS[family]
    rng = np.random.default_rng(seed)
    return Prepared(
        clean=rng.normal(size=latents_shape).astype(np.float32),
        noise=rng.normal(size=latents_shape).astype(np.float32),
        mask=image_edit.latent_mask(family, _paint()),
    )


class TestBlend:
    @pytest.mark.parametrize("family", LAYOUTS)
    @pytest.mark.parametrize("sigma", [0.0, 0.37, 1.0])
    def test_outside_the_paint_it_is_the_source_noised_to_sigma_and_inside_it_is_the_latents(self, family, sigma):
        prepared = _prepared(family)
        latents = np.random.default_rng(5).normal(size=prepared.clean.shape).astype(np.float32)

        blended = image_edit.blend(latents, prepared, sigma)

        mask = np.broadcast_to(prepared.mask, latents.shape)
        noised = (1 - sigma) * prepared.clean + sigma * prepared.noise
        np.testing.assert_allclose(blended, (1 - mask) * noised + mask * latents, atol=1e-6)
        np.testing.assert_allclose(blended[mask == 1], latents[mask == 1])
        np.testing.assert_allclose(blended[mask == 0], noised[mask == 0], atol=1e-6)
        assert (mask == 0).any() and (mask == 1).any() and ((mask > 0) & (mask < 1)).any()

    def test_sigma_zero_pins_the_clean_latents_outside_the_paint(self):
        prepared = _prepared("krea2")
        blended = image_edit.blend(np.zeros_like(prepared.clean), prepared, 0.0)

        outside = np.broadcast_to(prepared.mask, blended.shape) == 0
        np.testing.assert_array_equal(blended[outside], prepared.clean[outside])

    def test_the_result_has_the_latents_dtype(self):
        prepared = _prepared("flux2_klein")

        blended = image_edit.blend(np.zeros(prepared.clean.shape, dtype=np.float16), prepared, 0.5)

        assert blended.dtype == np.float16


class _Scheduler:
    """A scheduler with mflux's step signature that records the keywords it was called with."""

    sigmas = np.array([1.0, 0.6, 0.3, 0.0], dtype=np.float32)

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def step(self, noise, timestep, latents, **kwargs):
        self.calls.append({"timestep": timestep, **kwargs})
        return latents + noise


def _loop(config, prepared: Prepared, *, with_sigmas: bool):
    """Run mflux's loop on *config*: the scheduler step is called by keyword, with ``sigmas=`` as Klein does."""
    latents = prepared.noise
    seen = []
    for t in range(3):
        extra = {"sigmas": config.scheduler.sigmas} if with_sigmas else {}
        latents = config.scheduler.step(noise=np.ones_like(latents), timestep=t, latents=latents, **extra)
        seen.append(latents)
    return seen


class TestSchedulerInstaller:
    @pytest.mark.parametrize("with_sigmas", [False, True], ids=["zimage", "klein"])
    def test_a_loop_through_the_wrapper_blends_at_the_next_sigma(self, with_sigmas):
        prepared = _prepared("flux2_klein")
        config = SimpleNamespace(scheduler=_Scheduler())
        original = config.scheduler.step
        installer = image_edit._BlendInstaller(prepared)

        # mflux's GenerationContext passes these keywords to every before-loop callback.
        installer.call_before_loop(seed=1, prompt="p", latents=prepared.noise, config=config, canny_image=None, depth_image=None, control_images=None)
        seen = _loop(config, prepared, with_sigmas=with_sigmas)

        previous = prepared.noise
        for t, latents in enumerate(seen):
            stepped = previous + 1.0
            expected = image_edit.blend(stepped, prepared, float(_Scheduler.sigmas[t + 1]))
            np.testing.assert_allclose(latents, expected, atol=1e-6)
            previous = latents
        assert [call["timestep"] for call in config.scheduler.calls] == [0, 1, 2]
        if with_sigmas:  # the keyword Klein passes reaches the scheduler's own step
            assert all(call["sigmas"] is config.scheduler.sigmas for call in config.scheduler.calls)
        else:
            assert all("sigmas" not in call for call in config.scheduler.calls)
        assert config.scheduler.step is not original

    def test_the_wrapper_takes_only_keywords(self):
        config = SimpleNamespace(scheduler=_Scheduler())
        image_edit._BlendInstaller(_prepared("zimage")).call_before_loop(seed=1, prompt="p", latents=None, config=config)

        with pytest.raises(TypeError):
            config.scheduler.step(np.zeros(1), 0, np.zeros(1))

    def test_only_the_runs_own_scheduler_is_wrapped(self):
        first, second = SimpleNamespace(scheduler=_Scheduler()), SimpleNamespace(scheduler=_Scheduler())
        image_edit._BlendInstaller(_prepared("zimage")).call_before_loop(seed=1, prompt="p", latents=None, config=first)

        assert "step" in vars(first.scheduler)
        assert "step" not in vars(second.scheduler)


class _Callbacks:
    """The slice of mflux's callback registry the edit hook uses."""

    def __init__(self) -> None:
        self.before_loop: list[object] = []

    def register(self, callback) -> None:
        if hasattr(callback, "call_before_loop"):
            self.before_loop.append(callback)


class TestEditBlendForSchedulerFamilies:
    @pytest.mark.parametrize("family", ["zimage", "flux2_klein"])
    def test_the_installer_is_registered_inside_the_block_and_gone_after(self, family):
        model = SimpleNamespace(callbacks=_Callbacks())

        with image_edit.edit_blend(model, family, _prepared(family)):
            assert [type(callback) for callback in model.callbacks.before_loop] == [image_edit._BlendInstaller]

        assert model.callbacks.before_loop == []

    def test_the_installer_is_gone_after_the_block_raises(self):
        model = SimpleNamespace(callbacks=_Callbacks())

        with pytest.raises(_Stop), image_edit.edit_blend(model, "flux2_klein", _prepared("flux2_klein")):
            raise _Stop

        assert model.callbacks.before_loop == []

    def test_other_callbacks_stay_registered(self):
        other = SimpleNamespace(call_before_loop=lambda **_: None)
        model = SimpleNamespace(callbacks=_Callbacks())
        model.callbacks.register(other)

        with image_edit.edit_blend(model, "zimage", _prepared("zimage")):
            pass

        assert model.callbacks.before_loop == [other]


class _Stepper:
    def __init__(self, sigmas) -> None:
        self.sigmas = sigmas

    def step(self, i, x, v, denoised):
        return x + (self.sigmas[i + 1] - self.sigmas[i]) * v


@pytest.fixture
def sampler(monkeypatch):
    """Return a stand-in for Krea 2's sampler, which keeps a static ``make_stepper(name, sigmas, seed)`` in its own class body."""

    class Sampler:
        @staticmethod
        def make_stepper(name, sigmas, seed):
            return _Stepper(sigmas)

    monkeypatch.setattr(image_edit, "_krea2_sampler", lambda: Sampler)
    return Sampler


class TestKrea2Stepper:
    def test_a_stepper_built_inside_the_block_blends_at_the_next_sigma(self, sampler):
        prepared = _prepared("krea2")
        sigmas = np.array([1.0, 0.5, 0.0], dtype=np.float32)
        v = np.ones(prepared.clean.shape, dtype=np.float32)

        with image_edit.edit_blend(SimpleNamespace(), "krea2", prepared):
            stepper = sampler.make_stepper("euler", sigmas, 3)
            first = stepper.step(0, prepared.noise, v, None)
            last = stepper.step(1, first, v, None)

        np.testing.assert_allclose(first, image_edit.blend(prepared.noise + (0.5 - 1.0) * v, prepared, 0.5), atol=1e-6)
        np.testing.assert_allclose(last, image_edit.blend(first + (0.0 - 0.5) * v, prepared, 0.0), atol=1e-6)

    def test_make_stepper_is_restored_after_the_block(self, sampler):
        saved = vars(sampler)["make_stepper"]

        with image_edit.edit_blend(SimpleNamespace(), "krea2", _prepared("krea2")):
            assert vars(sampler)["make_stepper"] is not saved

        assert vars(sampler)["make_stepper"] is saved
        assert "step" not in vars(sampler.make_stepper("euler", np.zeros(2), 0))  # a stepper built after the block is not wrapped

    @pytest.mark.parametrize("error", [_Stop, RuntimeError])
    def test_make_stepper_is_restored_when_the_block_raises(self, sampler, error):
        saved = vars(sampler)["make_stepper"]

        with pytest.raises(error), image_edit.blended_stepper(_prepared("krea2")):
            raise error

        assert vars(sampler)["make_stepper"] is saved
        assert "step" not in vars(sampler.make_stepper("euler", np.zeros(2), 0))


def test_the_module_imports_without_mlx_or_mflux(monkeypatch):
    """The Mac backend's tests run on Linux, so nothing in ``image_edit`` may import mlx or mflux when the module loads."""
    for name in ("mlx", "mlx.core", "mflux"):
        monkeypatch.setitem(sys.modules, name, None)
    spec = importlib.util.spec_from_file_location("_image_edit_without_mlx", REPO_ROOT / "zvisiongenerator" / "backends" / "mac" / "image_edit.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)  # dataclasses looks its module up here

    spec.loader.exec_module(module)

    assert module.latent_mask("zimage", _paint()).shape == LAYOUTS["zimage"][1]
