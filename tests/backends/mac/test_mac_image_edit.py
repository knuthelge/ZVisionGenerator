"""Tests for the mflux latent blend with fakes: the mask layout, the formula, and how the hook installs and comes off.

NumPy arrays stand in for MLX arrays and a small class for mflux's scheduler, so this runs everywhere. ``test_mac_image_edit_real.py`` runs the same blend through mflux's real loops.
"""

from __future__ import annotations

import importlib.util
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from tests._images import painted_box
from tests._paths import REPO_ROOT
from zvisiongenerator.backends.mac import image_edit
from zvisiongenerator.backends.mac.image_edit import Prepared
from zvisiongenerator.processing.edit_region import soft_cell_mask

HEIGHT, WIDTH = 32, 48
LATENTS_SHAPE = (1, 6, 128)


# One whole 16 px cell (0, 1), a quarter of cell (1, 0) and an eighth of cell (1, 1).
_PAINT_BOXES = ((16, 0, 32, 16), (0, 16, 8, 24), (24, 28, 32, 32))


class _Stop(Exception):
    """Stands in for mflux's ``StopImageGenerationException``."""


def _prepared(seed: int = 0) -> Prepared:
    rng = np.random.default_rng(seed)
    return Prepared(
        clean=rng.normal(size=LATENTS_SHAPE).astype(np.float32),
        noise=rng.normal(size=LATENTS_SHAPE).astype(np.float32),
        mask=soft_cell_mask(painted_box(HEIGHT, WIDTH, *_PAINT_BOXES)).reshape(1, -1, 1),
    )


class TestBlend:
    @pytest.mark.parametrize("sigma", [0.0, 0.37, 1.0])
    def test_outside_the_paint_it_is_the_source_noised_to_sigma_and_inside_it_is_the_latents(self, sigma):
        prepared = _prepared()
        latents = np.random.default_rng(5).normal(size=prepared.clean.shape).astype(np.float32)

        blended = image_edit.blend(latents, prepared, sigma)

        mask = np.broadcast_to(prepared.mask, latents.shape)
        noised = (1 - sigma) * prepared.clean + sigma * prepared.noise
        np.testing.assert_allclose(blended, (1 - mask) * noised + mask * latents, atol=1e-6)
        np.testing.assert_allclose(blended[mask == 1], latents[mask == 1])
        np.testing.assert_allclose(blended[mask == 0], noised[mask == 0], atol=1e-6)
        assert (mask == 0).any() and (mask == 1).any() and ((mask > 0) & (mask < 1)).any()

    def test_the_result_has_the_latents_dtype(self):
        prepared = _prepared()

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
    @pytest.mark.parametrize("with_sigmas", [False, True], ids=["plain", "klein"])
    def test_a_loop_through_the_wrapper_blends_at_the_next_sigma(self, with_sigmas):
        prepared = _prepared()
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

    def test_only_the_runs_own_scheduler_is_wrapped(self):
        first, second = SimpleNamespace(scheduler=_Scheduler()), SimpleNamespace(scheduler=_Scheduler())
        image_edit._BlendInstaller(_prepared()).call_before_loop(seed=1, prompt="p", latents=None, config=first)

        assert "step" in vars(first.scheduler)
        assert "step" not in vars(second.scheduler)


class _Callbacks:
    """The slice of mflux's callback registry the edit hook uses."""

    def __init__(self) -> None:
        self.before_loop: list[object] = []

    def register(self, callback) -> None:
        if hasattr(callback, "call_before_loop"):
            self.before_loop.append(callback)


class TestEditBlend:
    def test_the_installer_is_registered_inside_the_block_and_gone_after(self):
        model = SimpleNamespace(callbacks=_Callbacks())

        with image_edit.edit_blend(model, _prepared()):
            assert [type(callback) for callback in model.callbacks.before_loop] == [image_edit._BlendInstaller]

        assert model.callbacks.before_loop == []

    def test_the_installer_is_gone_after_the_block_raises(self):
        model = SimpleNamespace(callbacks=_Callbacks())

        with pytest.raises(_Stop), image_edit.edit_blend(model, _prepared()):
            raise _Stop

        assert model.callbacks.before_loop == []

    def test_other_callbacks_stay_registered(self):
        other = SimpleNamespace(call_before_loop=lambda **_: None)
        model = SimpleNamespace(callbacks=_Callbacks())
        model.callbacks.register(other)

        with image_edit.edit_blend(model, _prepared()):
            pass

        assert model.callbacks.before_loop == [other]


def test_a_stop_during_an_extend_returns_none_without_running_the_join_fix(monkeypatch):
    """A skip ends the run with no picture: the join fix, which needs one, must not run on the stopped run's latents."""
    pytest.importorskip("mflux")
    from contextlib import contextmanager

    from PIL import Image

    from zvisiongenerator.backends.mac import image as mac_image
    from zvisiongenerator.utils.image_model_detect import ImageModelInfo

    @contextmanager
    def blend_that_saw_a_step(model, prepared, final):
        final.append(prepared.clean)  # the blend records the last step's latents as the first step runs
        yield

    monkeypatch.setattr(mac_image, "prepare_edit", lambda model, image, paint, seed: _prepared())
    monkeypatch.setattr(mac_image, "edit_blend", blend_that_saw_a_step)
    monkeypatch.setattr(mac_image, "match_strip", lambda *args, **kwargs: pytest.fail("the join fix ran on a stopped run"))
    monkeypatch.setattr(mac_image.MfluxBackend, "text_to_image", lambda self, *args, **kwargs: None)  # Stop: no picture
    backend = mac_image.MfluxBackend()
    backend._model_info = ImageModelInfo(family="flux2_klein", is_distilled=True, size=None)
    crop = Image.new("RGB", (WIDTH, HEIGHT))

    result = backend.edit_region(object(), crop, crop, painted_box(HEIGHT, WIDTH, (0, 0, 16, HEIGHT)), "p", 1, 4, 1.0, fix_join=True)

    assert result is None


def test_the_module_imports_without_mlx_or_mflux(monkeypatch):
    """The Mac backend's tests run on Linux, so nothing in ``image_edit`` may import mlx or mflux when the module loads."""
    for name in ("mlx", "mlx.core", "mflux"):
        monkeypatch.setitem(sys.modules, name, None)
    spec = importlib.util.spec_from_file_location("_image_edit_without_mlx", REPO_ROOT / "zvisiongenerator" / "backends" / "mac" / "image_edit.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)  # dataclasses looks its module up here

    spec.loader.exec_module(module)

    assert callable(module.prepare) and callable(module.edit_blend)
