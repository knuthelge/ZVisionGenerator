"""Tests for cheap mflux latent previews and their wiring into the progress checker."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from zvisiongenerator.core.latent_preview import ZIMAGE_RGB_BIAS


pytestmark = pytest.mark.skipif(sys.platform != "darwin", reason="mflux/MLX backend is macOS-only")


@pytest.fixture()
def mx():
    pytest.importorskip("mflux")
    return pytest.importorskip("mlx.core")


@pytest.fixture()
def preview_module(mx):
    import zvisiongenerator.backends.image_mac_preview as module

    return module


def _klein_model(mx, channels: int = 128):
    bn = SimpleNamespace(running_mean=mx.zeros((channels,)), running_var=mx.ones((channels,)), eps=0.0)
    return SimpleNamespace(vae=SimpleNamespace(bn=bn))


class TestRenderLatentPreview:
    def test_zimage_preview_is_one_eighth_resolution_and_zero_latents_map_to_bias(self, mx, preview_module):
        latents = mx.zeros((16, 1, 32, 16))

        image = preview_module.render_latent_preview(None, "zimage", latents, height=256, width=128)

        assert image.size == (16, 32)
        expected = np.round(np.array(ZIMAGE_RGB_BIAS) * 255)
        assert np.allclose(np.asarray(image)[0, 0], expected, atol=1)

    def test_flux2_klein_unpacks_like_the_mflux_vae(self, mx, preview_module):
        from mflux.models.flux2.latent_creator.flux2_latent_creator import Flux2LatentCreator
        from mflux.models.flux2.model.flux2_vae.vae import Flux2VAE

        height, width = 128, 64
        latents = mx.random.normal((1, (height // 16) * (width // 16), 128))

        spatial = preview_module._flux2_klein_spatial(_klein_model(mx), latents, height, width)
        reference = Flux2VAE._unpatchify_latents(Flux2LatentCreator.unpack_latents(latents, height, width))[0]

        assert spatial.shape == (32, height // 8, width // 8)
        assert np.allclose(np.array(spatial), np.array(reference))

    def test_flux2_klein_and_ideogram4_previews_are_one_eighth_resolution(self, mx, preview_module):
        latents = mx.random.normal((1, 8 * 4, 128))

        klein = preview_module.render_latent_preview(_klein_model(mx), "flux2_klein", latents, height=128, width=64)
        ideogram = preview_module.render_latent_preview(None, "ideogram4", latents, height=128, width=64)

        assert klein.size == (8, 16)
        assert ideogram.size == (8, 16)

    def test_clean_estimate_recovers_x0_from_one_euler_step(self, mx, preview_module):
        x0 = mx.random.normal((16, 1, 8, 8))
        noise = mx.random.normal((16, 1, 8, 8))
        previous = 0.2 * x0 + 0.8 * noise
        current = 0.6 * x0 + 0.4 * noise

        estimate = preview_module.estimate_clean_latents(previous, current, 0.8, 0.4)

        assert np.allclose(np.array(estimate), np.array(x0), atol=1e-4)

    def test_clean_estimate_without_a_noise_change_returns_current(self, mx, preview_module):
        current = mx.ones((4, 4))
        assert preview_module.estimate_clean_latents(mx.zeros((4, 4)), current, 0.5, 0.5) is current

    def test_unknown_family_has_no_preview(self, mx, preview_module):
        assert preview_module.render_latent_preview(None, "flux1", mx.zeros((1, 4, 64)), height=64, width=64) is None


class TestProgressCheckerPreviews:
    @pytest.fixture()
    def image_mac(self, mx):
        import zvisiongenerator.backends.image_mac as module

        return module

    def _euler_scheduler(self, sigmas):
        from mflux.models.common.schedulers import LinearScheduler

        scheduler = object.__new__(LinearScheduler)
        scheduler._sigmas = sigmas
        return scheduler

    def _config(self, total_steps: int, *, init_time_step: int = 0):
        sigmas = np.linspace(1.0, 0.0, total_steps + 1)
        return SimpleNamespace(num_inference_steps=total_steps, init_time_step=init_time_step, height=64, width=64, scheduler=self._euler_scheduler(sigmas))

    def _run_steps(self, checker, *, total_steps: int, latents, init_time_step: int = 0):
        config = self._config(total_steps, init_time_step=init_time_step)
        checker.call_before_loop(1, "prompt", latents, config)
        for t in range(init_time_step, total_steps):
            checker.call_in_loop(t, 1, "prompt", latents, config, None)

    def test_previews_attach_only_at_quarter_milestones(self, mx, image_mac):
        events: list[dict] = []
        checker = image_mac._ProgressChecker(8, events.append, model=None, family="zimage")

        self._run_steps(checker, total_steps=8, latents=mx.zeros((16, 1, 8, 8)))

        assert [event["current_step"] for event in events if "preview" in event] == [2, 4, 6]
        assert all(event["preview"].size == (8, 8) for event in events if "preview" in event)

    def test_callbacks_accept_keywords_added_by_newer_mflux(self, mx, image_mac):
        # mflux 0.20 passes control_images to before-loop callbacks; a strict signature failed every generation.
        events: list[dict] = []
        checker = image_mac._ProgressChecker(4, events.append, model=None, family="zimage")
        config = self._config(4)
        latents = mx.zeros((16, 1, 8, 8))

        checker.call_before_loop(seed=1, prompt="prompt", latents=latents, config=config, canny_image=None, depth_image=None, control_images=None)
        checker.call_in_loop(t=0, seed=1, prompt="prompt", latents=latents, config=config, time_steps=None, future_keyword=None)
        image_mac._SkipChecker(SimpleNamespace(check=lambda: False)).call_in_loop(0, 1, "prompt", latents, config, None, future_keyword=None)

        assert events[-1]["current_step"] == 1

    def test_partial_runs_place_milestones_within_the_steps_actually_run(self, mx, image_mac):
        events: list[dict] = []
        checker = image_mac._ProgressChecker(8, events.append, model=None, family="zimage")

        # An upscale/refine at denoise 0.5 over 16 scheduled steps runs steps 9..16.
        self._run_steps(checker, total_steps=16, latents=mx.zeros((16, 1, 8, 8)), init_time_step=8)

        assert [event["current_step"] for event in events if "preview" in event] == [2, 4, 6]

    def test_partial_runs_report_only_the_steps_actually_run(self, mx, image_mac):
        events: list[dict] = []
        # Upscale refinement: 3 steps at denoise 0.4 are scheduled as 7, of which the last 3 run.
        checker = image_mac._ProgressChecker(7, events.append)

        self._run_steps(checker, total_steps=7, latents=mx.zeros((16, 1, 8, 8)), init_time_step=4)

        assert [(event["current_step"], event["total_steps"]) for event in events] == [(1, 3), (2, 3), (3, 3)]

    def test_previews_show_the_predicted_clean_image_not_the_noisy_latents(self, mx, image_mac, monkeypatch):
        rendered: list = []
        monkeypatch.setattr(image_mac, "render_latent_preview", lambda _model, _family, latents, *_size: rendered.append(latents))
        x0 = mx.random.normal((16, 1, 8, 8))
        noise = mx.random.normal((16, 1, 8, 8))
        config = self._config(4)
        sigmas = config.scheduler.sigmas
        checker = image_mac._ProgressChecker(4, lambda _event: None, model=None, family="zimage")

        checker.call_before_loop(1, "prompt", noise, config)
        for t in range(3):
            sigma = float(sigmas[t + 1])
            checker.call_in_loop(t, 1, "prompt", (1 - sigma) * x0 + sigma * noise, config, None)

        assert len(rendered) == 3
        assert all(np.allclose(np.array(latents), np.array(x0), atol=1e-4) for latents in rendered)

    def test_unknown_schedulers_preview_raw_latents(self, mx, image_mac, monkeypatch):
        rendered: list = []
        monkeypatch.setattr(image_mac, "render_latent_preview", lambda _model, _family, latents, *_size: rendered.append(latents))
        noisy = mx.random.normal((16, 1, 8, 8))
        config = self._config(4)
        config.scheduler = SimpleNamespace(sigmas=config.scheduler.sigmas)
        checker = image_mac._ProgressChecker(4, lambda _event: None, model=None, family="zimage")

        checker.call_before_loop(1, "prompt", mx.zeros((16, 1, 8, 8)), config)
        checker.call_in_loop(0, 1, "prompt", noisy, config, None)

        assert image_mac._step_noise_levels("zimage", 0, config) is None
        assert rendered == [noisy]

    def test_beta_scheduler_counts_as_euler(self, image_mac):
        from zvisiongenerator.schedulers.beta_scheduler import BetaScheduler

        assert image_mac._is_euler_scheduler(object.__new__(BetaScheduler))
        assert not image_mac._is_euler_scheduler(SimpleNamespace(sigmas=[1.0, 0.0]))

    def test_euler_check_never_imports_the_beta_scheduler(self, image_mac, monkeypatch):
        monkeypatch.delitem(sys.modules, "zvisiongenerator.schedulers.beta_scheduler", raising=False)

        assert not image_mac._is_euler_scheduler(SimpleNamespace(sigmas=[1.0, 0.0]))
        assert "zvisiongenerator.schedulers.beta_scheduler" not in sys.modules

    def test_previous_latents_are_kept_only_ahead_of_a_preview_step(self, mx, image_mac):
        checker = image_mac._ProgressChecker(8, lambda _event: None, model=None, family="zimage")
        config = self._config(8)
        latents = mx.zeros((16, 1, 8, 8))
        checker.call_before_loop(1, "prompt", latents, config)

        kept = []
        for t in range(8):
            checker.call_in_loop(t, 1, "prompt", latents, config, None)
            kept.append(checker._previous_latents is not None)

        # Previews land on steps 2, 4 and 6, so only steps 1, 3 and 5 keep their latents.
        assert kept == [True, False, True, False, True, False, False, False]

    def test_ideogram4_noise_levels_come_from_the_recorded_schedule(self, mx, image_mac, monkeypatch):
        monkeypatch.setattr(image_mac._ideogram4_timesteps, "value", (np.array([0.9, 0.5, 0.1]), np.array([1.0, 0.9, 0.5])), raising=False)

        config = SimpleNamespace(num_inference_steps=3)

        assert image_mac._step_noise_levels("ideogram4", 0, config) == pytest.approx((0.9, 0.5))
        assert image_mac._step_noise_levels("ideogram4", 2, config) == pytest.approx((0.1, 0.0))
        assert image_mac._step_noise_levels("ideogram4", 0, SimpleNamespace(num_inference_steps=20)) is None

    def test_text_to_image_clears_the_recorded_ideogram4_schedule(self, image_mac, monkeypatch):
        from unittest.mock import MagicMock

        monkeypatch.setattr(image_mac._ideogram4_timesteps, "value", (np.array([0.5]), np.array([1.0])), raising=False)
        backend = image_mac.MfluxBackend()
        backend._model_info = SimpleNamespace(family="zimage")

        backend.text_to_image(MagicMock(), "prompt", 64, 64, seed=1, steps=1, guidance=None, step_callback=lambda _event: None)

        assert image_mac._ideogram4_timesteps.value is None

    def test_unregister_callback_removes_it_from_every_registry_list(self, image_mac):
        from mflux.callbacks.callback_registry import CallbackRegistry

        checker = image_mac._ProgressChecker(4, lambda _event: None)
        model = SimpleNamespace(callbacks=CallbackRegistry())
        model.callbacks.register(checker)

        image_mac._unregister_callback(model, checker)
        image_mac._unregister_callback(model, checker)

        registry = model.callbacks
        assert checker not in registry.before_loop + registry.in_loop + registry.after_loop + registry.interrupt

    def test_no_family_means_no_previews(self, mx, image_mac):
        events: list[dict] = []
        checker = image_mac._ProgressChecker(4, events.append)

        self._run_steps(checker, total_steps=4, latents=mx.zeros((16, 1, 8, 8)))

        assert len(events) == 4
        assert not any("preview" in event for event in events)

    def test_no_latents_are_kept_once_previews_are_disabled(self, mx, image_mac):
        checker = image_mac._ProgressChecker(8, lambda _event: None, model=None, family="zimage")
        config = self._config(8)
        checker.call_before_loop(1, "prompt", mx.zeros((16, 1, 8, 8)), config)
        checker._previews_enabled = False

        # Step 3 precedes the step-4 preview milestone, which would normally keep its latents.
        checker.call_in_loop(2, 1, "prompt", mx.zeros((16, 1, 8, 8)), config, None)

        assert checker._previous_latents is None

    def test_a_before_loop_failure_disables_previews_without_failing_generation(self, mx, image_mac):
        events: list[dict] = []
        checker = image_mac._ProgressChecker(4, events.append, model=None, family="zimage")
        broken_config = SimpleNamespace(num_inference_steps=4, height=64, width=64)  # no init_time_step

        with pytest.warns(UserWarning, match="Live previews disabled"):
            checker.call_before_loop(1, "prompt", mx.zeros((16, 1, 8, 8)), broken_config)
        for t in range(4):
            checker.call_in_loop(t, 1, "prompt", mx.zeros((16, 1, 8, 8)), self._config(4), None)

        assert len(events) == 4
        assert not any("preview" in event for event in events)

    def test_preview_failure_warns_once_and_keeps_reporting_progress(self, mx, image_mac, monkeypatch):
        def _boom(*_args, **_kwargs):
            raise ValueError("bad latents")

        monkeypatch.setattr(image_mac, "render_latent_preview", _boom)
        events: list[dict] = []
        checker = image_mac._ProgressChecker(8, events.append, model=None, family="zimage")

        with pytest.warns(UserWarning, match="Live preview failed") as caught:
            self._run_steps(checker, total_steps=8, latents=mx.zeros((16, 1, 8, 8)))

        assert len(caught) == 1
        assert len(events) == 8
        assert not any("preview" in event for event in events)
