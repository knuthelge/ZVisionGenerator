"""Tests for cheap diffusers latent previews and their milestone tracking."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from zvisiongenerator.core.latent_preview import FLUX2_RGB_BIAS, QWEN_IMAGE_RGB_BIAS, ZIMAGE_RGB_BIAS

torch = pytest.importorskip("torch")
diffusers = pytest.importorskip("diffusers")


@pytest.fixture()
def preview_module():
    import zvisiongenerator.backends.image_win_preview as module

    return module


def _flux2_pipe(channels: int = 128, *, scheduler=None):
    bn = SimpleNamespace(running_mean=torch.zeros(channels), running_var=torch.ones(channels))
    vae = SimpleNamespace(bn=bn, config=SimpleNamespace(batch_norm_eps=0.0))
    return SimpleNamespace(vae=vae, vae_scale_factor=8, scheduler=scheduler, num_timesteps=None)


def _euler_scheduler(steps: int):
    scheduler = diffusers.FlowMatchEulerDiscreteScheduler()
    scheduler.set_timesteps(steps)
    return scheduler


def _pixel(image, bias):
    return np.asarray(image)[0, 0], np.round(np.array(bias) * 255)


class TestRenderLatentPreview:
    def test_zimage_preview_is_one_eighth_resolution_and_zero_latents_map_to_bias(self, preview_module):
        latents = torch.zeros((1, 16, 32, 16))

        image = preview_module.render_latent_preview(None, "zimage", latents, height=256, width=128)

        assert image.size == (16, 32)
        actual, expected = _pixel(image, ZIMAGE_RGB_BIAS)
        assert np.allclose(actual, expected, atol=1)

    @pytest.mark.parametrize("family", ["flux2", "flux2_klein"])
    def test_flux2_preview_is_one_eighth_resolution_and_zero_latents_map_to_bias(self, preview_module, family):
        latents = torch.zeros((1, 8 * 4, 128))

        image = preview_module.render_latent_preview(_flux2_pipe(), family, latents, height=128, width=64)

        assert image.size == (8, 16)
        actual, expected = _pixel(image, FLUX2_RGB_BIAS)
        assert np.allclose(actual, expected, atol=1)

    def test_flux2_unpacks_like_the_diffusers_pipeline(self, preview_module):
        from diffusers.pipelines.flux2.pipeline_flux2_klein import Flux2KleinPipeline

        height, width = 128, 64
        spatial_in = torch.randn((1, 32, height // 8, width // 8))
        packed = Flux2KleinPipeline._pack_latents(Flux2KleinPipeline._patchify_latents(spatial_in))

        spatial = preview_module._flux2_spatial(_flux2_pipe(), packed, height, width)

        assert spatial.shape == (32, height // 8, width // 8)
        assert torch.allclose(spatial, spatial_in[0])

    def test_flux2_undoes_batch_norm(self, preview_module):
        pipe = _flux2_pipe()
        pipe.vae.bn.running_mean = torch.full((128,), 0.5)
        pipe.vae.bn.running_var = torch.full((128,), 4.0)

        spatial = preview_module._flux2_spatial(pipe, torch.ones((1, 8 * 4, 128)), height=128, width=64)

        assert torch.allclose(spatial, torch.full_like(spatial, 2.5))

    def test_flux2_rejects_a_token_count_that_does_not_match_the_size(self, preview_module):
        with pytest.raises(ValueError, match="latent tokens"):
            preview_module._flux2_spatial(_flux2_pipe(), torch.zeros((1, 10, 128)), height=128, width=64)

    def test_krea2_preview_is_one_eighth_resolution_and_zero_latents_map_to_bias(self, preview_module):
        pipe = SimpleNamespace(patch_size=2, vae_scale_factor=8)
        latents = torch.zeros((1, 8 * 4, 64))

        image = preview_module.render_latent_preview(pipe, "krea2", latents, height=128, width=64)

        assert image.size == (8, 16)
        actual, expected = _pixel(image, QWEN_IMAGE_RGB_BIAS)
        assert np.allclose(actual, expected, atol=1)

    def test_krea2_unpacks_like_the_diffusers_pipeline(self, preview_module):
        from diffusers.pipelines.krea2.pipeline_krea2 import Krea2Pipeline

        pipe = SimpleNamespace(patch_size=2, vae_scale_factor=8)
        height, width = 128, 64
        spatial_in = torch.randn((1, 16, height // 8, width // 8))
        packed = Krea2Pipeline._pack_latents(pipe, spatial_in, 1, 16, height // 8, width // 8)

        spatial = preview_module._krea2_spatial(pipe, packed, height, width)

        assert torch.allclose(spatial, spatial_in[0])

    def test_unknown_family_has_no_preview(self, preview_module):
        assert preview_module.render_latent_preview(None, "flux1", torch.zeros((1, 16, 4, 4)), 32, 32) is None


class TestEstimateCleanLatents:
    def test_recovers_clean_latents_from_one_euler_step(self, preview_module):
        clean, noise = torch.randn(4, 4), torch.randn(4, 4)
        previous = 0.2 * clean + 0.8 * noise
        current = 0.5 * clean + 0.5 * noise

        estimate = preview_module.estimate_clean_latents(previous, current, 0.8, 0.5)

        assert torch.allclose(estimate, clean, atol=1e-5)

    def test_returns_current_when_noise_level_does_not_change(self, preview_module):
        current = torch.randn(4, 4)

        assert preview_module.estimate_clean_latents(torch.randn(4, 4), current, 0.5, 0.5) is current


class TestCreateLivePreview:
    @pytest.mark.parametrize("family", ["zimage", "flux2", "flux2_klein", "krea2"])
    def test_supported_families(self, preview_module, family):
        assert isinstance(preview_module.create_live_preview(family, 8, 64, 64), preview_module.LivePreview)

    @pytest.mark.parametrize("family", ["flux1", "ideogram4", "unknown"])
    def test_unsupported_families(self, preview_module, family):
        assert preview_module.create_live_preview(family, 8, 64, 64) is None


class TestLivePreview:
    def _run(self, preview, pipe, steps, latents_for_step):
        previews = {}
        for step in range(steps):
            image = preview.observe(pipe, step, latents_for_step(step))
            if image is not None:
                previews[step + 1] = image
        return previews

    def test_previews_only_at_milestone_steps(self, preview_module):
        pipe = SimpleNamespace(scheduler=None, num_timesteps=8)
        preview = preview_module.LivePreview("zimage", 8, 64, 64)

        previews = self._run(preview, pipe, 8, lambda _: torch.zeros((1, 16, 8, 8)))

        assert set(previews) == {2, 4, 6}

    def test_milestones_follow_the_steps_the_pipeline_actually_runs(self, preview_module):
        # img2img at strength 0.5 runs 4 of 8 requested steps.
        pipe = SimpleNamespace(scheduler=None, num_timesteps=4)
        preview = preview_module.LivePreview("zimage", 8, 64, 64)

        previews = self._run(preview, pipe, 4, lambda _: torch.zeros((1, 16, 8, 8)))

        assert set(previews) == {1, 2, 3}

    def test_milestone_previews_show_the_predicted_clean_image(self, preview_module):
        steps = 4
        scheduler = _euler_scheduler(steps)
        pipe = SimpleNamespace(scheduler=scheduler, num_timesteps=steps)
        clean = torch.full((1, 16, 8, 8), 3.0)
        noise = torch.randn((1, 16, 8, 8))

        def latents_for_step(step):
            scheduler._step_index = step + 1
            sigma = float(scheduler.sigmas[step + 1])
            return (1 - sigma) * clean + sigma * noise

        preview = preview_module.LivePreview("zimage", steps, 64, 64)
        previews = self._run(preview, pipe, steps, latents_for_step)
        expected = preview_module.render_latent_preview(None, "zimage", clean, 64, 64)

        # Step 1 has no earlier latents to estimate from; later milestones recover the clean image.
        for step in (2, 3):
            assert np.allclose(np.asarray(previews[step], dtype=int), np.asarray(expected, dtype=int), atol=1)

    def test_failure_warns_once_and_disables_previews(self, preview_module):
        pipe = SimpleNamespace(scheduler=None, num_timesteps=8, vae_scale_factor=8)
        preview = preview_module.LivePreview("flux2_klein", 8, 64, 64)

        with pytest.warns(UserWarning, match="Live preview failed"):
            first = self._run(preview, pipe, 8, lambda _: torch.zeros((1, 3, 128)))

        assert first == {}
        assert preview.observe(pipe, 1, torch.zeros((1, 16, 128))) is None

    def test_missing_latents_yield_no_preview(self, preview_module):
        preview = preview_module.LivePreview("zimage", 8, 64, 64)

        assert preview.observe(SimpleNamespace(scheduler=None, num_timesteps=8), 1, None) is None
