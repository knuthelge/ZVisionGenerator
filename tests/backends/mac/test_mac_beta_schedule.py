"""Tests for the beta sigma schedule on the mflux backend: Krea 2's Euler sampler and FLUX.2 Klein's shift."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest
from PIL import Image

from zvisiongenerator.utils.image_model_detect import ImageModelInfo


pytestmark = pytest.mark.skipif(sys.platform != "darwin", reason="mflux/MLX backend is macOS-only")

_BETA = "zvisiongenerator.schedulers.beta_scheduler.BetaScheduler"


@pytest.fixture()
def image_mac():
    pytest.importorskip("mflux")
    import zvisiongenerator.backends.mac.image as module

    return module


def _sigmas(model_config, scheduler: str, steps: int) -> tuple[list[float], list[float]]:
    from mflux.models.common.config.config import Config

    config = Config(model_config=model_config, num_inference_steps=steps, width=832, height=1248, scheduler=scheduler)
    return [float(s) for s in config.scheduler.sigmas], [float(t) for t in config.scheduler.timesteps]


def _recording_model(seen: list[tuple[object, object]]) -> MagicMock:
    """Return a model whose ``generate_image`` records the Euler registry entry and the scheduler it was given."""
    from mflux.models.common.schedulers import SCHEDULER_REGISTRY

    def generate_image(**kwargs):
        seen.append((SCHEDULER_REGISTRY["euler"], kwargs.get("scheduler")))
        return MagicMock(image=Image.new("RGB", (16, 16)))

    model = MagicMock()
    model.generate_image.side_effect = generate_image
    return model


def _backend(image_mac, family: str):
    backend = image_mac.MfluxBackend()
    backend._model_info = ImageModelInfo(family=family, is_distilled=True, size=None)
    return backend


class TestKrea2BetaSchedule:
    def test_beta_gives_comfyui_beta_shifted_by_turbo_mu(self, image_mac):
        """ComfyUI's beta(0.6, 0.6) grid at Turbo's fixed shift (mu 1.15), as the diffusers backend runs it."""
        with image_mac._krea2_sigma_schedule(_BETA):
            sigmas, _ = _sigmas(image_mac._krea2_turbo_config(), image_mac._KREA2_SAMPLER, 8)

        assert sigmas == pytest.approx([1.0, 0.981, 0.937, 0.865, 0.760, 0.608, 0.402, 0.161, 0.0], abs=2e-3)

    @pytest.mark.parametrize("scheduler", [_BETA, None])
    def test_registry_is_restored_after_the_run(self, image_mac, scheduler):
        from mflux.models.common.schedulers import SCHEDULER_REGISTRY

        original = SCHEDULER_REGISTRY[image_mac._KREA2_SAMPLER]
        with image_mac._krea2_sigma_schedule(scheduler):
            pass

        assert SCHEDULER_REGISTRY[image_mac._KREA2_SAMPLER] is original

    def test_text_to_image_samples_with_euler_over_beta_sigmas(self, image_mac):
        from zvisiongenerator.schedulers.beta_scheduler import BetaScheduler

        seen: list[tuple[object, object]] = []
        _backend(image_mac, "krea2").text_to_image(_recording_model(seen), "p", 64, 64, seed=1, steps=8, guidance=1.0, scheduler=_BETA)

        assert seen == [(BetaScheduler, image_mac._KREA2_SAMPLER)]

    def test_image_to_image_keeps_the_evenly_spaced_schedule(self, image_mac):
        """Krea's refine passes (image to image, upscale) stay evenly spaced; beta only shapes the first pass."""
        from mflux.models.common.schedulers import SCHEDULER_REGISTRY

        original = SCHEDULER_REGISTRY[image_mac._KREA2_SAMPLER]
        seen: list[tuple[object, object]] = []
        _backend(image_mac, "krea2").image_to_image(_recording_model(seen), Image.new("RGB", (64, 64)), "p", strength=0.4, steps=8, seed=1, guidance=1.0, scheduler=_BETA)

        assert seen == [(original, image_mac._KREA2_SAMPLER)]


class TestKleinBetaSchedule:
    def test_klein_beta_matches_the_diffusers_backend(self, image_mac):
        """Klein's beta keeps its step-dependent shift: the sigmas and timesteps diffusers runs at 832x1248, 4 steps."""
        from mflux.models.common.config.model_config import ModelConfig

        sigmas, timesteps = _sigmas(ModelConfig.flux2_klein_9b(), image_mac._KLEIN_BETA_SCHEDULER, 4)

        assert sigmas == pytest.approx([1.0, 0.9788, 0.9079, 0.6774, 0.0], abs=2e-3)
        assert timesteps == pytest.approx([1000.0, 978.8, 907.9, 677.4], abs=2.0)

    def test_text_to_image_gives_klein_its_own_beta(self, image_mac):
        seen: list[tuple[object, object]] = []
        _backend(image_mac, "flux2_klein").text_to_image(_recording_model(seen), "p", 64, 64, seed=1, steps=4, guidance=1.0, scheduler=_BETA)

        assert seen[0][1] == image_mac._KLEIN_BETA_SCHEDULER

    def test_image_to_image_keeps_klein_own_schedule(self, image_mac):
        """As on Windows/Linux, Klein's refine passes (image to image, upscale) ignore the scheduler."""
        seen: list[tuple[object, object]] = []
        _backend(image_mac, "flux2_klein").image_to_image(_recording_model(seen), Image.new("RGB", (64, 64)), "p", strength=0.4, steps=4, seed=1, guidance=1.0, scheduler=_BETA)

        assert seen[0][1] is None

    def test_beta_constant_matches_the_scheduler_classes(self, image_mac):
        from zvisiongenerator.schedulers import SCHEDULER_CLASSES

        assert SCHEDULER_CLASSES["beta"]["mflux"] == image_mac._BETA_SCHEDULER
