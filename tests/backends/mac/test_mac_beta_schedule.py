"""Tests for FLUX.2 Klein's beta sigma schedule on the mflux backend."""

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

    def test_beta_constant_matches_the_packaged_config(self, image_mac):
        from zvisiongenerator.utils.config import load_config

        assert load_config()["schedulers"]["beta"]["mflux_class"] == image_mac._BETA_SCHEDULER
