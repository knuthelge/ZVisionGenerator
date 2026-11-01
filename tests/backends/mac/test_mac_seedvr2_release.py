"""Tests for ``release_model`` on the mflux SeedVR2 upscaler, with fake models (no MLX needed)."""

from __future__ import annotations

from unittest.mock import MagicMock

from zvisiongenerator.backends.mac.seedvr2 import MfluxSeedVR2Upscaler


class TestReleaseModel:
    def test_drops_the_transformer_and_the_vae(self):
        model = MagicMock(transformer=MagicMock(), vae=MagicMock())

        MfluxSeedVR2Upscaler().release_model(model)

        assert (model.transformer, model.vae) == (None, None)

    def test_a_second_release_does_not_raise(self):
        model = MagicMock(transformer=MagicMock(), vae=MagicMock())
        upscaler = MfluxSeedVR2Upscaler()

        upscaler.release_model(model)
        upscaler.release_model(model)

        assert (model.transformer, model.vae) == (None, None)
