"""Tests for ``release_model`` on the mflux image backend, with fake mflux models."""

from __future__ import annotations

import sys

import pytest

pytestmark = pytest.mark.skipif(sys.platform != "darwin", reason="mflux/MLX backend is macOS-only")


@pytest.fixture
def backend():
    pytest.importorskip("mflux")
    pytest.importorskip("mlx.core")
    from zvisiongenerator.backends.mac.image import MfluxBackend

    return MfluxBackend()


@pytest.fixture
def nn():
    return pytest.importorskip("mlx.nn")


def _model(nn, *weights: str):
    """Build an mflux-shaped model: an ``nn.Module`` holding one child module per name and a prompt cache."""
    model = nn.Module()
    for name in weights:
        setattr(model, name, nn.Linear(2, 2))
    model.prompt_cache = {"prompt": object()}
    return model


def _modules_left(model, nn):
    return [name for name, value in model.items() if isinstance(value, nn.Module)]


class TestReleaseModel:
    def test_drops_the_weights_and_clears_the_prompt_cache(self, backend, nn):
        model = _model(nn, "transformer", "vae", "text_encoder")
        cache = model.prompt_cache

        backend.release_model(model)

        assert (model.transformer, model.vae, model.text_encoder) == (None, None, None)
        assert not _modules_left(model, nn)
        assert cache == {}

    def test_drops_weights_under_family_specific_names(self, backend, nn):
        model = _model(nn, "conditional_transformer", "unconditional_transformer", "vae", "text_encoder")

        backend.release_model(model)

        assert (model.conditional_transformer, model.unconditional_transformer) == (None, None)
        assert not _modules_left(model, nn)

    def test_keeps_attributes_that_are_not_modules(self, backend, nn):
        model = _model(nn, "transformer")
        model.model_path = "weights"

        backend.release_model(model)

        assert model.model_path == "weights"

    def test_a_second_release_does_not_raise(self, backend, nn):
        model = _model(nn, "transformer", "vae", "text_encoder")

        backend.release_model(model)
        backend.release_model(model)

        assert not _modules_left(model, nn)

    def test_a_plain_object_model_is_released(self, backend, nn):
        class Model:
            def __init__(self):
                self.vae = nn.Linear(2, 2)
                self.text_encoder = None
                self.prompt_cache = {}

        model = Model()

        backend.release_model(model)

        assert model.vae is None
        assert not hasattr(model, "transformer")
