"""Tests for ``release_model`` on the mflux image backend, with fake mflux models."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.skipif(sys.platform != "darwin", reason="mflux/MLX backend is macOS-only")


@pytest.fixture
def backend():
    pytest.importorskip("mflux")
    pytest.importorskip("mlx.core")
    from zvisiongenerator.backends.mac.image import MfluxBackend

    return MfluxBackend()


def _model(**overrides):
    attributes = {"transformer": MagicMock(), "vae": MagicMock(), "text_encoder": MagicMock(), "prompt_cache": {"prompt": object()}}
    return MagicMock(**{**attributes, **overrides})


class TestReleaseModel:
    def test_drops_the_weights_and_clears_the_prompt_cache(self, backend):
        model = _model()
        cache = model.prompt_cache

        backend.release_model(model)

        assert (model.transformer, model.vae, model.text_encoder) == (None, None, None)
        assert cache == {}

    def test_a_second_release_does_not_raise(self, backend):
        model = _model()

        backend.release_model(model)
        backend.release_model(model)

        assert (model.transformer, model.vae, model.text_encoder) == (None, None, None)

    def test_a_model_without_a_transformer_attribute_is_released(self, backend):
        class Model:
            def __init__(self):
                self.vae = object()
                self.text_encoder = None
                self.prompt_cache = {}

        model = Model()

        backend.release_model(model)

        assert model.vae is None
        assert not hasattr(model, "transformer")
