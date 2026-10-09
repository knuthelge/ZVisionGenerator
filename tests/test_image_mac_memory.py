"""Tests for the mflux backend's resident weights, buffer-cache policy and stored-quant hooks."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.skipif(sys.platform != "darwin", reason="mflux/MLX backend is macOS-only")


@pytest.fixture
def image_mac(monkeypatch):
    pytest.importorskip("mflux")
    pytest.importorskip("mlx.core")
    import zvisiongenerator.backends.image_mac as module

    fake_mx = MagicMock(name="mx")
    # load_model sets mflux's process-wide precision from mx; restore it so the fake never leaks into other tests.
    monkeypatch.setattr(module.ModelConfig, "precision", module.ModelConfig.precision)
    monkeypatch.setattr(module, "mx", fake_mx)
    return module, fake_mx


def _load_zimage(module, monkeypatch, *, quantize=None):
    info = module.ImageModelInfo(family="zimage", is_distilled=False, size=None)
    monkeypatch.setattr(module, "detect_image_model", lambda _path: info)
    monkeypatch.setattr(module, "_upcast_model_weights", MagicMock())
    monkeypatch.setattr(module.ModelConfig, "z_image", MagicMock(return_value="cfg"))
    model = MagicMock(name="model")
    monkeypatch.setattr(module, "ZImageTurbo", MagicMock(return_value=model))
    backend = module.MfluxBackend()
    backend.load_model("/models/zit", quantize=quantize)
    return backend, model


def test_load_model_materializes_weights_then_caps_and_clears_the_cache(image_mac, monkeypatch):
    module, mx = image_mac

    _backend, model = _load_zimage(module, monkeypatch, quantize=8)

    mx.eval.assert_called_once_with(model.parameters.return_value)
    mx.set_cache_limit.assert_called_once_with(module._BUFFER_CACHE_LIMIT_BYTES)
    mx.clear_cache.assert_called()
    names = [call[0] for call in mx.method_calls]
    assert names.index("eval") < names.index("set_cache_limit") < names.index("clear_cache")


def test_buffer_cache_limit_is_four_gib(image_mac):
    module, _mx = image_mac
    assert module._BUFFER_CACHE_LIMIT_BYTES == 4 * 1024**3


@pytest.mark.parametrize("skipped", [False, True])
def test_text_to_image_clears_the_cache_after_each_generation(image_mac, monkeypatch, skipped):
    module, mx = image_mac
    backend, model = _load_zimage(module, monkeypatch)
    mx.reset_mock()
    if skipped:
        model.generate_image.side_effect = module.StopImageGenerationException("skip")

    backend.text_to_image(model=model, prompt="p", width=64, height=64, seed=1, steps=1, guidance=0.0)

    mx.clear_cache.assert_called_once()


def test_stored_quant_format_tracks_the_mflux_version(image_mac):
    module, _mx = image_mac
    import importlib.metadata

    assert module.MfluxBackend().stored_quant_format(8) == f"mflux-{importlib.metadata.version('mflux')}"


def test_save_quantized_uses_mflux_saver(image_mac):
    module, _mx = image_mac
    model = MagicMock()

    module.MfluxBackend().save_quantized(model, "/models/zit@q8")

    model.save_model.assert_called_once_with("/models/zit@q8")
