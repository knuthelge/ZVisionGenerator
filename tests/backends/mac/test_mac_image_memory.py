"""Tests for the mflux backend's resident weights, buffer-cache policy and stored-quant hooks."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, call

import pytest

pytestmark = pytest.mark.skipif(sys.platform != "darwin", reason="mflux/MLX backend is macOS-only")


@pytest.fixture
def image_mac(monkeypatch):
    pytest.importorskip("mflux")
    pytest.importorskip("mlx.core")
    import zvisiongenerator.backends.mac.image as module

    fake_mx = MagicMock(name="mx")
    # load_model sets mflux's process-wide precision from mx; restore it so the fake never leaks into other tests.
    monkeypatch.setattr(module.ModelConfig, "precision", module.ModelConfig.precision)
    monkeypatch.setattr(module, "mx", fake_mx)
    monkeypatch.setattr(module, "load_resident", MagicMock(name="load_resident", side_effect=lambda build: build()))
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


def test_load_model_builds_and_upcasts_the_vae_inside_load_resident(image_mac, monkeypatch):
    module, _mx = image_mac
    order = MagicMock()
    info = module.ImageModelInfo(family="zimage", is_distilled=False, size=None)
    monkeypatch.setattr(module, "detect_image_model", lambda _path: info)
    constructor = MagicMock(name="ZImageTurbo")
    upcast = MagicMock(name="upcast")
    order.attach_mock(constructor, "build")
    order.attach_mock(upcast, "upcast")

    def load_resident(build):
        order.resident_enter()
        loaded = build()
        order.resident_exit(loaded)
        return loaded

    monkeypatch.setattr(module, "load_resident", load_resident)
    monkeypatch.setattr(module, "_upcast_model_weights", upcast)
    monkeypatch.setattr(module.ModelConfig, "z_image", MagicMock(return_value="cfg"))
    monkeypatch.setattr(module, "ZImageTurbo", constructor)

    model, _info = module.MfluxBackend().load_model("/models/zit", quantize=8)

    built = constructor.return_value
    assert model is built
    assert order.mock_calls == [
        call.resident_enter(),
        call.build(quantize=8, model_path="/models/zit", model_config="cfg"),
        call.upcast(built, ["vae"]),
        call.resident_exit(built),
    ]


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
