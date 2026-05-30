"""Test backend selection and load_model guard contracts."""

from __future__ import annotations

import sys
from collections.abc import Mapping
from unittest.mock import MagicMock

import pytest
from PIL import Image

from zvisiongenerator.core.image_backend import ImageBackend


class TestBackendRegistryLookup:
    def test_get_backend_name_uses_dict_lookup_for_darwin(self, monkeypatch):
        import zvisiongenerator.backends as backends

        monkeypatch.setattr(backends.sys, "platform", "darwin")
        monkeypatch.setattr(backends, "_IMAGE_BACKENDS_MAP", {"darwin": ("mflux", lambda: MagicMock(spec=ImageBackend))})

        assert backends.get_backend_name() == "mflux"

    def test_get_backend_name_uses_dict_lookup_for_win32(self, monkeypatch):
        import zvisiongenerator.backends as backends

        monkeypatch.setattr(backends.sys, "platform", "win32")
        monkeypatch.setattr(backends, "_IMAGE_BACKENDS_MAP", {"win32": ("diffusers", lambda: MagicMock(spec=ImageBackend))})

        assert backends.get_backend_name() == "diffusers"

    def test_get_backend_uses_dict_lookup_for_darwin(self, monkeypatch):
        import zvisiongenerator.backends as backends

        backend = MagicMock(spec=ImageBackend)
        monkeypatch.setattr(backends, "BACKENDS", {})
        monkeypatch.setattr(backends.sys, "platform", "darwin")
        monkeypatch.setattr(backends, "_IMAGE_BACKENDS_MAP", {"darwin": ("mflux", lambda: backend)})

        result = backends.get_backend()

        assert result is backend
        assert backends.BACKENDS["mflux"] is backend

    def test_get_backend_uses_dict_lookup_for_win32(self, monkeypatch):
        import zvisiongenerator.backends as backends

        backend = MagicMock(spec=ImageBackend)
        monkeypatch.setattr(backends, "BACKENDS", {})
        monkeypatch.setattr(backends.sys, "platform", "win32")
        monkeypatch.setattr(backends, "_IMAGE_BACKENDS_MAP", {"win32": ("diffusers", lambda: backend)})

        result = backends.get_backend()

        assert result is backend
        assert backends.BACKENDS["diffusers"] is backend


# ---------------------------------------------------------------------------
# Guard: generation without load_model() must raise RuntimeError
# ---------------------------------------------------------------------------


class TestLoadModelGuard:
    """Backends must raise RuntimeError if generation is called without load_model()."""

    def test_mflux_text_to_image_without_load_model(self):
        from zvisiongenerator.backends.image_mac import MfluxBackend

        backend = MfluxBackend()
        with pytest.raises(RuntimeError, match="load_model"):
            backend.text_to_image(
                model=MagicMock(),
                prompt="test",
                width=64,
                height=64,
                seed=42,
                steps=1,
                guidance=0.5,
            )

    def test_mflux_image_to_image_without_load_model(self):
        from zvisiongenerator.backends.image_mac import MfluxBackend

        backend = MfluxBackend()
        with pytest.raises(RuntimeError, match="load_model"):
            backend.image_to_image(
                model=MagicMock(),
                image=Image.new("RGB", (64, 64)),
                prompt="test",
                strength=0.5,
                steps=1,
                seed=42,
                guidance=0.5,
            )

    @pytest.mark.skipif(sys.platform == "darwin", reason="torch/CUDA not available on macOS CI")
    def test_diffusers_text_to_image_without_load_model(self):
        from zvisiongenerator.backends.image_win import DiffusersBackend

        backend = DiffusersBackend()
        with pytest.raises(RuntimeError, match="load_model"):
            backend.text_to_image(
                model=MagicMock(),
                prompt="test",
                width=64,
                height=64,
                seed=42,
                steps=1,
                guidance=0.5,
            )

    @pytest.mark.skipif(sys.platform == "darwin", reason="torch/CUDA not available on macOS CI")
    def test_diffusers_image_to_image_without_load_model(self):
        from zvisiongenerator.backends.image_win import DiffusersBackend

        backend = DiffusersBackend()
        with pytest.raises(RuntimeError, match="load_model"):
            backend.image_to_image(
                model=MagicMock(),
                image=Image.new("RGB", (64, 64)),
                prompt="test",
                strength=0.5,
                steps=1,
                seed=42,
                guidance=0.5,
            )


class _FakeModelComponent:
    def __init__(self, params):
        self._params = params
        self.updated = None

    def parameters(self):
        return self._params

    def update(self, params):
        self.updated = params


def _flatten_params(params):
    for value in params.values():
        if isinstance(value, Mapping):
            yield from _flatten_params(value)
        else:
            yield value


@pytest.mark.skipif(sys.platform != "darwin", reason="Selective MLX upcast only runs on macOS")
class TestSelectiveMacWeightUpcast:
    def test_upcast_model_weights_preserves_uint32_quantized_weights(self):
        mx = pytest.importorskip("mlx.core")

        from zvisiongenerator.backends.image_mac import _upcast_model_weights

        uint_weights = mx.array([1, 2, 3], dtype=mx.uint32)
        fp16_weights = mx.array([1.5, 2.5], dtype=mx.float16)
        bf16_weights = mx.array([3.0, 4.0], dtype=mx.bfloat16)
        float32_weights = mx.array([5.0], dtype=mx.float32)
        component = _FakeModelComponent(
            {
                "quantized": uint_weights,
                "nested": {
                    "fp16": fp16_weights,
                    "bf16": bf16_weights,
                    "fp32": float32_weights,
                },
            }
        )
        model = MagicMock(transformer=component)

        _upcast_model_weights(model, ["transformer"])

        assert component.updated is not None
        assert component.updated["quantized"] is uint_weights
        assert component.updated["quantized"].dtype == mx.uint32
        assert component.updated["nested"]["fp16"].dtype == mx.float32
        assert component.updated["nested"]["bf16"].dtype == mx.float32
        assert component.updated["nested"]["fp32"] is float32_weights
        assert all(value.dtype != mx.float16 for value in _flatten_params(component.updated))
        assert all(value.dtype != mx.bfloat16 for value in _flatten_params(component.updated))
