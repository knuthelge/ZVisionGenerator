"""Backend smoke test and load_model guard tests."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest
from PIL import Image

from zvisiongenerator.core.image_backend import ImageBackend


# ---------------------------------------------------------------------------
# Smoke: get_backend() returns a working backend for this platform
# ---------------------------------------------------------------------------


def test_get_backend_returns_backend_instance():
    from zvisiongenerator.backends import get_backend

    backend = get_backend()
    assert isinstance(backend, ImageBackend)


def test_backend_name_is_correct_for_platform():
    from zvisiongenerator.backends import get_backend

    backend = get_backend()
    expected = "mflux" if sys.platform == "darwin" else "diffusers"
    assert backend.name == expected


class TestBackendRegistryLookup:
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
