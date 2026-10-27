"""Tests for VideoBackend Protocol and backend registry."""

from __future__ import annotations

import inspect
import sys
from unittest.mock import MagicMock, patch

import pytest

from zvisiongenerator.core.video_backend import VideoBackend, VideoExtender


class TestVideoBackendProtocol:
    """Verify that VideoBackend is runtime_checkable and real backends satisfy it."""

    @pytest.mark.skipif(sys.platform != "darwin", reason="macOS-only backends")
    def test_ltx_backend_satisfies_protocol(self):
        from zvisiongenerator.backends.mac.video import LtxVideoBackend

        assert isinstance(LtxVideoBackend(), VideoBackend)

    def test_diffusers_video_backend_satisfies_protocol(self):
        from zvisiongenerator.backends.cuda.video.backend import DiffusersVideoBackend

        assert isinstance(DiffusersVideoBackend(), VideoBackend)


def _shape(fn) -> list[tuple[str, object, object]]:
    """Return a method's parameters as ``(name, kind, default)``, ignoring ``self`` and annotations."""
    return [(p.name, p.kind, p.default) for name, p in inspect.signature(fn).parameters.items() if name != "self"]


class TestVideoExtenderProtocol:
    """Verify extending is a separate, optional capability that the macOS backend provides."""

    @pytest.mark.skipif(sys.platform != "darwin", reason="macOS-only backends")
    def test_ltx_backend_provides_extend_video_with_the_protocol_signature(self):
        from zvisiongenerator.backends.mac.video import LtxVideoBackend

        assert _shape(LtxVideoBackend.extend_video) == _shape(VideoExtender.extend_video)

    @pytest.mark.skipif(sys.platform != "darwin", reason="macOS-only backends")
    def test_ltx_backend_is_still_a_video_backend(self):
        from zvisiongenerator.backends.mac.video import LtxVideoBackend

        assert isinstance(LtxVideoBackend(), VideoBackend)

    def test_diffusers_backend_is_a_video_backend_with_extend_video(self):
        from zvisiongenerator.backends.cuda.video.backend import DiffusersVideoBackend

        backend = DiffusersVideoBackend()

        assert isinstance(backend, VideoBackend)
        assert callable(getattr(backend, "extend_video", None))


class TestGetVideoBackend:
    """Verify get_video_backend() registry behaviour."""

    @pytest.mark.skipif(sys.platform != "darwin", reason="macOS-only backends")
    def test_get_ltx_backend(self):
        from zvisiongenerator.backends import get_video_backend

        backend = get_video_backend("ltx")
        assert backend.name == "ltx"
        assert isinstance(backend, VideoBackend)

    @pytest.mark.parametrize("platform_key", ["win32", "linux"])
    def test_get_ltx_backend_on_diffusers_platforms(self, monkeypatch, platform_key):
        import zvisiongenerator.backends as backends_module

        monkeypatch.setattr(backends_module.sys, "platform", platform_key)
        backends_module.VIDEO_BACKENDS.clear()

        backend = backends_module.get_video_backend("ltx")

        assert backend.name == "ltx"
        assert isinstance(backend, VideoBackend)

    def test_unsupported_platform_error_lists_supported_video_platforms(self, monkeypatch):
        import zvisiongenerator.backends as backends_module

        monkeypatch.setattr(backends_module.sys, "platform", "freebsd")
        backends_module.VIDEO_BACKENDS.clear()

        with pytest.raises(RuntimeError, match="macOS, Windows, and Linux"):
            backends_module.get_video_backend("ltx")

    def test_get_nonexistent_backend_raises(self):
        from zvisiongenerator.backends import VIDEO_BACKENDS, get_video_backend

        # Patch VIDEO_BACKENDS so _register_video_backends() is skipped
        with patch.dict(VIDEO_BACKENDS, {"ltx": MagicMock()}, clear=True):
            with pytest.raises(RuntimeError, match="No video backend"):
                get_video_backend("nonexistent")
