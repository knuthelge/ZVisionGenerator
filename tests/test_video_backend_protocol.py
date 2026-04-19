"""Tests for VideoBackend Protocol and backend registry."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from zvisiongenerator.core.video_backend import VideoBackend


class TestVideoBackendProtocol:
    """Verify that VideoBackend is runtime_checkable and real backends satisfy it."""

    def test_mock_satisfies_protocol(self):
        """A MagicMock with required attributes satisfies isinstance check."""
        mock = MagicMock()
        mock.name = "test"
        mock.load_model = MagicMock()
        mock.text_to_video = MagicMock()
        mock.image_to_video = MagicMock()
        assert isinstance(mock, VideoBackend)

    @pytest.mark.skipif(sys.platform != "darwin", reason="macOS-only backends")
    def test_ltx_backend_satisfies_protocol(self):
        from zvisiongenerator.backends.video_mac import LtxVideoBackend

        assert isinstance(LtxVideoBackend(), VideoBackend)

    @pytest.mark.skipif(sys.platform != "win32", reason="Windows-only backends")
    def test_ltx_cuda_backend_satisfies_protocol(self):
        from zvisiongenerator.backends.video_win import LtxCudaVideoBackend

        assert isinstance(LtxCudaVideoBackend(), VideoBackend)

    def test_protocol_requires_name(self):
        """An object missing 'name' does not satisfy the protocol."""

        class Incomplete:
            def load_model(self, model_path, **kwargs):
                pass

            def text_to_video(self, model, prompt, width, height, num_frames, seed, steps, output_path):
                pass

            def image_to_video(self, model, image_path, prompt, width, height, num_frames, seed, steps, output_path):
                pass

        assert not isinstance(Incomplete(), VideoBackend)


class TestGetVideoBackend:
    """Verify get_video_backend() registry behaviour."""

    @pytest.mark.skipif(sys.platform != "darwin", reason="macOS-only backends")
    def test_get_ltx_backend(self):
        from zvisiongenerator.backends import get_video_backend

        backend = get_video_backend("ltx")
        assert backend.name == "ltx"
        assert isinstance(backend, VideoBackend)

    def test_get_nonexistent_backend_raises(self):
        from zvisiongenerator.backends import VIDEO_BACKENDS, get_video_backend

        # Patch VIDEO_BACKENDS so _register_video_backends() is skipped
        with patch.dict(VIDEO_BACKENDS, {"ltx": MagicMock()}, clear=True):
            with pytest.raises(RuntimeError, match="No video backend"):
                get_video_backend("nonexistent")
