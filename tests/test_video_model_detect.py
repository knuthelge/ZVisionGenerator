"""Tests for video model type detection."""

from __future__ import annotations

from typing import get_type_hints

import pytest

from zvisiongenerator.utils.video_model_detect import VideoModelInfo, detect_video_model


class TestDetectVideoModel:
    """Verify detect_video_model() prefix matching and returned metadata."""

    _EXPECTED_LTX_TEXT_ENCODER = "Lightricks/gemma-3-12b-it-qat-q4_0-unquantized"

    @pytest.mark.parametrize(
        ("model_path", "expected_family", "expected_backend", "expected_i2v", "expected_fps", "expected_frame_align", "expected_res_align"),
        [
            ("dgrauet/ltx-2.3-mlx-q4", "ltx", "ltx", True, 24, 8, 32),
            ("dgrauet/ltx-2-mlx-q8", "ltx", "ltx", True, 24, 8, 32),
            ("Lightricks/LTX-2.3", "ltx", "ltx", True, 24, 8, 32),
            ("Lightricks/LTX-Video", "ltx", "ltx", True, 24, 8, 32),
            ("Lightricks/LTX-Video-fp8", "ltx", "ltx", True, 24, 8, 32),
        ],
        ids=["ltx-q4", "ltx-q8", "ltx-lightricks", "ltx-video", "ltx-video-fp8"],
    )
    def test_known_models(self, model_path, expected_family, expected_backend, expected_i2v, expected_fps, expected_frame_align, expected_res_align):
        info = detect_video_model(model_path)
        assert info.family == expected_family
        assert info.backend == expected_backend
        assert info.supports_i2v == expected_i2v
        assert info.default_fps == expected_fps
        assert info.frame_alignment == expected_frame_align
        assert info.resolution_alignment == expected_res_align
        assert info.default_text_encoder == self._EXPECTED_LTX_TEXT_ENCODER

    @pytest.mark.parametrize("model_path", ["some/random-model", "stable-diffusion/v2", ""])
    def test_unknown_models(self, model_path):
        info = detect_video_model(model_path)
        assert info.family == "unknown"
        assert info.backend == "unknown"
        assert info.supports_i2v is False
        assert info.default_text_encoder is None

    @pytest.mark.parametrize(
        ("model_path", "expected_family", "expected_i2v"),
        [
            ("/path/to/ltx-model", "ltx", True),
            ("./models/LTX-2.3-q4", "ltx", True),
            ("/path/to/random-model", "unknown", False),
        ],
        ids=["local-ltx-lower", "local-ltx-upper", "local-unknown"],
    )
    def test_local_path_detection(self, model_path, expected_family, expected_i2v):
        info = detect_video_model(model_path)
        assert info.family == expected_family
        assert info.supports_i2v == expected_i2v
        expected_text_encoder = self._EXPECTED_LTX_TEXT_ENCODER if expected_family == "ltx" else None
        assert info.default_text_encoder == expected_text_encoder

    def test_default_text_encoder_field_annotation(self):
        annotations = get_type_hints(VideoModelInfo)
        assert annotations["default_text_encoder"] == str | None

    def test_returns_frozen_dataclass(self):
        info = detect_video_model("dgrauet/ltx-2.3-mlx-q4")
        assert isinstance(info, VideoModelInfo)
        with pytest.raises(AttributeError):
            info.family = "changed"  # type: ignore[misc]
