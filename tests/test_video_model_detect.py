"""Tests for video model type detection."""

from __future__ import annotations

import pytest

from zvisiongenerator.utils.video_model_detect import VideoModelInfo, detect_video_model


class TestDetectVideoModel:
    """Verify detect_video_model() prefix matching and returned metadata."""

    @pytest.mark.parametrize(
        ("model_path", "expected_family", "expected_backend", "expected_i2v", "expected_fps", "expected_frame_align", "expected_res_align"),
        [
            ("dgrauet/ltx-2.3-mlx-q4", "ltx", "ltx", True, 24, 8, 32),
            ("dgrauet/ltx-2-mlx-q8", "ltx", "ltx", True, 24, 8, 32),
            ("dg845/LTX-2.3-Diffusers", "ltx", "ltx", True, 24, 8, 32),
            ("Lightricks/LTX-Video", "ltx", "ltx", True, 24, 8, 32),
        ],
        ids=["ltx-q4", "ltx-q8", "ltx-diffusers-default", "ltx-video"],
    )
    def test_known_models(self, model_path, expected_family, expected_backend, expected_i2v, expected_fps, expected_frame_align, expected_res_align):
        info = detect_video_model(model_path)
        assert info.family == expected_family
        assert info.backend == expected_backend
        assert info.supports_i2v == expected_i2v
        assert info.default_fps == expected_fps
        assert info.frame_alignment == expected_frame_align
        assert info.resolution_alignment == expected_res_align

    @pytest.mark.parametrize("model_path", ["some/random-model", "stable-diffusion/v2", "Lightricks/LTX-2.3-13B", ""])
    def test_unknown_models(self, model_path):
        info = detect_video_model(model_path)
        assert info.family == "unknown"
        assert info.backend == "unknown"
        assert info.supports_i2v is False

    @pytest.mark.parametrize(
        ("model_path", "expected_family", "expected_i2v"),
        [
            ("/path/to/ltx-model", "ltx", True),
            ("./models/LTX-2.3-q4", "ltx", True),
            ("models/ltx-mlx", "ltx", True),
            ("C:/models/ltx", "ltx", True),
            (r"C:\models\ltx", "ltx", True),
            ("owner/repo/ltx", "ltx", True),
            ("/path/to/random-model", "unknown", False),
            ("models/random-model", "unknown", False),
        ],
        ids=["local-ltx-lower", "local-ltx-upper", "reserved-local-ltx", "windows-forward-ltx", "windows-backslash-ltx", "multi-segment-local-ltx", "local-unknown", "reserved-local-unknown"],
    )
    def test_local_path_detection(self, model_path, expected_family, expected_i2v):
        info = detect_video_model(model_path)
        assert info.family == expected_family
        assert info.supports_i2v == expected_i2v

    def test_returns_frozen_dataclass(self):
        info = detect_video_model("dgrauet/ltx-2.3-mlx-q4")
        assert isinstance(info, VideoModelInfo)
        with pytest.raises(AttributeError):
            info.family = "changed"  # type: ignore[misc]

    def test_detects_configured_default_repo(self, monkeypatch):
        monkeypatch.setattr(
            "zvisiongenerator.utils.video_model_detect._configured_diffusers_ltx_repo",
            lambda: "custom/LTX-2.3",
        )

        info = detect_video_model("custom/LTX-2.3")

        assert info.family == "ltx"
        assert info.backend == "ltx"

    def test_configured_default_repo_is_read_fresh(self, monkeypatch):
        repos = iter(["custom/old-ltx", "custom/new-ltx"])
        monkeypatch.setattr(
            "zvisiongenerator.utils.config.load_config",
            lambda: {"video_model_presets": {"ltx": {"diffusers": {"default_repo": next(repos)}}}},
        )

        assert detect_video_model("custom/old-ltx").family == "ltx"
        assert detect_video_model("custom/new-ltx").family == "ltx"

    def test_arbitrary_huggingface_ltx_repo_stays_unknown(self):
        info = detect_video_model("your-org/ltx-custom-diffusers")

        assert info.family == "unknown"
