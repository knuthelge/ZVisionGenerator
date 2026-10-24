"""Tests for video model type detection."""

from __future__ import annotations

import json

import pytest

from zvisiongenerator.utils.video_model_detect import VideoModelInfo, detect_video_model, is_known_video_repo


class TestDetectVideoModel:
    """Verify detect_video_model() prefix matching and returned metadata."""

    @pytest.mark.parametrize(
        ("model_path", "expected_family", "expected_backend", "expected_i2v", "expected_fps", "expected_frame_align", "expected_res_align"),
        [
            ("dgrauet/ltx-2.3-mlx-q4", "ltx", "ltx", True, 24, 8, 32),
            ("dgrauet/ltx-2-mlx-q8", "ltx", "ltx", True, 24, 8, 32),
            ("diffusers/LTX-2.3-Distilled-Diffusers", "ltx", "ltx", True, 24, 8, 32),
            ("Lightricks/LTX-Video", "ltx", "ltx", True, 24, 8, 32),
            ("unsloth/LTX-2.5-FP8", "ltx", "ltx", True, 24, 8, 32),
            ("unsloth/LTX-2.5-FP8@7b40724fc39bbb01d9b4d72638c0d263531c88c2", "ltx", "ltx", True, 24, 8, 32),
            ("MarceloFSO/ltx-2.5-mlx-q4-fast", "ltx", "ltx", True, 24, 8, 32),
            ("MarceloFSO/ltx-2.5-mlx-q4-fast@281bf05bd23207d6feef708fc2e270f84ba992b5", "ltx", "ltx", True, 24, 8, 32),
        ],
        ids=["ltx-q4", "ltx-q8", "ltx-diffusers-default", "ltx-video", "ltx-2.5", "ltx-2.5-pinned", "ltx-2.5-mlx", "ltx-2.5-mlx-pinned"],
    )
    def test_known_models(self, model_path, expected_family, expected_backend, expected_i2v, expected_fps, expected_frame_align, expected_res_align):
        info = detect_video_model(model_path)
        assert info.family == expected_family
        assert info.backend == expected_backend
        assert info.supports_i2v == expected_i2v
        assert info.default_fps == expected_fps
        assert info.frame_alignment == expected_frame_align
        assert info.resolution_alignment == expected_res_align

    @pytest.mark.parametrize(
        "model_path",
        [
            "MiniMaxAI/MiniMax-H3",
            "MiniMaxAI/MiniMax-H3@42ed227ee7df40d41602854ae760620d6eb651fe",
            "unsloth/MiniMax-H3",
        ],
    )
    def test_minimax_h3_repos(self, model_path):
        info = detect_video_model(model_path)
        assert (info.family, info.backend, info.supports_i2v, info.default_fps, info.frame_alignment, info.resolution_alignment) == ("h3", "h3", True, 24, 17, 32)
        assert is_known_video_repo(model_path) is True

    @pytest.mark.parametrize("folder_name", ["h3-local", "my-ltx-style-h3"])
    def test_a_local_h3_folder_is_h3_even_when_its_path_says_ltx(self, tmp_path, folder_name):
        model_dir = tmp_path / folder_name
        model_dir.mkdir()
        (model_dir / "model_index.json").write_text(json.dumps({"_class_name": "MiniMaxH3ModularPipeline"}), encoding="utf-8")

        info = detect_video_model(str(model_dir))

        assert (info.family, info.backend, info.frame_alignment) == ("h3", "h3", 17)

    def test_an_h3_folder_below_an_ltx_named_parent_is_h3(self, tmp_path):
        model_dir = tmp_path / "ltx-collection" / "minimax"
        model_dir.mkdir(parents=True)
        (model_dir / "model_index.json").write_text(json.dumps({"_class_name": "MiniMaxH3ModularPipeline"}), encoding="utf-8")

        assert detect_video_model(str(model_dir)).family == "h3"

    @pytest.mark.parametrize("model_path", ["some/random-model", "stable-diffusion/v2", "Lightricks/LTX-2.3-13B", "dg845/LTX-2.3-Diffusers", ""])
    def test_unknown_models(self, model_path):
        info = detect_video_model(model_path)
        assert info.family == "unknown"
        assert info.backend == "unknown"
        assert info.supports_i2v is False

    def test_is_known_video_repo_recognizes_distilled_default(self):
        assert is_known_video_repo("diffusers/LTX-2.3-Distilled-Diffusers") is True
        assert is_known_video_repo("Lightricks/LTX-2.3-13B") is False
        assert is_known_video_repo("dg845/LTX-2.3-Diffusers") is False

    @pytest.mark.parametrize("model_path", ["unsloth/LTX-2.5-FP8", "unsloth/LTX-2.5-FP8@7b40724fc39bbb01d9b4d72638c0d263531c88c2"])
    def test_is_known_video_repo_recognizes_ltx_2_5(self, model_path):
        assert is_known_video_repo(model_path) is True

    @pytest.mark.parametrize("model_path", ["MarceloFSO/ltx-2.5-mlx-q4-fast", "MarceloFSO/ltx-2.5-mlx-q4-fast@281bf05bd23207d6feef708fc2e270f84ba992b5"])
    def test_is_known_video_repo_recognizes_the_ltx_2_5_mac_mirror(self, model_path):
        assert is_known_video_repo(model_path) is True

    @pytest.mark.parametrize("text_encoder", ["Gemma4UnifiedForConditionalGeneration", "Gemma3ForConditionalGeneration"])
    def test_a_local_folder_is_ltx_by_its_model_index(self, tmp_path, text_encoder):
        model_dir = tmp_path / "converted"
        model_dir.mkdir()
        index = {"_class_name": "LTX2Pipeline", "text_encoder": ["transformers", text_encoder]}
        (model_dir / "model_index.json").write_text(json.dumps(index), encoding="utf-8")

        info = detect_video_model(str(model_dir))

        assert info.family == "ltx"
        assert info.supports_i2v is True

    def test_a_local_folder_with_another_pipeline_index_stays_unknown(self, tmp_path):
        model_dir = tmp_path / "other"
        model_dir.mkdir()
        (model_dir / "model_index.json").write_text(json.dumps({"_class_name": "ZImagePipeline"}), encoding="utf-8")

        assert detect_video_model(str(model_dir)).family == "unknown"

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
