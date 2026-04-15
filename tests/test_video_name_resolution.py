"""Tests for friendly video model name resolution and video model listing."""

from __future__ import annotations

from unittest.mock import patch

from zvisiongenerator.utils.paths import resolve_model_path


# ── Friendly name resolution for video models ────────────────────────────────


class TestVideoModelNameResolution:
    """Verify resolve_model_path() works for video model names."""

    def test_bare_name_resolves_when_dir_exists(self, tmp_path):
        """Bare name like 'ltx-2.3-mlx-q4' resolves to full path when dir exists."""
        model_dir = tmp_path / "models" / "ltx-2.3-mlx-q4"
        model_dir.mkdir(parents=True)

        with patch("zvisiongenerator.utils.paths.get_ziv_data_dir", return_value=tmp_path):
            result = resolve_model_path("ltx-2.3-mlx-q4")

        assert result == str(model_dir)

    def test_bare_name_passthrough_when_not_exists(self, tmp_path):
        """Bare name without matching dir passes through unchanged."""
        (tmp_path / "models").mkdir(parents=True)

        with patch("zvisiongenerator.utils.paths.get_ziv_data_dir", return_value=tmp_path):
            result = resolve_model_path("ltx-2.3-mlx-q4")

        assert result == "ltx-2.3-mlx-q4"

    def test_full_path_unchanged(self):
        """Absolute path passes through unchanged."""
        full_path = "/home/user/.ziv/models/ltx-2.3-mlx-q4"
        result = resolve_model_path(full_path)
        assert result == full_path

    def test_hf_repo_unchanged(self):
        """HF repo ID (with '/') passes through unchanged."""
        hf_repo = "dgrauet/ltx-video-0.9.7"
        result = resolve_model_path(hf_repo)
        assert result == hf_repo

    def test_ltx_bare_name_resolves(self, tmp_path):
        """LTX bare name also resolves when dir exists."""
        model_dir = tmp_path / "models" / "ltx-2.3-mlx-q4"
        model_dir.mkdir(parents=True)

        with patch("zvisiongenerator.utils.paths.get_ziv_data_dir", return_value=tmp_path):
            result = resolve_model_path("ltx-2.3-mlx-q4")

        assert result == str(model_dir)


# ── video_cli.py calls resolve_model_path ────────────────────────────────────


class TestVideoCLIResolvesModelPath:
    """Verify video_cli.main() passes model through resolve_model_path."""

    @patch("zvisiongenerator.video_cli.ensure_ffmpeg")
    @patch("zvisiongenerator.video_cli.run_video_batch")
    @patch("zvisiongenerator.video_cli.build_video_workflow")
    @patch("zvisiongenerator.video_cli.load_prompts_file", return_value={"test": [("a dog", None)]})
    @patch(
        "zvisiongenerator.video_cli.load_config",
        return_value={
            "video_generation": {"default_ratio": "16:9", "default_size": "m"},
            "video_sizes": {"ltx": {"16:9": {"m": {"width": 704, "height": 480, "frames": 49}}}},
            "video_model_presets": {"ltx": {"default_steps": 8}},
        },
    )
    @patch("zvisiongenerator.video_cli.get_video_backend")
    @patch("zvisiongenerator.video_cli.detect_video_model")
    @patch("zvisiongenerator.video_cli.resolve_model_path")
    def test_resolve_model_path_called(
        self,
        mock_resolve,
        mock_detect,
        mock_get_backend,
        mock_config,
        mock_prompts,
        mock_workflow,
        mock_run,
        mock_ffmpeg,
    ):
        """video_cli.main() calls resolve_model_path on the model argument."""
        from zvisiongenerator.utils.video_model_detect import VideoModelInfo

        mock_resolve.return_value = "/resolved/path/ltx-2.3-mlx-q4"
        mock_detect.return_value = VideoModelInfo(
            family="ltx",
            backend="ltx",
            supports_i2v=True,
            default_fps=24,
            frame_alignment=8,
            resolution_alignment=32,
        )
        mock_backend = mock_get_backend.return_value
        mock_backend.load_model.return_value = ({"model": "test"}, mock_detect.return_value)

        with patch("sys.argv", ["ziv-video", "-m", "ltx-2.3-mlx-q4", "--prompt", "a dog"]):
            from zvisiongenerator.video_cli import main

            main()

        mock_resolve.assert_called_once()
        # The argument to resolve_model_path should be the tilde-expanded model name
        assert "ltx-2.3-mlx-q4" in mock_resolve.call_args[0][0]
