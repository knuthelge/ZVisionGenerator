"""Tests for zvisiongenerator.converters.list_assets."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch


from zvisiongenerator.converters.list_assets import (
    LoraEntry,
    ModelEntry,
    VideoModelEntry,
    format_asset_table,
    list_loras,
    list_models,
    list_video_models,
)
from zvisiongenerator.utils.image_model_detect import ImageModelInfo
from zvisiongenerator.utils.video_model_detect import VideoModelInfo


# ── list_models ──────────────────────────────────────────────────────────────


class TestListModels:
    def test_missing_models_dir(self, tmp_path: Path):
        assert list_models(tmp_path) == []

    def test_empty_models_dir(self, tmp_path: Path):
        (tmp_path / "models").mkdir()
        assert list_models(tmp_path) == []

    @patch("zvisiongenerator.converters.list_assets.detect_image_model")
    def test_detects_model_directories(self, mock_detect, tmp_path: Path):
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        (models_dir / "alpha").mkdir()
        (models_dir / "beta").mkdir()

        # detect_image_model is called per directory in iterdir() order (not sorted),
        # so we use a dict to map directory name → ModelInfo
        info_map = {
            "alpha": ImageModelInfo(family="zimage", is_distilled=False, size=None),
            "beta": ImageModelInfo(family="flux2_klein", is_distilled=True, size="4b"),
        }
        mock_detect.side_effect = lambda path: info_map[Path(path).name]

        result = list_models(tmp_path)

        assert len(result) == 2
        # Result is sorted by name
        assert result[0].name == "alpha"
        assert result[0].family == "zimage"
        assert result[1].name == "beta"
        assert result[1].family == "flux2_klein"
        assert result[1].size == "4b"
        assert result[1].is_distilled is True

    @patch("zvisiongenerator.converters.list_assets.detect_image_model")
    def test_sorted_by_name(self, mock_detect, tmp_path: Path):
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        (models_dir / "zebra").mkdir()
        (models_dir / "alpha").mkdir()

        mock_detect.return_value = ImageModelInfo(family="unknown", is_distilled=False, size=None)

        result = list_models(tmp_path)

        assert [e.name for e in result] == ["alpha", "zebra"]

    @patch("zvisiongenerator.converters.list_assets.detect_image_model")
    def test_detection_error_marks_unknown(self, mock_detect, tmp_path: Path):
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        (models_dir / "broken").mkdir()

        mock_detect.side_effect = RuntimeError("parse error")

        result = list_models(tmp_path)

        assert len(result) == 1
        assert result[0].family == "unknown"
        assert result[0].size is None

    def test_ignores_files_in_models_dir(self, tmp_path: Path):
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        (models_dir / "not-a-model.txt").write_text("hi")

        with patch("zvisiongenerator.converters.list_assets.detect_image_model") as mock_detect:
            result = list_models(tmp_path)

        assert result == []
        mock_detect.assert_not_called()


# ── list_loras ───────────────────────────────────────────────────────────────


class TestListLoras:
    def test_missing_loras_dir(self, tmp_path: Path):
        assert list_loras(tmp_path) == []

    def test_empty_loras_dir(self, tmp_path: Path):
        (tmp_path / "loras").mkdir()
        assert list_loras(tmp_path) == []

    def test_lists_safetensors_files(self, tmp_path: Path):
        loras_dir = tmp_path / "loras"
        loras_dir.mkdir()
        lora_file = loras_dir / "style.safetensors"
        lora_file.write_bytes(b"\x00" * (2 * 1024 * 1024))  # 2 MB

        result = list_loras(tmp_path)

        assert len(result) == 1
        assert result[0].name == "style"
        assert result[0].file_size_mb == 2.0

    def test_ignores_non_safetensors(self, tmp_path: Path):
        loras_dir = tmp_path / "loras"
        loras_dir.mkdir()
        (loras_dir / "readme.txt").write_text("info")
        (loras_dir / "model.bin").write_bytes(b"data")
        (loras_dir / "good.safetensors").write_bytes(b"\x00" * 1024)

        result = list_loras(tmp_path)

        assert len(result) == 1
        assert result[0].name == "good"

    def test_sorted_by_name(self, tmp_path: Path):
        loras_dir = tmp_path / "loras"
        loras_dir.mkdir()
        (loras_dir / "zebra.safetensors").write_bytes(b"z")
        (loras_dir / "alpha.safetensors").write_bytes(b"a")

        result = list_loras(tmp_path)

        assert [e.name for e in result] == ["alpha", "zebra"]


# ── format_asset_table ───────────────────────────────────────────────────────


class TestFormatAssetTable:
    def test_models_header_and_columns(self):
        models = [ModelEntry(name="m1", family="zimage", size=None, is_distilled=False)]
        output = format_asset_table(models=models)
        assert "Models:" in output
        assert "Name" in output
        assert "Family" in output

    def test_loras_header_and_columns(self):
        loras = [LoraEntry(name="l1", file_size_mb=1.5)]
        output = format_asset_table(loras=loras)
        assert "LoRAs:" in output
        assert "Name" in output
        assert "Size (MB)" in output

    def test_empty_models_shows_none(self):
        output = format_asset_table(models=[])
        assert "(none)" in output

    def test_empty_loras_shows_none(self):
        output = format_asset_table(loras=[])
        assert "(none)" in output

    def test_both_sections(self):
        models = [ModelEntry(name="m", family="zimage", size=None, is_distilled=False)]
        loras = [LoraEntry(name="l", file_size_mb=0.5)]
        output = format_asset_table(models=models, loras=loras)
        assert "Models:" in output
        assert "LoRAs:" in output

    def test_none_params_omit_sections(self):
        output = format_asset_table(models=None, loras=None)
        assert output == ""

    def test_only_models_omits_loras(self):
        models = [ModelEntry(name="m", family="flux2", size="4b", is_distilled=True)]
        output = format_asset_table(models=models, loras=None)
        assert "Models:" in output
        assert "LoRAs:" not in output

    def test_only_loras_omits_models(self):
        loras = [LoraEntry(name="l", file_size_mb=3.2)]
        output = format_asset_table(models=None, loras=loras)
        assert "LoRAs:" in output
        assert "Models:" not in output


# ── list_video_models ────────────────────────────────────────────────────────


class TestListVideoModels:
    def test_missing_models_dir(self, tmp_path: Path):
        assert list_video_models(tmp_path) == []

    def test_empty_models_dir(self, tmp_path: Path):
        (tmp_path / "models").mkdir()
        assert list_video_models(tmp_path) == []

    @patch("zvisiongenerator.converters.list_assets.detect_video_model")
    def test_detects_video_model_directories(self, mock_detect, tmp_path: Path):
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        (models_dir / "LTX-Video-0.9.1-mlx").mkdir()

        mock_detect.return_value = VideoModelInfo(
            family="ltx",
            backend="ltx",
            supports_i2v=True,
            default_fps=24,
            frame_alignment=8,
            resolution_alignment=32,
        )

        result = list_video_models(tmp_path)

        assert len(result) == 1
        assert result[0].name == "LTX-Video-0.9.1-mlx"
        assert result[0].family == "ltx"
        assert result[0].supports_i2v is True

    @patch("zvisiongenerator.converters.list_assets.detect_video_model")
    def test_skips_unknown_family(self, mock_detect, tmp_path: Path):
        """Dirs classified as unknown by detect_video_model are skipped."""
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        (models_dir / "some-image-model").mkdir()

        mock_detect.return_value = VideoModelInfo(
            family="unknown",
            backend="unknown",
            supports_i2v=False,
            default_fps=24,
            frame_alignment=1,
            resolution_alignment=1,
        )

        result = list_video_models(tmp_path)
        assert result == []

    @patch("zvisiongenerator.converters.list_assets.detect_video_model")
    def test_sorted_by_name(self, mock_detect, tmp_path: Path):
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        (models_dir / "zzz-ltx-model").mkdir()
        (models_dir / "aaa-ltx-model").mkdir()

        mock_detect.return_value = VideoModelInfo(family="ltx", backend="ltx", supports_i2v=True, default_fps=24, frame_alignment=8, resolution_alignment=32)

        result = list_video_models(tmp_path)
        assert [e.name for e in result] == ["aaa-ltx-model", "zzz-ltx-model"]

    @patch("zvisiongenerator.converters.list_assets.detect_video_model")
    def test_mixed_video_and_image_models(self, mock_detect, tmp_path: Path):
        """Only video models (non-unknown) are returned; image models are skipped."""
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        (models_dir / "LTX-Video-0.9.1-mlx").mkdir()
        (models_dir / "Z-Image-Turbo").mkdir()

        info_map = {
            "LTX-Video-0.9.1-mlx": VideoModelInfo(family="ltx", backend="ltx", supports_i2v=True, default_fps=24, frame_alignment=8, resolution_alignment=32),
            "Z-Image-Turbo": VideoModelInfo(family="unknown", backend="unknown", supports_i2v=False, default_fps=24, frame_alignment=1, resolution_alignment=1),
        }
        mock_detect.side_effect = lambda path: info_map[Path(path).name]

        result = list_video_models(tmp_path)
        assert len(result) == 1
        assert result[0].name == "LTX-Video-0.9.1-mlx"

    def test_ignores_files(self, tmp_path: Path):
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        (models_dir / "not-a-model.txt").write_text("hi")

        with patch("zvisiongenerator.converters.list_assets.detect_video_model") as mock_detect:
            result = list_video_models(tmp_path)

        assert result == []
        mock_detect.assert_not_called()


# ── format_asset_table with video models ─────────────────────────────────────


class TestFormatAssetTableVideoModels:
    def test_video_models_header_and_columns(self):
        vmodels = [VideoModelEntry(name="LTX-Video-0.9.1-mlx", family="ltx", supports_i2v=True)]
        output = format_asset_table(video_models=vmodels)
        assert "Video Models:" in output
        assert "Name" in output
        assert "Family" in output
        assert "I2V" in output
        assert "LTX-Video-0.9.1-mlx" in output
        assert "ltx" in output
        assert "yes" in output

    def test_empty_video_models_shows_none(self):
        output = format_asset_table(video_models=[])
        assert "Video Models:" in output
        assert "(none)" in output

    def test_video_models_no_i2v(self):
        vmodels = [VideoModelEntry(name="test-model", family="ltx", supports_i2v=False)]
        output = format_asset_table(video_models=vmodels)
        assert "no" in output

    def test_all_three_sections(self):
        models = [ModelEntry(name="m", family="zimage", size=None, is_distilled=False)]
        vmodels = [VideoModelEntry(name="v", family="ltx", supports_i2v=True)]
        loras = [LoraEntry(name="l", file_size_mb=0.5)]
        output = format_asset_table(models=models, video_models=vmodels, loras=loras)
        assert "Models:" in output
        assert "Video Models:" in output
        assert "LoRAs:" in output

    def test_only_video_models_omits_others(self):
        vmodels = [VideoModelEntry(name="v", family="ltx", supports_i2v=True)]
        output = format_asset_table(video_models=vmodels)
        assert "Video Models:" in output
        # "Models:" appears as part of "Video Models:" — check no standalone Models: section
        assert output.replace("Video Models:", "").count("Models:") == 0
        assert "LoRAs:" not in output

    def test_none_video_models_omits_section(self):
        output = format_asset_table(models=None, video_models=None, loras=None)
        assert "Video Models:" not in output


# ── format_asset_table with aliases ──────────────────────────────────────────


class TestFormatAssetTableAliases:
    def test_aliases_header_and_entries(self):
        aliases = {"ltx-4": "dgrauet/ltx-2.3-mlx-q4", "zit": "Tongyi-MAI/Z-Image-Turbo"}
        output = format_asset_table(aliases=aliases)
        assert "Model Aliases:" in output
        assert "ltx-4" in output
        assert "dgrauet/ltx-2.3-mlx-q4" in output
        assert "zit" in output
        assert "Tongyi-MAI/Z-Image-Turbo" in output

    def test_aliases_sorted_alphabetically(self):
        aliases = {"zit": "Tongyi-MAI/Z-Image-Turbo", "klein4b": "black-forest-labs/FLUX.2-klein-4B"}
        output = format_asset_table(aliases=aliases)
        lines = output.strip().split("\n")
        # After header, sorted entries: klein4b before zit
        alias_lines = [line.strip() for line in lines[1:] if line.strip()]
        assert alias_lines[0].startswith("klein4b")
        assert alias_lines[1].startswith("zit")

    def test_empty_aliases_shows_none(self):
        output = format_asset_table(aliases={})
        assert "Model Aliases:" in output
        assert "(none)" in output

    def test_none_aliases_omits_section(self):
        output = format_asset_table(aliases=None)
        assert "Model Aliases:" not in output

    def test_all_sections_with_aliases(self):
        models = [ModelEntry(name="m", family="zimage", size=None, is_distilled=False)]
        loras = [LoraEntry(name="l", file_size_mb=0.5)]
        aliases = {"ltx-4": "dgrauet/ltx-2.3-mlx-q4"}
        output = format_asset_table(models=models, loras=loras, aliases=aliases)
        assert "Models:" in output
        assert "LoRAs:" in output
        assert "Model Aliases:" in output

    def test_aliases_uses_arrow_separator(self):
        aliases = {"ltx-4": "dgrauet/ltx-2.3-mlx-q4"}
        output = format_asset_table(aliases=aliases)
        assert "→" in output

    def test_all_five_config_aliases_displayed(self):
        """All 5 configured aliases appear in the formatted output."""
        aliases = {
            "ltx-8": "dgrauet/ltx-2.3-mlx-q8",
            "ltx-4": "dgrauet/ltx-2.3-mlx-q4",
            "zit": "Tongyi-MAI/Z-Image-Turbo",
            "klein9b": "black-forest-labs/FLUX.2-klein-9B",
            "klein4b": "black-forest-labs/FLUX.2-klein-4B",
        }
        output = format_asset_table(aliases=aliases)
        for alias, target in aliases.items():
            assert alias in output
            assert target in output

    def test_dict_alias_shows_both_platforms(self):
        """Platform-aware dict alias displays both platform values."""
        aliases = {"ltx-8": {"darwin": "dgrauet/ltx-2.3-mlx-q8", "win32": "Lightricks/LTX-2.3-fp8"}}
        output = format_asset_table(aliases=aliases, platform_labels={"darwin": "macOS", "win32": "Windows"})
        assert "ltx-8" in output
        assert "dgrauet/ltx-2.3-mlx-q8" in output
        assert "Lightricks/LTX-2.3-fp8" in output
        assert "macOS" in output
        assert "Windows" in output

    def test_mixed_string_and_dict_aliases(self):
        """Mix of string and dict aliases renders correctly."""
        aliases = {
            "ltx-4": {"darwin": "dgrauet/ltx-2.3-mlx-q4"},
            "zit": "Tongyi-MAI/Z-Image-Turbo",
        }
        output = format_asset_table(aliases=aliases, platform_labels={"darwin": "macOS", "win32": "Windows"})
        assert "ltx-4" in output
        assert "macOS" in output
        assert "Windows coming soon" in output
        assert "zit" in output
        assert "Tongyi-MAI/Z-Image-Turbo" in output
        # String alias should NOT have platform labels
        zit_line = [line for line in output.split("\n") if "zit" in line][0]
        assert "macOS" not in zit_line
        assert "Windows" not in zit_line

    def test_dict_alias_unsupported_platform_value_only_shows_supported_entry(self):
        """Unsupported dict values render as coming soon rather than a bogus path entry."""
        aliases = {
            "ltx-4": {
                "darwin": "dgrauet/ltx-2.3-mlx-q4",
                "win32": {"message": "Use 'ltx-8' instead."},
            }
        }

        output = format_asset_table(aliases=aliases, platform_labels={"darwin": "macOS", "win32": "Windows"})

        assert "dgrauet/ltx-2.3-mlx-q4 (macOS)" in output
        assert "Windows coming soon" in output
        assert "Use 'ltx-8' instead." not in output
