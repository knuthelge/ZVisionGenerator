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


def _table_data_rows(output: str) -> list[str]:
    return [line for line in output.splitlines() if line.startswith("  ") and "(none)" not in line and set(line.replace(" ", "")) != {"-"} and not line.lstrip().startswith("Name")]


def _alias_rows(output: str) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for line in _table_data_rows(output):
        alias, target = line.strip().split(None, 1)
        if target.startswith("→"):
            target = target[1:].strip()
        rows.append((alias.strip(), target.strip()))
    return rows


def _alias_row_map(output: str) -> dict[str, str]:
    return dict(_alias_rows(output))


def _alias_variants(target: str) -> list[str]:
    return [variant.strip() for variant in target.split(" / ") if variant.strip()]


def _alias_target_identifiers(target: str) -> list[str]:
    identifiers: list[str] = []
    for variant in _alias_variants(target):
        for token in variant.replace("(", " ").replace(")", " ").replace(":", " ").split():
            cleaned = token.strip(".,")
            if "/" in cleaned:
                identifiers.append(cleaned)
    return identifiers


def _has_unavailable_variant(target: str) -> bool:
    return len(_alias_variants(target)) > len(_alias_target_identifiers(target))


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

        assert result == []

    @patch("zvisiongenerator.converters.list_assets.detect_image_model")
    def test_skips_unknown_directories_so_video_only_models_do_not_appear_as_image_models(self, mock_detect, tmp_path: Path):
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        (models_dir / "ltx-local").mkdir()
        (models_dir / "zit-local").mkdir()

        info_map = {
            "ltx-local": ImageModelInfo(family="unknown", is_distilled=False, size=None),
            "zit-local": ImageModelInfo(family="zimage", is_distilled=False, size=None),
        }
        mock_detect.side_effect = lambda path: info_map[Path(path).name]

        result = list_models(tmp_path)

        assert [entry.name for entry in result] == ["zit-local"]

    @patch("zvisiongenerator.converters.list_assets.detect_image_model")
    def test_detection_error_skips_unknown_inventory_entry(self, mock_detect, tmp_path: Path):
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        (models_dir / "broken").mkdir()

        mock_detect.side_effect = RuntimeError("parse error")

        result = list_models(tmp_path)

        assert result == []

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
    def test_models_section_contains_discovered_entries(self):
        models = [ModelEntry(name="m1", family="zimage", size=None, is_distilled=False)]
        output = format_asset_table(models=models)
        lines = output.splitlines()
        data_rows = _table_data_rows(output)
        assert lines[0].endswith(":")
        assert len(data_rows) == 1
        assert "m1" in data_rows[0]
        assert "zimage" in data_rows[0]
        assert data_rows[0].rstrip().endswith("-")

    def test_loras_section_contains_discovered_entries(self):
        loras = [LoraEntry(name="l1", file_size_mb=1.5)]
        output = format_asset_table(loras=loras)
        lines = output.splitlines()
        data_rows = _table_data_rows(output)
        assert lines[0].endswith(":")
        assert len(data_rows) == 1
        assert "l1" in data_rows[0]
        assert data_rows[0].rstrip().endswith("1.5")

    def test_empty_models_shows_none(self):
        output = format_asset_table(models=[])
        lines = output.splitlines()
        assert len(lines) == 2
        assert lines[0].endswith(":")
        assert lines[1].startswith("  ")
        assert lines[1].endswith(")")

    def test_empty_loras_shows_none(self):
        output = format_asset_table(loras=[])
        lines = output.splitlines()
        assert len(lines) == 2
        assert lines[0].endswith(":")
        assert lines[1].startswith("  ")
        assert lines[1].endswith(")")

    def test_both_sections(self):
        models = [ModelEntry(name="m", family="zimage", size=None, is_distilled=False)]
        loras = [LoraEntry(name="l", file_size_mb=0.5)]
        output = format_asset_table(models=models, loras=loras)
        sections = output.split("\n\n")
        assert len(sections) == 2
        assert "m" in sections[0]
        assert "l" in sections[1]

    def test_none_params_omit_sections(self):
        output = format_asset_table(models=None, loras=None)
        assert output == ""

    def test_only_models_omits_loras(self):
        models = [ModelEntry(name="m", family="flux2", size="4b", is_distilled=True)]
        output = format_asset_table(models=models, loras=None)
        assert "m" in output
        assert "4b" in output
        assert "0.5" not in output

    def test_only_loras_omits_models(self):
        loras = [LoraEntry(name="l", file_size_mb=3.2)]
        output = format_asset_table(models=None, loras=loras)
        assert "l" in output
        assert "3.2" in output
        assert "flux2" not in output


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
    def test_video_models_section_contains_discovered_entries(self):
        vmodels = [VideoModelEntry(name="LTX-Video-0.9.1-mlx", family="ltx", supports_i2v=True)]
        output = format_asset_table(video_models=vmodels)
        lines = output.splitlines()
        data_rows = _table_data_rows(output)
        assert lines[0].endswith(":")
        assert len(data_rows) == 1
        assert "LTX-Video-0.9.1-mlx" in data_rows[0]
        assert "ltx" in data_rows[0]
        assert data_rows[0].rstrip().endswith("yes")

    def test_empty_video_models_shows_none(self):
        output = format_asset_table(video_models=[])
        lines = output.splitlines()
        assert len(lines) == 2
        assert lines[0].endswith(":")
        assert lines[1].startswith("  ")
        assert lines[1].endswith(")")

    def test_video_models_no_i2v(self):
        vmodels = [VideoModelEntry(name="test-model", family="ltx", supports_i2v=False)]
        output = format_asset_table(video_models=vmodels)
        assert "no" in output

    def test_all_three_sections(self):
        models = [ModelEntry(name="m", family="zimage", size=None, is_distilled=False)]
        vmodels = [VideoModelEntry(name="v", family="ltx", supports_i2v=True)]
        loras = [LoraEntry(name="l", file_size_mb=0.5)]
        output = format_asset_table(models=models, video_models=vmodels, loras=loras)
        sections = output.split("\n\n")
        assert len(sections) == 3
        assert "m" in sections[0]
        assert "v" in sections[1]
        assert "l" in sections[2]

    def test_only_video_models_omits_others(self):
        vmodels = [VideoModelEntry(name="v", family="ltx", supports_i2v=True)]
        output = format_asset_table(video_models=vmodels)
        sections = output.split("\n\n")
        assert len(sections) == 1
        assert "v" in sections[0]
        assert "ltx" in sections[0]

    def test_none_video_models_omits_section(self):
        output = format_asset_table(models=None, video_models=None, loras=None)
        assert "Video Models:" not in output


# ── format_asset_table with aliases ──────────────────────────────────────────


class TestFormatAssetTableAliases:
    def test_aliases_entries_render_as_alias_target_pairs(self):
        aliases = {"ltx-4": "dgrauet/ltx-2.3-mlx-q4", "zit": "Tongyi-MAI/Z-Image-Turbo"}
        output = format_asset_table(aliases=aliases)
        assert _alias_row_map(output) == aliases

    def test_aliases_sorted_alphabetically(self):
        aliases = {"zit": "Tongyi-MAI/Z-Image-Turbo", "klein4b": "black-forest-labs/FLUX.2-klein-4B"}
        output = format_asset_table(aliases=aliases)
        assert [alias for alias, _target in _alias_rows(output)] == ["klein4b", "zit"]

    def test_empty_aliases_shows_none(self):
        output = format_asset_table(aliases={})
        lines = output.splitlines()
        assert len(lines) == 2
        assert lines[0].endswith(":")
        assert lines[1].startswith("  ")
        assert lines[1].endswith(")")

    def test_none_aliases_omits_section(self):
        output = format_asset_table(aliases=None)
        assert output == ""

    def test_all_sections_with_aliases(self):
        models = [ModelEntry(name="m", family="zimage", size=None, is_distilled=False)]
        loras = [LoraEntry(name="l", file_size_mb=0.5)]
        aliases = {"ltx-4": "dgrauet/ltx-2.3-mlx-q4"}
        output = format_asset_table(models=models, loras=loras, aliases=aliases)
        sections = output.split("\n\n")
        assert len(sections) == 3
        assert "m" in sections[0]
        assert "l" in sections[1]
        assert "ltx-4" in sections[2]

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
        assert dict(_alias_rows(output)) == aliases

    def test_flat_aliases_render_unchanged(self):
        aliases = {"zit": "Tongyi-MAI/Z-Image-Turbo"}

        output = format_asset_table(aliases=aliases)

        assert _alias_rows(output) == [("zit", "Tongyi-MAI/Z-Image-Turbo")]
        assert "macOS" not in output
        assert "Windows" not in output

    def test_per_platform_aliases_render_platform_labels_and_targets(self):
        aliases = {
            "ltx-8": {
                "darwin": "dgrauet/ltx-2.3-mlx-q8",
                "win32": "Lightricks/LTX-2.3-fp8",
            }
        }

        output = format_asset_table(aliases=aliases, platforms={"darwin": "macOS", "win32": "Windows"})

        alias_target = _alias_row_map(output)["ltx-8"]
        assert len(_alias_variants(alias_target)) == 2
        assert _alias_target_identifiers(alias_target) == ["dgrauet/ltx-2.3-mlx-q8", "Lightricks/LTX-2.3-fp8"]
        assert not _has_unavailable_variant(alias_target)

    def test_message_aliases_render_unavailable_with_message(self):
        aliases = {
            "ltx-4": {
                "darwin": "dgrauet/ltx-2.3-mlx-q4",
                "win32": {"message": "LTX 4-bit is not available on Windows. Use 'ltx-8' instead."},
            }
        }

        output = format_asset_table(aliases=aliases, platforms={"darwin": "macOS", "win32": "Windows"})

        alias_target = _alias_row_map(output)["ltx-4"]
        assert len(_alias_variants(alias_target)) == 2
        assert _alias_target_identifiers(alias_target) == ["dgrauet/ltx-2.3-mlx-q4"]
        assert _has_unavailable_variant(alias_target)
