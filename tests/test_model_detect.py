"""Tests for zvisiongenerator.utils.model_detect — detect_image_model and _detect_klein_size."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from zvisiongenerator.utils.image_model_detect import _detect_klein_size, detect_image_model


class TestDetectModelType:
    def test_zimage_pipeline(self, tmp_path):
        (tmp_path / "model_index.json").write_text(json.dumps({"_class_name": "ZImagePipeline"}))
        info = detect_image_model(str(tmp_path))
        assert info.family == "zimage"
        assert info.is_distilled is False
        assert info.size is None

    def test_flux2_klein_pipeline(self, tmp_path):
        (tmp_path / "model_index.json").write_text(json.dumps({"_class_name": "Flux2KleinPipeline", "is_distilled": True}))
        # Need "4b" in path for size detection
        klein_dir = tmp_path / "flux2-klein-4b"
        klein_dir.mkdir()
        (klein_dir / "model_index.json").write_text(json.dumps({"_class_name": "Flux2KleinPipeline", "is_distilled": True}))
        info = detect_image_model(str(klein_dir))
        assert info.family == "flux2_klein"
        assert info.is_distilled is True
        assert info.size == "4b"

    def test_flux1_pipeline(self, tmp_path):
        (tmp_path / "model_index.json").write_text(json.dumps({"_class_name": "FluxPipeline"}))
        info = detect_image_model(str(tmp_path))
        assert info.family == "flux1"
        assert info.is_distilled is False
        assert info.size is None

    def test_flux2_pipeline_raises(self, tmp_path):
        (tmp_path / "model_index.json").write_text(json.dumps({"_class_name": "Flux2Pipeline"}))
        with pytest.raises(ValueError, match="FLUX.2-dev.*not supported"):
            detect_image_model(str(tmp_path))

    def test_missing_model_index_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="model_index.json not found"):
            detect_image_model(str(tmp_path))

    def test_missing_class_name_raises(self, tmp_path):
        (tmp_path / "model_index.json").write_text(json.dumps({"other": "field"}))
        with pytest.raises(ValueError, match="missing the '_class_name' field"):
            detect_image_model(str(tmp_path))

    def test_unknown_class_name(self, tmp_path):
        (tmp_path / "model_index.json").write_text(json.dumps({"_class_name": "SomeOtherPipeline"}))
        info = detect_image_model(str(tmp_path))
        assert info.family == "unknown"

    def test_huggingface_repo_id(self, tmp_path):
        index_file = tmp_path / "model_index.json"
        index_file.write_text(json.dumps({"_class_name": "ZImagePipeline"}))
        mock_download = MagicMock(return_value=str(index_file))
        with patch(
            "huggingface_hub.hf_hub_download",
            mock_download,
        ):
            info = detect_image_model("org/some-model")
        mock_download.assert_called_once_with(repo_id="org/some-model", filename="model_index.json")
        assert info.family == "zimage"

    # ── Local path error handling ──

    def test_local_relative_path_not_found_raises(self):
        with pytest.raises(FileNotFoundError, match="Local model directory not found"):
            detect_image_model("./models/nonexistent")

    def test_local_absolute_path_not_found_raises(self, tmp_path):
        fake = str(tmp_path / "no_such_model")
        with pytest.raises(FileNotFoundError, match="Local model directory not found"):
            detect_image_model(fake)

    def test_tilde_path_not_found_raises(self):
        with pytest.raises(FileNotFoundError, match="Local model directory not found"):
            detect_image_model("~/no_such_model_dir_xyz")

    def test_hf_repo_id_not_treated_as_local(self, tmp_path):
        """A bare 'org/repo' without local-path prefixes should NOT raise FileNotFoundError."""
        index_file = tmp_path / "model_index.json"
        index_file.write_text(json.dumps({"_class_name": "ZImagePipeline"}))
        mock_download = MagicMock(return_value=str(index_file))
        with patch("huggingface_hub.hf_hub_download", mock_download):
            info = detect_image_model("org/some-model")
        assert info.family == "zimage"

    def test_hf_failure_with_local_first_segment_gives_hint(self, tmp_path, monkeypatch):
        """When HF download fails and first path segment is a local dir, give a helpful error."""
        # Create a local 'models' directory so the hint triggers
        monkeypatch.chdir(tmp_path)
        (tmp_path / "models").mkdir()
        with patch(
            "huggingface_hub.hf_hub_download",
            side_effect=Exception("repo not found"),
        ):
            with pytest.raises(FileNotFoundError, match="did you mean './models/foo'"):
                detect_image_model("models/foo")


class TestDetectKleinSize:
    def test_path_contains_4b(self, tmp_path):
        model_dir = tmp_path / "flux2-klein-4b"
        model_dir.mkdir()
        assert _detect_klein_size(str(model_dir), is_local=True) == "4b"

    def test_path_contains_9b(self, tmp_path):
        model_dir = tmp_path / "flux2-klein-9b"
        model_dir.mkdir()
        assert _detect_klein_size(str(model_dir), is_local=True) == "9b"

    def test_path_case_insensitive(self, tmp_path):
        model_dir = tmp_path / "flux2-klein-4B"
        model_dir.mkdir()
        assert _detect_klein_size(str(model_dir), is_local=True) == "4b"

    def test_fallback_to_config_json_small(self, tmp_path):
        transformer_dir = tmp_path / "transformer"
        transformer_dir.mkdir()
        (transformer_dir / "config.json").write_text(json.dumps({"num_single_layers": 18}))
        assert _detect_klein_size(str(tmp_path), is_local=True) == "4b"

    def test_fallback_to_config_json_large(self, tmp_path):
        transformer_dir = tmp_path / "transformer"
        transformer_dir.mkdir()
        (transformer_dir / "config.json").write_text(json.dumps({"num_single_layers": 24}))
        assert _detect_klein_size(str(tmp_path), is_local=True) == "9b"

    def test_no_config_returns_none(self, tmp_path):
        assert _detect_klein_size(str(tmp_path), is_local=True) is None

    def test_huggingface_fallback_calls_download(self, tmp_path):
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({"num_single_layers": 20}))
        mock_download = MagicMock(return_value=str(config_path))
        with patch.dict("sys.modules", {"huggingface_hub": MagicMock(hf_hub_download=mock_download)}):
            result = _detect_klein_size("org/model-klein", is_local=False)
        mock_download.assert_called_once()
        assert result == "4b"


class TestRelativePathDetection:
    """Tests for the improved local-path vs HF-repo-ID heuristic."""

    def test_relative_path_with_existing_parent_treated_as_hf(self, tmp_path, monkeypatch):
        """models/nonexistent with models/ existing locally → ambiguous, tries HF.

        When the path matches the HF repo ID pattern (org/repo), it's treated
        as an HF repo ID even if the first segment exists locally. This avoids
        blocking valid HF repos whose org name shadows a local directory.
        """
        monkeypatch.chdir(tmp_path)
        (tmp_path / "models").mkdir()
        index_file = tmp_path / "model_index.json"
        index_file.write_text(json.dumps({"_class_name": "ZImagePipeline"}))
        mock_download = MagicMock(return_value=str(index_file))
        with patch("huggingface_hub.hf_hub_download", mock_download):
            info = detect_image_model("models/nonexistent")
        assert info.family == "zimage"

    def test_hf_repo_works_when_org_dir_exists_locally(self, tmp_path, monkeypatch):
        """black-forest-labs/model should go to HF even if ./black-forest-labs/ exists."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / "black-forest-labs").mkdir()
        index_file = tmp_path / "model_index.json"
        index_file.write_text(json.dumps({"_class_name": "ZImagePipeline"}))
        mock_download = MagicMock(return_value=str(index_file))
        with patch("huggingface_hub.hf_hub_download", mock_download):
            info = detect_image_model("black-forest-labs/some-model")
        mock_download.assert_called_once_with(repo_id="black-forest-labs/some-model", filename="model_index.json")
        assert info.family == "zimage"

    def test_hf_repo_id_no_local_dir_passes(self, tmp_path):
        """user/repo where user/ doesn't exist locally → treated as HF repo."""
        index_file = tmp_path / "model_index.json"
        index_file.write_text(json.dumps({"_class_name": "ZImagePipeline"}))
        mock_download = MagicMock(return_value=str(index_file))
        with patch("huggingface_hub.hf_hub_download", mock_download):
            info = detect_image_model("someorg/some-model")
        mock_download.assert_called_once_with(repo_id="someorg/some-model", filename="model_index.json")
        assert info.family == "zimage"

    def test_backslash_path_raises(self):
        """Paths with backslashes are definitely local, not HF repo IDs."""
        with pytest.raises(FileNotFoundError, match="Local model directory not found"):
            detect_image_model("models\\nonexistent")

    def test_nested_path_raises(self):
        """Paths with multiple segments (a/b/c) don't match HF pattern."""
        with pytest.raises(FileNotFoundError, match="Local model directory not found"):
            detect_image_model("models/subdir/nested")
