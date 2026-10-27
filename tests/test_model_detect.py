"""Tests for zvisiongenerator.utils.model_detect — detect_image_model and _detect_klein_size."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, call, patch

import pytest

from zvisiongenerator.utils.image_model_detect import _detect_klein_size, detect_image_model


class TestDetectModelType:
    def test_zimage_pipeline(self, tmp_path):
        (tmp_path / "model_index.json").write_text(json.dumps({"_class_name": "ZImagePipeline"}))
        info = detect_image_model(str(tmp_path))
        assert info.family == "zimage"
        assert info.is_distilled is False
        assert info.size is None

    def test_ideogram4_pipeline(self, tmp_path):
        (tmp_path / "model_index.json").write_text(json.dumps({"_class_name": "Ideogram4Pipeline"}))
        info = detect_image_model(str(tmp_path))
        assert info.family == "ideogram4"
        assert info.is_distilled is False
        assert info.size is None

    def test_krea2_pipeline(self, tmp_path):
        (tmp_path / "model_index.json").write_text(json.dumps({"_class_name": "Krea2Pipeline", "is_distilled": True}))
        info = detect_image_model(str(tmp_path))
        assert info.family == "krea2"
        assert info.size is None

    @pytest.mark.parametrize("distilled", [True, False])
    def test_moss_pipeline(self, tmp_path, distilled):
        (tmp_path / "model_index.json").write_text(json.dumps({"_class_name": "MOSSImagePipeline", "is_distilled": distilled}))
        info = detect_image_model(str(tmp_path))
        assert info.family == "moss"
        assert info.is_distilled is distilled
        assert info.size is None

    def test_flux2_klein_pipeline(self, tmp_path):
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
        mock_download.assert_called_once_with(repo_id="org/some-model", filename="model_index.json", local_files_only=True)
        assert info.family == "zimage"

    def test_huggingface_repo_id_falls_back_to_network_when_not_cached(self, tmp_path):
        index_file = tmp_path / "model_index.json"
        index_file.write_text(json.dumps({"_class_name": "ZImagePipeline"}))
        mock_download = MagicMock(side_effect=[FileNotFoundError("not cached"), str(index_file)])
        with patch("huggingface_hub.hf_hub_download", mock_download):
            info = detect_image_model("org/some-model")
        assert mock_download.call_args_list == [
            call(repo_id="org/some-model", filename="model_index.json", local_files_only=True),
            call(repo_id="org/some-model", filename="model_index.json"),
        ]
        assert info.family == "zimage"

    # ── Local path error handling ──

    @pytest.mark.parametrize(
        "reference",
        ["./models/nonexistent", "{tmp}/no_such_model", "~/no_such_model_dir_xyz", "models\\nonexistent", "models/subdir/nested"],
        ids=["relative", "absolute", "tilde", "backslash", "nested"],
    )
    def test_a_local_path_that_does_not_exist_raises(self, tmp_path, reference):
        with pytest.raises(FileNotFoundError, match="Local model directory not found"):
            detect_image_model(reference.format(tmp=tmp_path))

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
    @pytest.mark.parametrize(
        ("folder", "expected"),
        [("flux2-klein-4b", "4b"), ("flux2-klein-9b", "9b"), ("models--unsloth--FLUX.2-klein-9B", "9b"), ("flux2-klein-4B", "4b")],
        ids=["4b", "9b", "hub-cache-mirror", "case-insensitive"],
    )
    def test_the_size_comes_from_the_folder_name(self, tmp_path, folder, expected):
        model_dir = tmp_path / folder
        model_dir.mkdir()
        assert _detect_klein_size(str(model_dir), is_local=True) == expected

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
        mock_download.assert_called_once_with(repo_id="black-forest-labs/some-model", filename="model_index.json", local_files_only=True)
        assert info.family == "zimage"


class TestDetectWithRevision:
    """A repo pinned as owner/repo@revision is looked up at that revision."""

    def test_the_revision_is_passed_to_the_hub(self, tmp_path):
        index = tmp_path / "model_index.json"
        index.write_text(json.dumps({"_class_name": "Ideogram4Pipeline"}))
        mock_download = MagicMock(return_value=str(index))

        with patch("huggingface_hub.hf_hub_download", mock_download):
            info = detect_image_model("org/some-model@abc123")

        assert info.family == "ideogram4"
        mock_download.assert_called_once_with(repo_id="org/some-model", filename="model_index.json", local_files_only=True, revision="abc123")

    def test_a_revision_is_not_mistaken_for_a_local_path(self, tmp_path):
        index = tmp_path / "model_index.json"
        index.write_text(json.dumps({"_class_name": "Ideogram4Pipeline"}))

        with patch("huggingface_hub.hf_hub_download", MagicMock(return_value=str(index))):
            assert detect_image_model("org/some-model@v1.0").family == "ideogram4"

    def test_a_revision_on_a_model_whose_loader_cannot_pin_is_refused(self, tmp_path):
        index = tmp_path / "model_index.json"
        index.write_text(json.dumps({"_class_name": "ZImagePipeline"}))

        with patch("huggingface_hub.hf_hub_download", MagicMock(return_value=str(index))), pytest.raises(ValueError, match="only supported for Ideogram 4"):
            detect_image_model("org/some-model@v1.0")
