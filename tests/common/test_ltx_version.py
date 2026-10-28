"""Tests for telling the LTX version of a video model: aliases, known repos, converted folders and the config files of a local folder."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from zvisiongenerator.utils.ltx_converted import write_converted_ltx
from zvisiongenerator.utils.ltx_version import ltx_folder_version, ltx_model_version

_CONFIG = {"video_model_presets": {"ltx": {"versions": {"ltx25": {"aliases": ["ltx25-4"], "model_type": "ltx-2.5"}}}}}


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch) -> Path:
    """Point the data folder at an empty temporary one, so a model name resolves to a folder only when a test makes it."""
    monkeypatch.setenv("ZIV_DATA_DIR", str(tmp_path / "data"))
    return tmp_path / "data"


def _folder(tmp_path: Path, name: str, **files: object) -> Path:
    folder = tmp_path / name
    folder.mkdir()
    for file, content in files.items():
        (folder / file.replace("__", ".")).write_text(json.dumps(content), encoding="utf-8")
    return folder


class TestAliasesAndRepos:
    def test_an_alias_of_a_version_block_gets_that_version_and_any_other_alias_is_2_3(self):
        assert ltx_model_version(_CONFIG, "ltx25-4", "unsloth/LTX-2.5-FP8@abc") == "2.5"
        assert ltx_model_version(_CONFIG, "ltx-8", "diffusers/LTX-2.3-Distilled-Diffusers@abc") == "2.3"

    def test_a_known_repo_gets_its_version_without_reading_a_file(self):
        assert ltx_model_version({}, "my-name", "dgrauet/ltx-2.5-mlx-q4") == "2.5"

    def test_an_unknown_repo_is_2_3(self):
        assert ltx_model_version({}, "x", "someone/some-ltx-merge") == "2.3"


class TestLocalFolders:
    def test_ff_bias_false_in_the_embedded_config_is_2_5(self, tmp_path):
        folder = _folder(tmp_path, "mine", embedded_config__json={"transformer": {"ff_bias": False}})

        assert ltx_model_version({}, "mine", str(folder)) == "2.5"

    def test_the_first_config_found_is_the_one_read_as_the_loader_does(self, tmp_path):
        folder = _folder(tmp_path, "mine", embedded_config__json={"transformer": {}}, config__json={"transformer": {"ff_bias": False}})

        assert ltx_folder_version(folder) == "2.3"

    def test_config_json_is_read_when_there_is_no_embedded_config_and_a_flat_config_counts(self, tmp_path):
        folder = _folder(tmp_path, "mine", config__json={"ff_bias": False})

        assert ltx_folder_version(folder) == "2.5"

    @pytest.mark.parametrize("config", [{"transformer": {"num_layers": 48}}, {"transformer": {"ff_bias": True}}, {}, [], {"transformer": 3}])
    def test_a_config_without_ff_bias_false_is_2_3(self, tmp_path, config):
        folder = _folder(tmp_path, "mine", embedded_config__json=config)

        assert ltx_folder_version(folder) == "2.3"

    def test_a_missing_or_unreadable_config_falls_back_to_2_3(self, tmp_path):
        empty = _folder(tmp_path, "empty")
        broken = _folder(tmp_path, "broken")
        (broken / "embedded_config.json").write_text("{not json", encoding="utf-8")

        assert (ltx_folder_version(empty), ltx_folder_version(broken)) == ("2.3", "2.3")
        assert ltx_model_version({}, "gone", str(tmp_path / "does-not-exist")) == "2.3"

    def test_a_converted_folder_says_its_version_in_its_manifest(self, tmp_path):
        folder = _folder(tmp_path, "converted")
        write_converted_ltx(folder, {"bits": 4, "block_matrices": 1, "transformer_rest": 1}, version="2.5")
        other = _folder(tmp_path, "converted23")
        write_converted_ltx(other, {"bits": 4, "block_matrices": 1, "transformer_rest": 1}, version="2.3")

        assert (ltx_model_version({}, "converted", str(folder)), ltx_model_version({}, "converted23", str(other))) == ("2.5", "2.3")

    def test_a_diffusers_folder_with_a_gemma_4_text_encoder_is_2_5(self, tmp_path):
        index = {"_class_name": "LTX2Pipeline", "text_encoder": ["transformers", "Gemma4ForConditionalGeneration"]}
        folder = _folder(tmp_path, "diffusers", model_index__json=index)
        old = _folder(tmp_path, "diffusers23", model_index__json={**index, "text_encoder": ["transformers", "Gemma3ForConditionalGeneration"]})

        assert (ltx_folder_version(folder), ltx_folder_version(old)) == ("2.5", "2.3")
