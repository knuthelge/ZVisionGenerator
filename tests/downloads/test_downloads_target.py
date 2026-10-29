"""Tests for the download target value objects."""

from __future__ import annotations

import pytest

from zvisiongenerator.downloads.target import DownloadTarget, is_literal


class TestDownloadTarget:
    def test_key_defaults_revision_to_main_and_local_dir_to_empty(self):
        assert DownloadTarget("o/r").key == ("o/r", "main", "", ())

    def test_key_uses_revision_and_local_dir(self):
        assert DownloadTarget("o/r", revision="v1", local_dir="/tmp/x").key == ("o/r", "v1", "/tmp/x", ())

    def test_key_of_a_local_folder_names_the_files_wanted(self):
        first = DownloadTarget("o/r", local_dir="/tmp/x", allow_patterns=("b.safetensors", "a.safetensors"))
        same = DownloadTarget("o/r", local_dir="/tmp/x", allow_patterns=("a.safetensors", "b.safetensors"))
        other = DownloadTarget("o/r", local_dir="/tmp/x", allow_patterns=("c.safetensors",))

        assert first.key == same.key
        assert first.key != other.key

    def test_key_of_a_cache_target_ignores_its_patterns(self):
        assert DownloadTarget("o/r", allow_patterns=("a",)).key == DownloadTarget("o/r", allow_patterns=("b",)).key

    @pytest.mark.parametrize(
        "target",
        [
            DownloadTarget("o/r"),
            DownloadTarget("o/r", anonymous=True),
            DownloadTarget("o/r", revision="abc", allow_patterns=("vae/*", "a.json"), ignore_patterns=("*.bin",), local_dir="/tmp/x"),
            DownloadTarget("o/r", allow_patterns=()),
        ],
    )
    def test_json_round_trip(self, target):
        assert DownloadTarget.from_json(target.to_json()) == target


class TestIsLiteral:
    @pytest.mark.parametrize("pattern", ["model_index.json", "transformer/config.json"])
    def test_plain_paths_are_literal(self, pattern):
        assert is_literal(pattern) is True

    @pytest.mark.parametrize("pattern", ["*.json", "a?.bin"])
    def test_glob_characters_are_not_literal(self, pattern):
        assert is_literal(pattern) is False
