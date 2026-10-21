"""Tests for the download target value objects."""

from __future__ import annotations

import dataclasses

import pytest

from zvisiongenerator.downloads.target import DownloadProgress, DownloadTarget, TargetCheck, is_literal


class TestDownloadTarget:
    def test_key_defaults_revision_to_main_and_local_dir_to_empty(self):
        assert DownloadTarget("o/r").key == ("o/r", "main", "")

    def test_key_uses_revision_and_local_dir(self):
        assert DownloadTarget("o/r", revision="v1", local_dir="/tmp/x").key == ("o/r", "v1", "/tmp/x")

    @pytest.mark.parametrize(
        "target",
        [
            DownloadTarget("o/r"),
            DownloadTarget("o/r", revision="abc", allow_patterns=("vae/*", "a.json"), ignore_patterns=("*.bin",), local_dir="/tmp/x"),
            DownloadTarget("o/r", allow_patterns=()),
        ],
    )
    def test_json_round_trip(self, target):
        assert DownloadTarget.from_json(target.to_json()) == target

    def test_is_frozen(self):
        with pytest.raises(dataclasses.FrozenInstanceError):
            DownloadTarget("o/r").repo_id = "x"  # type: ignore[misc]


class TestIsLiteral:
    @pytest.mark.parametrize("pattern", ["model_index.json", "transformer/config.json", "turbo.safetensors"])
    def test_plain_paths_are_literal(self, pattern):
        assert is_literal(pattern) is True

    @pytest.mark.parametrize("pattern", ["*.json", "vae/*", "a?.bin", "x[0-9].txt", "tokenizer/**"])
    def test_glob_characters_are_not_literal(self, pattern):
        assert is_literal(pattern) is False


def test_progress_and_check_are_frozen_values():
    assert DownloadProgress(1, None, None, None).bytes_total is None
    assert TargetCheck(2, None).missing_bytes is None
    with pytest.raises(dataclasses.FrozenInstanceError):
        TargetCheck(1, 2).missing_files = 0  # type: ignore[misc]
