"""Tests for sizing LoRA files on disk."""

from __future__ import annotations

from zvisiongenerator.utils.lora_size import lora_files_bytes


def test_files_are_summed_and_a_folder_counts_its_files(tmp_path):
    (tmp_path / "a.safetensors").write_bytes(b"x" * 10)
    folder = tmp_path / "folder"
    folder.mkdir()
    (folder / "b.safetensors").write_bytes(b"y" * 5)
    (folder / "nested").mkdir()

    assert lora_files_bytes([tmp_path / "a.safetensors", folder]) == 15


def test_a_path_that_is_not_there_counts_nothing(tmp_path):
    assert lora_files_bytes([tmp_path / "missing"]) == 0


def test_no_paths_count_nothing():
    assert lora_files_bytes([]) == 0
