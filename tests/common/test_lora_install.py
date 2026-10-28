"""Tests for verifying a staged LoRA file and renaming it into place."""

from __future__ import annotations

import hashlib

import pytest

from zvisiongenerator.utils.lora_install import install_staged, sha256_file

_DATA = b"safetensors-bytes" * 100
_SHA = hashlib.sha256(_DATA).hexdigest()


@pytest.fixture
def staged(tmp_path):
    folder = tmp_path / "loras" / ".download-style"
    folder.mkdir(parents=True)
    path = folder / "file.safetensors"
    path.write_bytes(_DATA)
    return path


def test_sha256_file_hashes_the_whole_file(tmp_path):
    path = tmp_path / "f.safetensors"
    data = b"x" * (8 * 1024 * 1024 + 1)  # one byte over the 8 MiB read chunk, so the hash spans two reads
    path.write_bytes(data)

    assert sha256_file(path) == hashlib.sha256(data).hexdigest()


def test_a_verified_file_is_renamed_to_the_destination_and_leaves_the_staging_path(staged, tmp_path):
    destination = tmp_path / "loras" / "style.safetensors"

    digest = install_staged(staged, destination, expected_sha256=_SHA.upper(), expected_size=len(_DATA))

    assert digest == _SHA
    assert destination.read_bytes() == _DATA
    assert not staged.exists()


def test_a_sha_mismatch_raises_and_leaves_no_file_in_the_destination_folder(staged, tmp_path):
    destination = tmp_path / "loras" / "style.safetensors"

    with pytest.raises(ValueError, match="SHA-256"):
        install_staged(staged, destination, expected_sha256="0" * 64)

    assert not destination.exists()
    assert sorted(path.name for path in (tmp_path / "loras").iterdir()) == [".download-style"]


def test_a_size_mismatch_raises_before_hashing(staged, tmp_path):
    with pytest.raises(ValueError, match="bytes"):
        install_staged(staged, tmp_path / "loras" / "style.safetensors", expected_size=len(_DATA) + 1)


def test_an_existing_destination_is_never_replaced(staged, tmp_path):
    destination = tmp_path / "loras" / "style.safetensors"
    destination.write_bytes(b"mine")

    with pytest.raises(FileExistsError):
        install_staged(staged, destination)

    assert destination.read_bytes() == b"mine"
    assert staged.exists()


def test_a_missing_staged_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        install_staged(tmp_path / "missing.safetensors", tmp_path / "x.safetensors")


def test_without_expectations_the_file_is_installed_and_hashed(staged, tmp_path):
    assert install_staged(staged, tmp_path / "loras" / "style.safetensors") == _SHA
