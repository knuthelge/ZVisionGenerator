"""Tests for zvisiongenerator.converters.lora_import."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from zvisiongenerator.converters.lora_import import (
    _validate_name,
    import_lora_hf,
    import_lora_local,
)


# ── _validate_name ───────────────────────────────────────────────────────────


class TestValidateName:
    def test_valid_name_passes(self):
        assert _validate_name("my-lora") == "my-lora"

    def test_strips_safetensors_suffix(self):
        assert _validate_name("my-lora.safetensors") == "my-lora"

    def test_empty_name_raises(self):
        with pytest.raises(ValueError, match="must not be empty"):
            _validate_name("")

    def test_only_extension_raises(self):
        # ".safetensors" stripped → empty string
        with pytest.raises(ValueError, match="must not be empty"):
            _validate_name(".safetensors")

    def test_slash_raises(self):
        with pytest.raises(ValueError, match="Invalid LoRA name"):
            _validate_name("path/name")

    def test_backslash_raises(self):
        with pytest.raises(ValueError, match="Invalid LoRA name"):
            _validate_name("path\\name")

    def test_dotdot_raises(self):
        with pytest.raises(ValueError, match="Invalid LoRA name"):
            _validate_name("..sneaky")

    def test_null_byte_raises(self):
        with pytest.raises(ValueError, match="Invalid LoRA name"):
            _validate_name("bad\x00name")

    def test_plain_dots_ok(self):
        # A single dot or dots that aren't ".." should be fine
        assert _validate_name("v1.0") == "v1.0"


# ── import_lora_local ────────────────────────────────────────────────────────


class TestImportLoraLocal:
    def test_copies_with_default_name(self, tmp_path: Path):
        src = tmp_path / "source" / "my_lora.safetensors"
        src.parent.mkdir()
        src.write_bytes(b"fake-tensor-data")
        dest_dir = tmp_path / "loras"

        result = import_lora_local(src, dest_dir)

        assert result == dest_dir / "my_lora.safetensors"
        assert result.read_bytes() == b"fake-tensor-data"

    def test_copies_with_custom_name(self, tmp_path: Path):
        src = tmp_path / "orig.safetensors"
        src.write_bytes(b"data")
        dest_dir = tmp_path / "loras"

        result = import_lora_local(src, dest_dir, name="custom")

        assert result == dest_dir / "custom.safetensors"
        assert result.exists()

    def test_custom_name_extension_stripped(self, tmp_path: Path):
        src = tmp_path / "orig.safetensors"
        src.write_bytes(b"data")
        dest_dir = tmp_path / "loras"

        result = import_lora_local(src, dest_dir, name="custom.safetensors")

        assert result == dest_dir / "custom.safetensors"

    def test_missing_source_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="Source file not found"):
            import_lora_local(tmp_path / "nope.safetensors", tmp_path / "loras")

    def test_non_safetensors_raises(self, tmp_path: Path):
        src = tmp_path / "model.bin"
        src.write_bytes(b"data")

        with pytest.raises(ValueError, match="must be a .safetensors file"):
            import_lora_local(src, tmp_path / "loras")

    def test_dest_exists_raises(self, tmp_path: Path):
        src = tmp_path / "my_lora.safetensors"
        src.write_bytes(b"data")
        dest_dir = tmp_path / "loras"
        dest_dir.mkdir()
        (dest_dir / "my_lora.safetensors").write_bytes(b"existing")

        with pytest.raises(FileExistsError, match="already exists"):
            import_lora_local(src, dest_dir)

    def test_creates_dest_dir(self, tmp_path: Path):
        src = tmp_path / "my.safetensors"
        src.write_bytes(b"data")
        dest_dir = tmp_path / "deep" / "nested" / "loras"

        result = import_lora_local(src, dest_dir)

        assert dest_dir.is_dir()
        assert result.exists()


# ── import_lora_hf ───────────────────────────────────────────────────────────


class TestImportLoraHf:
    @patch("huggingface_hub.hf_hub_download")
    @patch("huggingface_hub.list_repo_files")
    def test_single_safetensors_auto_selects(self, mock_list: MagicMock, mock_dl: MagicMock, tmp_path: Path):
        mock_list.return_value = ["README.md", "model.safetensors"]
        cached = tmp_path / "cached.safetensors"
        cached.write_bytes(b"tensor-data")
        mock_dl.return_value = str(cached)

        dest_dir = tmp_path / "loras"
        result = import_lora_hf("user/repo", dest_dir)

        assert result == dest_dir / "model.safetensors"
        assert result.read_bytes() == b"tensor-data"
        mock_dl.assert_called_once_with(repo_id="user/repo", filename="model.safetensors")

    @patch("huggingface_hub.hf_hub_download")
    @patch("huggingface_hub.list_repo_files")
    def test_multiple_safetensors_without_file_raises(self, mock_list: MagicMock, mock_dl: MagicMock, tmp_path: Path):
        mock_list.return_value = ["a.safetensors", "b.safetensors"]

        with pytest.raises(ValueError, match="Multiple .safetensors"):
            import_lora_hf("user/repo", tmp_path / "loras")

    @patch("huggingface_hub.hf_hub_download")
    @patch("huggingface_hub.list_repo_files")
    def test_no_safetensors_raises(self, mock_list: MagicMock, mock_dl: MagicMock, tmp_path: Path):
        mock_list.return_value = ["README.md", "model.bin"]

        with pytest.raises(ValueError, match="No .safetensors files found"):
            import_lora_hf("user/repo", tmp_path / "loras")

    @patch("huggingface_hub.hf_hub_download")
    @patch("huggingface_hub.list_repo_files")
    def test_explicit_filename(self, mock_list: MagicMock, mock_dl: MagicMock, tmp_path: Path):
        cached = tmp_path / "cached.safetensors"
        cached.write_bytes(b"data")
        mock_dl.return_value = str(cached)

        dest_dir = tmp_path / "loras"
        result = import_lora_hf("user/repo", dest_dir, filename="specific.safetensors")

        assert result == dest_dir / "specific.safetensors"
        mock_list.assert_not_called()  # should skip listing when filename given

    @patch("huggingface_hub.hf_hub_download")
    @patch("huggingface_hub.list_repo_files")
    def test_custom_name(self, mock_list: MagicMock, mock_dl: MagicMock, tmp_path: Path):
        cached = tmp_path / "cached.safetensors"
        cached.write_bytes(b"data")
        mock_dl.return_value = str(cached)

        dest_dir = tmp_path / "loras"
        result = import_lora_hf("user/repo", dest_dir, filename="orig.safetensors", name="renamed")

        assert result == dest_dir / "renamed.safetensors"

    @patch("huggingface_hub.hf_hub_download")
    @patch("huggingface_hub.list_repo_files")
    def test_dest_exists_raises(self, mock_list: MagicMock, mock_dl: MagicMock, tmp_path: Path):
        mock_list.return_value = ["model.safetensors"]
        dest_dir = tmp_path / "loras"
        dest_dir.mkdir(parents=True)
        (dest_dir / "model.safetensors").write_bytes(b"existing")

        with pytest.raises(FileExistsError, match="already exists"):
            import_lora_hf("user/repo", dest_dir)

    def test_explicit_non_safetensors_filename_raises(self, tmp_path: Path):
        """Explicit --file that is not .safetensors should be rejected."""
        with pytest.raises(ValueError, match="must be a .safetensors file"):
            import_lora_hf("user/repo", tmp_path / "loras", filename="model.bin")
