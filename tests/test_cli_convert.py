"""Tests for ziv-model CLI subcommand parsing."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from zvisiongenerator.converters.convert_checkpoint import _build_model_parser


# ── Helper: build the parser without importing torch/safetensors ─────────────


def _build_parser():
    """Build the argparse parser via the extracted _build_model_parser function."""
    return _build_model_parser()


# ── Top-level ────────────────────────────────────────────────────────────────


class TestCliNoArgs:
    def test_no_args_exits_zero(self):
        """ziv-model with no args prints help and exits 0."""
        parser = _build_parser()
        args = parser.parse_args([])
        assert args.command is None


# ── model subcommand ─────────────────────────────────────────────────────────


class TestModelSubcommand:
    def test_basic_model(self):
        parser = _build_parser()
        args = parser.parse_args(["model", "-i", "foo.safetensors"])
        assert args.command == "model"
        assert args.input == "foo.safetensors"

    def test_model_with_custom_name(self):
        parser = _build_parser()
        args = parser.parse_args(["model", "-i", "foo.safetensors", "--name", "mymodel"])
        assert args.name == "mymodel"

    def test_model_defaults(self):
        parser = _build_parser()
        args = parser.parse_args(["model", "-i", "x.safetensors"])
        assert args.model_type == "zimage"
        assert args.base_model == "Tongyi-MAI/Z-Image-Turbo"
        assert args.copy is False

    def test_model_accepts_krea2_turbo_type(self):
        args = _build_parser().parse_args(["model", "-i", "x.safetensors", "--model-type", "krea2-turbo"])
        assert args.model_type == "krea2-turbo"

    def test_model_missing_input_exits(self):
        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["model"])

    def test_model_name_path_traversal_rejected(self, tmp_path):
        """model --name with path traversal characters should be rejected."""
        from zvisiongenerator.converters.convert_checkpoint import _cmd_model

        # Create a dummy input file so _cmd_model gets past the file check
        dummy = tmp_path / "dummy.safetensors"
        dummy.write_bytes(b"fake")

        for bad_name in ["../../tmp/evil", "foo/bar", "..sneaky", "a\\b"]:
            args = SimpleNamespace(
                input=str(dummy),
                name=bad_name,
                model_type="zimage",
                base_model="Tongyi-MAI/Z-Image-Turbo",
                copy=False,
            )
            with pytest.raises(SystemExit):
                _cmd_model(args)


class TestModelQuantize:
    def test_quantize_defaults_to_none(self):
        args = _build_parser().parse_args(["model", "-i", "x.safetensors"])
        assert args.quantize is None

    def test_quantize_accepts_supported_levels(self):
        assert _build_parser().parse_args(["model", "-i", "x.safetensors", "--quantize", "8"]).quantize == 8
        assert _build_parser().parse_args(["model", "-i", "x.safetensors", "--quantize", "4"]).quantize == 4

    def test_quantize_rejects_other_levels(self):
        with pytest.raises(SystemExit):
            _build_parser().parse_args(["model", "-i", "x.safetensors", "--quantize", "6"])

    def test_stored_quant_names_are_reserved(self, tmp_path, monkeypatch):
        from zvisiongenerator.converters.convert_checkpoint import _cmd_model

        monkeypatch.setenv("ZIV_DATA_DIR", str(tmp_path / "data"))
        dummy = tmp_path / "dummy.safetensors"
        dummy.write_bytes(b"fake")
        args = SimpleNamespace(input=str(dummy), name="mine@q8", model_type="zimage", base_model="Tongyi-MAI/Z-Image-Turbo", copy=False, quantize=None)

        with pytest.raises(SystemExit):
            _cmd_model(args)
        assert not (tmp_path / "data" / "models" / "mine@q8").exists()

    def test_store_quantized_copy_saves_next_to_the_model(self, tmp_path, monkeypatch):
        import json

        import zvisiongenerator.backends as backends_module
        from zvisiongenerator.converters.convert_checkpoint import _store_quantized_copy
        from zvisiongenerator.utils.stored_quant import is_current

        model_dir = tmp_path / "models" / "mine"
        (model_dir / "transformer").mkdir(parents=True)
        (model_dir / "model_index.json").write_text(json.dumps({"transformer": ["diffusers", "ZImageTransformer2DModel"]}))
        (model_dir / "transformer" / "w.safetensors").write_bytes(b"x")
        backend = MagicMock()
        backend.stored_quant_format.return_value = "fmt"
        backend.quantizes_from_files.return_value = False
        backend.load_model.return_value = (MagicMock(), MagicMock())
        backend.save_quantized.side_effect = lambda _model, path: (Path(path) / "transformer").mkdir(parents=True)
        monkeypatch.setattr(backends_module, "get_backend", lambda: backend)

        target = _store_quantized_copy(model_dir, 8)

        assert target == tmp_path / "models" / "mine@q8"
        assert backend.load_model.call_args.kwargs["quantize"] == 8
        assert is_current(target, model_dir, 8, "fmt")

    def test_store_quantized_copy_fails_where_unsupported(self, tmp_path, monkeypatch):
        import zvisiongenerator.backends as backends_module
        from zvisiongenerator.converters.convert_checkpoint import _store_quantized_copy

        backend = MagicMock()
        backend.stored_quant_format.return_value = None
        monkeypatch.setattr(backends_module, "get_backend", lambda: backend)

        with pytest.raises(RuntimeError):
            _store_quantized_copy(tmp_path / "mine", 8)
        backend.load_model.assert_not_called()


# ── lora subcommand ──────────────────────────────────────────────────────────


class TestLoraSubcommand:
    def test_lora_local(self):
        parser = _build_parser()
        args = parser.parse_args(["lora", "-i", "foo.safetensors"])
        assert args.command == "lora"
        assert args.input == "foo.safetensors"

    def test_lora_hf(self):
        parser = _build_parser()
        args = parser.parse_args(["lora", "--hf", "user/repo"])
        assert args.command == "lora"
        assert args.hf == "user/repo"

    def test_lora_hf_with_file_and_name(self):
        parser = _build_parser()
        args = parser.parse_args(
            [
                "lora",
                "--hf",
                "user/repo",
                "--file",
                "model.safetensors",
                "--name",
                "custom",
            ]
        )
        assert args.hf == "user/repo"
        assert args.file == "model.safetensors"
        assert args.name == "custom"

    def test_lora_no_source_exits(self):
        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["lora"])

    def test_lora_both_sources_exits(self):
        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["lora", "-i", "foo", "--hf", "bar"])


# ── list subcommand ──────────────────────────────────────────────────────────


class TestListSubcommand:
    def test_list_defaults(self):
        parser = _build_parser()
        args = parser.parse_args(["list"])
        assert args.command == "list"
        assert args.models is False
        assert args.loras is False

    def test_list_models_flag(self):
        parser = _build_parser()
        args = parser.parse_args(["list", "--models"])
        assert args.models is True
        assert args.loras is False

    def test_list_loras_flag(self):
        parser = _build_parser()
        args = parser.parse_args(["list", "--loras"])
        assert args.loras is True
        assert args.models is False

    def test_list_both_flags(self):
        parser = _build_parser()
        args = parser.parse_args(["list", "--models", "--loras"])
        assert args.models is True
        assert args.loras is True
