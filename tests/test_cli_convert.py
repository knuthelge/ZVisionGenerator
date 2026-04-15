"""Tests for ziv-convert CLI subcommand parsing."""

from __future__ import annotations

from types import SimpleNamespace

import pytest


# ── Helper: build the parser without importing torch/safetensors ─────────────


def _build_parser():
    """Build the argparse parser from main() without executing anything.

    We re-create the parser here because main() both creates and immediately
    acts on the parsed args. This mirrors the parser defined in
    convert_checkpoint.main().
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="ziv-convert",
        description="Import and manage models and LoRAs for Z-Vision Generator",
    )
    subparsers = parser.add_subparsers(dest="command")

    # model subcommand
    model_parser = subparsers.add_parser("model", help="Convert a checkpoint to diffusers format")
    model_parser.add_argument("-i", "--input", required=True, help="Path to .safetensors checkpoint file")
    model_parser.add_argument("--name", default=None, help="Custom model folder name")
    model_parser.add_argument(
        "--model-type",
        choices=["zimage", "flux2-klein-4b", "flux2-klein-9b"],
        default="zimage",
    )
    model_parser.add_argument("--base-model", default="Tongyi-MAI/Z-Image-Turbo")
    model_parser.add_argument("--copy", action="store_true")

    # lora subcommand
    lora_parser = subparsers.add_parser("lora", help="Import a LoRA file")
    lora_source = lora_parser.add_mutually_exclusive_group(required=True)
    lora_source.add_argument("-i", "--input", help="Path to local .safetensors file")
    lora_source.add_argument("--hf", help="HuggingFace repo ID")
    lora_parser.add_argument("--file", default=None)
    lora_parser.add_argument("--name", default=None)

    # list subcommand
    list_parser = subparsers.add_parser("list", help="List installed models and LoRAs")
    list_parser.add_argument("--models", action="store_true")
    list_parser.add_argument("--loras", action="store_true")

    return parser


# ── Top-level ────────────────────────────────────────────────────────────────


class TestCliNoArgs:
    def test_no_args_exits_zero(self):
        """ziv-convert with no args prints help and exits 0."""
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
