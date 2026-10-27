"""Tests for ziv-model CLI subcommand parsing."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from zvisiongenerator.converters.convert_checkpoint import _build_model_parser


# ── Parser: argv -> attributes, or SystemExit ────────────────────────────────

PARSER_CASES = [
    pytest.param([], {"command": None}, id="no-args"),
    pytest.param(
        ["model", "-i", "x.safetensors"],
        {"command": "model", "input": "x.safetensors", "model_type": "zimage", "base_model": "Tongyi-MAI/Z-Image-Turbo", "copy": False, "quantize": None},
        id="model-defaults",
    ),
    pytest.param(
        ["model", "-i", "x.safetensors", "--name", "mymodel", "--model-type", "krea2-turbo", "--quantize", "8"],
        {"name": "mymodel", "model_type": "krea2-turbo", "quantize": 8},
        id="model-options",
    ),
    pytest.param(["model"], SystemExit, id="model-needs-input"),
    pytest.param(["model", "-i", "x.safetensors", "--quantize", "6"], SystemExit, id="model-unsupported-quantize"),
    pytest.param(["lora", "-i", "foo.safetensors"], {"command": "lora", "input": "foo.safetensors"}, id="lora-local"),
    pytest.param(["lora", "--hf", "user/repo", "--file", "model.safetensors", "--name", "custom"], {"hf": "user/repo", "file": "model.safetensors", "name": "custom"}, id="lora-hub"),
    pytest.param(["lora"], SystemExit, id="lora-needs-a-source"),
    pytest.param(["lora", "-i", "foo", "--hf", "bar"], SystemExit, id="lora-takes-one-source"),
    pytest.param(["list"], {"command": "list", "models": False, "loras": False}, id="list-defaults"),
    pytest.param(["list", "--models", "--loras"], {"models": True, "loras": True}, id="list-both"),
]


@pytest.mark.parametrize(("argv", "expected"), PARSER_CASES)
def test_the_parser_reads_the_documented_flags(argv, expected):
    parser = _build_model_parser()
    if expected is SystemExit:
        with pytest.raises(SystemExit):
            parser.parse_args(argv)
        return
    args = parser.parse_args(argv)
    assert {key: getattr(args, key) for key in expected} == expected


class TestModelSubcommand:
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
    @pytest.mark.parametrize("name", ["mine@q8", "mine@mlx-te4"])
    def test_stored_quant_names_are_reserved(self, tmp_path, monkeypatch, name):
        from zvisiongenerator.converters.convert_checkpoint import _cmd_model

        monkeypatch.setenv("ZIV_DATA_DIR", str(tmp_path / "data"))
        dummy = tmp_path / "dummy.safetensors"
        dummy.write_bytes(b"fake")
        args = SimpleNamespace(input=str(dummy), name=name, model_type="zimage", base_model="Tongyi-MAI/Z-Image-Turbo", copy=False, quantize=None)

        with pytest.raises(SystemExit):
            _cmd_model(args)
        assert not (tmp_path / "data" / "models" / name).exists()

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
