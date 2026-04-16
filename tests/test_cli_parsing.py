"""Tests for CLI parsing — parse_lora_arg and argument validation."""

from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock

from zvisiongenerator.image_cli import _build_parser, main
from zvisiongenerator.utils.lora import parse_lora_arg


# ── parse_lora_arg ──────────────────────────────────────────────────────────


class TestParseLoraArg:
    def test_single_name_default_weight(self):
        result = parse_lora_arg("style")
        assert result == [("style", 1.0)]

    def test_name_with_weight(self):
        result = parse_lora_arg("style:0.8")
        assert result == [("style", 0.8)]

    def test_comma_separated_multiple(self):
        result = parse_lora_arg("style:0.8,detail:0.5")
        assert result == [("style", 0.8), ("detail", 0.5)]

    def test_mixed_with_and_without_weight(self):
        result = parse_lora_arg("style:0.7,detail")
        assert result == [("style", 0.7), ("detail", 1.0)]

    def test_invalid_weight_treated_as_name(self):
        result = parse_lora_arg("style:notanumber")
        assert result == [("style:notanumber", 1.0)]

    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="Empty LoRA entry"):
            parse_lora_arg("")

    def test_empty_name_raises(self):
        with pytest.raises(ValueError, match="Empty LoRA name"):
            parse_lora_arg(":0.5")

    def test_whitespace_in_entries_stripped(self):
        result = parse_lora_arg(" style : 0.8 , detail : 0.5 ")
        assert result == [("style", 0.8), ("detail", 0.5)]

    def test_name_with_path_separator(self):
        result = parse_lora_arg("models/style.safetensors:0.9")
        assert result == [("models/style.safetensors", 0.9)]

    def test_zero_weight(self):
        result = parse_lora_arg("style:0.0")
        assert result == [("style", 0.0)]

    def test_negative_weight(self):
        result = parse_lora_arg("style:-0.5")
        assert result == [("style", -0.5)]


# ── CLI validation via parser ───────────────────────────────────────────────


class TestCLIValidation:
    """Test CLI argument validation by calling main() with mocked sys.argv.

    main() calls parser.error() for invalid args, which raises SystemExit(2).
    Heavy dependencies (load_model, run_batch, etc.) are mocked out.
    """

    _MAIN_MOCKS = {
        "zvisiongenerator.image_cli.load_config": lambda: {
            "sizes": {"2:3": {"m": {"width": 832, "height": 1216}}},
            "model_presets": {},
            "schedulers": {"beta": {}},
        },
        "zvisiongenerator.image_cli.detect_image_model": lambda _: MagicMock(family="zimage", size=None),
        "zvisiongenerator.image_cli.resolve_model_path": lambda p, **kw: p,
        "zvisiongenerator.image_cli.resolve_defaults": lambda *a, **kw: {"steps": 10, "guidance": 0.5, "scheduler": None},
        "zvisiongenerator.image_cli.validate_scheduler": lambda *a: None,
        "zvisiongenerator.image_cli.load_prompts_file": lambda _: {"set": [("a cat", None)]},
        "zvisiongenerator.image_cli.get_backend": lambda: MagicMock(name="mflux", load_model=MagicMock(return_value=(MagicMock(), MagicMock(family="zimage")))),
        "zvisiongenerator.image_cli.run_batch": lambda *a, **kw: None,
    }

    def _run_main(self, argv: list[str]):
        """Call main() with mocked argv and dependencies."""
        with patch("sys.argv", ["ziv-image"] + argv):
            with patch.multiple("zvisiongenerator.image_cli", **{k.split(".")[-1]: v for k, v in self._MAIN_MOCKS.items()}):
                main()

    def test_runs_zero_exits(self):
        with pytest.raises(SystemExit, match="2"):
            self._run_main(["--runs", "0", "-m", "fake"])

    def test_upscale_3_exits(self):
        with pytest.raises(SystemExit, match="2"):
            self._run_main(["--upscale", "3", "-m", "fake"])

    def test_upscale_guidance_negative_rejected(self):
        """--upscale-guidance with negative value must be rejected by the CLI."""
        with pytest.raises(SystemExit):
            self._run_main(["--upscale-guidance", "-1.0", "-m", "fake"])

    def test_negative_width_exits(self):
        with pytest.raises(SystemExit, match="2"):
            self._run_main(["--width", "-100", "-m", "fake"])

    def test_width_not_multiple_of_16_exits(self):
        with pytest.raises(SystemExit, match="2"):
            self._run_main(["--width", "500", "-m", "fake"])

    def test_height_not_multiple_of_16_exits(self):
        with pytest.raises(SystemExit, match="2"):
            self._run_main(["--height", "500", "-m", "fake"])

    def test_default_runs_is_1(self):
        parser = _build_parser()
        args = parser.parse_args(["-m", "fake"])
        assert args.runs == 1

    def test_default_size_is_none(self):
        parser = _build_parser()
        args = parser.parse_args(["-m", "fake"])
        assert args.size is None

    # ── Upscale size drift validation ───────────────────────────────────

    def test_upscale_incompatible_width_exits(self):
        """width=528 with 4x upscale drifts to 576 — should be rejected."""
        with pytest.raises(SystemExit, match="2"):
            self._run_main(["--width", "528", "--upscale", "4", "-m", "fake"])

    def test_upscale_incompatible_height_exits(self):
        """height=528 with 4x upscale drifts to 576 — should be rejected."""
        with pytest.raises(SystemExit, match="2"):
            self._run_main(["--height", "528", "--upscale", "4", "-m", "fake"])

    def test_upscale_compatible_width_ok(self):
        """width=512 with 4x upscale: 512//4=128, round16(128)=128, 128*4=512 — ok."""
        self._run_main(["--width", "512", "--upscale", "4", "-m", "fake"])

    def test_upscale_without_explicit_dims_ok(self):
        """Upscale without explicit width/height should not error."""
        self._run_main(["--upscale", "2", "-m", "fake"])

    # ── Steps and guidance validation ──

    def test_steps_zero_exits(self):
        with pytest.raises(SystemExit, match="2"):
            self._run_main(["--steps", "0", "-m", "fake"])

    def test_steps_negative_exits(self):
        with pytest.raises(SystemExit, match="2"):
            self._run_main(["--steps", "-1", "-m", "fake"])

    def test_upscale_steps_zero_exits(self):
        with pytest.raises(SystemExit, match="2"):
            self._run_main(["--upscale-steps", "0", "--upscale", "2", "-m", "fake"])

    def test_guidance_negative_exits(self):
        with pytest.raises(SystemExit, match="2"):
            self._run_main(["--guidance", "-1.0", "-m", "fake"])

    def test_guidance_zero_ok(self):
        self._run_main(["--guidance", "0.0", "-m", "fake"])

    # ── Backend and model error handling ──

    def test_get_backend_runtime_error_exits(self):
        with pytest.raises(SystemExit, match="2"):
            with patch("sys.argv", ["ziv-image", "-m", "fake"]):
                with patch.multiple("zvisiongenerator.image_cli", **{k.split(".")[-1]: v for k, v in self._MAIN_MOCKS.items() if k.split(".")[-1] != "get_backend"}):
                    with patch("zvisiongenerator.image_cli.get_backend", side_effect=RuntimeError("unsupported")):
                        main()

    def test_load_model_os_error_exits(self):
        def _bad_load(*a, **kw):
            raise OSError("model file corrupt")

        mock_backend = MagicMock()
        mock_backend.name = "mflux"
        mock_backend.load_model = _bad_load
        with pytest.raises(SystemExit, match="2"):
            with patch("sys.argv", ["ziv-image", "-m", "fake"]):
                with patch.multiple("zvisiongenerator.image_cli", **{k.split(".")[-1]: v for k, v in self._MAIN_MOCKS.items() if k.split(".")[-1] != "get_backend"}):
                    with patch("zvisiongenerator.image_cli.get_backend", return_value=mock_backend):
                        main()

    # ── Empty --prompt validation ──────────────────────────────────

    def test_empty_prompt_string_exits(self):
        """--prompt '' should be rejected."""
        with pytest.raises(SystemExit, match="2"):
            self._run_main(["--prompt", "", "-m", "fake"])

    def test_whitespace_only_prompt_exits(self):
        """--prompt '   ' should be rejected."""
        with pytest.raises(SystemExit, match="2"):
            self._run_main(["--prompt", "   ", "-m", "fake"])

    # ── --size defaults to config value ─────────────────────────────

    def test_size_defaults_to_config_default_size(self):
        """--size should default to generation.default_size from config."""
        recorded = {}

        def _capture_run_batch(_backend, _model, _prompts, _config, args, **kw):
            recorded["size"] = args.size
            recorded["ratio"] = args.ratio

        mocks = {k.split(".")[-1]: v for k, v in self._MAIN_MOCKS.items() if k.split(".")[-1] != "run_batch"}
        mocks["run_batch"] = _capture_run_batch
        # Config with generation.default_size = "s" and both sizes defined (nested under ratio)
        mocks["load_config"] = lambda: {
            "sizes": {"2:3": {"s": {"width": 576, "height": 864}, "m": {"width": 832, "height": 1216}}},
            "model_presets": {},
            "schedulers": {},
            "generation": {"default_size": "s", "default_ratio": "2:3"},
        }

        with patch("sys.argv", ["ziv-image", "-m", "fake"]):
            with patch.multiple("zvisiongenerator.image_cli", **mocks):
                main()
        assert recorded["size"] == "s"


# ── Post-processing flag parsing ────────────────────────────────────────────


class TestPostProcessingFlags:
    """Test --sharpen/--no-sharpen, --contrast/--no-contrast, --saturation/--no-saturation."""

    def _parse(self, argv: list[str]):
        parser = _build_parser()
        return parser.parse_args(["-m", "fake"] + argv)

    # -- defaults --

    def test_default_sharpen_is_true(self):
        args = self._parse([])
        assert args.sharpen is True

    def test_default_contrast_is_false(self):
        args = self._parse([])
        assert args.contrast is False

    def test_default_saturation_is_false(self):
        args = self._parse([])
        assert args.saturation is False

    # -- bare flags --

    def test_bare_sharpen(self):
        args = self._parse(["--sharpen"])
        assert args.sharpen is True

    def test_bare_contrast(self):
        args = self._parse(["--contrast"])
        assert args.contrast is True

    def test_bare_saturation(self):
        args = self._parse(["--saturation"])
        assert args.saturation is True

    # -- with amount --

    def test_sharpen_with_amount(self):
        args = self._parse(["--sharpen", "0.6"])
        assert args.sharpen == 0.6

    def test_contrast_with_amount(self):
        args = self._parse(["--contrast", "1.3"])
        assert args.contrast == 1.3

    def test_saturation_with_amount(self):
        args = self._parse(["--saturation", "1.2"])
        assert args.saturation == 1.2

    # -- negation --

    def test_no_sharpen(self):
        args = self._parse(["--no-sharpen"])
        assert args.sharpen is False

    def test_no_contrast(self):
        args = self._parse(["--no-contrast"])
        assert args.contrast is False

    def test_no_saturation(self):
        args = self._parse(["--no-saturation"])
        assert args.saturation is False

    # -- 0.0 edge case --

    def test_sharpen_zero_is_not_false(self):
        args = self._parse(["--sharpen", "0.0"])
        assert args.sharpen == 0.0
        assert args.sharpen is not False

    def test_contrast_zero_is_accepted(self):
        args = self._parse(["--contrast", "0.0"])
        assert args.contrast == 0.0
        assert args.contrast is not False

    def test_saturation_zero_is_accepted(self):
        args = self._parse(["--saturation", "0.0"])
        assert args.saturation == 0.0
        assert args.saturation is not False

    # -- negative amounts rejected --

    def test_negative_sharpen_rejected(self):
        with pytest.raises(SystemExit):
            with patch("sys.argv", ["ziv-image", "-m", "fake", "--sharpen", "-1"]):
                with patch.multiple("zvisiongenerator.image_cli", **{k.split(".")[-1]: v for k, v in TestCLIValidation._MAIN_MOCKS.items()}):
                    main()

    def test_negative_contrast_rejected(self):
        with pytest.raises(SystemExit):
            with patch("sys.argv", ["ziv-image", "-m", "fake", "--contrast", "-0.5"]):
                with patch.multiple("zvisiongenerator.image_cli", **{k.split(".")[-1]: v for k, v in TestCLIValidation._MAIN_MOCKS.items()}):
                    main()

    def test_negative_saturation_rejected(self):
        with pytest.raises(SystemExit):
            with patch("sys.argv", ["ziv-image", "-m", "fake", "--saturation", "-2"]):
                with patch.multiple("zvisiongenerator.image_cli", **{k.split(".")[-1]: v for k, v in TestCLIValidation._MAIN_MOCKS.items()}):
                    main()
