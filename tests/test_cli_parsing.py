"""Tests for CLI parsing — parse_lora_arg and argument validation."""

from __future__ import annotations

import re
import warnings

import pytest
from unittest.mock import MagicMock, patch

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

    def _run_main_with_overrides(self, argv: list[str], **overrides):
        mocks = {k.split(".")[-1]: v for k, v in self._MAIN_MOCKS.items()}
        mocks.update(overrides)
        with patch("sys.argv", ["ziv-image"] + argv):
            with patch.multiple("zvisiongenerator.image_cli", **mocks):
                main()

    @staticmethod
    def _flag_misuse_warning_messages(caught_warnings: list[warnings.WarningMessage]) -> list[str]:
        return [str(warning.message) for warning in caught_warnings if "--first-sigma only affects Ideogram 4" in str(warning.message) or "passed as a literal prompt" in str(warning.message)]

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

    @pytest.mark.parametrize("width", [240, 3072])
    def test_ideogram4_dimension_bounds_error_on_in_grid_out_of_range_widths(self, width, capsys):
        mock_backend = MagicMock()
        mock_backend.name = "mflux"
        mock_backend.load_model = MagicMock(return_value=(MagicMock(), MagicMock(family="ideogram4")))

        with pytest.raises(SystemExit, match="2"):
            self._run_main_with_overrides(
                ["--prompt", "ok", "--width", str(width), "--height", "1024", "-m", "fake"],
                detect_image_model=lambda _path: MagicMock(family="ideogram4", size=None),
                get_backend=lambda: mock_backend,
                resolve_defaults=lambda *a, **kw: {"steps": 20, "guidance": 7.0, "scheduler": None},
            )

        assert "between 256 and 2048" in capsys.readouterr().err
        mock_backend.load_model.assert_not_called()

    def test_ideogram4_dimension_bounds_accept_1024_square(self):
        mock_backend = MagicMock()
        mock_backend.name = "mflux"
        mock_backend.load_model = MagicMock(return_value=(MagicMock(), MagicMock(family="ideogram4")))

        self._run_main_with_overrides(
            ["--prompt", "ok", "--width", "1024", "--height", "1024", "-m", "fake"],
            detect_image_model=lambda _path: MagicMock(family="ideogram4", size=None),
            get_backend=lambda: mock_backend,
            resolve_defaults=lambda *a, **kw: {"steps": 20, "guidance": 7.0, "scheduler": None},
        )

        mock_backend.load_model.assert_called_once()

    def test_default_runs_is_1(self):
        parser = _build_parser()
        args = parser.parse_args(["-m", "fake"])
        assert args.runs == 1

    def test_default_size_is_none(self):
        parser = _build_parser()
        args = parser.parse_args(["-m", "fake"])
        assert args.size is None

    def test_first_sigma_parses_as_float(self):
        parser = _build_parser()
        args = parser.parse_args(["--first-sigma", "1.005", "-m", "fake"])

        assert args.first_sigma == 1.005

    def test_first_sigma_default_is_none(self):
        parser = _build_parser()
        args = parser.parse_args(["-m", "fake"])

        assert args.first_sigma is None

    def test_first_sigma_help_mentions_valid_range(self):
        parser = _build_parser()

        first_sigma_action = next(action for action in parser._actions if "--first-sigma" in action.option_strings)

        assert "(0.0, 2.0]" in first_sigma_action.help

    def test_first_sigma_invalid_float_exits_before_load_model(self):
        mock_backend = MagicMock()
        mock_backend.name = "mflux"
        mock_backend.load_model = MagicMock(return_value=(MagicMock(), MagicMock(family="ideogram4")))

        with pytest.raises(SystemExit, match="2"):
            self._run_main_with_overrides(
                ["--first-sigma", "abc", "-m", "fake"],
                get_backend=lambda: mock_backend,
            )

        mock_backend.load_model.assert_not_called()

    @pytest.mark.parametrize("value", ["-1", "0", "2.5"])
    def test_first_sigma_out_of_band_exits_before_load_model(self, value):
        mock_backend = MagicMock()
        mock_backend.name = "mflux"
        mock_backend.load_model = MagicMock(return_value=(MagicMock(), MagicMock(family="ideogram4")))

        with pytest.raises(SystemExit, match="2"):
            self._run_main_with_overrides(
                ["--first-sigma", value, "--prompt", "ok", "-m", "fake"],
                get_backend=lambda: mock_backend,
            )

        mock_backend.load_model.assert_not_called()

    @pytest.mark.parametrize("value", [1.006, 2.0])
    def test_first_sigma_accepts_in_band_values(self, value):
        mock_backend = MagicMock()
        mock_backend.name = "mflux"
        mock_backend.load_model = MagicMock(return_value=(MagicMock(), MagicMock(family="ideogram4")))
        recorded = {}

        def _capture_run_batch(_backend, _model, _prompts_data, _config, args, **_kwargs):
            recorded["first_sigma"] = args.first_sigma

        self._run_main_with_overrides(
            ["--first-sigma", str(value), "--prompt", "ok", "-m", "fake"],
            detect_image_model=lambda _path: MagicMock(family="ideogram4", size=None),
            get_backend=lambda: mock_backend,
            run_batch=_capture_run_batch,
            resolve_defaults=lambda *a, **kw: {"steps": 20, "guidance": 7.0, "scheduler": None},
        )

        assert recorded["first_sigma"] == value
        mock_backend.load_model.assert_called_once()

    def test_non_ideogram_first_sigma_warns_before_load_model(self):
        mock_backend = MagicMock()
        mock_backend.name = "mflux"
        mock_backend.load_model = MagicMock(side_effect=AssertionError("load_model reached after warning"))

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")

            with pytest.raises(AssertionError, match="load_model reached after warning"):
                self._run_main_with_overrides(
                    ["--first-sigma", "1.006", "--prompt", "ok", "-m", "fake"],
                    detect_image_model=lambda _path: MagicMock(family="zimage", size=None),
                    get_backend=lambda: mock_backend,
                )

        warning_messages = self._flag_misuse_warning_messages(caught)

        assert any("--first-sigma only affects Ideogram 4" in message and "zimage" in message for message in warning_messages)

    def test_non_ideogram_json_prompt_warns_before_load_model(self):
        mock_backend = MagicMock()
        mock_backend.name = "mflux"
        mock_backend.load_model = MagicMock(side_effect=AssertionError("load_model reached after warning"))

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")

            with pytest.raises(AssertionError, match="load_model reached after warning"):
                self._run_main_with_overrides(
                    ["--json-prompt", '{"a": 1}', "-m", "fake"],
                    detect_image_model=lambda _path: MagicMock(family="zimage", size=None),
                    get_backend=lambda: mock_backend,
                )

        warning_messages = self._flag_misuse_warning_messages(caught)

        assert any("passed as a literal prompt" in message and "skips {a|b|c}" in message and "zimage" in message for message in warning_messages)

    @pytest.mark.parametrize(
        "argv",
        [
            ["--first-sigma", "1.006", "--prompt", "ok", "-m", "fake"],
            ["--json-prompt", '{"a": 1}', "-m", "fake"],
        ],
    )
    def test_ideogram4_flag_usage_does_not_emit_non_ideogram_warnings(self, argv):
        mock_backend = MagicMock()
        mock_backend.name = "mflux"
        mock_backend.load_model = MagicMock(return_value=(MagicMock(), MagicMock(family="ideogram4")))

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self._run_main_with_overrides(
                argv,
                detect_image_model=lambda _path: MagicMock(family="ideogram4", size=None),
                get_backend=lambda: mock_backend,
                resolve_defaults=lambda *a, **kw: {"steps": 20, "guidance": 7.0, "scheduler": None},
            )

        warning_messages = self._flag_misuse_warning_messages(caught)

        assert not any("--first-sigma only affects Ideogram 4" in message for message in warning_messages)
        assert not any("passed as a literal prompt" in message for message in warning_messages)

    def test_non_ideogram_without_first_sigma_or_json_prompt_does_not_emit_flag_warnings(self):
        mock_backend = MagicMock()
        mock_backend.name = "mflux"
        mock_backend.load_model = MagicMock(return_value=(MagicMock(), MagicMock(family="zimage")))

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self._run_main_with_overrides(
                ["--prompt", "ok", "-m", "fake"],
                detect_image_model=lambda _path: MagicMock(family="zimage", size=None),
                get_backend=lambda: mock_backend,
            )

        warning_messages = self._flag_misuse_warning_messages(caught)

        assert not any("--first-sigma only affects Ideogram 4" in message for message in warning_messages)
        assert not any("passed as a literal prompt" in message for message in warning_messages)

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

    def test_json_prompt_accepts_inline_json_object_value(self):
        recorded = {}
        mock_backend = MagicMock()
        mock_backend.name = "mflux"
        mock_backend.load_model = MagicMock(return_value=(MagicMock(), MagicMock(family="ideogram4")))

        def _capture_run_batch(_backend, _model, prompts_data, _config, args, **_kwargs):
            recorded["prompt"] = prompts_data["prompt"][0][0]
            recorded["json_prompt"] = args.json_prompt
            recorded["json_prompt_enabled"] = args.json_prompt_enabled

        self._run_main_with_overrides(
            ["--json-prompt", '{"a": 1}', "-m", "fake"],
            detect_image_model=lambda _path: MagicMock(family="ideogram4", size=None),
            get_backend=lambda: mock_backend,
            run_batch=_capture_run_batch,
            resolve_defaults=lambda *a, **kw: {"steps": 20, "guidance": 7.0, "scheduler": None},
        )

        assert recorded == {
            "prompt": '{"a": 1}',
            "json_prompt": '{"a": 1}',
            "json_prompt_enabled": True,
        }

    def test_json_prompt_rejects_prompt_argument_combination_before_load_model(self, capsys):
        mock_backend = MagicMock()
        mock_backend.name = "mflux"
        mock_backend.load_model = MagicMock(return_value=(MagicMock(), MagicMock(family="zimage")))

        with pytest.raises(SystemExit, match="2"):
            self._run_main_with_overrides(
                ["--prompt", "x", "--json-prompt", '{"a": 1}', "-m", "fake"],
                get_backend=lambda: mock_backend,
            )

        assert "not allowed with argument" in capsys.readouterr().err
        mock_backend.load_model.assert_not_called()

    def test_json_prompt_rejects_empty_value_before_load_model(self, capsys):
        mock_backend = MagicMock()
        mock_backend.name = "mflux"
        mock_backend.load_model = MagicMock(return_value=(MagicMock(), MagicMock(family="zimage")))

        with pytest.raises(SystemExit, match="2"):
            self._run_main_with_overrides(
                ["--json-prompt", "   ", "-m", "fake"],
                get_backend=lambda: mock_backend,
            )

        assert "must not be empty" in capsys.readouterr().err
        mock_backend.load_model.assert_not_called()

    def test_json_prompt_rejects_invalid_json_before_load_model(self, capsys):
        mock_backend = MagicMock()
        mock_backend.name = "mflux"
        mock_backend.load_model = MagicMock(return_value=(MagicMock(), MagicMock(family="zimage")))

        with pytest.raises(SystemExit, match="2"):
            self._run_main_with_overrides(
                ["--json-prompt", "a red car", "-m", "fake"],
                get_backend=lambda: mock_backend,
            )

        assert "must be a JSON object" in capsys.readouterr().err
        mock_backend.load_model.assert_not_called()

    @pytest.mark.parametrize(
        ("json_value", "type_name"),
        [
            ("[1,2,3]", "list"),
            ('"hi"', "str"),
        ],
    )
    def test_json_prompt_rejects_non_object_json_before_load_model(self, json_value, type_name, capsys):
        mock_backend = MagicMock()
        mock_backend.name = "mflux"
        mock_backend.load_model = MagicMock(return_value=(MagicMock(), MagicMock(family="zimage")))

        with pytest.raises(SystemExit, match="2"):
            self._run_main_with_overrides(
                ["--json-prompt", json_value, "-m", "fake"],
                get_backend=lambda: mock_backend,
            )

        err = capsys.readouterr().err
        assert "must be a JSON object" in err
        assert type_name in err
        mock_backend.load_model.assert_not_called()

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

    def test_upscale_steps_defaults_from_resolved_model_defaults(self):
        """--upscale uses resolve_defaults()['upscale_steps'] when --upscale-steps is omitted."""
        recorded = {}

        def _capture_run_batch(_backend, _model, _prompts, _config, args, **kw):
            recorded["upscale_steps"] = args.upscale_steps

        mocks = {k.split(".")[-1]: v for k, v in self._MAIN_MOCKS.items() if k.split(".")[-1] not in ("run_batch", "resolve_defaults")}
        mocks["run_batch"] = _capture_run_batch
        mocks["resolve_defaults"] = lambda *a, **kw: {"steps": 10, "guidance": 0.5, "scheduler": None, "upscale_steps": 12}

        with patch("sys.argv", ["ziv-image", "-m", "fake", "--upscale", "2"]):
            with patch.multiple("zvisiongenerator.image_cli", **mocks):
                main()

        assert recorded["upscale_steps"] == 12

    def test_explicit_upscale_steps_wins_over_model_default(self):
        """Explicit --upscale-steps must not be overwritten by resolve_defaults upscale default."""
        recorded = {}

        def _capture_run_batch(_backend, _model, _prompts, _config, args, **kw):
            recorded["upscale_steps"] = args.upscale_steps

        mocks = {k.split(".")[-1]: v for k, v in self._MAIN_MOCKS.items() if k.split(".")[-1] not in ("run_batch", "resolve_defaults")}
        mocks["run_batch"] = _capture_run_batch
        mocks["resolve_defaults"] = lambda *a, **kw: {"steps": 10, "guidance": 0.5, "scheduler": None, "upscale_steps": 12}

        with patch("sys.argv", ["ziv-image", "-m", "fake", "--upscale", "2", "--upscale-steps", "21"]):
            with patch.multiple("zvisiongenerator.image_cli", **mocks):
                main()

        assert recorded["upscale_steps"] == 21

    def test_ideogram4_preset_size_guard_errors_before_load_model(self, capsys):
        mock_backend = MagicMock()
        mock_backend.name = "mflux"
        mock_backend.load_model = MagicMock(return_value=(MagicMock(), MagicMock(family="ideogram4")))

        def _load_config():
            return {
                "sizes": {
                    "16:9": {"xl": {"width": 2112, "height": 1184}},
                    "2:3": {"m": {"width": 832, "height": 1216}},
                },
                "model_presets": {},
                "schedulers": {},
                "generation": {"default_ratio": "2:3", "default_size": "m"},
            }

        with pytest.raises(SystemExit, match="2"):
            self._run_main_with_overrides(
                ["--prompt", "ok", "--size", "xl", "--ratio", "16:9", "-m", "ideo"],
                load_config=_load_config,
                detect_image_model=lambda _path: MagicMock(family="ideogram4", size=None),
                get_backend=lambda: mock_backend,
                resolve_defaults=lambda *a, **kw: {"steps": 20, "guidance": 7.0, "scheduler": None},
            )

        assert "2048" in capsys.readouterr().err
        mock_backend.load_model.assert_not_called()

    def test_ideogram4_img2img_early_guard_errors_before_load_model(self, tmp_path, capsys):
        image_path = tmp_path / "reference.png"
        image_path.write_bytes(b"fake")
        mock_backend = MagicMock()
        mock_backend.name = "mflux"
        mock_backend.load_model = MagicMock(return_value=(MagicMock(), MagicMock(family="ideogram4")))

        with pytest.raises(SystemExit, match="2"):
            self._run_main_with_overrides(
                ["--prompt", "ok", "--image", str(image_path), "-m", "ideo"],
                detect_image_model=lambda _path: MagicMock(family="ideogram4", size=None),
                get_backend=lambda: mock_backend,
                resolve_defaults=lambda *a, **kw: {"steps": 20, "guidance": 7.0, "scheduler": None},
            )

        assert "img2img is not supported" in capsys.readouterr().err.lower()
        mock_backend.load_model.assert_not_called()

    def test_ideogram4_upscale_early_guard_errors_before_load_model(self, capsys):
        mock_backend = MagicMock()
        mock_backend.name = "mflux"
        mock_backend.load_model = MagicMock(return_value=(MagicMock(), MagicMock(family="ideogram4")))

        with pytest.raises(SystemExit, match="2"):
            self._run_main_with_overrides(
                ["--prompt", "ok", "--upscale", "2", "-m", "ideo"],
                detect_image_model=lambda _path: MagicMock(family="ideogram4", size=None),
                get_backend=lambda: mock_backend,
                resolve_defaults=lambda *a, **kw: {"steps": 20, "guidance": 7.0, "scheduler": None},
            )

        assert re.search(r"upscal.*not supported", capsys.readouterr().err, re.IGNORECASE)
        mock_backend.load_model.assert_not_called()


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
