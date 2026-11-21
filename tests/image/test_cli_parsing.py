"""Tests for CLI argument validation."""

from __future__ import annotations

import argparse
import re
import warnings

import pytest
from unittest.mock import MagicMock, patch

from tests._factories import _patch_image_main_dependencies
from PIL import Image

from zvisiongenerator.utils.memory_verdict import MemoryJob
from zvisiongenerator.image_cli import _build_parser, _warn_if_memory_tight, main
from zvisiongenerator.utils.memory_verdict import MemoryVerdict, MemoryWeights
from zvisiongenerator.utils.model_memory import GIB, MemoryBudget
from zvisiongenerator.utils.matte import matte_pins
from zvisiongenerator.utils.prompts import PromptFileInspection

BIREFNET, VITMATTE = matte_pins()

_IDEOGRAM = {
    "detect_image_model": lambda _path: MagicMock(family="ideogram4", size=None),
    "resolve_defaults": lambda *a, **kw: {"steps": 20, "guidance": 7.0, "scheduler": None},
}
"""``main()`` overrides that make the detected model Ideogram 4 with its preset steps and guidance."""


def _backend(family: str = "zimage") -> MagicMock:
    """Return a mock mflux backend whose ``load_model`` returns a model of *family*."""
    backend = MagicMock()
    backend.name = "mflux"
    backend.load_model = MagicMock(return_value=(MagicMock(), MagicMock(family=family)))
    return backend


# ── CLI validation via parser ───────────────────────────────────────────────


class TestCLIValidation:
    """Test CLI argument validation by calling main() with mocked sys.argv.

    main() calls parser.error() for invalid args, which raises SystemExit(2).
    Heavy dependencies (load_model, run_batch, etc.) are mocked out.
    """

    _MAIN_MOCKS = {
        "zvisiongenerator.image_cli.load_config": lambda: {
            "sizes": {"2:3": {"m": {"width": 832, "height": 1216}}},
        },
        "zvisiongenerator.image_cli.detect_image_model": lambda _: MagicMock(family="zimage", size=None),
        "zvisiongenerator.image_cli.resolve_model_path": lambda p, **kw: p,
        "zvisiongenerator.image_cli.resolve_defaults": lambda *a, **kw: {"steps": 10, "guidance": 0.5, "scheduler": None},
        "zvisiongenerator.image_cli.validate_scheduler": lambda *a: None,
        "zvisiongenerator.image_cli.inspect_prompts_file": lambda _: PromptFileInspection(prompts_data={"set": [("a cat", None)]}, options=[]),
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
        return [str(warning.message) for warning in caught_warnings if "--first-sigma" in str(warning.message) or "--json-prompt" in str(warning.message)]

    def test_quantize_rejected_for_families_without_quantization(self):
        backend = MagicMock(name="mflux")
        with pytest.raises(SystemExit, match="2"):
            self._run_main_with_overrides(
                ["-m", "ideo", "-q", "8"],
                detect_image_model=lambda _: MagicMock(family="ideogram4", size=None),
                resolve_defaults=lambda *a, **kw: {"steps": 20, "guidance": 7.0, "scheduler": None, "supports_quantize": False},
                get_backend=lambda: backend,
            )
        backend.load_model.assert_not_called()

    def test_quantize_accepted_for_families_with_quantization(self):
        backend = MagicMock(name="mflux", load_model=MagicMock(return_value=(MagicMock(), MagicMock(family="zimage"))))
        backend.stored_quant_format.return_value = None
        self._run_main_with_overrides(
            ["-m", "fake", "-q", "8"],
            resolve_defaults=lambda *a, **kw: {"steps": 10, "guidance": 0.5, "scheduler": None, "supports_quantize": True},
            get_backend=lambda: backend,
        )
        assert backend.load_model.call_args.kwargs["quantize"] == 8

    @pytest.mark.parametrize(
        "argv",
        [
            ["--runs", "0"],
            ["--upscale", "3"],
            ["--width", "-100"],
            ["--width", "500"],
            ["--height", "500"],
            ["--steps", "0"],
            ["--steps", "-1"],
            ["--upscale-steps", "0", "--upscale", "2"],
            ["--guidance", "-1.0"],
            ["--upscale-guidance", "-1.0"],
            ["--prompt", ""],
            ["--prompt", "   "],
            # 528 with a 4x upscale drifts to 576.
            ["--width", "528", "--upscale", "4"],
            ["--height", "528", "--upscale", "4"],
        ],
        ids=lambda argv: " ".join(argv),
    )
    def test_invalid_values_exit_2(self, argv):
        with pytest.raises(SystemExit) as excinfo:
            self._run_main([*argv, "-m", "fake"])

        assert excinfo.value.code == 2

    def test_review_without_upscale_exits(self):
        with patch("sys.stdin.isatty", create=True, return_value=True), pytest.raises(SystemExit, match="2"):
            self._run_main(["--review", "-m", "fake"])

    def test_review_without_a_terminal_exits(self):
        with patch("sys.stdin") as stdin, pytest.raises(SystemExit, match="2"):
            stdin.isatty.return_value = False
            self._run_main(["--review", "--upscale", "2", "-m", "fake"])

    def test_review_with_upscale_in_a_terminal_reaches_the_batch(self):
        run_batch = MagicMock()
        with patch("sys.stdin") as stdin:
            stdin.isatty.return_value = True
            stdin.fileno.side_effect = OSError("no fd")  # the key listener gives up quietly instead of dying on the mock
            self._run_main_with_overrides(["--review", "--upscale", "2", "-m", "fake"], run_batch=run_batch)
        assert run_batch.call_args.args[4].review is True

    @pytest.mark.parametrize(("flags", "expected"), [([], False), (["--upscale-sharpen"], True)])
    def test_upscale_sharpen_follows_the_model_default_unless_given(self, flags, expected):
        run_batch = MagicMock()
        self._run_main_with_overrides(
            ["--upscale", "2", "-m", "fake", *flags],
            resolve_defaults=lambda *a, **kw: {"steps": 8, "guidance": 1.0, "scheduler": "beta", "upscale_sharpen": False},
            run_batch=run_batch,
        )
        assert run_batch.call_args.args[4].upscale_sharpen is expected

    @pytest.mark.parametrize("width", [240, 3072])
    def test_ideogram4_dimension_bounds_error_on_in_grid_out_of_range_widths(self, width, capsys):
        mock_backend = _backend("ideogram4")

        with pytest.raises(SystemExit, match="2"):
            self._run_main_with_overrides(
                ["--prompt", "ok", "--width", str(width), "--height", "1024", "-m", "fake"],
                **_IDEOGRAM,
                get_backend=lambda: mock_backend,
            )

        assert "between 256 and 2048" in capsys.readouterr().err
        mock_backend.load_model.assert_not_called()

    def test_ideogram4_dimension_bounds_accept_1024_square(self):
        mock_backend = _backend("ideogram4")

        self._run_main_with_overrides(
            ["--prompt", "ok", "--width", "1024", "--height", "1024", "-m", "fake"],
            **_IDEOGRAM,
            get_backend=lambda: mock_backend,
        )

        mock_backend.load_model.assert_called_once()

    def test_first_sigma_parses_as_float(self):
        parser = _build_parser()
        args = parser.parse_args(["--first-sigma", "1.005", "-m", "fake"])

        assert args.first_sigma == 1.005

    def test_first_sigma_invalid_float_exits_before_load_model(self):
        mock_backend = _backend("ideogram4")

        with pytest.raises(SystemExit, match="2"):
            self._run_main_with_overrides(
                ["--first-sigma", "abc", "-m", "fake"],
                get_backend=lambda: mock_backend,
            )

        mock_backend.load_model.assert_not_called()

    @pytest.mark.parametrize("value", ["-1", "0", "2.5"])
    def test_first_sigma_out_of_band_exits_before_load_model(self, value):
        mock_backend = _backend("ideogram4")

        with pytest.raises(SystemExit, match="2"):
            self._run_main_with_overrides(
                ["--first-sigma", value, "--prompt", "ok", "-m", "fake"],
                get_backend=lambda: mock_backend,
            )

        mock_backend.load_model.assert_not_called()

    @pytest.mark.parametrize("value", [1.006, 2.0])
    def test_first_sigma_accepts_in_band_values(self, value):
        mock_backend = _backend("ideogram4")
        recorded = {}

        def _capture_run_batch(_backend, _model, _prompts_data, _config, args, **_kwargs):
            recorded["first_sigma"] = args.first_sigma

        self._run_main_with_overrides(
            ["--first-sigma", str(value), "--prompt", "ok", "-m", "fake"],
            **_IDEOGRAM,
            get_backend=lambda: mock_backend,
            run_batch=_capture_run_batch,
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

        assert any("--first-sigma" in message and "zimage" in message for message in warning_messages)

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

        assert any("--json-prompt" in message and "zimage" in message for message in warning_messages)

    @pytest.mark.parametrize(
        "argv",
        [
            ["--first-sigma", "1.006", "--prompt", "ok", "-m", "fake"],
            ["--json-prompt", '{"a": 1}', "-m", "fake"],
        ],
    )
    def test_ideogram4_flag_usage_does_not_emit_non_ideogram_warnings(self, argv):
        mock_backend = _backend("ideogram4")

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self._run_main_with_overrides(
                argv,
                **_IDEOGRAM,
                get_backend=lambda: mock_backend,
            )

        warning_messages = self._flag_misuse_warning_messages(caught)

        assert warning_messages == []

    def test_non_ideogram_without_first_sigma_or_json_prompt_does_not_emit_flag_warnings(self):
        mock_backend = _backend("zimage")

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self._run_main_with_overrides(
                ["--prompt", "ok", "-m", "fake"],
                detect_image_model=lambda _path: MagicMock(family="zimage", size=None),
                get_backend=lambda: mock_backend,
            )

        warning_messages = self._flag_misuse_warning_messages(caught)

        assert warning_messages == []

    # ── Upscale size drift validation ───────────────────────────────────

    def test_upscale_compatible_width_ok(self):
        """width=512 with 4x upscale: 512//4=128, round16(128)=128, 128*4=512 — ok."""
        self._run_main(["--width", "512", "--upscale", "4", "-m", "fake"])

    def test_upscale_without_explicit_dims_ok(self):
        """Upscale without explicit width/height should not error."""
        self._run_main(["--upscale", "2", "-m", "fake"])

    # ── Steps and guidance validation ──

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

    def test_missing_prompts_file_is_refused_before_model_detection(self, capsys):
        """No --prompt and no prompts file ends with a hint, before the model is detected or the backend loads."""
        detect = MagicMock(side_effect=AssertionError("detect_image_model must not run"))
        get_backend = MagicMock(side_effect=AssertionError("get_backend must not run"))

        with pytest.raises(SystemExit, match="2"):
            self._run_main_with_overrides(
                ["-m", "fake", "-p", "missing.yaml"],
                inspect_prompts_file=MagicMock(side_effect=FileNotFoundError(2, "No such file or directory", "missing.yaml")),
                detect_image_model=detect,
                get_backend=get_backend,
            )

        error = capsys.readouterr().err
        assert "Prompts file not found: missing.yaml" in error
        assert "Provide --prompt or a valid --prompts-file" in error
        detect.assert_not_called()
        get_backend.assert_not_called()

    def test_json_prompt_accepts_inline_json_object_value(self):
        recorded = {}
        mock_backend = _backend("ideogram4")

        def _capture_run_batch(_backend, _model, prompts_data, _config, args, **_kwargs):
            recorded["prompt"] = prompts_data["prompt"][0][0]
            recorded["json_prompt"] = args.json_prompt
            recorded["json_prompt_enabled"] = args.json_prompt_enabled

        self._run_main_with_overrides(
            ["--json-prompt", '{"a": 1}', "-m", "fake"],
            **_IDEOGRAM,
            get_backend=lambda: mock_backend,
            run_batch=_capture_run_batch,
        )

        assert recorded == {
            "prompt": '{"a": 1}',
            "json_prompt": '{"a": 1}',
            "json_prompt_enabled": True,
        }

    def test_json_prompt_rejects_prompt_argument_combination_before_load_model(self, capsys):
        mock_backend = _backend("zimage")

        with pytest.raises(SystemExit, match="2"):
            self._run_main_with_overrides(
                ["--prompt", "x", "--json-prompt", '{"a": 1}', "-m", "fake"],
                get_backend=lambda: mock_backend,
            )

        assert "not allowed with argument" in capsys.readouterr().err
        mock_backend.load_model.assert_not_called()

    def test_json_prompt_rejects_empty_value_before_load_model(self, capsys):
        mock_backend = _backend("zimage")

        with pytest.raises(SystemExit, match="2"):
            self._run_main_with_overrides(
                ["--json-prompt", "   ", "-m", "fake"],
                get_backend=lambda: mock_backend,
            )

        assert "must not be empty" in capsys.readouterr().err
        mock_backend.load_model.assert_not_called()

    def test_json_prompt_rejects_invalid_json_before_load_model(self, capsys):
        mock_backend = _backend("zimage")

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
        mock_backend = _backend("zimage")

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
        mock_backend = _backend("ideogram4")

        def _load_config():
            return {
                "sizes": {
                    "16:9": {"xl": {"width": 2112, "height": 1184}},
                    "2:3": {"m": {"width": 832, "height": 1216}},
                },
                "generation": {"default_ratio": "2:3", "default_size": "m"},
            }

        with pytest.raises(SystemExit, match="2"):
            self._run_main_with_overrides(
                ["--prompt", "ok", "--size", "xl", "--ratio", "16:9", "-m", "ideo"],
                load_config=_load_config,
                **_IDEOGRAM,
                get_backend=lambda: mock_backend,
            )

        assert "2048" in capsys.readouterr().err
        mock_backend.load_model.assert_not_called()

    def test_ideogram4_img2img_early_guard_errors_before_load_model(self, tmp_path, capsys):
        image_path = tmp_path / "reference.png"
        image_path.write_bytes(b"fake")
        mock_backend = _backend("ideogram4")

        with pytest.raises(SystemExit, match="2"):
            self._run_main_with_overrides(
                ["--prompt", "ok", "--image", str(image_path), "-m", "ideo"],
                **_IDEOGRAM,
                get_backend=lambda: mock_backend,
            )

        assert "img2img is not supported" in capsys.readouterr().err.lower()
        mock_backend.load_model.assert_not_called()

    def test_ideogram4_upscale_early_guard_errors_before_load_model(self, capsys):
        mock_backend = _backend("ideogram4")

        with pytest.raises(SystemExit, match="2"):
            self._run_main_with_overrides(
                ["--prompt", "ok", "--upscale", "2", "-m", "ideo"],
                **_IDEOGRAM,
                get_backend=lambda: mock_backend,
            )

        assert re.search(r"upscal.*not supported", capsys.readouterr().err, re.IGNORECASE)
        mock_backend.load_model.assert_not_called()

    def test_model_with_its_own_sampler_rejects_scheduler(self, capsys):
        mock_backend = MagicMock()
        mock_backend.name = "mflux"

        with pytest.raises(SystemExit, match="2"):
            self._run_main_with_overrides(
                ["--prompt", "ok", "--scheduler", "beta", "-m", "krea2"],
                detect_image_model=lambda _path: MagicMock(family="krea2", size=None),
                get_backend=lambda: mock_backend,
                resolve_defaults=lambda *a, **kw: {"steps": 8, "guidance": 1.0, "scheduler": "beta", "supports_scheduler": False},
            )

        assert "own sampler" in capsys.readouterr().err
        mock_backend.load_model.assert_not_called()


class TestImageMemoryWarning:
    """``ziv-image`` warns once, before any download, when the memory verdict is Tight or Too large."""

    _MAIN_MOCKS = TestCLIValidation._MAIN_MOCKS
    _run_main_with_overrides = TestCLIValidation._run_main_with_overrides

    _KLEIN = {
        "detect_image_model": lambda _path: MagicMock(family="flux2_klein", size="9b"),
        "resolve_defaults": lambda *a, **kw: {"steps": 4, "guidance": 1.0, "scheduler": None, "max_reference_images": 4},
    }
    _MACHINE = MemoryBudget(10 * GIB, system_bytes=30 * GIB)

    @staticmethod
    def _verdict(status="too_large", resource="gpu", gpu_gb=10.4, ram_gb=None):
        return MemoryVerdict(round(gpu_gb * GIB), None if ram_gb is None else round(ram_gb * GIB), "resident", status, resource)

    @staticmethod
    def _resolved(**extra):
        return MagicMock(family="flux2_klein", model_size="9b", stored_bits=None, weights=MemoryWeights(), **extra)

    @staticmethod
    def _row_judge(by_level):
        """Return the engine stand-in the rows of a job's other levels are judged with; a level it has no verdict for cannot be judged."""

        def judge(job, *_):
            if job.quantize not in (by_level or {}):
                raise ValueError("no weights")
            return by_level[job.quantize]

        return judge

    def _run(self, argv, verdict, tmp_path, *, machine=_MACHINE, resolved="default", judge_error=None, by_level=None):
        """Run a Klein generation with ``judge`` mocked; return the judge and resolver mocks, the backend and the memory warnings caught.

        *by_level* maps a quantize level to the verdict the engine gives the job at it, for the rows of the job's other levels.
        """
        refs = []
        for index in range(2):
            path = tmp_path / f"ref{index}.png"
            Image.new("RGB", (1024, 512)).save(path)
            refs.append(str(path))
        backend = _backend("flux2_klein")
        judge = MagicMock(return_value=verdict, side_effect=judge_error)
        resolve = MagicMock(return_value=self._resolved() if resolved == "default" else resolved)
        argv = [value.replace("REFS", ",".join(refs)) for value in argv]
        with warnings.catch_warnings(record=True) as caught, patch("zvisiongenerator.utils.memory_rows.judge", side_effect=self._row_judge(by_level)):
            warnings.simplefilter("always")
            self._run_main_with_overrides(
                ["--prompt", "ok", "-m", "klein", *argv],
                **self._KLEIN,
                get_backend=lambda: backend,
                accelerator_memory_budget_or_none=lambda: machine,
                resolve_image_weights=resolve,
                judge=judge,
            )
        return judge, resolve, backend, [str(w.message) for w in caught if w.message.args and str(w.message).startswith("Memory for")]

    def test_a_too_large_job_warns_once_with_the_sentence_and_the_figures(self, tmp_path):
        judge, resolve, backend, messages = self._run(["--width", "1600", "--height", "1600", "-q", "4"], self._verdict(), tmp_path)

        assert messages == ["Memory for this image (1600×1600 · q4): Needs more GPU memory than this card has (about 10.4 of 10 GB). Continuing."]
        job, machine, weights = judge.call_args.args
        assert (job.media, job.family, job.model_size, job.quantize, job.width, job.height) == ("image", "flux2_klein", "9b", 4, 1600, 1600)
        assert machine == self._MACHINE and weights is resolve.return_value.weights
        assert resolve.call_args.kwargs["hub"] is True
        backend.load_model.assert_called_once()

    def test_a_tight_job_warns_with_its_sentence(self, tmp_path):
        _, _, _, messages = self._run([], self._verdict("tight", gpu_gb=9.6), tmp_path)

        assert messages == ["Memory for this image (832×1216): Fits, but close to the GPU's limit. It may run out of memory (about 9.6 of 10 GB). Continuing."]

    def test_a_ram_verdict_names_the_ram(self, tmp_path):
        _, _, _, messages = self._run([], self._verdict(resource="ram", ram_gb=31.5), tmp_path)

        assert messages[0].endswith("Needs more RAM than this computer has (about 31.5 of 30 GB). Continuing.")

    def test_the_references_are_in_the_heading_and_the_job(self, tmp_path):
        judge, _, _, messages = self._run(["--width", "1024", "--height", "1024", "--reference", "REFS"], self._verdict(), tmp_path)

        assert messages[0].startswith("Memory for this image (1024×1024 · 2 references):")
        assert judge.call_args.args[0].reference_megapixels == pytest.approx((0.524288, 0.524288))

    _LEVELS = {None: _verdict("too_large", gpu_gb=17.0), 8: _verdict("too_large", gpu_gb=10.4), 4: _verdict("fits", gpu_gb=6.0)}

    def test_a_too_large_job_names_how_the_other_quants_fit(self, tmp_path):
        _, _, _, messages = self._run(["--width", "1600", "--height", "1600"], self._LEVELS[None], tmp_path, by_level=self._LEVELS)

        assert messages == [
            "Memory for this image (1600×1600): Needs more GPU memory than this card has (about 17 of 10 GB). Continuing. Other quants: q8 Too large (10.4 GB) · q4 Fits (6 GB)",
        ]

    def test_the_level_the_job_runs_at_is_left_out_of_the_other_quants(self, tmp_path):
        _, _, _, messages = self._run(["-q", "8"], self._LEVELS[8], tmp_path, by_level=self._LEVELS)

        assert messages[0].endswith("Continuing. Other quants: None Too large (17 GB) · q4 Fits (6 GB)")

    def test_a_tight_job_names_the_other_quants_too(self, tmp_path):
        levels = {**self._LEVELS, None: self._verdict("tight", gpu_gb=9.6)}

        _, _, _, messages = self._run([], levels[None], tmp_path, by_level=levels)

        assert messages[0].endswith("Continuing. Other quants: q8 Too large (10.4 GB) · q4 Fits (6 GB)")

    def test_a_model_stored_at_one_level_has_no_other_quants(self, tmp_path):
        _, _, _, messages = self._run([], self._verdict(), tmp_path, resolved=MagicMock(family="flux2_klein", model_size="9b", stored_bits=4, weights=MemoryWeights()), by_level=self._LEVELS)

        assert messages == ["Memory for this image (832×1216 · q4): Needs more GPU memory than this card has (about 10.4 of 10 GB). Continuing."]

    def test_a_model_that_cannot_be_quantized_has_no_other_quants(self, tmp_path):
        with patch("zvisiongenerator.utils.memory_rows.model_capabilities", return_value={"supports_quantize": False}):
            _, _, _, messages = self._run([], self._verdict(), tmp_path, by_level=self._LEVELS)

        assert messages == ["Memory for this image (832×1216): Needs more GPU memory than this card has (about 10.4 of 10 GB). Continuing."]

    def test_an_edit_is_given_its_loras_and_its_other_quants(self):
        args = argparse.Namespace(model_name="klein", model="klein", quantize=None, edit_source="a.png", reference_paths=[], lora_paths=["a.safetensors", "b.safetensors"])
        judge = MagicMock(return_value=self._verdict())

        with patch.multiple("zvisiongenerator.image_cli", accelerator_memory_budget_or_none=lambda: self._MACHINE, resolve_image_weights=MagicMock(return_value=self._resolved()), judge=judge):
            with patch("zvisiongenerator.utils.memory_rows.judge", side_effect=self._row_judge(self._LEVELS)):
                with pytest.warns(UserWarning, match=r"^Memory for this edit \(1024×768 · 2 LoRAs\): .* Other quants: q8 Too large \(10\.4 GB\) · q4 Fits \(6 GB\)$"):
                    _warn_if_memory_tight(args, MemoryJob(media="image", width=1024, height=768))

    def test_one_lora_is_counted_in_the_singular(self):
        args = argparse.Namespace(model_name="klein", model="klein", quantize=4, edit_source=None, reference_paths=[], lora_paths=["a.safetensors"])

        with patch.multiple(
            "zvisiongenerator.image_cli",
            accelerator_memory_budget_or_none=lambda: self._MACHINE,
            resolve_image_weights=MagicMock(return_value=self._resolved()),
            judge=MagicMock(return_value=self._verdict()),
        ):
            with pytest.warns(UserWarning, match=r"\(1024×768 · q4 · 1 LoRA\)"):
                _warn_if_memory_tight(args, MemoryJob(media="image", width=1024, height=768))

    def test_a_fitting_job_does_not_warn(self, tmp_path):
        judge, _, backend, messages = self._run([], self._verdict("fits", gpu_gb=5.0), tmp_path)

        assert messages == []
        judge.assert_called_once()
        backend.load_model.assert_called_once()

    def test_without_a_memory_budget_nothing_is_judged(self, tmp_path):
        judge, resolve, _, messages = self._run([], self._verdict(), tmp_path, machine=None)

        assert messages == []
        judge.assert_not_called()
        resolve.assert_not_called()

    def test_weights_that_cannot_be_sized_stay_silent(self, tmp_path):
        judge, _, backend, messages = self._run([], self._verdict(), tmp_path, resolved=None)

        assert messages == []
        judge.assert_not_called()
        backend.load_model.assert_called_once()

    def test_a_job_the_engine_cannot_judge_stays_silent(self, tmp_path):
        _, _, backend, messages = self._run([], self._verdict(), tmp_path, judge_error=ValueError("no weights"))

        assert messages == []
        backend.load_model.assert_called_once()

    def test_the_warning_comes_before_the_download_and_the_load(self, tmp_path):
        backend = _backend("flux2_klein")
        seen: dict[str, bool] = {}
        built = MagicMock(return_value=True)

        def _judge(*_args, **_kwargs):
            seen["built"], seen["loaded"] = built.called, backend.load_model.called
            return self._verdict()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._run_main_with_overrides(
                ["--prompt", "ok", "-m", "klein"],
                **self._KLEIN,
                get_backend=lambda: backend,
                ensure_built_model=built,
                accelerator_memory_budget_or_none=lambda: self._MACHINE,
                resolve_image_weights=MagicMock(return_value=self._resolved()),
                judge=_judge,
            )

        assert seen == {"built": False, "loaded": False}
        built.assert_called_once()

    def test_an_edit_is_judged_at_its_canvas_with_its_picture_and_named_an_edit(self, tmp_path):
        args = argparse.Namespace(model_name="klein", model="klein", quantize=None, edit_source="a.png", reference_paths=[])
        judge = MagicMock(return_value=self._verdict())

        with patch.multiple("zvisiongenerator.image_cli", accelerator_memory_budget_or_none=lambda: self._MACHINE, resolve_image_weights=MagicMock(return_value=self._resolved()), judge=judge):
            with pytest.warns(UserWarning, match=r"^Memory for this edit \(1024×768\): Needs more GPU memory"):
                _warn_if_memory_tight(args, MemoryJob(media="image", width=1024, height=768, reference_megapixels=(0.79,)))

        assert judge.call_args.args[0].reference_megapixels == (0.79,)


# ── Post-processing flag parsing ────────────────────────────────────────────


class TestPostProcessingFlags:
    """Test --sharpen/--no-sharpen, --contrast/--no-contrast, --saturation/--no-saturation."""

    @pytest.mark.parametrize(("flag", "default", "amount"), [("sharpen", True, 0.6), ("contrast", False, 1.3), ("saturation", False, 1.2)])
    def test_flag_forms(self, flag, default, amount):
        def parse(*argv: str):
            return getattr(_build_parser().parse_args(["-m", "fake", *argv]), flag)

        assert parse() is default
        assert parse(f"--{flag}") is True
        assert parse(f"--{flag}", str(amount)) == amount
        zero = parse(f"--{flag}", "0.0")
        assert zero == 0.0 and zero is not False
        assert parse(f"--no-{flag}") is False

    @pytest.mark.parametrize("argv", [["--sharpen", "-1"], ["--sharpen", "1.6"], ["--contrast", "-0.5"], ["--saturation", "-2"]], ids=lambda argv: " ".join(argv))
    def test_out_of_range_amounts_are_rejected(self, argv):
        with pytest.raises(SystemExit) as excinfo:
            TestCLIValidation()._run_main(["-m", "fake", *argv])

        assert excinfo.value.code == 2


class TestTransparentFlag:
    """--transparent parses to a flag and, once set, fetches both pinned cut-out snapshots before the image model loads."""

    def _run(self, argv, *, snapshot_with, order):
        loaded = (MagicMock(name="model"), MagicMock(family="zimage"))

        def _load(*_a, **_kw):
            order.append("load")
            return loaded

        with patch("zvisiongenerator.image_cli.cached_snapshot_dirs", return_value=[]):
            TestCLIValidation()._run_main_with_overrides(["-m", "fake", *argv], snapshot_with=snapshot_with, load_image_model=_load)

    def test_the_flag_defaults_to_false_and_parses_to_true(self):
        assert _build_parser().parse_args(["-m", "fake"]).transparent is False
        assert _build_parser().parse_args(["-m", "fake", "--transparent"]).transparent is True

    def test_both_pinned_snapshots_are_fetched_once_before_the_model_loads(self):
        order: list[str] = []
        fetch = MagicMock(side_effect=lambda files, _download: order.append(files))

        self._run(["--transparent"], snapshot_with=fetch, order=order)

        assert order == [BIREFNET, VITMATTE, "load"]

    def test_a_failed_fetch_stops_the_run_before_any_load(self):
        order: list[str] = []
        fetch = MagicMock(side_effect=OSError("offline"))

        with pytest.raises(SystemExit) as excinfo:
            self._run(["--transparent"], snapshot_with=fetch, order=order)

        assert excinfo.value.code == 2
        assert order == []

    def test_without_the_flag_nothing_is_fetched(self):
        order: list[str] = []
        fetch = MagicMock()

        self._run([], snapshot_with=fetch, order=order)

        fetch.assert_not_called()
        assert order == ["load"]

    def test_the_run_gets_a_reloader_that_loads_the_model_again(self):
        loads: list[str] = []
        captured = {}

        def _load(*_a, **_kw):
            loads.append("load")
            return (MagicMock(name="model"), MagicMock(family="zimage"))

        def _capture(*_a, reload_model=None, **_kw):
            captured["loads_before"] = list(loads)
            captured["reloaded"] = reload_model()

        with patch("zvisiongenerator.image_cli.cached_snapshot_dirs", return_value=[]):
            TestCLIValidation()._run_main_with_overrides(["-m", "fake", "--transparent"], snapshot_with=MagicMock(), load_image_model=_load, run_batch=_capture)

        assert captured["loads_before"] == ["load"]
        assert captured["reloaded"] is not None
        assert loads == ["load", "load"]


class TestReferenceFlag:
    """``--reference`` takes a comma-separated list, checked against the model's cap and the files."""

    _MAIN_MOCKS = TestCLIValidation._MAIN_MOCKS
    _run_main_with_overrides = TestCLIValidation._run_main_with_overrides

    _KLEIN = {
        "detect_image_model": lambda _path: MagicMock(family="flux2_klein", size="4b"),
        "resolve_defaults": lambda *a, **kw: {"steps": 4, "guidance": 1.0, "scheduler": None, "max_reference_images": 4},
        "get_backend": lambda: _backend("flux2_klein"),
    }

    @staticmethod
    def _images(tmp_path, count):
        paths = []
        for index in range(count):
            path = tmp_path / f"ref{index}.png"
            path.write_bytes(b"fake")
            paths.append(str(path))
        return paths

    def _run_klein(self, argv, **overrides):
        self._run_main_with_overrides(["--prompt", "ok", "-m", "klein", *argv], **{**self._KLEIN, **overrides})

    def test_the_list_reaches_the_runner_in_order_and_combines_with_a_start_image(self, tmp_path):
        refs = self._images(tmp_path, 3)
        start = tmp_path / "start.png"
        start.write_bytes(b"fake")
        seen = {}

        self._run_klein(["--reference", ",".join(refs), "--image", str(start)], run_batch=lambda *a, **kw: seen.update(args=a[4]))

        assert seen["args"].reference_paths == refs
        assert seen["args"].image_path == str(start)

    def test_no_flag_means_no_references(self):
        seen = {}

        self._run_klein([], run_batch=lambda *a, **kw: seen.update(args=a[4]))

        assert seen["args"].reference_paths == []

    def test_a_model_without_references_is_refused_before_loading(self, tmp_path):
        backend = _backend("zimage")

        with pytest.raises(SystemExit) as excinfo:
            self._run_main_with_overrides(["--prompt", "ok", "--reference", self._images(tmp_path, 1)[0], "-m", "zit"], get_backend=lambda: backend)

        assert excinfo.value.code == 2
        backend.load_model.assert_not_called()

    def test_a_file_that_is_not_an_existing_image_is_refused(self, tmp_path):
        ran = []

        with pytest.raises(SystemExit) as excinfo:
            self._run_klein(["--reference", str(tmp_path / "gone.png")], run_batch=lambda *a, **kw: ran.append(a))

        assert excinfo.value.code == 2
        assert ran == []


class TestMemoryJobHandoff:
    """The model loads for the size the run draws, so the backend can place its weights for it."""

    @staticmethod
    def _run(monkeypatch, *argv):
        backend = _patch_image_main_dependencies(monkeypatch)
        monkeypatch.setattr("sys.argv", ["ziv-image", "-m", "model", "--prompt", "a cat", *argv])
        main()
        return backend.load_model.call_args.kwargs["memory_job"]

    def test_the_preset_size_is_the_jobs_size(self, monkeypatch):
        job = self._run(monkeypatch)

        assert (job.width, job.height, job.reference_megapixels) == (832, 1216, ())

    def test_explicit_dimensions_override_the_preset(self, monkeypatch):
        job = self._run(monkeypatch, "--width", "1600", "--height", "1024")

        assert (job.width, job.height) == (1600, 1024)

    def test_an_upscale_draws_its_final_size(self, monkeypatch):
        job = self._run(monkeypatch, "--width", "1600", "--height", "1024", "--upscale", "2")

        assert (job.width, job.height) == (1600, 1024)  # the size after the upscale, the largest the model draws

    def test_the_references_and_loras_are_counted(self, tmp_path):
        from PIL import Image

        from zvisiongenerator.image_cli import _image_memory_job

        reference = tmp_path / "ref.png"
        Image.new("RGB", (2000, 1000)).save(reference)
        lora = tmp_path / "style.safetensors"
        lora.write_bytes(b"x" * 40)
        args = argparse.Namespace(edit_source=None, ratio="2:3", size="m", width=None, height=None, reference_paths=[str(reference)])

        job = _image_memory_job(args, {"2:3": {"m": {"width": 832, "height": 1216}}}, [str(lora)])

        assert (job.width, job.height) == (832, 1216)
        assert job.reference_megapixels == pytest.approx((2.0,))
        assert job.lora_bytes == 40

    def test_an_extend_draws_its_canvas_and_holds_it_as_its_one_reference_capped_at_a_megapixel(self):
        from zvisiongenerator.image_cli import _image_memory_job
        from zvisiongenerator.processing.edit_region import edit_model_size

        args = argparse.Namespace(edit_source="/in/a.png", edit_size=(3000, 2000), edit_paint=None, edit_extend=("right", 512), reference_paths=[])
        job = _image_memory_job(args, {}, None)

        assert (job.width, job.height) == edit_model_size(3000, 2000, masked=False, extend=("right", 512))
        assert job.reference_megapixels == pytest.approx((1024 * 1024 / 1e6,))
