"""Tests verifying --lora flag unification between image and video CLIs.

Ensures both CLIs share the same parse_lora_arg() from utils/lora.py
and accept the identical comma-separated 'name1:0.8,name2:0.5' syntax.
"""

from __future__ import annotations

import inspect
import pathlib

import pytest

from zvisiongenerator.utils.lora import parse_lora_arg
from zvisiongenerator.video_cli import _build_video_parser
from zvisiongenerator.image_cli import _build_parser


# ---------------------------------------------------------------------------
# SC-1: parse_lora_arg lives in utils/lora.py and both CLIs import from there
# ---------------------------------------------------------------------------


class TestSharedImportLocation:
    """Verify parse_lora_arg is sourced from zvisiongenerator.utils.lora."""

    def test_parse_lora_arg_module(self):
        assert parse_lora_arg.__module__ == "zvisiongenerator.utils.lora"

    def test_image_cli_imports_from_utils_lora(self):
        import zvisiongenerator.image_cli as mod

        src = inspect.getsource(mod)
        assert "from zvisiongenerator.utils.lora import parse_lora_arg" in src

    def test_video_cli_imports_from_utils_lora(self):
        import zvisiongenerator.video_cli as mod

        src = inspect.getsource(mod)
        assert "from zvisiongenerator.utils.lora import parse_lora_arg" in src


# ---------------------------------------------------------------------------
# SC-2: utils/__init__.py re-exports parse_lora_arg in __all__
# ---------------------------------------------------------------------------


class TestUtilsReExport:
    def test_parse_lora_arg_in_utils_all(self):
        import zvisiongenerator.utils

        assert "parse_lora_arg" in zvisiongenerator.utils.__all__

    def test_parse_lora_arg_importable_from_utils(self):
        from zvisiongenerator.utils import parse_lora_arg as fn

        assert callable(fn)
        assert fn is parse_lora_arg


# ---------------------------------------------------------------------------
# SC-3: Video CLI's --lora argparse definition matches image CLI's
# ---------------------------------------------------------------------------


class TestArgparseDefinitionMatch:
    """Both parsers must declare --lora identically (type=str, default=None)."""

    def _get_lora_action(self, parser):
        for action in parser._actions:
            if "--lora" in getattr(action, "option_strings", []):
                return action
        pytest.fail("--lora not found in parser")

    def test_video_lora_type_is_str(self):
        action = self._get_lora_action(_build_video_parser())
        assert action.type is str

    def test_video_lora_default_is_none(self):
        action = self._get_lora_action(_build_video_parser())
        assert action.default is None

    def test_image_lora_type_is_str(self):
        action = self._get_lora_action(_build_parser())
        assert action.type is str

    def test_image_lora_default_is_none(self):
        action = self._get_lora_action(_build_parser())
        assert action.default is None

    def test_video_lora_not_append_action(self):
        """Ensure --lora is NOT using the old append/nargs syntax."""
        action = self._get_lora_action(_build_video_parser())
        # Must not be an _AppendAction
        assert type(action).__name__ != "_AppendAction"
        # Must not have nargs set (old repeatable syntax used nargs="+")
        assert action.nargs is None

    def test_both_parsers_identical_lora_config(self):
        img_action = self._get_lora_action(_build_parser())
        vid_action = self._get_lora_action(_build_video_parser())
        assert img_action.type is vid_action.type
        assert img_action.default == vid_action.default
        assert img_action.nargs == vid_action.nargs


# ---------------------------------------------------------------------------
# SC-4: Video CLI's _parse_loras() is completely removed
# ---------------------------------------------------------------------------


class TestOldFunctionRemoved:
    def test_no_parse_loras_in_video_cli(self):
        import zvisiongenerator.video_cli as mod

        assert not hasattr(mod, "_parse_loras"), "_parse_loras should be removed from video_cli"

    def test_no_parse_loras_in_source(self):
        import zvisiongenerator.video_cli as mod

        src = inspect.getsource(mod)
        assert "def _parse_loras" not in src


# ---------------------------------------------------------------------------
# SC-5: Video CLI main() uses parse_lora_arg() and resolve_lora_path()
# ---------------------------------------------------------------------------


class TestVideoMainUsesSharedHelpers:
    def test_video_main_calls_parse_lora_arg(self):
        import zvisiongenerator.video_cli as mod

        src = inspect.getsource(mod.main)
        assert "parse_lora_arg" in src

    def test_video_main_calls_resolve_lora_path(self):
        import zvisiongenerator.video_cli as mod

        src = inspect.getsource(mod.main)
        assert "resolve_lora_path" in src


# ---------------------------------------------------------------------------
# SC-6: Existing parse_lora_arg tests pass from new import location
# ---------------------------------------------------------------------------


class TestParseLoraArgFromNewLocation:
    """Mirror of existing TestParseLoraArg from test_cli_parsing, imported from utils.lora."""

    def test_single_name(self):
        assert parse_lora_arg("style") == [("style", 1.0)]

    def test_name_with_weight(self):
        assert parse_lora_arg("style:0.8") == [("style", 0.8)]

    def test_comma_separated(self):
        assert parse_lora_arg("style:0.8,detail:0.5") == [("style", 0.8), ("detail", 0.5)]

    def test_mixed(self):
        assert parse_lora_arg("style:0.7,detail") == [("style", 0.7), ("detail", 1.0)]

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="Empty LoRA entry"):
            parse_lora_arg("")

    def test_empty_name_raises(self):
        with pytest.raises(ValueError, match="Empty LoRA name"):
            parse_lora_arg(":0.5")


# ---------------------------------------------------------------------------
# SC-7: Video CLI tests have no references to old append/nargs syntax
# ---------------------------------------------------------------------------


class TestNoOldSyntaxInVideoTests:
    def test_video_test_file_has_no_append_nargs(self):
        import tests.test_video_cli as test_mod

        src = inspect.getsource(test_mod)
        assert "append" not in src.lower() or "append" not in src
        assert "nargs" not in src


# ---------------------------------------------------------------------------
# SC-8: No test imports _parse_loras from video_cli
# ---------------------------------------------------------------------------


class TestNoImportOfOldFunction:
    def test_no_test_imports_parse_loras(self):
        tests_dir = pathlib.Path(__file__).parent
        this_file = pathlib.Path(__file__).name
        for test_file in tests_dir.glob("test_*.py"):
            if test_file.name == this_file:
                continue
            src = test_file.read_text()
            assert "_parse_loras" not in src, f"{test_file.name} still references _parse_loras"


# ---------------------------------------------------------------------------
# Functional: Both CLIs parse --lora identically via the parser
# ---------------------------------------------------------------------------


class TestFunctionalParsing:
    """Verify both parsers produce identical parsed LoRA strings."""

    def test_image_parser_accepts_comma_syntax(self):
        parser = _build_parser()
        args = parser.parse_args(["-m", "model", "--lora", "style:0.8,detail:0.5"])
        assert args.lora == "style:0.8,detail:0.5"
        parsed = parse_lora_arg(args.lora)
        assert parsed == [("style", 0.8), ("detail", 0.5)]

    def test_video_parser_accepts_comma_syntax(self):
        parser = _build_video_parser()
        args = parser.parse_args(["-m", "model", "--lora", "style:0.8,detail:0.5"])
        assert args.lora == "style:0.8,detail:0.5"
        parsed = parse_lora_arg(args.lora)
        assert parsed == [("style", 0.8), ("detail", 0.5)]

    def test_both_parsers_same_result(self):
        lora_str = "lora1:0.9,lora2,lora3:0.3"
        img_args = _build_parser().parse_args(["-m", "model", "--lora", lora_str])
        vid_args = _build_video_parser().parse_args(["-m", "model", "--lora", lora_str])
        assert img_args.lora == vid_args.lora
        assert parse_lora_arg(img_args.lora) == parse_lora_arg(vid_args.lora)
