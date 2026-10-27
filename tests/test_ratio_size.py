"""Tests for the aspect-ratio + size-scale sizing system.

Covers:
- The packaged size table (shape, upscale survival) and its copy in the docs
- CLI --ratio/--size choices match the packaged table, and invalid values are refused
- Default ratio/size resolution when only some flags are given
- Runner resolves nested lookup correctly, and --width/--height overrides take priority
- Filenames are the set name plus a timestamp
"""

from __future__ import annotations

import re
from unittest.mock import MagicMock, patch

import pytest

from tests._factories import _make_plan
from tests._paths import REPO_ROOT

from zvisiongenerator.image_cli import _build_parser, main
from zvisiongenerator.utils.alignment import round_to_alignment
from zvisiongenerator.utils.prompts import PromptFileInspection


_IMAGES_DOC = REPO_ROOT / "docs" / "concepts" / "images.md"

# The few presets the runner and default-resolution tests look up, as (width, height).
_SIZES_UNDER_TEST = {
    "1:1": {"m": (1024, 1024)},
    "16:9": {"l": (1920, 1088)},
    "9:16": {"xs": (384, 672)},
    "2:3": {"m": (832, 1216), "xl": (1280, 1920)},
}


def _sizes_config() -> dict:
    """Return ``_SIZES_UNDER_TEST`` in config form."""
    return {ratio: {size: {"width": w, "height": h} for size, (w, h) in scale_map.items()} for ratio, scale_map in _SIZES_UNDER_TEST.items()}


def _documented_sizes() -> dict[str, dict[str, tuple[int, int]]]:
    """Parse the size table in docs/concepts/images.md into ``{ratio: {size: (width, height)}}``."""
    rows: list[str] = []
    for line in _IMAGES_DOC.read_text(encoding="utf-8").splitlines():
        if line.startswith("|"):
            rows.append(line)
        elif rows:
            break  # the size table is the first table on the page
    ratios = [cell.strip() for cell in rows[0].strip("|").split("|")[1:]]
    table: dict[str, dict[str, tuple[int, int]]] = {ratio: {} for ratio in ratios}
    for row in rows[2:]:
        cells = [cell.strip() for cell in row.strip("|").split("|")]
        size = cells[0].strip("`")
        for ratio, cell in zip(ratios, cells[1:]):
            width, height = (int(part) for part in cell.split("\u00d7"))
            table[ratio][size] = (width, height)
    return table


# ── The packaged size table ─────────────────────────────────────────────────


class TestPackagedSizeTable:
    def test_five_ratios_with_five_sizes_each(self, packaged_config_shared):
        sizes = packaged_config_shared["sizes"]

        assert set(sizes) == {"1:1", "16:9", "9:16", "3:2", "2:3"}
        for ratio, scale_map in sizes.items():
            assert set(scale_map) == {"xs", "s", "m", "l", "xl"}, ratio

    def test_the_docs_table_matches_the_packaged_sizes(self, packaged_config_shared):
        packaged = {ratio: {size: (dims["width"], dims["height"]) for size, dims in scale_map.items()} for ratio, scale_map in packaged_config_shared["sizes"].items()}

        assert _documented_sizes() == packaged

    def test_every_preset_survives_upscale(self, packaged_config_shared):
        """xs and s survive 2x; m, l and xl survive 2x and 4x (the base render is a multiple of 16)."""
        for ratio, scale_map in packaged_config_shared["sizes"].items():
            for size, dims in scale_map.items():
                factors = (2,) if size in ("xs", "s") else (2, 4)
                for factor in factors:
                    for value in (dims["width"], dims["height"]):
                        assert round_to_alignment(value // factor) * factor == value, f"{ratio}/{size} {value} drifts under {factor}x upscale"


# ── CLI flag choices ────────────────────────────────────────────────────────


class TestCLIRatioSizeFlags:
    def test_the_choices_are_the_packaged_ratios_and_sizes(self, packaged_config_shared):
        parser = _build_parser()
        choices = {option: set(action.choices) for action in parser._actions for option in action.option_strings if action.choices}
        sizes = packaged_config_shared["sizes"]

        assert choices["--ratio"] == set(sizes)
        assert choices["--size"] == {size for scale_map in sizes.values() for size in scale_map}

    @pytest.mark.parametrize("argv", [["--ratio", "4:3"], ["--size", "xxl"]])
    def test_an_unknown_ratio_or_size_exits_2(self, argv):
        with pytest.raises(SystemExit) as excinfo:
            _build_parser().parse_args([*argv, "-m", "fake"])

        assert excinfo.value.code == 2


# ── Default resolution via main() ───────────────────────────────────────────


class TestDefaultResolution:
    """Verify the config defaults fill in whichever of --ratio and --size is missing."""

    _MAIN_MOCKS = {
        "detect_image_model": staticmethod(lambda _: MagicMock(family="zimage", size=None)),
        "resolve_model_path": staticmethod(lambda p, **kw: p),
        "resolve_defaults": staticmethod(lambda *a, **kw: {"steps": 10, "guidance": 0.5, "scheduler": None}),
        "validate_scheduler": staticmethod(lambda *a: None),
        "inspect_prompts_file": staticmethod(lambda _: PromptFileInspection(prompts_data={"set": [("a cat", None)]}, options=[])),
        "get_backend": staticmethod(lambda: MagicMock(name="mflux", load_model=MagicMock(return_value=(MagicMock(), MagicMock(family="zimage"))))),
    }

    def _run_capturing(self, argv: list[str]) -> dict:
        """Run main() capturing the args passed to run_batch."""
        recorded = {}

        def _capture(backend, model, prompts, config, args, **kw):
            recorded["ratio"] = args.ratio
            recorded["size"] = args.size

        mocks = dict(self._MAIN_MOCKS)
        mocks["run_batch"] = _capture
        mocks["load_config"] = lambda: {
            "sizes": _sizes_config(),
            "model_presets": {},
            "schedulers": {},
            "generation": {"default_ratio": "2:3", "default_size": "m", "default_steps": 10, "default_guidance": 3.5},
        }

        with patch("sys.argv", ["ziv-image"] + argv):
            with patch.multiple("zvisiongenerator.image_cli", **mocks):
                main()
        return recorded

    @pytest.mark.parametrize(
        ("argv", "expected"),
        [
            (["-m", "fake"], ("2:3", "m")),
            (["-m", "fake", "--size", "xl"], ("2:3", "xl")),
            (["-m", "fake", "--ratio", "1:1"], ("1:1", "m")),
        ],
        ids=["no-flags", "only-size", "only-ratio"],
    )
    def test_the_missing_flag_takes_the_config_default(self, argv, expected):
        result = self._run_capturing(argv)

        assert (result["ratio"], result["size"]) == expected


# ── Runner nested lookup (SC-1, SC-2) ──────────────────────────────────────


class TestRunnerSizeLookup:
    """Verify runner.run_batch() resolves dimensions from ratio+size."""

    def _capture_request(self, ratio, size, width_override=None, height_override=None):
        """Run a single-prompt batch and capture the GenerationRequest."""
        from zvisiongenerator.core.types import StageOutcome
        from zvisiongenerator.core.workflow import GenerationWorkflow
        from zvisiongenerator.image_runner import run_batch
        from zvisiongenerator.utils.image_model_detect import ImageModelInfo
        from tests._factories import _make_args

        captured = {}

        def _stage(request, artifacts):
            captured["request"] = request
            return StageOutcome.success

        wf = GenerationWorkflow(name="test", stages=[_stage])

        config = {
            "sizes": _sizes_config(),
            "generation": {"seed_min": 1, "seed_max": 100},
            "sharpening": {"normal": 0.8, "upscaled": 1.2, "pre_upscale": 0.4},
            "model_presets": {"zimage": {"supports_negative_prompt": True}},
        }
        args = _make_args(ratio=ratio, size=size, width=width_override, height=height_override)
        model_info = ImageModelInfo(family="zimage", is_distilled=False, size=None)

        with patch("zvisiongenerator.image_runner.build_workflow", return_value=wf):
            run_batch(MagicMock(), MagicMock(spec=[]), {"s": [("cat", None)]}, config, args, model_info=model_info, plan=_make_plan({"s": [("cat", None)]}, args, config=config))

        return captured["request"]

    @pytest.mark.parametrize(
        ("ratio", "size", "expected"),
        [("16:9", "l", (1920, 1088)), ("2:3", "m", (832, 1216)), ("1:1", "m", (1024, 1024)), ("2:3", "xl", (1280, 1920))],
    )
    def test_ratio_and_size_resolve_to_the_preset_dimensions(self, ratio, size, expected):
        req = self._capture_request(ratio, size)

        assert (req.width, req.height) == expected

    def test_width_override_takes_priority(self):
        """SC-5: --width overrides preset width."""
        req = self._capture_request("16:9", "l", width_override=800)
        assert req.width == 800
        assert req.height == 1088  # height from preset

    def test_height_override_takes_priority(self):
        """SC-5: --height overrides preset height."""
        req = self._capture_request("16:9", "l", height_override=600)
        assert req.width == 1920  # width from preset
        assert req.height == 600

    def test_both_overrides_ignore_preset(self):
        """SC-5: --width + --height override both preset dimensions."""
        req = self._capture_request("16:9", "l", width_override=800, height_override=600)
        assert req.width == 800
        assert req.height == 600

    def test_ratio_and_size_stored_on_request(self):
        """GenerationRequest carries ratio and size metadata when using presets."""
        req = self._capture_request("9:16", "xs")
        assert req.ratio == "9:16"
        assert req.size == "xs"

    def test_ratio_and_size_none_with_explicit_dims(self):
        """GenerationRequest has None ratio/size when explicit dims used."""
        req = self._capture_request("9:16", "xs", width_override=800, height_override=600)
        assert req.ratio is None
        assert req.size is None


# ── Runner filename integration (SC-9) ──────────────────────────────────────


class TestRunnerFilenameIntegration:
    """Runner names each file after its prompt set and the time."""

    def _capture_filename(self, width_override=None, height_override=None):
        """Run batch and capture the filename base from WorkingArtifacts."""
        from zvisiongenerator.core.types import StageOutcome
        from zvisiongenerator.core.workflow import GenerationWorkflow
        from zvisiongenerator.image_runner import run_batch
        from zvisiongenerator.utils.image_model_detect import ImageModelInfo
        from tests._factories import _make_args

        captured = {}

        def _stage(request, artifacts):
            captured["filename"] = artifacts.filename
            return StageOutcome.success

        wf = GenerationWorkflow(name="test", stages=[_stage])

        config = {
            "sizes": {"2:3": {"m": {"width": 832, "height": 1216}}},
            "generation": {"seed_min": 1, "seed_max": 100},
            "sharpening": {"normal": 0.8, "upscaled": 1.2, "pre_upscale": 0.4},
            "model_presets": {"zimage": {"supports_negative_prompt": True}},
        }
        args = _make_args(ratio="2:3", size="m", width=width_override, height=height_override)
        model_info = ImageModelInfo(family="zimage", is_distilled=False, size=None)

        with patch("zvisiongenerator.image_runner.build_workflow", return_value=wf):
            run_batch(MagicMock(), MagicMock(spec=[]), {"s": [("cat", None)]}, config, args, model_info=model_info, plan=_make_plan({"s": [("cat", None)]}, args, config=config))

        return captured["filename"]

    def test_filename_is_set_name_and_timestamp(self):
        """Dimensions and settings live in the embedded metadata, not the filename."""
        filename = self._capture_filename(width_override=800, height_override=600)
        assert re.fullmatch(r"s_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}", filename)
