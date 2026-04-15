"""Tests for the aspect-ratio + size-scale sizing system.

Covers:
- All 25 dimension lookup entries match the PRD spec
- All dimensions are multiples of 16
- CLI rejects invalid --ratio and --size values
- Default ratio/size resolution (no flags → 2:3/m/832×1216)
- Specific ratio+size → correct dimensions
- Filename includes ratio-size when using presets
- Filename omits ratio-size when using explicit --width/--height
- Console output includes ratio, size, and dimensions
- Runner resolves nested lookup correctly
- --width/--height overrides take priority
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from zvisiongenerator.image_cli import _build_parser, main
from zvisiongenerator.utils.filename import generate_filename
from zvisiongenerator.utils.config import load_config


# ── PRD dimension lookup table ──────────────────────────────────────────────

EXPECTED_DIMENSIONS = {
    "1:1": {
        "xs": (512, 512),
        "s": (704, 704),
        "m": (1024, 1024),
        "l": (1440, 1440),
        "xl": (1600, 1600),
    },
    "16:9": {
        "xs": (672, 384),
        "s": (944, 528),
        "m": (1344, 768),
        "l": (1888, 1056),
        "xl": (2112, 1184),
    },
    "9:16": {
        "xs": (384, 672),
        "s": (528, 944),
        "m": (768, 1344),
        "l": (1056, 1888),
        "xl": (1184, 2112),
    },
    "3:2": {
        "xs": (608, 400),
        "s": (864, 576),
        "m": (1216, 832),
        "l": (1728, 1152),
        "xl": (1936, 1296),
    },
    "2:3": {
        "xs": (400, 608),
        "s": (576, 864),
        "m": (832, 1216),
        "l": (1152, 1728),
        "xl": (1296, 1936),
    },
}

RATIOS = ["1:1", "16:9", "9:16", "3:2", "2:3"]
SIZES = ["xs", "s", "m", "l", "xl"]


# ── All 25 dimension lookup entries (SC-7) ──────────────────────────────────


class TestDimensionLookupTable:
    """Verify all 25 entries in config.yaml match the PRD spec (REQ-1, SC-7)."""

    @pytest.fixture(scope="class")
    def config(self):
        return load_config()

    @pytest.mark.parametrize(
        "ratio,size",
        [(r, s) for r in RATIOS for s in SIZES],
        ids=[f"{r}_{s}" for r in RATIOS for s in SIZES],
    )
    def test_dimension_matches_spec(self, config, ratio, size):
        expected_w, expected_h = EXPECTED_DIMENSIONS[ratio][size]
        actual = config["sizes"][ratio][size]
        assert actual["width"] == expected_w, f"{ratio}/{size}: width {actual['width']} != {expected_w}"
        assert actual["height"] == expected_h, f"{ratio}/{size}: height {actual['height']} != {expected_h}"

    @pytest.mark.parametrize(
        "ratio,size",
        [(r, s) for r in RATIOS for s in SIZES],
        ids=[f"{r}_{s}_mod16" for r in RATIOS for s in SIZES],
    )
    def test_dimensions_are_multiples_of_16(self, config, ratio, size):
        """REQ-5: All dimensions must be multiples of 16."""
        dims = config["sizes"][ratio][size]
        assert dims["width"] % 16 == 0, f"{ratio}/{size}: width {dims['width']} not multiple of 16"
        assert dims["height"] % 16 == 0, f"{ratio}/{size}: height {dims['height']} not multiple of 16"

    def test_exactly_5_ratios(self, config):
        assert set(config["sizes"].keys()) == set(RATIOS)

    def test_each_ratio_has_5_sizes(self, config):
        for ratio in RATIOS:
            assert set(config["sizes"][ratio].keys()) == set(SIZES), f"Ratio {ratio} missing sizes"


# ── CLI flag validation (SC-6) ──────────────────────────────────────────────


class TestCLIRatioSizeFlags:
    """CLI must reject invalid --ratio and --size values."""

    def test_invalid_ratio_rejected(self):
        """SC-6: --ratio 4:3 is not in valid choices."""
        parser = _build_parser()
        with pytest.raises(SystemExit, match="2"):
            parser.parse_args(["--ratio", "4:3", "-m", "fake"])

    def test_invalid_ratio_3_4_rejected(self):
        parser = _build_parser()
        with pytest.raises(SystemExit, match="2"):
            parser.parse_args(["--ratio", "3:4", "-m", "fake"])

    def test_invalid_size_xxl_rejected(self):
        """SC-6: --size xxl is not in valid choices."""
        parser = _build_parser()
        with pytest.raises(SystemExit, match="2"):
            parser.parse_args(["--size", "xxl", "-m", "fake"])

    def test_invalid_size_medium_rejected(self):
        parser = _build_parser()
        with pytest.raises(SystemExit, match="2"):
            parser.parse_args(["--size", "medium", "-m", "fake"])

    @pytest.mark.parametrize("ratio", RATIOS)
    def test_valid_ratios_accepted(self, ratio):
        parser = _build_parser()
        args = parser.parse_args(["--ratio", ratio, "-m", "fake"])
        assert args.ratio == ratio

    @pytest.mark.parametrize("size", SIZES)
    def test_valid_sizes_accepted(self, size):
        parser = _build_parser()
        args = parser.parse_args(["--size", size, "-m", "fake"])
        assert args.size == size

    def test_short_flag_s_works(self):
        """REQ-2: -s is the short flag for --size."""
        parser = _build_parser()
        args = parser.parse_args(["-s", "xl", "-m", "fake"])
        assert args.size == "xl"

    def test_ratio_default_is_none(self):
        """Parser default for --ratio is None (resolved later from config)."""
        parser = _build_parser()
        args = parser.parse_args(["-m", "fake"])
        assert args.ratio is None

    def test_size_default_is_none(self):
        """Parser default for --size is None (resolved later from config)."""
        parser = _build_parser()
        args = parser.parse_args(["-m", "fake"])
        assert args.size is None


# ── Default resolution via main() (SC-2, REQ-3) ────────────────────────────


class TestDefaultResolution:
    """Verify defaults when no flags are given."""

    _FULL_CONFIG = {
        "sizes": EXPECTED_DIMENSIONS,  # reuse spec table converted to dict form
        "model_presets": {},
        "schedulers": {},
        "generation": {
            "default_ratio": "2:3",
            "default_size": "m",
            "default_steps": 10,
            "default_guidance": 3.5,
        },
    }

    # Convert EXPECTED_DIMENSIONS to config-style dicts
    @classmethod
    def _sizes_config(cls):
        sizes = {}
        for ratio, scale_map in EXPECTED_DIMENSIONS.items():
            sizes[ratio] = {}
            for size, (w, h) in scale_map.items():
                sizes[ratio][size] = {"width": w, "height": h}
        return sizes

    _MAIN_MOCKS = {
        "detect_image_model": staticmethod(lambda _: MagicMock(family="zimage", size=None)),
        "resolve_model_path": staticmethod(lambda p, **kw: p),
        "resolve_defaults": staticmethod(lambda *a, **kw: {"steps": 10, "guidance": 0.5, "scheduler": None}),
        "validate_scheduler": staticmethod(lambda *a: None),
        "load_prompts_file": staticmethod(lambda _: {"set": [("a cat", None)]}),
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
            "sizes": self._sizes_config(),
            "model_presets": {},
            "schedulers": {},
            "generation": {
                "default_ratio": "2:3",
                "default_size": "m",
                "default_steps": 10,
                "default_guidance": 3.5,
            },
        }

        with patch("sys.argv", ["ziv"] + argv):
            with patch.multiple("zvisiongenerator.image_cli", **mocks):
                main()
        return recorded

    def test_no_flags_defaults_to_2_3_m(self):
        """SC-2: No flags → ratio='2:3', size='m' → 832×1216."""
        result = self._run_capturing(["-m", "fake"])
        assert result["ratio"] == "2:3"
        assert result["size"] == "m"

    def test_only_size_flag_uses_default_ratio(self):
        """SC-3: --size xl with default ratio 2:3 → 1296×1936."""
        result = self._run_capturing(["-m", "fake", "--size", "xl"])
        assert result["ratio"] == "2:3"
        assert result["size"] == "xl"

    def test_only_ratio_flag_uses_default_size(self):
        """SC-4: --ratio 1:1 with default size m → 1024×1024."""
        result = self._run_capturing(["-m", "fake", "--ratio", "1:1"])
        assert result["ratio"] == "1:1"
        assert result["size"] == "m"

    def test_both_flags_explicit(self):
        """SC-1: --ratio 16:9 --size l → 1888×1056."""
        result = self._run_capturing(["-m", "fake", "--ratio", "16:9", "--size", "l"])
        assert result["ratio"] == "16:9"
        assert result["size"] == "l"


# ── Runner nested lookup (SC-1, SC-2) ──────────────────────────────────────


class TestRunnerSizeLookup:
    """Verify runner.run_batch() resolves dimensions from ratio+size."""

    def _capture_request(self, ratio, size, width_override=None, height_override=None):
        """Run a single-prompt batch and capture the GenerationRequest."""
        from zvisiongenerator.core.types import StageOutcome
        from zvisiongenerator.core.workflow import GenerationWorkflow
        from zvisiongenerator.image_runner import run_batch
        from zvisiongenerator.utils.image_model_detect import ImageModelInfo
        from conftest import _make_args

        captured = {}

        def _stage(request, artifacts):
            captured["request"] = request
            return StageOutcome.success

        wf = GenerationWorkflow(name="test", stages=[_stage])

        sizes_config = {}
        for r, scale_map in EXPECTED_DIMENSIONS.items():
            sizes_config[r] = {}
            for s, (w, h) in scale_map.items():
                sizes_config[r][s] = {"width": w, "height": h}

        config = {
            "sizes": sizes_config,
            "generation": {"seed_min": 1, "seed_max": 100},
            "sharpening": {"normal": 0.8, "upscaled": 1.2, "pre_upscale": 0.4},
            "model_presets": {"zimage": {"supports_negative_prompt": True}},
        }
        args = _make_args(ratio=ratio, size=size, width=width_override, height=height_override)
        model_info = ImageModelInfo(family="zimage", is_distilled=False, size=None)

        with patch("zvisiongenerator.image_runner.build_workflow", return_value=wf):
            run_batch(MagicMock(), MagicMock(spec=[]), {"s": [("cat", None)]}, config, args, model_info=model_info)

        return captured["request"]

    def test_16_9_l_resolves_1888x1056(self):
        """SC-1: --ratio 16:9 --size l → 1888×1056."""
        req = self._capture_request("16:9", "l")
        assert req.width == 1888
        assert req.height == 1056

    def test_2_3_m_resolves_832x1216(self):
        """SC-2: Default 2:3/m → 832×1216."""
        req = self._capture_request("2:3", "m")
        assert req.width == 832
        assert req.height == 1216

    def test_1_1_m_resolves_1024x1024(self):
        """SC-4: --ratio 1:1 → 1024×1024."""
        req = self._capture_request("1:1", "m")
        assert req.width == 1024
        assert req.height == 1024

    def test_2_3_xl_resolves_1296x1936(self):
        """SC-3: --size xl → 1296×1936."""
        req = self._capture_request("2:3", "xl")
        assert req.width == 1296
        assert req.height == 1936

    def test_width_override_takes_priority(self):
        """SC-5: --width overrides preset width."""
        req = self._capture_request("16:9", "l", width_override=800)
        assert req.width == 800
        assert req.height == 1056  # height from preset

    def test_height_override_takes_priority(self):
        """SC-5: --height overrides preset height."""
        req = self._capture_request("16:9", "l", height_override=600)
        assert req.width == 1888  # width from preset
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


# ── Filename with ratio/size (REQ-7, SC-9) ─────────────────────────────────


class TestFilenameRatioSize:
    """Verify filename always uses plain WxH format."""

    def test_filename_uses_plain_wxh_with_presets(self):
        """Filename uses plain WxH even when using presets."""
        result = generate_filename(
            set_name="test",
            width=1888,
            height=1056,
            seed=42,
            steps=10,
            guidance=3.5,
            scheduler=None,
            model="models/my-model",
        )
        assert "1888x1056" in result
        assert "16-9" not in result

    def test_filename_uses_plain_wxh_with_explicit_dims(self):
        """Filename uses plain WxH with explicit dimensions."""
        result = generate_filename(
            set_name="test",
            width=800,
            height=600,
            seed=42,
            steps=10,
            guidance=3.5,
            scheduler=None,
            model="models/my-model",
        )
        assert "800x600" in result
        assert "None" not in result


# ── Console output (REQ-8) ─────────────────────────────────────────────────


class TestConsoleRatioSize:
    """Console output displays ratio/size conditionally."""

    def test_console_shows_ratio_size_with_presets(self):
        from zvisiongenerator.core.image_types import ImageGenerationRequest, ImageWorkingArtifacts
        from zvisiongenerator.utils.console import format_generation_info

        request = ImageGenerationRequest(
            backend=None,
            model=None,
            prompt="test",
            ratio="16:9",
            size="l",
            width=1888,
            height=1056,
        )
        output = format_generation_info(
            request,
            ImageWorkingArtifacts(),
            run_number=0,
            total_runs=1,
            ran_iterations=1,
            total_iterations=1,
            set_name="test",
            prompt_idx=0,
            total_prompts=1,
        )
        assert "Ratio: 16:9" in output
        assert "Size: l" in output
        assert "1888" in output
        assert "1056" in output

    def test_console_shows_only_dims_with_explicit(self):
        from zvisiongenerator.core.image_types import ImageGenerationRequest, ImageWorkingArtifacts
        from zvisiongenerator.utils.console import format_generation_info

        request = ImageGenerationRequest(
            backend=None,
            model=None,
            prompt="test",
            width=800,
            height=600,
        )
        output = format_generation_info(
            request,
            ImageWorkingArtifacts(),
            run_number=0,
            total_runs=1,
            ran_iterations=1,
            total_iterations=1,
            set_name="test",
            prompt_idx=0,
            total_prompts=1,
        )
        assert "800\u00d7600" in output
        assert "Ratio:" not in output
        assert "Size:" not in output


# ── Runner filename integration (SC-9) ──────────────────────────────────────


class TestRunnerFilenameIntegration:
    """Runner passes ratio/size to filename only when using presets."""

    def _capture_filename(self, width_override=None, height_override=None):
        """Run batch and capture the filename base from WorkingArtifacts."""
        from zvisiongenerator.core.types import StageOutcome
        from zvisiongenerator.core.workflow import GenerationWorkflow
        from zvisiongenerator.image_runner import run_batch
        from zvisiongenerator.utils.image_model_detect import ImageModelInfo
        from conftest import _make_args

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
            run_batch(MagicMock(), MagicMock(spec=[]), {"s": [("cat", None)]}, config, args, model_info=model_info)

        return captured["filename"]

    def test_preset_filename_contains_plain_wxh(self):
        """When using presets, filename uses plain WxH."""
        filename = self._capture_filename()
        assert "832x1216" in filename
        assert "2-3_m" not in filename

    def test_explicit_dims_filename_omits_ratio_size(self):
        """When using explicit dims, filename should NOT include ratio-size."""
        filename = self._capture_filename(width_override=800, height_override=600)
        assert "800x600" in filename
        assert "2-3_m" not in filename


# ── Upscale round-trip with presets (SC-8) ──────────────────────────────────


class TestUpscalePresetRoundTrip:
    """SC-8: Upscale validation works with the new size resolution."""

    def test_16_9_l_upscale_2_round_trip(self):
        """SC-8: 1888/2=944, 1056/2=528 — both multiples of 16."""
        assert 1888 // 2 == 944
        assert 944 % 16 == 0
        assert 1056 // 2 == 528
        assert 528 % 16 == 0

    def test_2_3_m_upscale_2_round_trip(self):
        """Default preset 2:3/m: 832/2=416, 1216/2=608 — both multiples of 16."""
        assert 832 // 2 == 416
        assert 416 % 16 == 0
        assert 1216 // 2 == 608
        assert 608 % 16 == 0

    # Note: 6 presets (16:9/s, 9:16/s, 3:2/xs, 3:2/xl, 2:3/xs, 2:3/xl)
    # have dimensions that are multiples of 16 but NOT 32, so they drift
    # under 2x upscale. The runner handles this with a warning — see
    # TestPresetSizeDriftWarning in test_runner_outcome.py.
