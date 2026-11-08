"""Tests for config loading, parity with hardcoded values, and resolution logic."""

from __future__ import annotations

import re
from copy import deepcopy

import pytest

from zvisiongenerator.utils.ltx_facts import ltx_load_options
from zvisiongenerator.utils.config import _deep_merge, get_variant_key, load_config, resolve_defaults, resolve_video_defaults, validate_video_size_choice, video_load_options
from zvisiongenerator.utils.image_model_detect import ImageModelInfo
from zvisiongenerator.utils.model_metadata import alias_map, image_model_family, model_capabilities, model_display_name
from zvisiongenerator.utils.paths import resolve_model_path


_SEEDVR2_BLOCK = {
    "model": "auto",
    "softness": 0.75,
    "video_softness": 0.0,
    "reference_megapixels": 16,
}


def _write_packaged_config(tmp_path, data):
    import yaml

    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump(data), encoding="utf-8")
    return config_file


# ---------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# ── Nested config validation ──
# ---------------------------------------------------------------------------


_GENERATION = {"default_steps": 10, "default_guidance": 3.5}


def _load_packaged(tmp_path, data, *, user_dir=None):
    """Load ``data`` as the packaged config, with no user override unless ``user_dir`` has one."""
    from unittest.mock import patch

    config_file = _write_packaged_config(tmp_path, data)
    with patch("importlib.resources.files") as mock_files, patch("zvisiongenerator.utils.config.get_ziv_data_dir") as mock_dir:
        mock_dir.return_value = user_dir or tmp_path / "no-user-config"
        mock_files.return_value.joinpath.return_value = config_file
        return load_config()


class TestNestedConfigValidation:
    """Ensure load_config rejects configs with bad nested values."""

    @pytest.mark.parametrize(
        ("sizes", "generation", "match"),
        [
            ({"1:1": {"m": {"width": 100, "height": 100}}}, {**_GENERATION, "default_steps": "ten"}, "default_steps.*integer"),
            ({"1:1": {"m": {"width": 100, "height": 100}}}, {**_GENERATION, "default_guidance": "high"}, "default_guidance.*number"),
            ({"1:1": {"m": {"width": "wide", "height": 100}}}, _GENERATION, r"sizes\.1:1\.m\.width.*integer"),
            ({"1:1": "medium"}, _GENERATION, r"sizes\.1:1.*mapping.*size"),
            ({"1:1": {"m": {"width": 1024, "height": 1024}}}, {**_GENERATION, "default_ratio": "4:3"}, r"default_ratio.*4:3.*not a valid ratio"),
            ({"1:1": {"m": {"width": 1024, "height": 1024}}}, {**_GENERATION, "default_ratio": "1:1", "default_size": "xxl"}, r"default_size.*xxl.*not a valid size.*1:1"),
        ],
        ids=["steps-not-int", "guidance-not-number", "width-not-int", "size-not-mapping", "unknown-ratio", "unknown-size"],
    )
    def test_a_bad_size_or_generation_value_names_the_key(self, tmp_path, sizes, generation, match):
        data = {"sizes": sizes, "generation": generation, "sharpening": {}, "upscale": {}, "model_presets": {}}

        with pytest.raises(ValueError, match=match):
            _load_packaged(tmp_path, data)


def test_packaged_config_resolves_ideo_to_its_models_folder_on_every_platform(monkeypatch, tmp_path):
    from zvisiongenerator.utils import config as config_module

    monkeypatch.setattr(config_module, "get_ziv_data_dir", lambda: tmp_path / ".ziv")
    monkeypatch.setenv("ZIV_DATA_DIR", str(tmp_path / ".ziv"))

    config = load_config()

    for platform_key in ("darwin", "win32", "linux"):
        assert resolve_model_path("ideo", aliases=alias_map(config), platform_key=platform_key) == str(tmp_path / ".ziv" / "models" / "ideo")
    assert image_model_family("ideo") == "ideogram4"


def test_metadata_pins_every_ideogram4_instant_source():
    from zvisiongenerator.utils.ideogram4_instant import instant_source

    formats = {}
    for platform_key in ("darwin", "win32", "linux"):
        source = instant_source(platform_key)
        assert re.fullmatch(r"[0-9a-f]{40}", source.transformer_revision)
        assert source.components_revision is not None and re.fullmatch(r"[0-9a-f]{40}", source.components_revision)
        formats[platform_key] = source.transformer_format
    assert formats == {"darwin": "fp8", "win32": "bf16", "linux": "bf16"}


def test_packaged_config_ideogram4_preset_is_instant(packaged_config):
    defaults = resolve_defaults(ImageModelInfo(family="ideogram4", is_distilled=False, size=None), packaged_config, {}, "diffusers")

    assert (defaults["steps"], defaults["guidance"], defaults["supports_quantize"]) == (8, 1.0, True)


def test_packaged_config_ships_krea2_alias_and_defaults(packaged_config):
    config = packaged_config
    turbo = ImageModelInfo(family="krea2", is_distilled=False, size=None)

    for platform_key in ("darwin", "win32", "linux"):
        assert resolve_model_path("krea2", aliases=alias_map(config), platform_key=platform_key) == "unsloth/Krea-2-Turbo"
    assert image_model_family("krea2") == "krea2"
    for backend_name in ("mflux", "diffusers"):
        defaults = resolve_defaults(turbo, config, {}, backend_name)
        assert (defaults["steps"], defaults["guidance"], defaults["scheduler"]) == (8, 1.0, "beta")
        assert defaults["supports_negative_prompt"] is False
        assert defaults["supports_scheduler"] is True
        assert (defaults["upscale_steps"], defaults["upscale_sharpen"]) == (5, False)


def test_packaged_config_klein9b_uses_ungated_mirror(packaged_config):
    aliases = alias_map(packaged_config)

    assert aliases["klein9b"] == "unsloth/FLUX.2-klein-9B"
    for platform_key in ("darwin", "win32", "linux"):
        resolved = set()
        for name in aliases:
            try:
                resolved.add(resolve_model_path(name, aliases=aliases, platform_key=platform_key))
            except ValueError:  # an alias that is not available on this platform
                continue
        assert "black-forest-labs/FLUX.2-klein-9B" not in resolved


def test_packaged_config_krea2_capabilities(packaged_config):
    caps = model_capabilities("krea2")

    assert (caps["supports_img2img"], caps["supports_upscale"]) == (True, True)


def test_packaged_config_references_only_on_klein(packaged_config):
    assert model_capabilities("flux2_klein")["max_reference_images"] == 4
    assert model_capabilities("flux2_klein", "distilled")["max_reference_images"] == 4
    assert model_capabilities("zimage")["max_reference_images"] == 0


# ---------------------------------------------------------------------------
# _deep_merge
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("base", "override", "expected"),
    [
        ({"a": 1, "b": 2}, {"b": 99}, {"a": 1, "b": 99}),
        ({"a": 1}, {"b": 2}, {"a": 1, "b": 2}),
        ({"a": {"x": 1, "y": 2}}, {"a": {"y": 99, "z": 3}}, {"a": {"x": 1, "y": 99, "z": 3}}),
        ({"a": {"b": {"c": {"d": 1}}}}, {"a": {"b": {"c": {"d": 2, "e": 3}}}}, {"a": {"b": {"c": {"d": 2, "e": 3}}}}),
        ({"a": {"x": 1}}, {"a": 42}, {"a": 42}),
        ({"a": 42}, {"a": {"x": 1}}, {"a": {"x": 1}}),
        ({"a": [1, 2, 3]}, {"a": [4, 5]}, {"a": [4, 5]}),
        ({"a": 1}, {}, {"a": 1}),
    ],
    ids=["flat", "new-key", "nested", "deeply-nested", "dict-to-scalar", "scalar-to-dict", "list-replaced", "empty"],
)
def test_deep_merge(base, override, expected):
    _deep_merge(base, override)

    assert base == expected


# ---------------------------------------------------------------------------
# get_variant_key
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("family", "is_distilled", "size", "expected"),
    [("flux2_klein", True, "4b", "distilled"), ("flux2_klein", False, "9b", "base"), ("zimage", False, None, None), ("flux1", False, None, None), ("unknown", False, None, None)],
)
def test_get_variant_key(family, is_distilled, size, expected):
    assert get_variant_key(ImageModelInfo(family=family, is_distilled=is_distilled, size=size)) == expected


# ---------------------------------------------------------------------------
# resolve_defaults
# ---------------------------------------------------------------------------


class TestResolveDefaults:
    _SYNTHETIC_CONFIG = {
        "generation": {
            "default_steps": 15,
            "default_guidance": 4.0,
        },
        "model_presets": {
            "zimage": {
                "default_steps": 8,
                "default_guidance": 0.7,
                "upscale": {
                    "default_upscale_steps": 8,
                },
                "default_scheduler": {
                    "mflux": "beta",
                    "diffusers": None,
                },
            },
            "flux2_klein": {
                "default_scheduler": {
                    "mflux": None,
                    "diffusers": None,
                },
                "variants": {
                    "distilled": {
                        "default_steps": 6,
                        "default_guidance": 1.2,
                    },
                    "base": {
                        "default_steps": 40,
                        "default_guidance": 1.8,
                    },
                },
            },
            "flux2": {
                "default_steps": 25,
                "default_guidance": 4.5,
                "default_scheduler": {
                    "mflux": None,
                    "diffusers": None,
                },
            },
            "flux1": {
                "default_steps": 22,
                "default_guidance": 3.0,
                "default_scheduler": {
                    "mflux": None,
                    "diffusers": None,
                },
            },
            "ideogram4": {
                "supports_img2img": False,
                "default_steps": 20,
                "default_guidance": 7.0,
                "default_scheduler": {
                    "mflux": None,
                    "diffusers": None,
                },
            },
        },
    }

    @pytest.fixture(autouse=True)
    def _load(self):
        self.config = deepcopy(self._SYNTHETIC_CONFIG)

    @pytest.mark.parametrize(
        ("family", "is_distilled", "size", "backend", "expected"),
        [
            (
                "zimage",
                False,
                None,
                "mflux",
                {
                    "steps": 8,
                    "guidance": 0.7,
                    "upscale_steps": 8,
                    "scheduler": "beta",
                    "supports_quantize": True,
                    "supports_img2img": True,
                    "supports_upscale": True,
                    "supports_json_prompt": False,
                    "supports_first_sigma": False,
                    "dimension_min": 16,
                    "dimension_max": None,
                    "dimension_step": 16,
                },
            ),
            ("zimage", False, None, "diffusers", {"steps": 8, "guidance": 0.7, "upscale_steps": 8, "scheduler": None}),
            ("flux2_klein", True, "4b", "mflux", {"steps": 6, "guidance": 1.2, "scheduler": None}),
            ("flux2_klein", False, "9b", "mflux", {"steps": 40, "guidance": 1.8}),
            ("flux1", False, None, "mflux", {"steps": 22, "guidance": 3.0}),
            ("flux2", False, None, "mflux", {"steps": 25, "guidance": 4.5}),
            (
                "ideogram4",
                False,
                None,
                "mflux",
                {
                    "steps": 20,
                    "guidance": 7.0,
                    "scheduler": None,
                    "supports_negative_prompt": False,
                    "supports_quantize": True,
                    "supports_img2img": False,
                    "supports_upscale": False,
                    "supports_json_prompt": True,
                    "supports_first_sigma": True,
                    "dimension_min": 256,
                    "dimension_max": 2048,
                    "dimension_step": 16,
                },
            ),
            ("unknown", False, None, "mflux", {"steps": 15, "guidance": 4.0, "scheduler": None}),
        ],
        ids=["zimage-mflux", "zimage-diffusers", "klein-distilled", "klein-base", "flux1", "flux2", "ideogram4", "family-without-a-preset"],
    )
    def test_the_family_preset_resolves(self, family, is_distilled, size, backend, expected):
        result = resolve_defaults(ImageModelInfo(family=family, is_distilled=is_distilled, size=size), self.config, {}, backend)

        assert {key: result[key] for key in expected} == expected

    @pytest.mark.parametrize(
        ("family", "is_distilled", "size", "overrides", "expected"),
        [
            ("flux2_klein", True, "4b", {"steps": 20}, {"steps": 20, "guidance": 1.2}),
            ("zimage", False, None, {"upscale_steps": 20}, {"upscale_steps": 20}),
            ("zimage", False, None, {"steps": 30, "guidance": 2.0, "scheduler": "euler"}, {"steps": 30, "guidance": 2.0, "scheduler": "euler"}),
            ("ideogram4", False, None, {"steps": 12, "guidance": 3.5}, {"steps": 12, "guidance": 3.5}),
            ("zimage", False, None, {"steps": None}, {"steps": 8}),
        ],
        ids=["klein-steps", "upscale-steps", "several", "ideogram4", "none-is-ignored"],
    )
    def test_cli_overrides_win_over_the_preset(self, family, is_distilled, size, overrides, expected):
        result = resolve_defaults(ImageModelInfo(family=family, is_distilled=is_distilled, size=size), self.config, overrides, "mflux")

        assert {key: result[key] for key in expected} == expected

    def test_string_default_scheduler_does_not_crash(self):
        """resolve_defaults should not crash if default_scheduler is a string."""
        config = deepcopy(self._SYNTHETIC_CONFIG)
        # Inject a string where a dict is expected
        preset = config["model_presets"]["zimage"]
        preset["default_scheduler"] = "beta"
        info = ImageModelInfo(family="zimage", is_distilled=False, size=None)
        result = resolve_defaults(info, config, {}, "mflux")
        assert result["scheduler"] is None

    def test_klein_variant_missing_key_falls_to_family(self):
        """If variant exists but has no 'default_steps', family default is used."""
        cfg = deepcopy(self.config)
        del cfg["model_presets"]["flux2_klein"]["variants"]["distilled"]["default_steps"]
        info = ImageModelInfo(family="flux2_klein", is_distilled=True, size="4b")
        result = resolve_defaults(info, cfg, {}, "mflux")
        assert result["steps"] == cfg["generation"]["default_steps"]


# ---------------------------------------------------------------------------
# Config schema validation after merge
# ---------------------------------------------------------------------------


class TestConfigSchemaValidation:
    """load_config() should reject configs where known sections have wrong types."""

    @staticmethod
    def _load_with_user_config(tmp_path, monkeypatch, text):
        (tmp_path / "config.yaml").write_text(text)
        monkeypatch.setattr("zvisiongenerator.utils.config.get_ziv_data_dir", lambda: tmp_path)
        return load_config()

    @pytest.mark.parametrize(
        ("text", "match"),
        [
            ("sizes:\n  - bad\n", "Config section 'sizes' must be a mapping"),
            ("generation: 42\n", "Config section 'generation' must be a mapping"),
            ("seedvr2: 42\n", "Config section 'seedvr2' must be a mapping"),
            ("seedvr2:\n  softness: 2\n", "seedvr2.softness"),
        ],
        ids=["sizes-list", "generation-scalar", "seedvr2-scalar", "seedvr2-softness"],
    )
    def test_a_bad_user_override_is_rejected(self, tmp_path, monkeypatch, text, match):
        with pytest.raises(ValueError, match=match):
            self._load_with_user_config(tmp_path, monkeypatch, text)

    def test_seedvr2_model_override_is_merged(self, tmp_path, monkeypatch):
        cfg = self._load_with_user_config(tmp_path, monkeypatch, "seedvr2:\n  model: seedvr2-3b\n")
        assert cfg["seedvr2"]["model"] == "seedvr2-3b"
        assert cfg["seedvr2"]["softness"] == 0.75

    def test_valid_user_override_accepted(self, tmp_path, monkeypatch):
        cfg = self._load_with_user_config(tmp_path, monkeypatch, "generation:\n  default_steps: 99\n")
        assert cfg["generation"]["default_steps"] == 99


def _platform_config(**sections):
    """Return a minimal config with every section the platform checks read, overridden by ``sections``."""
    return {
        "sizes": {"1:1": {"m": {"width": 100, "height": 100}}},
        "generation": _GENERATION,
        "sharpening": {},
        "upscale": {},
        "contrast": {},
        "saturation": {},
        "model_aliases": {},
        "model_presets": {},
        "video_sizes": {},
        "video_generation": {},
        "video_model_presets": {},
        **sections,
    }


class TestPlatformConfigValidation:
    def test_model_aliases_section_must_be_mapping(self, tmp_path):
        with pytest.raises(ValueError, match="Config section 'model_aliases' must be a mapping"):
            _load_packaged(tmp_path, _platform_config(model_aliases=["ltx"]))

    def test_sharpening_amounts_above_the_cap_are_rejected(self, tmp_path):
        data = {"sizes": {"1:1": {"m": {"width": 100, "height": 100}}}, "generation": _GENERATION, "sharpening": {"normal": 1.0, "upscaled": 2.0}}

        with pytest.raises(ValueError, match="sharpening.upscaled"):
            _load_packaged(tmp_path, data)

    @pytest.mark.parametrize(
        "alias_value",
        [
            "Tongyi-MAI/Z-Image-Turbo",
            {"win32": "dg845/LTX-2.3-Diffusers", "linux": "dg845/LTX-2.3-Diffusers"},
            {"darwin": "dgrauet/ltx-2.3-mlx-q4"},
        ],
    )
    def test_polymorphic_alias_values_accept_supported_shapes(self, tmp_path, alias_value):
        cfg = _load_packaged(tmp_path, _platform_config(seedvr2=_SEEDVR2_BLOCK, model_aliases={"ltx": alias_value}))

        assert cfg["model_aliases"]["ltx"] == alias_value

    @pytest.mark.parametrize(
        ("alias_value", "match"),
        [
            (42, r"config 'model_aliases\.ltx' must be a string or mapping"),
            ({1: "repo/model"}, r"platform keys must be strings"),
            ({"win32": 42}, r"config 'model_aliases\.ltx\.win32' must be a string"),
            ({"win32": {"message": "Use ltx-8 instead."}}, r"config 'model_aliases\.ltx\.win32' must be a string"),
        ],
    )
    def test_polymorphic_alias_values_reject_invalid_shapes(self, tmp_path, alias_value, match):
        with pytest.raises(ValueError, match=match):
            _load_packaged(tmp_path, _platform_config(model_aliases={"ltx": alias_value}))


# ---------------------------------------------------------------------------
# Scheduler resolution via runner helper
# ---------------------------------------------------------------------------


class TestSchedulerResolution:
    """resolve_scheduler_class resolves scheduler names to backend class paths."""

    def test_beta_scheduler_class(self):
        from zvisiongenerator.utils.config import resolve_scheduler_class

        assert resolve_scheduler_class("beta", "mflux") == "zvisiongenerator.schedulers.beta_scheduler.BetaScheduler"

    def test_beta_on_diffusers_receives_its_name(self):
        from zvisiongenerator.utils.config import resolve_scheduler_class

        assert resolve_scheduler_class("beta", "diffusers") == "beta"

    @pytest.mark.parametrize("backend", ["mflux", "diffusers"])
    def test_simple_resolves_to_the_model_own_schedule(self, backend):
        from zvisiongenerator.utils.config import resolve_scheduler_class

        assert resolve_scheduler_class("simple", backend) is None

    def test_resolve_unknown_scheduler_passthrough(self):
        from zvisiongenerator.utils.config import resolve_scheduler_class

        assert resolve_scheduler_class("euler", "mflux") == "euler"

    def test_resolve_none_scheduler(self):
        from zvisiongenerator.utils.config import resolve_scheduler_class

        assert resolve_scheduler_class(None, "mflux") is None


# ---------------------------------------------------------------------------
# Scheduler validation
# ---------------------------------------------------------------------------


class TestSchedulerValidation:
    """Verify validate_scheduler works."""

    def test_validate_known_scheduler(self):
        from zvisiongenerator.utils.config import validate_scheduler

        validate_scheduler("beta")  # Should not raise
        validate_scheduler("simple")

    def test_validate_none_scheduler(self):
        from zvisiongenerator.utils.config import validate_scheduler

        validate_scheduler(None)  # Should not raise

    def test_validate_unknown_scheduler(self):
        from zvisiongenerator.utils.config import validate_scheduler

        with pytest.raises(ValueError, match="Unknown scheduler"):
            validate_scheduler("nonexistent")


# ---------------------------------------------------------------------------
# Malformed YAML config
# ---------------------------------------------------------------------------


def test_malformed_yaml_config_raises_valueerror(tmp_path, monkeypatch):
    """Malformed YAML in user config must raise ValueError."""
    bad_config = tmp_path / "config.yaml"
    bad_config.write_text("sizes:\n  s:\n  bad: [unterminated", encoding="utf-8")

    monkeypatch.setattr(
        "zvisiongenerator.utils.config.get_ziv_data_dir",
        lambda: tmp_path,
    )

    with pytest.raises(ValueError, match="Failed to parse config file"):
        load_config()


# ---------------------------------------------------------------------------
# resolve_video_defaults
# ---------------------------------------------------------------------------


class TestResolveVideoDefaults:
    """Verify video config resolution: preset → global → CLI overrides."""

    _VIDEO_CONFIG = {
        "video_generation": {
            "default_ratio": "16:9",
            "default_size": "m",
        },
        "video_sizes": {
            "16:9": {
                "s": {"width": 512, "height": 288, "frames": 49},
                "m": {"width": 704, "height": 480, "frames": 65},
            },
            "9:16": {
                "s": {"width": 288, "height": 512, "frames": 49},
                "m": {"width": 480, "height": 704, "frames": 49},
            },
            "1:1": {
                "m": {"width": 512, "height": 512, "frames": 33},
            },
        },
        "video_model_presets": {
            "ltx": {
                "default_steps": 8,
            },
            "other": {
                "default_steps": 12,
            },
        },
    }

    def test_cli_overrides_take_precedence(self):
        overrides = {"steps": 10}
        result = resolve_video_defaults("ltx", self._VIDEO_CONFIG, overrides)
        assert result["steps"] == 10
        # Non-overridden values still come from preset
        assert result["width"] == 704

    def test_model_family_selects_preset_but_not_dimensions(self):
        result = resolve_video_defaults("other", self._VIDEO_CONFIG, {})
        assert result["steps"] == 12
        assert result["width"] == 704
        assert result["height"] == 480
        assert result["num_frames"] == 65

    def test_cli_none_values_ignored(self):
        overrides = {"steps": None}
        result = resolve_video_defaults("ltx", self._VIDEO_CONFIG, overrides)
        assert result["steps"] == 8

    def test_every_packaged_video_size_is_121_frames(self, packaged_config):
        sizes = packaged_config["video_sizes"]

        frames = {(ratio, size): dims["frames"] for ratio, entries in sizes.items() for size, dims in entries.items()}
        assert len(frames) == 15
        assert set(frames.values()) == {121}

    @pytest.mark.parametrize(
        ("overrides", "expected"),
        [
            ({"ratio": "9:16", "size": "m"}, {"width": 480, "height": 704, "num_frames": 49, "ratio": "9:16", "size": "m"}),
            ({"ratio": "16:9", "size": "s"}, {"width": 512, "height": 288, "num_frames": 49}),
            ({"ratio": "1:1", "size": "m"}, {"width": 512, "height": 512, "num_frames": 33}),
            # Explicit values override the preset's; the rest still come from it.
            ({"ratio": "16:9", "size": "m", "width": 640}, {"width": 640, "height": 480, "num_frames": 65}),
            ({"ratio": "16:9", "size": "m", "width": 640, "height": 360, "num_frames": 33}, {"width": 640, "height": 360, "num_frames": 33}),
        ],
        ids=["9:16-m", "16:9-s", "1:1-m", "width", "all-dimensions"],
    )
    def test_ratio_and_size_select_the_preset_and_explicit_dimensions_win(self, overrides, expected):
        result = resolve_video_defaults("ltx", self._VIDEO_CONFIG, overrides)

        assert {key: result[key] for key in expected} == expected


def test_sharpening_amounts_fill_defaults_and_follow_upscaled():
    from zvisiongenerator.utils.config import sharpening_amounts

    amounts = sharpening_amounts({"sharpening": {"upscaled": 1.3}})

    assert amounts["normal"] == 1.0
    assert amounts["existing_pre_upscale"] == 0.0
    assert amounts["existing_upscaled"] == 1.3
    assert sharpening_amounts({"sharpening": {"existing_upscaled": 1.1}})["existing_upscaled"] == 1.1


def test_model_capabilities_match_resolve_defaults():
    from zvisiongenerator.utils.model_metadata import model_capabilities

    config = {"generation": {"default_steps": 10, "default_guidance": 3.5}}
    capabilities = model_capabilities("ideogram4")
    resolved = resolve_defaults(ImageModelInfo(family="ideogram4", is_distilled=False, size=None), config, {}, "mflux")

    assert capabilities == {key: resolved[key] for key in capabilities}
    assert capabilities["dimension_max"] == 2048
    assert model_capabilities("unknown")["supports_upscale"] is True


# ---------------------------------------------------------------------------
# validate_video_size_choice
# ---------------------------------------------------------------------------


class TestValidateVideoSizeChoice:
    _CONFIG = {"video_sizes": {"16:9": {"m": {}, "l": {}}, "1:1": {"m": {}}}}

    def test_a_listed_ratio_and_size_pass(self):
        validate_video_size_choice(self._CONFIG, "16:9", "l")

    @pytest.mark.parametrize(("ratio", "size", "match"), [("4:3", "m", "ratio"), ("1:1", "l", "size")])
    def test_an_unlisted_ratio_or_size_is_refused(self, ratio, size, match):
        with pytest.raises(ValueError, match=match):
            validate_video_size_choice(self._CONFIG, ratio, size)

    def test_a_config_without_a_size_table_accepts_anything(self):
        validate_video_size_choice({}, "4:3", "xl")


class TestShippedVideoSizes:
    """Verify the shipped video size table suits LTX with and without --upscale 2."""

    def test_every_ratio_has_the_five_tiers(self, packaged_config):
        for ratio, tiers in packaged_config["video_sizes"].items():
            assert list(tiers) == ["xs", "s", "m", "l", "xl"], ratio

    def test_sizes_are_64_aligned_and_frames_are_8k_plus_1(self, packaged_config):
        for ratio, tiers in packaged_config["video_sizes"].items():
            for size, dims in tiers.items():
                assert dims["width"] % 64 == 0 and dims["height"] % 64 == 0, f"{ratio}/{size}"
                assert (dims["frames"] - 1) % 8 == 0, f"{ratio}/{size}"

    def test_the_default_size_is_listed(self, packaged_config):
        video = packaged_config["video_generation"]
        assert video["default_size"] in packaged_config["video_sizes"][video["default_ratio"]]


class TestVideoLoadOptionsGolden:
    """G4: ``video_load_options("ltx", ...)`` is exactly ``ltx_load_options`` for every LTX alias and both upscale settings."""

    @pytest.mark.parametrize(("name", "upscale"), [("ltx-4", False), ("ltx25-4", True)])
    def test_ltx_options_are_the_ltx_options(self, packaged_config, name, upscale):
        assert video_load_options("ltx", packaged_config, name, platform_key="linux", upscale=upscale) == ltx_load_options(name, "linux", upscale=upscale)


class TestVideoFamilyDispatch:
    """R5: load options and companions are asked of the family; default frames come from the family."""

    def test_a_family_without_load_options_is_refused(self):
        with pytest.raises(ValueError, match="No video load options for model family 'wan'"):
            video_load_options("wan", {}, "m", platform_key="linux", upscale=False)

    def test_a_family_without_load_options_is_refused_on_any_platform(self):
        with pytest.raises(ValueError, match="No video load options for model family 'zimage'"):
            video_load_options("zimage", {}, "m", platform_key="darwin", upscale=False)

    def _h3_config(self, **settings):
        return {"video_sizes": {"16:9": {"s": {"width": 768, "height": 448, "frames": 121}}}, "video_model_presets": {"h3": {"default_steps": 8, **settings}}}

    def test_a_family_default_frame_count_replaces_the_size_tables(self):
        defaults = resolve_video_defaults("h3", self._h3_config(default_frames=124), {"ratio": "16:9", "size": "s"})

        assert (defaults["num_frames"], defaults["width"], defaults["height"]) == (124, 768, 448)

    def test_explicit_frames_win_over_the_family_default(self):
        defaults = resolve_video_defaults("h3", self._h3_config(default_frames=124), {"ratio": "16:9", "size": "s", "num_frames": 22})

        assert defaults["num_frames"] == 22

    def test_a_family_without_default_frames_takes_the_size_table(self):
        assert resolve_video_defaults("h3", self._h3_config(), {"ratio": "16:9", "size": "s"})["num_frames"] == 121


class TestModelDisplayName:
    """An alias reads by the name its metadata entry gives, else by itself."""

    def test_a_built_in_alias_reads_by_its_name(self):
        assert model_display_name("klein9b") == "FLUX.2 Klein 9B"

    def test_a_name_without_an_entry_reads_as_itself(self):
        assert model_display_name("my-own-alias") == "my-own-alias"

    def test_every_built_in_alias_has_a_name(self):
        for alias in alias_map({}):
            name = model_display_name(alias)
            assert isinstance(name, str) and name.strip(), alias
            assert name != alias or alias == "unknown", alias


# ---------------------------------------------------------------------------
# G7: video defaults for LTX
# ---------------------------------------------------------------------------


class TestResolveVideoDefaultsGolden:
    """The packaged LTX defaults and the fallbacks for a size the table does not have."""

    def test_packaged_ltx_defaults(self, packaged_config):
        assert resolve_video_defaults("ltx", packaged_config, {}) == {"steps": 8, "width": 768, "height": 448, "num_frames": 121, "ratio": "16:9", "size": "s"}

    def test_explicit_values_win_over_the_size_entry(self, packaged_config):
        overrides = {"ratio": "16:9", "size": "m", "steps": 4, "width": 640}

        assert resolve_video_defaults("ltx", packaged_config, overrides) == {"steps": 4, "width": 640, "height": 576, "num_frames": 121, "ratio": "16:9", "size": "m"}

    def test_a_missing_size_falls_back_to_the_built_in_size(self):
        config = {"video_sizes": {}, "video_model_presets": {"ltx": {"default_steps": 6}}}

        assert resolve_video_defaults("ltx", config, {"ratio": "1:1", "size": "zz"}) == {"steps": 6, "width": 768, "height": 448, "num_frames": 121, "ratio": "1:1", "size": "zz"}

    def test_a_family_without_a_preset_gets_eight_steps(self):
        assert resolve_video_defaults("unknown", {}, {}) == {"steps": 8, "width": 768, "height": 448, "num_frames": 121, "ratio": "16:9", "size": "s"}
