"""Tests for config loading, parity with hardcoded values, and resolution logic."""

from __future__ import annotations

import re
from copy import deepcopy

import pytest

from zvisiongenerator.utils.config import (
    EULER_ANCESTRAL,
    _deep_merge,
    get_variant_key,
    load_config,
    ltx_diffusers_quantize,
    ltx_latent_upsampler_source,
    ltx_load_options,
    ltx_mlx_packs,
    ltx_mlx_text_encoder,
    ltx_transformer_source,
    ltx_version,
    model_capabilities,
    model_display_name,
    resolve_defaults,
    resolve_video_defaults,
    validate_video_size_choice,
    video_companion_repos,
    video_load_options,
    video_preset,
)
from zvisiongenerator.utils.h3_preset import HubFiles
from zvisiongenerator.utils.image_model_detect import ImageModelInfo
from zvisiongenerator.utils.paths import parse_huggingface_repo_reference, resolve_model_path


_SEEDVR2_BLOCK = {
    "model": "auto",
    "softness": 0.75,
    "video_softness": 0.0,
    "reference_megapixels": 16,
    "tiers": [{"model": "seedvr2-3b", "weights_gb": 4.72, "activation_floor_gb": 10.79, "activation_base_gb": 1.1, "activation_gb_per_mp": 0.89}],
    "discrete_tiers": [{"model": "seedvr2-3b", "weights_gb": 0.3, "activation_floor_gb": 5.0, "activation_base_gb": 0.6, "activation_gb_per_mp": 0.6, "system_gb": 12.5}],
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
        data = {"sizes": sizes, "generation": generation, "sharpening": {}, "upscale": {}, "schedulers": {}, "model_presets": {}}

        with pytest.raises(ValueError, match=match):
            _load_packaged(tmp_path, data)

    def test_model_alias_reference_resolves_after_user_override(self, tmp_path):
        """Config references should follow the merged default repo value."""
        import yaml

        packaged = {
            "sizes": {"1:1": {"m": {"width": 100, "height": 100}}},
            "generation": {**_GENERATION, "default_ratio": "1:1", "default_size": "m"},
            "sharpening": {},
            "upscale": {},
            "seedvr2": _SEEDVR2_BLOCK,
            "contrast": {},
            "saturation": {},
            "schedulers": {},
            "platforms": {"darwin": "macOS", "win32": "Windows", "linux": "Linux"},
            "model_aliases": {
                "ltx-2.3": {
                    "win32": "${video_model_presets.ltx.diffusers.default_repo}",
                    "linux": "${video_model_presets.ltx.diffusers.default_repo}",
                }
            },
            "model_presets": {},
            "video_sizes": {"16:9": {"m": {"width": 704, "height": 448, "frames": 49}}},
            "video_generation": {"default_ratio": "16:9", "default_size": "m"},
            "video_model_presets": {"ltx": {"diffusers": {"default_repo": "dg845/LTX-2.3-Diffusers"}}},
        }
        user_dir = tmp_path / ".ziv"
        user_dir.mkdir()
        (user_dir / "config.yaml").write_text(
            yaml.dump({"video_model_presets": {"ltx": {"diffusers": {"default_repo": "custom/LTX-2.3"}}}}),
            encoding="utf-8",
        )

        config = _load_packaged(tmp_path, packaged, user_dir=user_dir)

        assert config["model_aliases"]["ltx-2.3"]["win32"] == "custom/LTX-2.3"
        assert config["model_aliases"]["ltx-2.3"]["linux"] == "custom/LTX-2.3"


def test_packaged_config_resolves_ideo_to_its_models_folder_on_every_platform(monkeypatch, tmp_path):
    from zvisiongenerator.utils import config as config_module

    monkeypatch.setattr(config_module, "get_ziv_data_dir", lambda: tmp_path / ".ziv")
    monkeypatch.setenv("ZIV_DATA_DIR", str(tmp_path / ".ziv"))

    config = load_config()

    for platform_key in ("darwin", "win32", "linux"):
        assert resolve_model_path("ideo", aliases=config["model_aliases"], platform_key=platform_key) == str(tmp_path / ".ziv" / "models" / "ideo")
    assert config["model_alias_families"]["ideo"] == "ideogram4"


def test_packaged_config_pins_every_ideogram4_instant_source(packaged_config):
    from zvisiongenerator.utils.ideogram4_instant import instant_source

    formats = {}
    for platform_key in ("darwin", "win32", "linux"):
        source = instant_source(packaged_config, platform_key)
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
        assert resolve_model_path("krea2", aliases=config["model_aliases"], platform_key=platform_key) == "unsloth/Krea-2-Turbo"
    assert config["model_alias_families"]["krea2"] == "krea2"
    for backend_name in ("mflux", "diffusers"):
        defaults = resolve_defaults(turbo, config, {}, backend_name)
        assert (defaults["steps"], defaults["guidance"], defaults["scheduler"]) == (8, 1.0, None)
        assert defaults["supports_negative_prompt"] is False
        assert defaults["supports_scheduler"] is False


def test_packaged_config_klein9b_uses_ungated_mirror(packaged_config):
    aliases = packaged_config["model_aliases"]

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
    caps = model_capabilities(packaged_config, "krea2")

    assert (caps["supports_img2img"], caps["supports_upscale"]) == (True, True)


def test_packaged_config_references_only_on_klein(packaged_config):
    assert model_capabilities(packaged_config, "flux2_klein")["max_reference_images"] == 4
    assert model_capabilities(packaged_config, "flux2_klein", "distilled")["max_reference_images"] == 4
    assert model_capabilities(packaged_config, "zimage")["max_reference_images"] == 0


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
                "supports_negative_prompt": True,
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
                "supports_negative_prompt": False,
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
                "supports_negative_prompt": False,
                "default_steps": 25,
                "default_guidance": 4.5,
                "default_scheduler": {
                    "mflux": None,
                    "diffusers": None,
                },
            },
            "flux1": {
                "supports_negative_prompt": False,
                "default_steps": 22,
                "default_guidance": 3.0,
                "default_scheduler": {
                    "mflux": None,
                    "diffusers": None,
                },
            },
            "ideogram4": {
                "supports_negative_prompt": False,
                "supports_quantize": False,
                "supports_img2img": False,
                "supports_upscale": False,
                "supports_json_prompt": True,
                "supports_first_sigma": True,
                "dimension_min": 256,
                "dimension_max": 2048,
                "dimension_step": 16,
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
                    "supports_quantize": False,
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
        assert [row["model"] for row in cfg["seedvr2"]["tiers"]] == ["seedvr2-7b", "seedvr2-3b"]

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
        "schedulers": {},
        "platforms": {"darwin": "macOS", "win32": "Windows", "linux": "Linux"},
        "model_aliases": {},
        "model_presets": {},
        "video_sizes": {},
        "video_generation": {},
        "video_model_presets": {},
        **sections,
    }


class TestPlatformConfigValidation:
    def test_platforms_section_must_be_mapping(self, tmp_path):
        with pytest.raises(ValueError, match="Config section 'platforms' must be a mapping"):
            _load_packaged(tmp_path, _platform_config(platforms=["darwin"]))

    def test_sharpening_amounts_above_the_cap_are_rejected(self, tmp_path):
        data = {"sizes": {"1:1": {"m": {"width": 100, "height": 100}}}, "generation": _GENERATION, "sharpening": {"normal": 1.0, "upscaled": 2.0}}

        with pytest.raises(ValueError, match="sharpening.upscaled"):
            _load_packaged(tmp_path, data)

    @pytest.mark.parametrize(
        "alias_value",
        [
            "Tongyi-MAI/Z-Image-Turbo",
            {"win32": "dg845/LTX-2.3-Diffusers", "linux": "dg845/LTX-2.3-Diffusers"},
            {"darwin": "dgrauet/ltx-2.3-mlx-q4", "win32": {"message": "Use ltx-8 instead."}},
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
            ({"win32": 42}, r"config 'model_aliases\.ltx\.win32' must be a string or mapping with a message"),
            ({"win32": {}}, r"config 'model_aliases\.ltx\.win32\.message' must be a non-empty string"),
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

    def test_beta_scheduler_class(self, packaged_config):
        from zvisiongenerator.utils.config import resolve_scheduler_class

        resolved = resolve_scheduler_class("beta", packaged_config, "mflux")
        assert resolved == "zvisiongenerator.schedulers.beta_scheduler.BetaScheduler"
        assert resolved == packaged_config["schedulers"]["beta"]["mflux_class"]

    def test_resolve_unknown_scheduler_passthrough(self, packaged_config):
        from zvisiongenerator.utils.config import resolve_scheduler_class

        assert resolve_scheduler_class("euler", packaged_config, "mflux") == "euler"

    def test_resolve_none_scheduler(self, packaged_config):
        from zvisiongenerator.utils.config import resolve_scheduler_class

        assert resolve_scheduler_class(None, packaged_config, "mflux") is None


# ---------------------------------------------------------------------------
# Scheduler validation
# ---------------------------------------------------------------------------


class TestSchedulerValidation:
    """Verify validate_scheduler works."""

    def test_validate_known_scheduler(self, packaged_config):
        from zvisiongenerator.utils.config import validate_scheduler

        validate_scheduler("beta", packaged_config)  # Should not raise

    def test_validate_none_scheduler(self):
        from zvisiongenerator.utils.config import validate_scheduler

        validate_scheduler(None, {"schedulers": {"beta": {}}})  # Should not raise

    def test_validate_unknown_scheduler(self):
        from zvisiongenerator.utils.config import validate_scheduler

        with pytest.raises(ValueError, match="Unknown scheduler"):
            validate_scheduler("nonexistent", {"schedulers": {"beta": {}}})


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
    from zvisiongenerator.utils.config import model_capabilities

    config = {
        "generation": {"default_steps": 10, "default_guidance": 3.5},
        "model_presets": {"ideogram4": {"supports_upscale": False, "supports_img2img": False, "dimension_max": 2048}},
    }
    capabilities = model_capabilities(config, "ideogram4")
    resolved = resolve_defaults(ImageModelInfo(family="ideogram4", is_distilled=False, size=None), config, {}, "mflux")

    assert capabilities == {key: resolved[key] for key in capabilities}
    assert capabilities["dimension_max"] == 2048
    assert model_capabilities(config, "unknown")["supports_upscale"] is True


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


class TestLtxLatentUpsamplerSource:
    """Verify the latent upsampler source helper reads and validates its config key."""

    @staticmethod
    def _config(value):
        return {"video_model_presets": {"ltx": {"diffusers": {"latent_upsampler": value}}}}

    def test_packaged_config_ships_upsampler_and_distilled_default_repo(self, packaged_config):
        config = packaged_config

        assert ltx_latent_upsampler_source(config) == ("Lightricks/LTX-2.3@c1521179de480464f366a98a9d9b2acc4cc097b9", "ltx-2.3-spatial-upscaler-x2-1.1.safetensors")
        assert config["video_model_presets"]["ltx"]["diffusers"]["default_repo"] == "diffusers/LTX-2.3-Distilled-Diffusers@432e0d3c2d1769aaa4d295f9243f7062bf6b47ee"

    @pytest.mark.parametrize("config", [{}, {"video_model_presets": {}}, {"video_model_presets": {"ltx": {"diffusers": {}}}}])
    def test_absent_key_returns_none(self, config):
        assert ltx_latent_upsampler_source(config) is None

    def test_returns_stripped_values(self):
        assert ltx_latent_upsampler_source(self._config({"repo": " your-org/upsampler ", "filename": " up.safetensors "})) == ("your-org/upsampler", "up.safetensors")

    @pytest.mark.parametrize(
        "value",
        [None, "not-a-mapping", {"repo": " ", "filename": "x"}, {"repo": "r", "filename": ""}, {"repo": "r"}, {"filename": "x"}, {"repo": 3, "filename": "x"}],
    )
    def test_malformed_value_raises_value_error(self, value):
        with pytest.raises(ValueError, match="latent_upsampler"):
            ltx_latent_upsampler_source(self._config(value))


class TestLtxTransformerSource:
    """Verify the single-file transformer source helper reads and validates its config key."""

    @staticmethod
    def _config(value):
        return {"video_model_presets": {"ltx": {"diffusers": {"transformer": value}}}}

    def test_packaged_config_ships_the_distilled_1_1_transformer(self, packaged_config):
        assert ltx_transformer_source(packaged_config) == ("Lightricks/LTX-2.3@c1521179de480464f366a98a9d9b2acc4cc097b9", "ltx-2.3-22b-distilled-1.1.safetensors")

    @pytest.mark.parametrize("config", [{}, {"video_model_presets": {"ltx": {"diffusers": {}}}}, _config.__func__(None)])
    def test_absent_or_null_key_uses_the_repos_own_transformer(self, config):
        assert ltx_transformer_source(config) is None

    def test_a_user_null_turns_the_packaged_transformer_off(self, monkeypatch, tmp_path):
        from zvisiongenerator.utils import config as config_module

        (tmp_path / ".ziv").mkdir()
        (tmp_path / ".ziv" / "config.yaml").write_text("video_model_presets:\n  ltx:\n    diffusers:\n      transformer: null\n")
        monkeypatch.setattr(config_module, "get_ziv_data_dir", lambda: tmp_path / ".ziv")

        assert ltx_transformer_source(load_config()) is None

    @pytest.mark.parametrize("value", ["not-a-mapping", {"repo": "r"}, {"repo": " ", "filename": "x"}])
    def test_malformed_value_raises_value_error(self, value):
        with pytest.raises(ValueError, match="transformer"):
            ltx_transformer_source(self._config(value))


class TestLtxDiffusersQuantize:
    """Verify the per-alias quantize levels of the diffusers LTX backend."""

    @staticmethod
    def _config(value):
        return {"video_model_presets": {"ltx": {"diffusers": {"quantize": value}}}}

    def test_packaged_config_quantizes_ltx_4_and_ltx_8_only(self, packaged_config):
        config = packaged_config

        assert [ltx_diffusers_quantize(config, name) for name in ("ltx-4", "ltx-8", "other", None)] == [4, 8, None, None]

    @pytest.mark.parametrize("platform_key", ["win32", "linux"])
    def test_packaged_ltx_aliases_resolve_to_the_diffusers_repo_off_macos(self, packaged_config, platform_key):
        config = packaged_config
        repo = config["video_model_presets"]["ltx"]["diffusers"]["default_repo"]

        assert "ltx-2.3" not in config["model_aliases"]
        for alias in ("ltx-4", "ltx-8"):
            assert resolve_model_path(alias, aliases=config["model_aliases"], platform_key=platform_key) == repo

    @pytest.mark.parametrize("value", ["4", {"ltx-4": 6}, {"ltx-4": "4"}])
    def test_malformed_value_raises_value_error(self, value):
        with pytest.raises(ValueError, match="quantize"):
            ltx_diffusers_quantize(self._config(value), "ltx-4")


class TestLtxLoadOptions:
    """Verify the load options both the CLI and the Web UI pass to a video backend."""

    _CONFIG = {
        "video_model_presets": {
            "ltx": {
                "diffusers": {
                    "transformer": {"repo": "org/ltx", "filename": "ltx.safetensors"},
                    "latent_upsampler": {"repo": "org/up", "filename": "up.safetensors"},
                    "quantize": {"ltx-4": 4},
                }
            }
        }
    }

    def test_options_carry_the_level_and_the_transformer_source(self):
        assert ltx_load_options(self._CONFIG, "ltx-4", upscale=False) == {"quantize": 4, "transformer_source": ("org/ltx", "ltx.safetensors"), "model_name": "ltx-4", "mlx_packs": {}}

    def test_upscale_adds_the_upsampler(self):
        options = ltx_load_options(self._CONFIG, "other", upscale=True)

        assert options["quantize"] is None
        assert options["upscale"] is True
        assert options["latent_upsampler"] == ("org/up", "up.safetensors")


_LTX25_PIN = "unsloth/LTX-2.5-FP8@7b40724fc39bbb01d9b4d72638c0d263531c88c2"
_LTX25_UPSAMPLER = (_LTX25_PIN, "latent_upsampler/diffusion_pytorch_model.safetensors")
_LTX25_MAC_PIN = "MarceloFSO/ltx-2.5-mlx-q4-fast@281bf05bd23207d6feef708fc2e270f84ba992b5"
_LTX23_REPO = "diffusers/LTX-2.3-Distilled-Diffusers@432e0d3c2d1769aaa4d295f9243f7062bf6b47ee"
_LTX23_LIGHTRICKS = "Lightricks/LTX-2.3@c1521179de480464f366a98a9d9b2acc4cc097b9"
_LTX23_TRANSFORMER = (_LTX23_LIGHTRICKS, "ltx-2.3-22b-distilled-1.1.safetensors")
_LTX23_UPSAMPLER = (_LTX23_LIGHTRICKS, "ltx-2.3-spatial-upscaler-x2-1.1.safetensors")


def _versioned(**version):
    """Return a config whose ``ltx`` family has a ``v`` version (listing ``alias-v``) holding *version*."""
    return {
        "video_model_presets": {
            "ltx": {
                "default_steps": 8,
                "diffusers": {"default_repo": "org/family", "quantize": {"ltx-4": 4}, "transformer": {"repo": "org/ltx", "filename": "ltx.safetensors"}},
                "versions": {"v": {"aliases": ["alias-v"], **version}},
            }
        }
    }


class TestVideoPreset:
    """R1: a version's block is merged over the family preset for the aliases it lists."""

    def test_version_merges_nested_dicts_and_replaces_scalars_and_none(self):
        config = _versioned(default_steps=4, diffusers={"default_repo": "org/v", "transformer": None, "quantize": {"alias-v": 8}})

        preset = video_preset(config, "ltx", "alias-v")

        assert preset["default_steps"] == 4
        assert preset["diffusers"] == {"default_repo": "org/v", "transformer": None, "quantize": {"ltx-4": 4, "alias-v": 8}}
        assert "versions" not in preset
        assert "aliases" not in preset

    @pytest.mark.parametrize("model_name", [None, "ltx-4", "unknown"])
    def test_an_unlisted_name_or_none_returns_the_family_preset(self, model_name):
        config = _versioned(default_steps=4)

        preset = video_preset(config, "ltx", model_name)

        assert preset == {"default_steps": 8, "diffusers": config["video_model_presets"]["ltx"]["diffusers"]}

    def test_the_result_is_independent_of_the_config(self):
        config = _versioned(diffusers={"quantize": {"alias-v": 8}})
        before = deepcopy(config)

        for name in ("alias-v", "ltx-4"):
            preset = video_preset(config, "ltx", name)
            preset["diffusers"]["quantize"]["mutated"] = 1
            preset["diffusers"]["default_repo"] = "mutated"

        assert config == before

    def test_it_is_family_generic(self):
        config = {"video_model_presets": {"other": {"default_steps": 6, "versions": {"b": {"aliases": ["o-b"], "default_steps": 9}}}}}

        assert video_preset(config, "other", "o-b") == {"default_steps": 9}
        assert video_preset(config, "other", "o-a") == {"default_steps": 6}
        assert video_preset(config, "missing", "o-b") == {}

    @pytest.mark.parametrize("config", [{}, {"video_model_presets": None}, {"video_model_presets": {"ltx": {"default_steps": 8}}}])
    def test_no_versions_returns_the_family_preset(self, config):
        assert video_preset(config, "ltx", "ltx-4") == (config.get("video_model_presets") or {}).get("ltx", {})

    def test_an_alias_in_two_versions_raises_naming_both(self):
        config = _versioned()
        config["video_model_presets"]["ltx"]["versions"]["w"] = {"aliases": ["alias-v"]}

        with pytest.raises(ValueError, match=r"alias-v.*'v'.*'w'"):
            video_preset(config, "ltx", "alias-v")

    @pytest.mark.parametrize("versions", ["x", ["v"], {"v": "x"}, {"v": None}])
    def test_versions_that_are_not_a_mapping_of_mappings_raise(self, versions):
        config = {"video_model_presets": {"ltx": {"versions": versions}}}

        with pytest.raises(ValueError, match=r"video_model_presets\.ltx\.versions"):
            video_preset(config, "ltx", "x")

    @pytest.mark.parametrize("aliases", [None, "alias-v", ["alias-v", " "], ["alias-v", 3], [["alias-v"]]])
    def test_aliases_that_are_not_a_list_of_strings_raise(self, aliases):
        config = _versioned()
        config["video_model_presets"]["ltx"]["versions"]["v"]["aliases"] = aliases

        with pytest.raises(ValueError, match=r"versions\.v\.aliases"):
            video_preset(config, "ltx", "alias-v")

    def test_a_version_missing_its_aliases_raises(self):
        config = {"video_model_presets": {"ltx": {"versions": {"v": {"default_steps": 4}}}}}

        with pytest.raises(ValueError, match=r"versions\.v\.aliases"):
            video_preset(config, "ltx", None)


class TestPackagedLtx25Config:
    """R2: the packaged ``ltx25-4`` entries, and the LTX-2.3 entries left as they were."""

    @pytest.mark.parametrize("platform_key", ["win32", "linux"])
    def test_ltx25_4_resolves_to_the_pinned_repo(self, packaged_config, platform_key):
        config = packaged_config

        assert resolve_model_path("ltx25-4", aliases=config["model_aliases"], platform_key=platform_key) == _LTX25_PIN

    def test_ltx25_4_resolves_to_the_pinned_mirror_on_macos(self, packaged_config):
        config = packaged_config

        assert resolve_model_path("ltx25-4", aliases=config["model_aliases"], platform_key="darwin") == _LTX25_MAC_PIN

    def test_the_pin_is_written_once_and_referenced(self, packaged_config):
        config = packaged_config
        version = config["video_model_presets"]["ltx"]["versions"]["ltx25"]

        assert version["aliases"] == ["ltx25-4"]
        assert version["diffusers"]["default_repo"] == _LTX25_PIN
        assert version["diffusers"]["latent_upsampler"]["repo"] == _LTX25_PIN
        assert config["model_aliases"]["ltx25-4"]["win32"] == config["model_aliases"]["ltx25-4"]["linux"] == _LTX25_PIN
        assert config["model_aliases"]["ltx25-4"]["darwin"] == _LTX25_MAC_PIN
        assert config["video_model_presets"]["ltx"]["diffusers"]["default_repo"] == _LTX23_REPO


class TestLtxVersionAwareAccessors:
    """R3: the ``ltx_*`` accessors and load options read the version that lists the model."""

    def test_ltx25_4_load_options_with_upscale(self, packaged_config):
        config = packaged_config

        assert ltx_load_options(config, "ltx25-4", upscale=True) == {
            "quantize": 4,
            "transformer_source": None,
            "model_name": "ltx25-4",
            "repo_ignore": ("LTX-2.5-*.safetensors", "samples/*"),
            "sampler": "euler_ancestral",
            "image_crf": 18,
            "upscale": True,
            "latent_upsampler": _LTX25_UPSAMPLER,
            "mlx_packs": ltx_mlx_packs(config),
            "mlx_text_encoder": ltx_mlx_text_encoder(config),
        }

    def test_ltx25_4_load_options_without_upscale_have_no_upsampler(self, packaged_config):
        options = ltx_load_options(packaged_config, "ltx25-4", upscale=False)

        assert "upscale" not in options and "latent_upsampler" not in options
        assert options["transformer_source"] is None

    @pytest.mark.parametrize("name", ["ltx-4", "ltx-8"])
    def test_ltx_2_3_load_options_are_unchanged(self, packaged_config, name):
        """2.3 guard: the options of ``ltx-4`` and ``ltx-8`` carry no version key."""
        config = packaged_config
        bits = {"ltx-4": 4, "ltx-8": 8}[name]
        plain = {"quantize": bits, "transformer_source": _LTX23_TRANSFORMER, "model_name": name, "mlx_packs": ltx_mlx_packs(config), "mlx_text_encoder": ltx_mlx_text_encoder(config)}

        assert ltx_load_options(config, name, upscale=False) == plain
        assert ltx_load_options(config, name, upscale=True) == {**plain, "upscale": True, "latent_upsampler": _LTX23_UPSAMPLER}

    def test_ltx_2_3_sources_do_not_see_the_version(self, packaged_config):
        config = packaged_config

        for name in (None, "ltx-4", "ltx-8", "unknown"):
            assert ltx_transformer_source(config, name) == _LTX23_TRANSFORMER
            assert ltx_latent_upsampler_source(config, name) == _LTX23_UPSAMPLER
        assert ltx_transformer_source(config) == _LTX23_TRANSFORMER
        assert ltx_latent_upsampler_source(config) == _LTX23_UPSAMPLER

    def test_ltx25_4_sources_and_quantize_come_from_the_version(self, packaged_config):
        config = packaged_config

        assert ltx_transformer_source(config, "ltx25-4") is None
        assert ltx_latent_upsampler_source(config, "ltx25-4") == _LTX25_UPSAMPLER
        assert ltx_diffusers_quantize(config, "ltx25-4") == 4
        assert [ltx_diffusers_quantize(config, name) for name in ("ltx-4", "ltx-8")] == [4, 8]

    def test_a_family_transformer_override_does_not_reach_ltx25_4(self, packaged_config):
        config = packaged_config
        config["video_model_presets"]["ltx"]["diffusers"]["transformer"] = {"repo": "me/ltx", "filename": "mine.safetensors"}

        assert ltx_transformer_source(config, "ltx-4") == ("me/ltx", "mine.safetensors")
        assert ltx_transformer_source(config, "ltx25-4") is None
        assert ltx_load_options(config, "ltx25-4", upscale=False)["transformer_source"] is None

    @pytest.mark.parametrize("name", ["ltx-4", "ltx-8", None, "unknown", "org/some-ltx-repo"])
    def test_ancestral_sampler_outside_a_version_is_refused(self, packaged_config, name):
        config = packaged_config
        config["video_model_presets"]["ltx"]["diffusers"]["sampler"] = EULER_ANCESTRAL

        with pytest.raises(ValueError, match=r"video_model_presets\.ltx\.diffusers\.sampler") as excinfo:
            ltx_load_options(config, name, upscale=False)
        assert (name or "this model") in str(excinfo.value)

    def test_ancestral_sampler_is_accepted_for_a_version_alias(self, packaged_config):
        config = packaged_config
        config["video_model_presets"]["ltx"]["diffusers"]["sampler"] = EULER_ANCESTRAL

        assert ltx_load_options(config, "ltx25-4", upscale=False)["sampler"] == EULER_ANCESTRAL

    def test_an_invalid_sampler_still_reports_the_allowed_values(self, packaged_config):
        config = packaged_config
        config["video_model_presets"]["ltx"]["diffusers"]["sampler"] = "ddim"

        with pytest.raises(ValueError, match="must be one of"):
            ltx_load_options(config, "ltx-4", upscale=False)

    def test_ltx_version_reports_version_membership(self, packaged_config):
        config = packaged_config

        assert ltx_version(config, "ltx25-4") == "ltx25"
        assert ltx_version(config, "ltx-4") is None
        assert ltx_version(config, None) is None

    def test_a_converted_ltx25_folder_takes_its_versions_settings(self, packaged_config, monkeypatch, tmp_path):
        from zvisiongenerator.utils.ltx_converted import write_converted_ltx

        config = packaged_config
        data = tmp_path / "data"
        for name, version in (("my25", "2.5"), ("my23", "2.3")):
            (data / "models" / name).mkdir(parents=True)
            write_converted_ltx(data / "models" / name, {"bits": 4, "storage": "nf4", "block_matrices": 1, "transformer_rest": 1}, version=version)
        monkeypatch.setattr("zvisiongenerator.utils.config.get_ziv_data_dir", lambda: data)

        assert ltx_version(config, "my25") == "ltx25"
        assert ltx_version(config, str(data / "models" / "my25")) == "ltx25"
        assert ltx_version(config, "my23") is None
        assert ltx_load_options(config, "my25", upscale=False)["sampler"] == "euler_ancestral"

    def test_the_converter_types_map_to_version_blocks_with_their_own_sources(self, packaged_config):
        from zvisiongenerator.utils.config import ltx_diffusers_repo, ltx_distilled_lora_source, ltx_mlx_base_repo, ltx_version_for_type

        config = packaged_config

        assert ltx_version_for_type(config, "ltx-2.3") is None
        assert ltx_version_for_type(config, "ltx-2.5") == "ltx25"
        repo, filename = ltx_distilled_lora_source(config, "ltx25")
        assert repo.partition("@")[2] and "2.5" in filename  # pinned at a revision
        assert ltx_distilled_lora_source(config, None) != ltx_distilled_lora_source(config, "ltx25")
        assert ltx_diffusers_repo(config, "ltx25") != ltx_diffusers_repo(config, None)
        assert ltx_mlx_base_repo(config, 8, "ltx25") != ltx_mlx_base_repo(config, 8, None)
        with pytest.raises(ValueError):
            ltx_version_for_type(config, "ltx-9")

    def test_the_repo_ignore_of_a_version_skips_its_root_checkpoints(self, packaged_config):
        from zvisiongenerator.utils.config import ltx_repo_ignore

        config = packaged_config

        assert "LTX-2.5-*.safetensors" in ltx_repo_ignore(config, "ltx25")
        assert ltx_repo_ignore(config, None) == ()

    def test_a_version_without_the_new_keys_adds_none(self):
        config = _versioned(diffusers={"quantize": {"alias-v": 4}})

        assert ltx_load_options(config, "alias-v", upscale=False) == {"quantize": 4, "transformer_source": ("org/ltx", "ltx.safetensors"), "model_name": "alias-v", "mlx_packs": {}}

    def test_a_null_new_key_is_unset(self):
        config = _versioned(diffusers={"sampler": None, "image_crf": None, "repo_ignore": None})

        assert set(ltx_load_options(config, "alias-v", upscale=False)) == {"quantize", "transformer_source", "model_name", "mlx_packs"}

    def test_image_crf_zero_and_the_bounds_are_valid(self):
        for crf in (0, 51):
            assert ltx_load_options(_versioned(diffusers={"image_crf": crf}), "alias-v", upscale=False)["image_crf"] == crf

    @pytest.mark.parametrize(
        ("key", "value"),
        [
            ("sampler", "ddim"),
            ("sampler", 3),
            ("image_crf", True),
            ("image_crf", 52),
            ("image_crf", -1),
            ("image_crf", "18"),
            ("image_crf", 18.0),
            ("repo_ignore", "samples/*"),
            ("repo_ignore", ["samples/*", " "]),
            ("repo_ignore", ["samples/*", 3]),
        ],
    )
    def test_an_invalid_version_value_raises_naming_the_key(self, key, value):
        config = _versioned(diffusers={key: value})

        with pytest.raises(ValueError, match=rf"video_model_presets\.ltx\.versions\.v\.diffusers\.{key}"):
            ltx_load_options(config, "alias-v", upscale=False)

    def test_a_malformed_version_source_names_the_version_key(self):
        config = _versioned(diffusers={"latent_upsampler": {"repo": "org/up"}})

        with pytest.raises(ValueError, match=r"video_model_presets\.ltx\.versions\.v\.diffusers\.latent_upsampler\.filename"):
            ltx_latent_upsampler_source(config, "alias-v")
        assert ltx_latent_upsampler_source(config, "ltx-4") is None

    def test_a_malformed_family_source_still_names_the_family_key(self):
        config = _versioned()
        config["video_model_presets"]["ltx"]["diffusers"]["latent_upsampler"] = "x"

        with pytest.raises(ValueError, match=r"^video_model_presets\.ltx\.diffusers\.latent_upsampler "):
            ltx_latent_upsampler_source(config, "ltx-4")

    def test_distilled_lora_and_family_helpers_stay_family_level(self, packaged_config):
        from zvisiongenerator.utils.config import ltx_companion_repos, ltx_diffusers_repo, ltx_distilled_lora_source

        config = packaged_config

        assert ltx_distilled_lora_source(config) == (_LTX23_LIGHTRICKS, "ltx-2.3-22b-distilled-lora-384-1.1.safetensors")
        assert ltx_diffusers_repo(config) == _LTX23_REPO
        assert ltx_companion_repos(config, _LTX25_PIN) == ()
        assert ltx_companion_repos(config, _LTX23_REPO) == ("Lightricks/LTX-2.3",)


class TestVideoLoadOptionsGolden:
    """G4: ``video_load_options("ltx", ...)`` is exactly ``ltx_load_options`` for every LTX alias and both upscale settings."""

    @pytest.mark.parametrize(("name", "upscale"), [("ltx-4", False), ("ltx25-4", True)])
    def test_ltx_options_are_the_ltx_options(self, packaged_config, name, upscale):
        config = packaged_config

        assert video_load_options("ltx", config, name, upscale=upscale) == ltx_load_options(config, name, upscale=upscale)


class TestVideoFamilyDispatch:
    """R5: load options and companions are asked of the family; default frames come from the family."""

    def test_a_family_without_load_options_is_refused(self):
        with pytest.raises(ValueError, match="No video load options for model family 'wan'"):
            video_load_options("wan", {}, "m", upscale=False)

    def test_ltx_companions_are_the_ltx_companions(self, packaged_config):
        config = packaged_config

        assert video_companion_repos("ltx", config, _LTX23_REPO) == ("Lightricks/LTX-2.3",)
        assert video_companion_repos("ltx", config, _LTX25_PIN) == ()

    @pytest.mark.parametrize("family", ["zimage", "flux2_klein", "wan"])
    def test_another_family_has_no_companions(self, packaged_config, family):
        assert video_companion_repos(family, packaged_config, _LTX23_REPO) == ()

    def _h3_config(self, **preset):
        block = {
            "label": "MiniMax H3",
            "default_steps": 8,
            "frame_rule": {"alignment": 17, "offset": 5, "round": "up"},
            "extend_frame_step": 17,
            "carry_frames": 22,
            "capabilities": {"i2v": True, "end_frame": True, "extend": True, "upscale": True, "upscale_review": True, "lora": True, "audio": True, "low_memory": False},
            **preset,
        }
        return {"video_sizes": {"16:9": {"s": {"width": 768, "height": 448, "frames": 121}}}, "video_model_presets": {"h3": block}}

    def test_a_family_default_frame_count_replaces_the_size_tables(self):
        defaults = resolve_video_defaults("h3", self._h3_config(default_frames=124), {"ratio": "16:9", "size": "s"})

        assert (defaults["num_frames"], defaults["width"], defaults["height"]) == (124, 768, 448)

    def test_explicit_frames_win_over_the_family_default(self):
        defaults = resolve_video_defaults("h3", self._h3_config(default_frames=124), {"ratio": "16:9", "size": "s", "num_frames": 22})

        assert defaults["num_frames"] == 22

    def test_a_family_without_default_frames_takes_the_size_table(self):
        assert resolve_video_defaults("h3", self._h3_config(), {"ratio": "16:9", "size": "s"})["num_frames"] == 121


class TestModelDisplayName:
    """AC4.1/AC4.2: an alias reads by its configured name in the Web UI, else by itself."""

    def test_a_configured_alias_reads_by_its_name(self):
        assert model_display_name({"model_labels": {"klein9b": {"name": "FLUX.2 Klein 9B"}}}, "klein9b") == "FLUX.2 Klein 9B"

    @pytest.mark.parametrize(
        "config",
        [
            {},
            {"model_labels": None},
            {"model_labels": {}},
            {"model_labels": {"other": {"name": "Other"}}},
            {"model_labels": {"zit": {"name": "   "}}},
            {"model_labels": {"zit": {"name": 7}}},
            {"model_labels": {"zit": {}}},
            {"model_labels": {"zit": "Z-Image Turbo"}},
        ],
    )
    def test_anything_unusable_reads_as_the_alias(self, config):
        assert model_display_name(config, "zit") == "zit"

    def test_every_packaged_alias_has_a_name(self, packaged_config):
        config = packaged_config

        for alias in config["model_aliases"]:
            name = config["model_labels"][alias]["name"]
            assert isinstance(name, str) and name.strip(), alias
            assert model_display_name(config, alias) == name
        assert model_display_name(config, "klein9b") == "FLUX.2 Klein 9B"


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


class TestLtxMlxPacks:
    """The pinned MLX packs and the Gemma 3 reference macOS LTX downloads and loads from."""

    _PIN = "a" * 40

    def _config(self, **block):
        return {"video_model_presets": {"ltx": block}}

    def test_a_pinned_base_is_a_pack_of_its_blocks_files(self):
        config = self._config(mlx_base={"q4": f"o/p@{self._PIN}", "q8": f"o/q@{self._PIN}"}, mlx_files=["t.safetensors", "c.json"])

        assert ltx_mlx_packs(config) == {
            f"o/p@{self._PIN}": HubFiles("o/p", self._PIN, ("t.safetensors", "c.json")),
            f"o/q@{self._PIN}": HubFiles("o/q", self._PIN, ("t.safetensors", "c.json")),
        }

    def test_an_unpinned_base_is_skipped_and_needs_no_file_list(self):
        assert ltx_mlx_packs(self._config(mlx_base={"q4": "o/p", "q8": "o/q"})) == {}
        assert ltx_mlx_packs({}) == {}

    def test_a_version_block_has_its_own_files(self):
        config = self._config(
            mlx_base={"q4": f"o/p@{self._PIN}"},
            mlx_files=["a"],
            versions={"v": {"aliases": ["x"], "mlx_base": {"q4": f"o/v@{self._PIN}", "q8": f"o/v@{self._PIN}"}, "mlx_files": ["b"]}},
        )

        assert ltx_mlx_packs(config) == {f"o/p@{self._PIN}": HubFiles("o/p", self._PIN, ("a",)), f"o/v@{self._PIN}": HubFiles("o/v", self._PIN, ("b",))}

    @pytest.mark.parametrize("files", [None, [], [""], "a", ["a", 3]])
    def test_a_pinned_base_needs_a_valid_file_list(self, files):
        config = self._config(mlx_base={"q4": f"o/p@{self._PIN}"}, **({} if files is None else {"mlx_files": files}))

        with pytest.raises(ValueError, match="mlx_files"):
            ltx_mlx_packs(config)

    @pytest.mark.parametrize("bases", ["o/p", {"q4": ""}, {"q4": 3}])
    def test_a_malformed_base_is_refused(self, bases):
        with pytest.raises(ValueError, match="mlx_base"):
            ltx_mlx_packs(self._config(mlx_base=bases, mlx_files=["a"]))

    def test_one_reference_with_two_file_lists_is_refused(self):
        config = self._config(mlx_base={"q4": f"o/p@{self._PIN}"}, mlx_files=["a"], versions={"v": {"aliases": ["x"], "mlx_base": {"q4": f"o/p@{self._PIN}"}, "mlx_files": ["b"]}})

        with pytest.raises(ValueError, match="other files"):
            ltx_mlx_packs(config)

    def test_the_text_encoder_is_a_repo_reference(self):
        assert ltx_mlx_text_encoder(self._config(mlx_text_encoder=f" o/gemma@{self._PIN} ")) == f"o/gemma@{self._PIN}"
        for value in (None, "", 3, "/local/folder"):
            with pytest.raises(ValueError, match="mlx_text_encoder"):
                ltx_mlx_text_encoder(self._config(mlx_text_encoder=value))
        with pytest.raises(ValueError, match="mlx_text_encoder"):
            ltx_mlx_text_encoder({})

    def test_the_options_carry_the_text_encoder_only_when_configured(self):
        assert "mlx_text_encoder" not in ltx_load_options(self._config(), "ltx-4", upscale=False)
        assert ltx_load_options(self._config(mlx_text_encoder="o/gemma"), "ltx-4", upscale=False)["mlx_text_encoder"] == "o/gemma"

    def test_the_packaged_packs_are_pinned_with_their_files(self, packaged_config):
        config = packaged_config
        ltx = config["video_model_presets"]["ltx"]

        packs = ltx_mlx_packs(config)

        assert set(packs) == {ltx["mlx_base"]["q4"], ltx["mlx_base"]["q8"], ltx["versions"]["ltx25"]["mlx_base"]["q4"]}
        assert all(len(pack.revision) == 40 for pack in packs.values())
        assert packs[ltx["mlx_base"]["q4"]].files == tuple(ltx["mlx_files"])
        assert packs[ltx["versions"]["ltx25"]["mlx_base"]["q4"]].files == tuple(ltx["versions"]["ltx25"]["mlx_files"])
        assert parse_huggingface_repo_reference(ltx_mlx_text_encoder(config)).revision is not None  # type: ignore[union-attr]

    def test_the_macos_aliases_are_the_pinned_bases(self, packaged_config):
        """The aliases read the packs' references, so an alias, the plan and the run cannot name different revisions."""
        config = packaged_config
        ltx = config["video_model_presets"]["ltx"]
        aliases = config["model_aliases"]

        assert aliases["ltx-4"]["darwin"] == ltx["mlx_base"]["q4"]
        assert aliases["ltx-8"]["darwin"] == ltx["mlx_base"]["q8"]
        assert aliases["ltx25-4"]["darwin"] == ltx["versions"]["ltx25"]["mlx_base"]["q4"]
        assert all(alias["darwin"] in ltx_mlx_packs(config) for alias in (aliases["ltx-4"], aliases["ltx-8"], aliases["ltx25-4"]))
