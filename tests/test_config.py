"""Tests for config loading, parity with hardcoded values, and resolution logic."""

from __future__ import annotations

import pytest

from zvisiongenerator.utils.config import (
    _deep_merge,
    get_variant_key,
    load_config,
    resolve_defaults,
    resolve_video_defaults,
)
from zvisiongenerator.utils.image_model_detect import ImageModelInfo


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


class TestNestedConfigValidation:
    """Ensure load_config rejects configs with bad nested values."""

    def test_default_steps_not_int_raises(self, tmp_path):
        from unittest.mock import patch
        import yaml

        bad_yaml = {
            "sizes": {"1:1": {"m": {"width": 100, "height": 100}}},
            "generation": {"default_steps": "ten", "default_guidance": 3.5},
            "sharpening": {},
            "upscale": {},
            "schedulers": {},
            "model_presets": {},
        }
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump(bad_yaml))

        with patch("importlib.resources.files") as mock_files, patch("zvisiongenerator.utils.config.get_ziv_data_dir") as mock_dir:
            mock_dir.return_value = tmp_path / "nope"
            mock_files.return_value.joinpath.return_value = config_file
            with pytest.raises(ValueError, match="default_steps.*integer"):
                load_config()

    def test_default_guidance_not_number_raises(self, tmp_path):
        """Guidance set to a string should be rejected."""
        from unittest.mock import patch
        import yaml

        bad_yaml = {
            "sizes": {"1:1": {"m": {"width": 100, "height": 100}}},
            "generation": {"default_steps": 10, "default_guidance": "high"},
            "sharpening": {},
            "upscale": {},
            "schedulers": {},
            "model_presets": {},
        }
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump(bad_yaml))

        with patch("importlib.resources.files") as mock_files, patch("zvisiongenerator.utils.config.get_ziv_data_dir") as mock_dir:
            mock_dir.return_value = tmp_path / "nope"  # no user override
            mock_files.return_value.joinpath.return_value = config_file
            with pytest.raises(ValueError, match="default_guidance.*number"):
                load_config()

    def test_size_width_not_int_raises(self, tmp_path):
        """A size entry with string width should be rejected."""
        from unittest.mock import patch
        import yaml

        bad_yaml = {
            "sizes": {"1:1": {"m": {"width": "wide", "height": 100}}},
            "generation": {"default_steps": 10, "default_guidance": 3.5},
            "sharpening": {},
            "upscale": {},
            "schedulers": {},
            "model_presets": {},
        }
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump(bad_yaml))

        with patch("importlib.resources.files") as mock_files, patch("zvisiongenerator.utils.config.get_ziv_data_dir") as mock_dir:
            mock_dir.return_value = tmp_path / "nope"
            mock_files.return_value.joinpath.return_value = config_file
            with pytest.raises(ValueError, match=r"sizes\.1:1\.m\.width.*integer"):
                load_config()

    def test_size_not_dict_raises(self, tmp_path):
        """A size entry that is a scalar should be rejected."""
        from unittest.mock import patch
        import yaml

        bad_yaml = {
            "sizes": {"1:1": "medium"},
            "generation": {"default_steps": 10, "default_guidance": 3.5},
            "sharpening": {},
            "upscale": {},
            "schedulers": {},
            "model_presets": {},
        }
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump(bad_yaml))

        with patch("importlib.resources.files") as mock_files, patch("zvisiongenerator.utils.config.get_ziv_data_dir") as mock_dir:
            mock_dir.return_value = tmp_path / "nope"
            mock_files.return_value.joinpath.return_value = config_file
            with pytest.raises(ValueError, match=r"sizes\.1:1.*mapping.*size"):
                load_config()

    def test_unknown_default_ratio_raises(self, tmp_path):
        """default_ratio referencing a ratio not in sizes should be rejected."""
        from unittest.mock import patch
        import yaml

        bad_yaml = {
            "sizes": {"1:1": {"m": {"width": 1024, "height": 1024}}},
            "generation": {"default_steps": 10, "default_guidance": 3.5, "default_ratio": "4:3"},
            "sharpening": {},
            "upscale": {},
            "schedulers": {},
            "model_presets": {},
        }
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump(bad_yaml))

        with patch("importlib.resources.files") as mock_files, patch("zvisiongenerator.utils.config.get_ziv_data_dir") as mock_dir:
            mock_dir.return_value = tmp_path / "nope"
            mock_files.return_value.joinpath.return_value = config_file
            with pytest.raises(ValueError, match=r"default_ratio.*4:3.*not a valid ratio"):
                load_config()

    def test_unknown_default_size_raises(self, tmp_path):
        """default_size referencing a size not in the ratio's entries should be rejected."""
        from unittest.mock import patch
        import yaml

        bad_yaml = {
            "sizes": {"1:1": {"m": {"width": 1024, "height": 1024}}},
            "generation": {"default_steps": 10, "default_guidance": 3.5, "default_ratio": "1:1", "default_size": "xxl"},
            "sharpening": {},
            "upscale": {},
            "schedulers": {},
            "model_presets": {},
        }
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump(bad_yaml))

        with patch("importlib.resources.files") as mock_files, patch("zvisiongenerator.utils.config.get_ziv_data_dir") as mock_dir:
            mock_dir.return_value = tmp_path / "nope"
            mock_files.return_value.joinpath.return_value = config_file
            with pytest.raises(ValueError, match=r"default_size.*xxl.*not a valid size.*1:1"):
                load_config()


# ---------------------------------------------------------------------------
# _deep_merge
# ---------------------------------------------------------------------------


class TestDeepMerge:
    def test_flat_override(self):
        base = {"a": 1, "b": 2}
        _deep_merge(base, {"b": 99})
        assert base == {"a": 1, "b": 99}

    def test_new_key(self):
        base = {"a": 1}
        _deep_merge(base, {"b": 2})
        assert base == {"a": 1, "b": 2}

    def test_nested_merge(self):
        base = {"a": {"x": 1, "y": 2}}
        _deep_merge(base, {"a": {"y": 99, "z": 3}})
        assert base == {"a": {"x": 1, "y": 99, "z": 3}}

    def test_override_dict_with_scalar(self):
        base = {"a": {"x": 1}}
        _deep_merge(base, {"a": 42})
        assert base == {"a": 42}

    def test_override_scalar_with_dict(self):
        base = {"a": 42}
        _deep_merge(base, {"a": {"x": 1}})
        assert base == {"a": {"x": 1}}

    def test_empty_override(self):
        base = {"a": 1}
        _deep_merge(base, {})
        assert base == {"a": 1}


# ---------------------------------------------------------------------------
# get_variant_key
# ---------------------------------------------------------------------------


class TestGetVariantKey:
    def test_flux2_klein_distilled(self):
        info = ImageModelInfo(family="flux2_klein", is_distilled=True, size="4b")
        assert get_variant_key(info) == "distilled"

    def test_flux2_klein_base(self):
        info = ImageModelInfo(family="flux2_klein", is_distilled=False, size="9b")
        assert get_variant_key(info) == "base"

    def test_zimage_returns_none(self):
        info = ImageModelInfo(family="zimage", is_distilled=False, size=None)
        assert get_variant_key(info) is None

    def test_flux1_returns_none(self):
        info = ImageModelInfo(family="flux1", is_distilled=False, size=None)
        assert get_variant_key(info) is None

    def test_unknown_returns_none(self):
        info = ImageModelInfo(family="unknown", is_distilled=False, size=None)
        assert get_variant_key(info) is None


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
        },
    }

    @pytest.fixture(autouse=True)
    def _load(self):
        import copy

        self.config = copy.deepcopy(self._SYNTHETIC_CONFIG)

    def test_zimage_mflux(self):
        info = ImageModelInfo(family="zimage", is_distilled=False, size=None)
        result = resolve_defaults(info, self.config, {}, "mflux")
        assert result["steps"] == 8
        assert result["guidance"] == 0.7
        assert result["scheduler"] == "beta"

    def test_zimage_diffusers(self):
        info = ImageModelInfo(family="zimage", is_distilled=False, size=None)
        result = resolve_defaults(info, self.config, {}, "diffusers")
        assert result["steps"] == 8
        assert result["guidance"] == 0.7
        assert result["scheduler"] is None

    def test_flux2_klein_distilled(self):
        info = ImageModelInfo(family="flux2_klein", is_distilled=True, size="4b")
        result = resolve_defaults(info, self.config, {}, "mflux")
        assert result["steps"] == 6
        assert result["guidance"] == 1.2
        assert result["scheduler"] is None

    def test_flux2_klein_base(self):
        info = ImageModelInfo(family="flux2_klein", is_distilled=False, size="9b")
        result = resolve_defaults(info, self.config, {}, "mflux")
        assert result["steps"] == 40
        assert result["guidance"] == 1.8

    def test_flux1(self):
        info = ImageModelInfo(family="flux1", is_distilled=False, size=None)
        result = resolve_defaults(info, self.config, {}, "mflux")
        assert result["steps"] == 22
        assert result["guidance"] == 3.0

    def test_unknown_falls_to_global(self):
        info = ImageModelInfo(family="unknown", is_distilled=False, size=None)
        result = resolve_defaults(info, self.config, {}, "mflux")
        assert result["steps"] == 15
        assert result["guidance"] == 4.0

    def test_cli_overrides_win(self):
        """CLI --steps 20 overrides Klein distilled preset (default_steps=6)."""
        info = ImageModelInfo(family="flux2_klein", is_distilled=True, size="4b")
        result = resolve_defaults(info, self.config, {"steps": 20}, "mflux")
        assert result["steps"] == 20
        # guidance still from preset variant
        assert result["guidance"] == 1.2

    def test_cli_none_ignored(self):
        """CLI overrides with None values do not replace preset defaults."""
        info = ImageModelInfo(family="zimage", is_distilled=False, size=None)
        result = resolve_defaults(info, self.config, {"steps": None}, "mflux")
        assert result["steps"] == 8

    def test_nonexistent_family_uses_global(self):
        """A family not in model_presets falls back to global defaults."""
        info = ImageModelInfo(family="totally_new", is_distilled=False, size=None)
        result = resolve_defaults(info, self.config, {}, "mflux")
        assert result["steps"] == 15
        assert result["guidance"] == 4.0
        assert result["scheduler"] is None

    def test_string_default_scheduler_does_not_crash(self):
        """resolve_defaults should not crash if default_scheduler is a string."""
        import copy

        config = copy.deepcopy(self._SYNTHETIC_CONFIG)
        # Inject a string where a dict is expected
        preset = config["model_presets"]["zimage"]
        preset["default_scheduler"] = "beta"
        info = ImageModelInfo(family="zimage", is_distilled=False, size=None)
        result = resolve_defaults(info, config, {}, "mflux")
        assert result["scheduler"] is None

    def test_cli_overrides_scheduler(self):
        """CLI --scheduler override takes precedence over preset."""
        info = ImageModelInfo(family="zimage", is_distilled=False, size=None)
        result = resolve_defaults(info, self.config, {"scheduler": "euler"}, "mflux")
        assert result["scheduler"] == "euler"

    def test_flux2_preset_resolves(self):
        """flux2 family resolves from config."""
        info = ImageModelInfo(family="flux2", is_distilled=False, size=None)
        result = resolve_defaults(info, self.config, {}, "mflux")
        assert result["steps"] == 25
        assert result["guidance"] == 4.5

    def test_multiple_cli_overrides(self):
        """Multiple CLI overrides all apply."""
        info = ImageModelInfo(family="zimage", is_distilled=False, size=None)
        result = resolve_defaults(info, self.config, {"steps": 30, "guidance": 2.0, "scheduler": "euler"}, "mflux")
        assert result["steps"] == 30
        assert result["guidance"] == 2.0
        assert result["scheduler"] == "euler"

    def test_klein_variant_missing_key_falls_to_family(self):
        """If variant exists but has no 'default_steps', family default is used."""
        import copy

        cfg = copy.deepcopy(self.config)
        del cfg["model_presets"]["flux2_klein"]["variants"]["distilled"]["default_steps"]
        info = ImageModelInfo(family="flux2_klein", is_distilled=True, size="4b")
        result = resolve_defaults(info, cfg, {}, "mflux")
        assert result["steps"] == cfg["generation"]["default_steps"]


# ---------------------------------------------------------------------------
# Config schema validation after merge
# ---------------------------------------------------------------------------


class TestConfigSchemaValidation:
    """load_config() should reject configs where known sections have wrong types."""

    def test_sizes_replaced_with_list_raises(self, tmp_path, monkeypatch):
        from zvisiongenerator.utils import config as config_mod

        # Write a user config that replaces 'sizes' with a list
        user_cfg = tmp_path / "config.yaml"
        user_cfg.write_text("sizes:\n  - bad\n")
        monkeypatch.setattr(config_mod, "get_ziv_data_dir", lambda: tmp_path)
        with pytest.raises(ValueError, match="Config section 'sizes' must be a mapping"):
            load_config()

    def test_generation_replaced_with_scalar_raises(self, tmp_path, monkeypatch):
        from zvisiongenerator.utils import config as config_mod

        user_cfg = tmp_path / "config.yaml"
        user_cfg.write_text("generation: 42\n")
        monkeypatch.setattr(config_mod, "get_ziv_data_dir", lambda: tmp_path)
        with pytest.raises(ValueError, match="Config section 'generation' must be a mapping"):
            load_config()

    def test_valid_user_override_accepted(self, tmp_path, monkeypatch):
        from zvisiongenerator.utils import config as config_mod

        user_cfg = tmp_path / "config.yaml"
        user_cfg.write_text("generation:\n  default_steps: 99\n")
        monkeypatch.setattr(config_mod, "get_ziv_data_dir", lambda: tmp_path)
        cfg = load_config()
        assert cfg["generation"]["default_steps"] == 99


class TestPlatformConfigValidation:
    def test_platforms_section_must_be_mapping(self, tmp_path):
        from unittest.mock import patch

        packaged = _write_packaged_config(
            tmp_path,
            {
                "sizes": {"1:1": {"m": {"width": 100, "height": 100}}},
                "generation": {"default_steps": 10, "default_guidance": 3.5},
                "sharpening": {},
                "upscale": {},
                "contrast": {},
                "saturation": {},
                "schedulers": {},
                "platforms": ["darwin"],
                "model_aliases": {},
                "model_presets": {},
                "video_sizes": {},
                "video_generation": {},
                "video_model_presets": {},
            },
        )

        with patch("importlib.resources.files") as mock_files, patch("zvisiongenerator.utils.config.get_ziv_data_dir") as mock_dir:
            mock_dir.return_value = tmp_path / "no-user-config"
            mock_files.return_value.joinpath.return_value = packaged
            with pytest.raises(ValueError, match="Config section 'platforms' must be a mapping"):
                load_config()

    @pytest.mark.parametrize(
        "alias_value",
        [
            "Tongyi-MAI/Z-Image-Turbo",
            {"darwin": "dgrauet/ltx-2.3-mlx-q8", "win32": "Lightricks/LTX-2.3-fp8"},
            {"darwin": "dgrauet/ltx-2.3-mlx-q4", "win32": {"message": "Use ltx-8 instead."}},
        ],
    )
    def test_polymorphic_alias_values_accept_supported_shapes(self, tmp_path, alias_value):
        from unittest.mock import patch

        packaged = _write_packaged_config(
            tmp_path,
            {
                "sizes": {"1:1": {"m": {"width": 100, "height": 100}}},
                "generation": {"default_steps": 10, "default_guidance": 3.5},
                "sharpening": {},
                "upscale": {},
                "contrast": {},
                "saturation": {},
                "schedulers": {},
                "platforms": {"darwin": "macOS", "win32": "Windows"},
                "model_aliases": {"ltx": alias_value},
                "model_presets": {},
                "video_sizes": {},
                "video_generation": {},
                "video_model_presets": {},
            },
        )

        with patch("importlib.resources.files") as mock_files, patch("zvisiongenerator.utils.config.get_ziv_data_dir") as mock_dir:
            mock_dir.return_value = tmp_path / "no-user-config"
            mock_files.return_value.joinpath.return_value = packaged

            cfg = load_config()

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
        from unittest.mock import patch

        packaged = _write_packaged_config(
            tmp_path,
            {
                "sizes": {"1:1": {"m": {"width": 100, "height": 100}}},
                "generation": {"default_steps": 10, "default_guidance": 3.5},
                "sharpening": {},
                "upscale": {},
                "contrast": {},
                "saturation": {},
                "schedulers": {},
                "platforms": {"darwin": "macOS", "win32": "Windows"},
                "model_aliases": {"ltx": alias_value},
                "model_presets": {},
                "video_sizes": {},
                "video_generation": {},
                "video_model_presets": {},
            },
        )

        with patch("importlib.resources.files") as mock_files, patch("zvisiongenerator.utils.config.get_ziv_data_dir") as mock_dir:
            mock_dir.return_value = tmp_path / "no-user-config"
            mock_files.return_value.joinpath.return_value = packaged
            with pytest.raises(ValueError, match=match):
                load_config()


# ---------------------------------------------------------------------------
# Scheduler resolution via runner helper
# ---------------------------------------------------------------------------


class TestSchedulerResolution:
    """_resolve_scheduler_class resolves scheduler names to backend class paths."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.config = load_config()

    def test_beta_scheduler_class(self):
        from zvisiongenerator.image_runner import _resolve_scheduler_class

        resolved = _resolve_scheduler_class("beta", self.config, "mflux")
        assert resolved == "zvisiongenerator.schedulers.beta_scheduler.BetaScheduler"
        assert resolved == self.config["schedulers"]["beta"]["mflux_class"]

    def test_resolve_unknown_scheduler_passthrough(self):
        from zvisiongenerator.image_runner import _resolve_scheduler_class

        assert _resolve_scheduler_class("euler", self.config, "mflux") == "euler"

    def test_resolve_none_scheduler(self):
        from zvisiongenerator.image_runner import _resolve_scheduler_class

        assert _resolve_scheduler_class(None, self.config, "mflux") is None


# ---------------------------------------------------------------------------
# _deep_merge — additional edge cases
# ---------------------------------------------------------------------------


class TestDeepMergeEdgeCases:
    def test_deeply_nested(self):
        base = {"a": {"b": {"c": {"d": 1}}}}
        _deep_merge(base, {"a": {"b": {"c": {"d": 2, "e": 3}}}})
        assert base == {"a": {"b": {"c": {"d": 2, "e": 3}}}}

    def test_list_replaced_not_merged(self):
        """Lists are replaced, not merged element-by-element."""
        base = {"a": [1, 2, 3]}
        _deep_merge(base, {"a": [4, 5]})
        assert base == {"a": [4, 5]}


# ---------------------------------------------------------------------------
# Scheduler validation
# ---------------------------------------------------------------------------


class TestSchedulerValidation:
    """Verify validate_scheduler works."""

    def test_validate_known_scheduler(self):
        from zvisiongenerator.utils.config import validate_scheduler

        config = load_config()
        validate_scheduler("beta", config)  # Should not raise

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
            "ltx": {
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
        },
        "video_model_presets": {
            "ltx": {
                "default_steps": 8,
            },
        },
    }

    def test_ltx_defaults(self):
        result = resolve_video_defaults("ltx", self._VIDEO_CONFIG, {})
        assert result["steps"] == 8
        assert result["width"] == 704
        assert result["height"] == 480
        assert result["num_frames"] == 65
        assert result["ratio"] == "16:9"
        assert result["size"] == "m"

    def test_cli_overrides_take_precedence(self):
        overrides = {"steps": 10}
        result = resolve_video_defaults("ltx", self._VIDEO_CONFIG, overrides)
        assert result["steps"] == 10
        # Non-overridden values still come from preset
        assert result["width"] == 704

    def test_cli_none_values_ignored(self):
        overrides = {"steps": None}
        result = resolve_video_defaults("ltx", self._VIDEO_CONFIG, overrides)
        assert result["steps"] == 8

    @pytest.mark.parametrize(
        "family",
        ["unknown_family"],
    )
    def test_missing_config_falls_to_hardcoded(self, family):
        empty_config: dict = {}
        result = resolve_video_defaults(family, empty_config, {})
        assert result["steps"] == 8
        assert result["width"] == 704
        assert result["height"] == 448
        assert result["num_frames"] == 49

    def test_ratio_override_selects_preset(self):
        overrides = {"ratio": "9:16", "size": "m"}
        result = resolve_video_defaults("ltx", self._VIDEO_CONFIG, overrides)
        assert result["width"] == 480
        assert result["height"] == 704
        assert result["num_frames"] == 49
        assert result["ratio"] == "9:16"
        assert result["size"] == "m"

    def test_size_override_selects_preset(self):
        overrides = {"ratio": "16:9", "size": "s"}
        result = resolve_video_defaults("ltx", self._VIDEO_CONFIG, overrides)
        assert result["width"] == 512
        assert result["height"] == 288
        assert result["num_frames"] == 49

    def test_ratio_size_with_width_override(self):
        """Explicit --width overrides preset width but height/frames come from preset."""
        overrides = {"ratio": "16:9", "size": "m", "width": 640}
        result = resolve_video_defaults("ltx", self._VIDEO_CONFIG, overrides)
        assert result["width"] == 640
        assert result["height"] == 480
        assert result["num_frames"] == 65

    def test_ratio_size_with_all_dim_overrides(self):
        """Explicit width/height/num_frames override all preset dimensions."""
        overrides = {"ratio": "16:9", "size": "m", "width": 640, "height": 360, "num_frames": 33}
        result = resolve_video_defaults("ltx", self._VIDEO_CONFIG, overrides)
        assert result["width"] == 640
        assert result["height"] == 360
        assert result["num_frames"] == 33

    def test_default_ratio_size_from_config(self):
        """When no ratio/size in overrides, config defaults are used."""
        result = resolve_video_defaults("ltx", self._VIDEO_CONFIG, {})
        assert result["ratio"] == "16:9"
        assert result["size"] == "m"

    def test_ratio_1_1_preset(self):
        overrides = {"ratio": "1:1", "size": "m"}
        result = resolve_video_defaults("ltx", self._VIDEO_CONFIG, overrides)
        assert result["width"] == 512
        assert result["height"] == 512
        assert result["num_frames"] == 33
