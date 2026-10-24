"""Tests for config loading, parity with hardcoded values, and resolution logic."""

from __future__ import annotations

import re
from copy import deepcopy

import pytest

from zvisiongenerator.utils.config import (
    EULER_ANCESTRAL,
    LTX_SAMPLERS,
    _deep_merge,
    get_variant_key,
    load_config,
    ltx_diffusers_quantize,
    ltx_latent_upsampler_source,
    ltx_load_options,
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
from zvisiongenerator.utils.image_model_detect import ImageModelInfo
from zvisiongenerator.utils.paths import resolve_model_path


_SEEDVR2_BLOCK = {
    "model": "auto",
    "softness": 0.75,
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

    def test_model_alias_reference_resolves_after_user_override(self, tmp_path):
        """Config references should follow the merged default repo value."""
        from unittest.mock import patch
        import yaml

        packaged = {
            "sizes": {"1:1": {"m": {"width": 100, "height": 100}}},
            "generation": {"default_steps": 10, "default_guidance": 3.5, "default_ratio": "1:1", "default_size": "m"},
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
        config_file = _write_packaged_config(tmp_path, packaged)
        user_dir = tmp_path / ".ziv"
        user_dir.mkdir()
        (user_dir / "config.yaml").write_text(
            yaml.dump({"video_model_presets": {"ltx": {"diffusers": {"default_repo": "custom/LTX-2.3"}}}}),
            encoding="utf-8",
        )

        with patch("importlib.resources.files") as mock_files, patch("zvisiongenerator.utils.config.get_ziv_data_dir", return_value=user_dir):
            mock_files.return_value.joinpath.return_value = config_file
            config = load_config()

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


def test_packaged_config_pins_every_ideogram4_instant_source(monkeypatch, tmp_path):
    from zvisiongenerator.utils import config as config_module
    from zvisiongenerator.utils.ideogram4_instant import instant_source

    monkeypatch.setattr(config_module, "get_ziv_data_dir", lambda: tmp_path / ".ziv")

    config = load_config()

    formats = {}
    for platform_key in ("darwin", "win32", "linux"):
        source = instant_source(config, platform_key)
        assert re.fullmatch(r"[0-9a-f]{40}", source.transformer_revision)
        assert source.components_revision is not None and re.fullmatch(r"[0-9a-f]{40}", source.components_revision)
        formats[platform_key] = source.transformer_format
    assert formats == {"darwin": "fp8", "win32": "bf16", "linux": "bf16"}


def test_packaged_config_ideogram4_preset_is_instant(monkeypatch, tmp_path):
    from zvisiongenerator.utils import config as config_module

    monkeypatch.setattr(config_module, "get_ziv_data_dir", lambda: tmp_path / ".ziv")

    defaults = resolve_defaults(ImageModelInfo(family="ideogram4", is_distilled=False, size=None), load_config(), {}, "diffusers")

    assert (defaults["steps"], defaults["guidance"], defaults["supports_quantize"]) == (8, 1.0, True)


def test_packaged_config_ships_krea2_alias_and_defaults(monkeypatch, tmp_path):
    from zvisiongenerator.utils import config as config_module

    monkeypatch.setattr(config_module, "get_ziv_data_dir", lambda: tmp_path / ".ziv")

    config = load_config()
    turbo = ImageModelInfo(family="krea2", is_distilled=False, size=None)

    for platform_key in ("darwin", "win32", "linux"):
        assert resolve_model_path("krea2", aliases=config["model_aliases"], platform_key=platform_key) == "unsloth/Krea-2-Turbo"
    assert config["model_alias_families"]["krea2"] == "krea2"
    for backend_name in ("mflux", "diffusers"):
        defaults = resolve_defaults(turbo, config, {}, backend_name)
        assert (defaults["steps"], defaults["guidance"], defaults["scheduler"]) == (8, 1.0, None)
        assert defaults["supports_negative_prompt"] is False
        assert defaults["supports_scheduler"] is False


def test_packaged_config_klein9b_uses_ungated_mirror(monkeypatch, tmp_path):
    from zvisiongenerator.utils import config as config_module

    monkeypatch.setattr(config_module, "get_ziv_data_dir", lambda: tmp_path / ".ziv")

    aliases = load_config()["model_aliases"]

    assert aliases["klein9b"] == "unsloth/FLUX.2-klein-9B"
    for platform_key in ("darwin", "win32", "linux"):
        resolved = set()
        for name in aliases:
            try:
                resolved.add(resolve_model_path(name, aliases=aliases, platform_key=platform_key))
            except ValueError:  # an alias that is not available on this platform
                continue
        assert "black-forest-labs/FLUX.2-klein-9B" not in resolved


def test_packaged_config_krea2_capabilities(monkeypatch, tmp_path):
    from zvisiongenerator.utils import config as config_module

    monkeypatch.setattr(config_module, "get_ziv_data_dir", lambda: tmp_path / ".ziv")
    config = load_config()

    caps = model_capabilities(config, "krea2")

    assert (caps["supports_img2img"], caps["supports_upscale"]) == (True, True)


def test_new_capability_flags_default_permissive():
    caps = model_capabilities({}, "unknown")
    assert caps["supports_scheduler"] is True


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
        import copy

        self.config = copy.deepcopy(self._SYNTHETIC_CONFIG)

    def test_zimage_mflux(self):
        info = ImageModelInfo(family="zimage", is_distilled=False, size=None)
        result = resolve_defaults(info, self.config, {}, "mflux")
        assert result["steps"] == 8
        assert result["guidance"] == 0.7
        assert result["upscale_steps"] == 8
        assert result["scheduler"] == "beta"
        assert result["supports_quantize"] is True
        assert result["supports_img2img"] is True
        assert result["supports_upscale"] is True
        assert result["supports_json_prompt"] is False
        assert result["supports_first_sigma"] is False
        assert result["dimension_min"] == 16
        assert result["dimension_max"] is None
        assert result["dimension_step"] == 16

    def test_zimage_diffusers(self):
        info = ImageModelInfo(family="zimage", is_distilled=False, size=None)
        result = resolve_defaults(info, self.config, {}, "diffusers")
        assert result["steps"] == 8
        assert result["guidance"] == 0.7
        assert result["upscale_steps"] == 8
        assert result["scheduler"] is None

    def test_cli_overrides_upscale_steps(self):
        """CLI --upscale-steps override takes precedence over preset upscale default."""
        info = ImageModelInfo(family="zimage", is_distilled=False, size=None)
        result = resolve_defaults(info, self.config, {"upscale_steps": 20}, "mflux")
        assert result["upscale_steps"] == 20

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

    def test_ideogram4(self):
        info = ImageModelInfo(family="ideogram4", is_distilled=False, size=None)
        result = resolve_defaults(info, self.config, {}, "mflux")
        assert result["steps"] == 20
        assert result["guidance"] == 7.0
        assert result["scheduler"] is None
        assert result["supports_negative_prompt"] is False
        assert result["supports_quantize"] is False
        assert result["supports_img2img"] is False
        assert result["supports_upscale"] is False
        assert result["supports_json_prompt"] is True
        assert result["supports_first_sigma"] is True
        assert result["dimension_min"] == 256
        assert result["dimension_max"] == 2048
        assert result["dimension_step"] == 16

    def test_ideogram4_cli_overrides_win(self):
        info = ImageModelInfo(family="ideogram4", is_distilled=False, size=None)
        result = resolve_defaults(info, self.config, {"steps": 12, "guidance": 3.5}, "mflux")
        assert result["steps"] == 12
        assert result["guidance"] == 3.5

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

    def test_seedvr2_replaced_with_scalar_raises(self, tmp_path, monkeypatch):
        from zvisiongenerator.utils import config as config_mod

        (tmp_path / "config.yaml").write_text("seedvr2: 42\n")
        monkeypatch.setattr(config_mod, "get_ziv_data_dir", lambda: tmp_path)
        with pytest.raises(ValueError, match="Config section 'seedvr2' must be a mapping"):
            load_config()

    def test_seedvr2_user_override_is_validated(self, tmp_path, monkeypatch):
        from zvisiongenerator.utils import config as config_mod

        (tmp_path / "config.yaml").write_text("seedvr2:\n  softness: 2\n")
        monkeypatch.setattr(config_mod, "get_ziv_data_dir", lambda: tmp_path)
        with pytest.raises(ValueError, match="seedvr2.softness"):
            load_config()

    def test_seedvr2_model_override_is_merged(self, tmp_path, monkeypatch):
        from zvisiongenerator.utils import config as config_mod

        (tmp_path / "config.yaml").write_text("seedvr2:\n  model: seedvr2-3b\n")
        monkeypatch.setattr(config_mod, "get_ziv_data_dir", lambda: tmp_path)
        cfg = load_config()
        assert cfg["seedvr2"]["model"] == "seedvr2-3b"
        assert [row["model"] for row in cfg["seedvr2"]["tiers"]] == ["seedvr2-7b", "seedvr2-3b"]

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

    def test_sharpening_amounts_above_the_cap_are_rejected(self, tmp_path):
        from unittest.mock import patch

        packaged = _write_packaged_config(
            tmp_path,
            {
                "sizes": {"1:1": {"m": {"width": 100, "height": 100}}},
                "generation": {"default_steps": 10, "default_guidance": 3.5},
                "sharpening": {"normal": 1.0, "upscaled": 2.0},
            },
        )

        with patch("importlib.resources.files") as mock_files, patch("zvisiongenerator.utils.config.get_ziv_data_dir") as mock_dir:
            mock_dir.return_value = tmp_path / "no-user-config"
            mock_files.return_value.joinpath.return_value = packaged
            with pytest.raises(ValueError, match="sharpening.upscaled"):
                load_config()

    @pytest.mark.parametrize(
        "alias_value",
        [
            "Tongyi-MAI/Z-Image-Turbo",
            {"win32": "dg845/LTX-2.3-Diffusers", "linux": "dg845/LTX-2.3-Diffusers"},
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
                "seedvr2": _SEEDVR2_BLOCK,
                "contrast": {},
                "saturation": {},
                "schedulers": {},
                "platforms": {"darwin": "macOS", "win32": "Windows", "linux": "Linux"},
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
                "platforms": {"darwin": "macOS", "win32": "Windows", "linux": "Linux"},
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
    """resolve_scheduler_class resolves scheduler names to backend class paths."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.config = load_config()

    def test_beta_scheduler_class(self):
        from zvisiongenerator.utils.config import resolve_scheduler_class

        resolved = resolve_scheduler_class("beta", self.config, "mflux")
        assert resolved == "zvisiongenerator.schedulers.beta_scheduler.BetaScheduler"
        assert resolved == self.config["schedulers"]["beta"]["mflux_class"]

    def test_resolve_unknown_scheduler_passthrough(self):
        from zvisiongenerator.utils.config import resolve_scheduler_class

        assert resolve_scheduler_class("euler", self.config, "mflux") == "euler"

    def test_resolve_none_scheduler(self):
        from zvisiongenerator.utils.config import resolve_scheduler_class

        assert resolve_scheduler_class(None, self.config, "mflux") is None


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

    @pytest.mark.parametrize(
        "family",
        ["unknown_family"],
    )
    def test_missing_config_falls_to_hardcoded(self, family):
        empty_config: dict = {}
        result = resolve_video_defaults(family, empty_config, {})
        assert result["steps"] == 8
        assert result["width"] == 768
        assert result["height"] == 448
        assert result["num_frames"] == 121

    def test_every_packaged_video_size_is_121_frames(self, monkeypatch, tmp_path):
        sizes = _packaged_config(monkeypatch, tmp_path)["video_sizes"]

        frames = {(ratio, size): dims["frames"] for ratio, entries in sizes.items() for size, dims in entries.items()}
        assert len(frames) == 15
        assert set(frames.values()) == {121}

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

    @pytest.fixture(scope="class")
    def config(self):
        return load_config()

    def test_every_ratio_has_the_five_tiers(self, config):
        for ratio, tiers in config["video_sizes"].items():
            assert list(tiers) == ["xs", "s", "m", "l", "xl"], ratio

    def test_sizes_are_64_aligned_and_frames_are_8k_plus_1(self, config):
        for ratio, tiers in config["video_sizes"].items():
            for size, dims in tiers.items():
                assert dims["width"] % 64 == 0 and dims["height"] % 64 == 0, f"{ratio}/{size}"
                assert (dims["frames"] - 1) % 8 == 0, f"{ratio}/{size}"

    def test_the_default_size_is_listed(self, config):
        video = config["video_generation"]
        assert video["default_size"] in config["video_sizes"][video["default_ratio"]]


class TestLtxLatentUpsamplerSource:
    """Verify the latent upsampler source helper reads and validates its config key."""

    @staticmethod
    def _config(value):
        return {"video_model_presets": {"ltx": {"diffusers": {"latent_upsampler": value}}}}

    def test_packaged_config_ships_upsampler_and_distilled_default_repo(self, monkeypatch, tmp_path):
        from zvisiongenerator.utils import config as config_module

        monkeypatch.setattr(config_module, "get_ziv_data_dir", lambda: tmp_path / ".ziv")

        config = load_config()

        assert ltx_latent_upsampler_source(config) == ("Lightricks/LTX-2.3", "ltx-2.3-spatial-upscaler-x2-1.1.safetensors")
        assert config["video_model_presets"]["ltx"]["diffusers"]["default_repo"] == "diffusers/LTX-2.3-Distilled-Diffusers"

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

    def test_packaged_config_ships_the_distilled_1_1_transformer(self, monkeypatch, tmp_path):
        from zvisiongenerator.utils import config as config_module

        monkeypatch.setattr(config_module, "get_ziv_data_dir", lambda: tmp_path / ".ziv")

        assert ltx_transformer_source(load_config()) == ("Lightricks/LTX-2.3", "ltx-2.3-22b-distilled-1.1.safetensors")

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

    def test_packaged_config_quantizes_ltx_4_and_ltx_8_only(self, monkeypatch, tmp_path):
        from zvisiongenerator.utils import config as config_module

        monkeypatch.setattr(config_module, "get_ziv_data_dir", lambda: tmp_path / ".ziv")
        config = load_config()

        assert [ltx_diffusers_quantize(config, name) for name in ("ltx-4", "ltx-8", "other", None)] == [4, 8, None, None]

    @pytest.mark.parametrize("platform_key", ["win32", "linux"])
    def test_packaged_ltx_aliases_resolve_to_the_diffusers_repo_off_macos(self, monkeypatch, tmp_path, platform_key):
        from zvisiongenerator.utils import config as config_module

        monkeypatch.setattr(config_module, "get_ziv_data_dir", lambda: tmp_path / ".ziv")
        monkeypatch.setattr("zvisiongenerator.utils.paths.get_ziv_data_dir", lambda: tmp_path / ".ziv")
        config = load_config()
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
        assert ltx_load_options(self._CONFIG, "ltx-4", upscale=False) == {"quantize": 4, "transformer_source": ("org/ltx", "ltx.safetensors"), "model_name": "ltx-4"}

    def test_upscale_adds_the_upsampler(self):
        options = ltx_load_options(self._CONFIG, "other", upscale=True)

        assert options["quantize"] is None
        assert options["upscale"] is True
        assert options["latent_upsampler"] == ("org/up", "up.safetensors")


_LTX25_PIN = "unsloth/LTX-2.5-FP8@7b40724fc39bbb01d9b4d72638c0d263531c88c2"
_LTX25_UPSAMPLER = (_LTX25_PIN, "latent_upsampler/diffusion_pytorch_model.safetensors")
_LTX25_MAC_PIN = "MarceloFSO/ltx-2.5-mlx-q4-fast@281bf05bd23207d6feef708fc2e270f84ba992b5"
_LTX23_REPO = "diffusers/LTX-2.3-Distilled-Diffusers"
_LTX23_TRANSFORMER = ("Lightricks/LTX-2.3", "ltx-2.3-22b-distilled-1.1.safetensors")
_LTX23_UPSAMPLER = ("Lightricks/LTX-2.3", "ltx-2.3-spatial-upscaler-x2-1.1.safetensors")


def _packaged_config(monkeypatch, tmp_path):
    """Load the packaged config with no user override."""
    from zvisiongenerator.utils import config as config_module

    monkeypatch.setattr(config_module, "get_ziv_data_dir", lambda: tmp_path / ".ziv")
    monkeypatch.setattr("zvisiongenerator.utils.paths.get_ziv_data_dir", lambda: tmp_path / ".ziv")
    return load_config()


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
    def test_ltx25_4_resolves_to_the_pinned_repo(self, monkeypatch, tmp_path, platform_key):
        config = _packaged_config(monkeypatch, tmp_path)

        assert resolve_model_path("ltx25-4", aliases=config["model_aliases"], platform_key=platform_key) == _LTX25_PIN

    def test_ltx25_4_resolves_to_the_pinned_mirror_on_macos(self, monkeypatch, tmp_path):
        config = _packaged_config(monkeypatch, tmp_path)

        assert resolve_model_path("ltx25-4", aliases=config["model_aliases"], platform_key="darwin") == _LTX25_MAC_PIN

    def test_ltx25_4_is_labelled(self, monkeypatch, tmp_path):
        config = _packaged_config(monkeypatch, tmp_path)

        assert model_display_name(config, "ltx25-4") == "LTX-2.5 4-bit"

    def test_the_pin_is_written_once_and_referenced(self, monkeypatch, tmp_path):
        config = _packaged_config(monkeypatch, tmp_path)
        version = config["video_model_presets"]["ltx"]["versions"]["ltx25"]

        assert version["aliases"] == ["ltx25-4"]
        assert version["diffusers"]["default_repo"] == _LTX25_PIN
        assert version["diffusers"]["latent_upsampler"]["repo"] == _LTX25_PIN
        assert config["model_aliases"]["ltx25-4"]["win32"] == config["model_aliases"]["ltx25-4"]["linux"] == _LTX25_PIN
        assert config["model_aliases"]["ltx25-4"]["darwin"] == _LTX25_MAC_PIN
        assert config["video_model_presets"]["ltx"]["diffusers"]["default_repo"] == _LTX23_REPO

    def test_the_version_block_carries_the_2_5_settings(self, monkeypatch, tmp_path):
        config = _packaged_config(monkeypatch, tmp_path)
        diffusers = config["video_model_presets"]["ltx"]["versions"]["ltx25"]["diffusers"]

        assert "transformer" in diffusers and diffusers["transformer"] is None
        assert diffusers["quantize"] == {"ltx25-4": 4}
        assert diffusers["repo_ignore"] == ["LTX-2.5-*.safetensors", "samples/*"]
        assert (diffusers["image_crf"], diffusers["sampler"]) == (18, EULER_ANCESTRAL)

    def test_the_ltx_2_3_aliases_labels_and_family_block_are_unchanged(self, monkeypatch, tmp_path):
        config = _packaged_config(monkeypatch, tmp_path)
        assert config["model_aliases"]["ltx-8"] == {"darwin": "dgrauet/ltx-2.3-mlx-q8", "win32": _LTX23_REPO, "linux": _LTX23_REPO}
        assert config["model_aliases"]["ltx-4"] == {"darwin": "dgrauet/ltx-2.3-mlx-q4", "win32": _LTX23_REPO, "linux": _LTX23_REPO}
        assert config["model_labels"]["ltx-8"] == {"name": "LTX-2.3 8-bit"}
        assert config["model_labels"]["ltx-4"] == {"name": "LTX-2.3 4-bit"}
        family = config["video_model_presets"]["ltx"]
        assert family["diffusers"] == {
            "default_repo": _LTX23_REPO,
            "transformer": {"repo": "Lightricks/LTX-2.3", "filename": "ltx-2.3-22b-distilled-1.1.safetensors"},
            "quantize": {"ltx-4": 4, "ltx-8": 8},
            "latent_upsampler": {"repo": "Lightricks/LTX-2.3", "filename": "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"},
        }

    def test_the_first_run_and_default_models_are_untouched(self, monkeypatch, tmp_path):
        config = _packaged_config(monkeypatch, tmp_path)

        assert config["ui"]["default_models"]["video"] == "ltx-8"
        assert "ltx25-4" not in str(config["first_run"])


class TestLtxVersionAwareAccessors:
    """R3: the ``ltx_*`` accessors and load options read the version that lists the model."""

    def test_ltx25_4_load_options_with_upscale(self, monkeypatch, tmp_path):
        config = _packaged_config(monkeypatch, tmp_path)

        assert ltx_load_options(config, "ltx25-4", upscale=True) == {
            "quantize": 4,
            "transformer_source": None,
            "model_name": "ltx25-4",
            "repo_ignore": ("LTX-2.5-*.safetensors", "samples/*"),
            "sampler": "euler_ancestral",
            "image_crf": 18,
            "upscale": True,
            "latent_upsampler": _LTX25_UPSAMPLER,
        }

    def test_ltx25_4_load_options_without_upscale_have_no_upsampler(self, monkeypatch, tmp_path):
        options = ltx_load_options(_packaged_config(monkeypatch, tmp_path), "ltx25-4", upscale=False)

        assert "upscale" not in options and "latent_upsampler" not in options
        assert options["transformer_source"] is None

    @pytest.mark.parametrize("name", ["ltx-4", "ltx-8"])
    def test_ltx_2_3_load_options_are_unchanged(self, monkeypatch, tmp_path, name):
        """2.3 guard: the options of ``ltx-4`` and ``ltx-8`` carry no version key."""
        config = _packaged_config(monkeypatch, tmp_path)
        bits = {"ltx-4": 4, "ltx-8": 8}[name]
        plain = {"quantize": bits, "transformer_source": _LTX23_TRANSFORMER, "model_name": name}

        assert ltx_load_options(config, name, upscale=False) == plain
        assert ltx_load_options(config, name, upscale=True) == {**plain, "upscale": True, "latent_upsampler": _LTX23_UPSAMPLER}

    def test_ltx_2_3_sources_do_not_see_the_version(self, monkeypatch, tmp_path):
        config = _packaged_config(monkeypatch, tmp_path)

        for name in (None, "ltx-4", "ltx-8", "unknown"):
            assert ltx_transformer_source(config, name) == _LTX23_TRANSFORMER
            assert ltx_latent_upsampler_source(config, name) == _LTX23_UPSAMPLER
        assert ltx_transformer_source(config) == _LTX23_TRANSFORMER
        assert ltx_latent_upsampler_source(config) == _LTX23_UPSAMPLER

    def test_ltx25_4_sources_and_quantize_come_from_the_version(self, monkeypatch, tmp_path):
        config = _packaged_config(monkeypatch, tmp_path)

        assert ltx_transformer_source(config, "ltx25-4") is None
        assert ltx_latent_upsampler_source(config, "ltx25-4") == _LTX25_UPSAMPLER
        assert ltx_diffusers_quantize(config, "ltx25-4") == 4
        assert [ltx_diffusers_quantize(config, name) for name in ("ltx-4", "ltx-8")] == [4, 8]

    def test_a_family_transformer_override_does_not_reach_ltx25_4(self, monkeypatch, tmp_path):
        config = _packaged_config(monkeypatch, tmp_path)
        config["video_model_presets"]["ltx"]["diffusers"]["transformer"] = {"repo": "me/ltx", "filename": "mine.safetensors"}

        assert ltx_transformer_source(config, "ltx-4") == ("me/ltx", "mine.safetensors")
        assert ltx_transformer_source(config, "ltx25-4") is None
        assert ltx_load_options(config, "ltx25-4", upscale=False)["transformer_source"] is None

    @pytest.mark.parametrize("name", ["ltx-4", "ltx-8", None, "unknown", "org/some-ltx-repo"])
    def test_ancestral_sampler_outside_a_version_is_refused(self, monkeypatch, tmp_path, name):
        config = _packaged_config(monkeypatch, tmp_path)
        config["video_model_presets"]["ltx"]["diffusers"]["sampler"] = EULER_ANCESTRAL

        with pytest.raises(ValueError, match=r"video_model_presets\.ltx\.diffusers\.sampler") as excinfo:
            ltx_load_options(config, name, upscale=False)
        assert (name or "this model") in str(excinfo.value)

    def test_ancestral_sampler_is_accepted_for_a_version_alias(self, monkeypatch, tmp_path):
        config = _packaged_config(monkeypatch, tmp_path)
        config["video_model_presets"]["ltx"]["diffusers"]["sampler"] = EULER_ANCESTRAL

        assert ltx_load_options(config, "ltx25-4", upscale=False)["sampler"] == EULER_ANCESTRAL

    def test_an_invalid_sampler_still_reports_the_allowed_values(self, monkeypatch, tmp_path):
        config = _packaged_config(monkeypatch, tmp_path)
        config["video_model_presets"]["ltx"]["diffusers"]["sampler"] = "ddim"

        with pytest.raises(ValueError, match="must be one of"):
            ltx_load_options(config, "ltx-4", upscale=False)

    def test_ltx_version_reports_version_membership(self, monkeypatch, tmp_path):
        config = _packaged_config(monkeypatch, tmp_path)

        assert ltx_version(config, "ltx25-4") == "ltx25"
        assert ltx_version(config, "ltx-4") is None
        assert ltx_version(config, None) is None

    def test_a_version_without_the_new_keys_adds_none(self):
        config = _versioned(diffusers={"quantize": {"alias-v": 4}})

        assert ltx_load_options(config, "alias-v", upscale=False) == {"quantize": 4, "transformer_source": ("org/ltx", "ltx.safetensors"), "model_name": "alias-v"}

    def test_a_null_new_key_is_unset(self):
        config = _versioned(diffusers={"sampler": None, "image_crf": None, "repo_ignore": None})

        assert set(ltx_load_options(config, "alias-v", upscale=False)) == {"quantize", "transformer_source", "model_name"}

    def test_image_crf_zero_and_the_bounds_are_valid(self):
        for crf in (0, 51):
            assert ltx_load_options(_versioned(diffusers={"image_crf": crf}), "alias-v", upscale=False)["image_crf"] == crf

    def test_the_sampler_names_are_defined_once(self):
        assert EULER_ANCESTRAL == "euler_ancestral"
        assert LTX_SAMPLERS == frozenset({EULER_ANCESTRAL})

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

    def test_distilled_lora_and_family_helpers_stay_family_level(self, monkeypatch, tmp_path):
        from zvisiongenerator.utils.config import ltx_companion_repos, ltx_diffusers_repo, ltx_distilled_lora_source

        config = _packaged_config(monkeypatch, tmp_path)

        assert ltx_distilled_lora_source(config) == ("Lightricks/LTX-2.3", "ltx-2.3-22b-distilled-lora-384-1.1.safetensors")
        assert ltx_diffusers_repo(config) == _LTX23_REPO
        assert ltx_companion_repos(config, _LTX25_PIN) == ()
        assert ltx_companion_repos(config, _LTX23_REPO) == ("Lightricks/LTX-2.3",)


class TestVideoLoadOptionsGolden:
    """G4: ``video_load_options("ltx", ...)`` is exactly ``ltx_load_options`` for every LTX alias and both upscale settings."""

    @pytest.mark.parametrize("upscale", [False, True])
    @pytest.mark.parametrize("name", ["ltx-4", "ltx-8", "ltx25-4", "unknown", None])
    def test_ltx_options_are_the_ltx_options(self, monkeypatch, tmp_path, name, upscale):
        config = _packaged_config(monkeypatch, tmp_path)

        assert video_load_options("ltx", config, name, upscale=upscale) == ltx_load_options(config, name, upscale=upscale)


class TestVideoFamilyDispatch:
    """R5: load options and companions are asked of the family; default frames come from the family."""

    def test_a_family_without_load_options_is_refused(self):
        with pytest.raises(ValueError, match="No video load options for model family 'wan'"):
            video_load_options("wan", {}, "m", upscale=False)

    def test_ltx_companions_are_the_ltx_companions(self, monkeypatch, tmp_path):
        config = _packaged_config(monkeypatch, tmp_path)

        assert video_companion_repos("ltx", config, _LTX23_REPO) == ("Lightricks/LTX-2.3",)
        assert video_companion_repos("ltx", config, _LTX25_PIN) == ()

    @pytest.mark.parametrize("family", ["zimage", "flux2_klein", "wan"])
    def test_another_family_has_no_companions(self, monkeypatch, tmp_path, family):
        assert video_companion_repos(family, _packaged_config(monkeypatch, tmp_path), _LTX23_REPO) == ()

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

    def test_every_packaged_alias_has_a_name(self, monkeypatch, tmp_path):
        from zvisiongenerator.utils import config as config_module

        monkeypatch.setattr(config_module, "get_ziv_data_dir", lambda: tmp_path / ".ziv")
        config = load_config()

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

    def test_packaged_ltx_defaults(self):
        assert resolve_video_defaults("ltx", load_config(), {}) == {"steps": 8, "width": 768, "height": 448, "num_frames": 121, "ratio": "16:9", "size": "s"}

    def test_explicit_values_win_over_the_size_entry(self):
        overrides = {"ratio": "16:9", "size": "m", "steps": 4, "width": 640}

        assert resolve_video_defaults("ltx", load_config(), overrides) == {"steps": 4, "width": 640, "height": 576, "num_frames": 121, "ratio": "16:9", "size": "m"}

    def test_a_missing_size_falls_back_to_the_built_in_size(self):
        config = {"video_sizes": {}, "video_model_presets": {"ltx": {"default_steps": 6}}}

        assert resolve_video_defaults("ltx", config, {"ratio": "1:1", "size": "zz"}) == {"steps": 6, "width": 768, "height": 448, "num_frames": 121, "ratio": "1:1", "size": "zz"}

    def test_a_family_without_a_preset_gets_eight_steps(self):
        assert resolve_video_defaults("unknown", {}, {}) == {"steps": 8, "width": 768, "height": 448, "num_frames": 121, "ratio": "16:9", "size": "s"}
