"""Tests for platform-aware alias resolution helpers."""

from __future__ import annotations

import pytest

from zvisiongenerator.utils.platform import PlatformInfo, get_platform_info, resolve_alias


class TestResolveAlias:
    def test_string_value_returns_universal_target(self):
        assert resolve_alias("Tongyi-MAI/Z-Image-Turbo", "darwin") == "Tongyi-MAI/Z-Image-Turbo"

    def test_dict_value_returns_platform_target(self):
        alias_value = {"darwin": "dgrauet/ltx-2.3-mlx-q8", "win32": "dg845/LTX-2.3-Diffusers"}

        assert resolve_alias(alias_value, "win32") == "dg845/LTX-2.3-Diffusers"

    def test_message_dict_raises_value_error_with_message_text(self):
        alias_value = {
            "darwin": "dgrauet/ltx-2.3-mlx-q4",
            "win32": {"message": "LTX 4-bit is not available on Windows. Use 'ltx-8' instead."},
        }

        with pytest.raises(ValueError, match="LTX 4-bit is not available on Windows"):
            resolve_alias(alias_value, "win32")

    def test_a_missing_platform_raises_naming_its_label(self):
        alias_value = {
            "darwin": "dgrauet/ltx-2.3-mlx-q8",
            "win32": "dg845/LTX-2.3-Diffusers",
        }

        with pytest.raises(ValueError, match="Linux"):
            resolve_alias(alias_value, "linux", platform_labels={"darwin": "macOS", "win32": "Windows", "linux": "Linux"})


class TestGetPlatformInfo:
    def test_happy_path_returns_platform_info_with_label_from_config(self):
        config = {"platforms": {"darwin": "macOS", "win32": "Windows"}}

        info = get_platform_info(config, "darwin")

        assert info == PlatformInfo(key="darwin", label="macOS")

    def test_unknown_platform_returns_fallback_info_with_raw_key_label(self):
        config = {"platforms": {"darwin": "macOS", "win32": "Windows"}}

        info = get_platform_info(config, "linux")

        assert info == PlatformInfo(key="linux", label="linux")
