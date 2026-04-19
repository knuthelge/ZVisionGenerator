"""Tests for platform-aware alias resolution helpers."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from zvisiongenerator.utils.platform import PlatformInfo, get_platform_info, resolve_alias


class TestPlatformInfo:
    def test_construction_sets_fields(self):
        info = PlatformInfo(key="darwin", label="macOS")

        assert info.key == "darwin"
        assert info.label == "macOS"

    def test_is_frozen(self):
        info = PlatformInfo(key="darwin", label="macOS")

        with pytest.raises(FrozenInstanceError):
            info.label = "Windows"


class TestResolveAlias:
    def test_string_value_returns_universal_target(self):
        assert resolve_alias("Tongyi-MAI/Z-Image-Turbo", "darwin") == "Tongyi-MAI/Z-Image-Turbo"

    @pytest.mark.parametrize(
        ("platform_key", "expected"),
        [
            ("darwin", "dgrauet/ltx-2.3-mlx-q8"),
            ("win32", "Lightricks/LTX-2.3-fp8"),
        ],
    )
    def test_dict_value_returns_platform_target(self, platform_key, expected):
        alias_value = {
            "darwin": "dgrauet/ltx-2.3-mlx-q8",
            "win32": "Lightricks/LTX-2.3-fp8",
        }

        assert resolve_alias(alias_value, platform_key) == expected

    def test_message_dict_raises_value_error_with_message_text(self):
        alias_value = {
            "darwin": "dgrauet/ltx-2.3-mlx-q4",
            "win32": {"message": "LTX 4-bit is not available on Windows. Use 'ltx-8' instead."},
        }

        with pytest.raises(ValueError, match="LTX 4-bit is not available on Windows"):
            resolve_alias(alias_value, "win32")

    def test_missing_platform_raises_value_error_listing_available_platforms(self):
        alias_value = {
            "darwin": "dgrauet/ltx-2.3-mlx-q8",
            "win32": "Lightricks/LTX-2.3-fp8",
        }

        with pytest.raises(ValueError, match=r"Available platforms: darwin, win32"):
            resolve_alias(alias_value, "linux")

    def test_platform_labels_are_used_in_missing_platform_errors(self):
        alias_value = {
            "darwin": "dgrauet/ltx-2.3-mlx-q8",
            "win32": "Lightricks/LTX-2.3-fp8",
        }

        with pytest.raises(ValueError, match=r"Model alias is not available for Linux\. Available platforms: macOS, Windows"):
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
