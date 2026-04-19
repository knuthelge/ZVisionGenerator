"""Tests for zvisiongenerator.utils.platform — PlatformInfo, get_platform_info, labels."""

from __future__ import annotations

import dataclasses
from unittest.mock import patch

import pytest

from zvisiongenerator.utils.platform import PlatformInfo, get_all_platform_labels, get_platform_info, get_platform_label


@pytest.fixture()
def _platform_config() -> dict:
    """Minimal config dict with a platforms section — no load_config() needed."""
    return {
        "platforms": {
            "darwin": {"label": "macOS"},
            "win32": {"label": "Windows"},
        },
    }


class TestPlatformInfoFrozen:
    def test_platform_info_is_frozen(self):
        """PlatformInfo should be immutable (frozen dataclass)."""
        info = PlatformInfo(key="darwin", label="macOS", message="")
        with pytest.raises(dataclasses.FrozenInstanceError):
            info.key = "win32"  # type: ignore[misc]


class TestGetPlatformInfo:
    @pytest.mark.parametrize(
        ("platform", "expected_key", "expected_label"),
        [
            ("darwin", "darwin", "macOS"),
            ("win32", "win32", "Windows"),
        ],
    )
    def test_get_platform_info_known(self, _platform_config, platform, expected_key, expected_label):
        """Known platforms return correct key, label, and empty message."""
        info = get_platform_info(_platform_config, platform=platform)
        assert info.key == expected_key
        assert info.label == expected_label
        assert info.message == ""

    def test_get_platform_info_unknown_platform(self, _platform_config):
        """Unknown platform key returns fallback with label=raw_key and descriptive message."""
        info = get_platform_info(_platform_config, platform="freebsd")
        assert info.key == "freebsd"
        assert info.label == "freebsd"
        assert "freebsd" in info.message
        assert "not supported" in info.message.lower()

    def test_get_platform_info_with_message(self):
        """Message field is populated when present in config."""
        config = {
            "platforms": {
                "linux": {"label": "Linux", "message": "Linux is experimental."},
            },
        }
        info = get_platform_info(config, platform="linux")
        assert info.key == "linux"
        assert info.label == "Linux"
        assert info.message == "Linux is experimental."

    def test_get_platform_info_missing_platforms_section(self):
        """Graceful handling when config has no 'platforms' key."""
        config: dict = {}
        info = get_platform_info(config, platform="darwin")
        assert info.key == "darwin"
        assert info.label == "darwin"
        assert "not supported" in info.message.lower()

    def test_get_platform_info_defaults_to_sys_platform(self, _platform_config):
        """When platform parameter is omitted, defaults to sys.platform."""
        with patch("zvisiongenerator.utils.platform.sys") as mock_sys:
            mock_sys.platform = "win32"
            info = get_platform_info(_platform_config)
        assert info.key == "win32"
        assert info.label == "Windows"


class TestGetPlatformLabel:
    def test_get_platform_label_known(self, _platform_config):
        """Returns configured label for a known platform."""
        assert get_platform_label(_platform_config, platform="darwin") == "macOS"
        assert get_platform_label(_platform_config, platform="win32") == "Windows"

    def test_get_platform_label_unknown(self, _platform_config):
        """Returns raw platform key for an unknown platform."""
        assert get_platform_label(_platform_config, platform="linux") == "linux"


class TestGetAllPlatformLabels:
    def test_get_all_platform_labels(self, _platform_config):
        """Returns full {key: label} dict for all configured platforms."""
        labels = get_all_platform_labels(_platform_config)
        assert labels == {"darwin": "macOS", "win32": "Windows"}

    def test_get_all_platform_labels_empty_config(self):
        """Returns empty dict when config has no platforms section."""
        assert get_all_platform_labels({}) == {}
