"""Tests for platform-aware alias resolution helpers."""

from __future__ import annotations

import pytest

from zvisiongenerator.utils.platform import resolve_alias


class TestResolveAlias:
    def test_string_value_returns_universal_target(self):
        assert resolve_alias("Tongyi-MAI/Z-Image-Turbo", "darwin") == "Tongyi-MAI/Z-Image-Turbo"

    def test_dict_value_returns_platform_target(self):
        alias_value = {"darwin": "dgrauet/ltx-2.3-mlx-q8", "win32": "dg845/LTX-2.3-Diffusers"}

        assert resolve_alias(alias_value, "win32") == "dg845/LTX-2.3-Diffusers"

    def test_a_missing_platform_raises_naming_it_and_the_available_ones(self):
        alias_value = {
            "darwin": "dgrauet/ltx-2.3-mlx-q8",
            "win32": "dg845/LTX-2.3-Diffusers",
        }

        with pytest.raises(ValueError, match="linux.*darwin, win32"):
            resolve_alias(alias_value, "linux")
