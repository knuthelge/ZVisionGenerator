"""Tests for naming hex colours with plain colour words."""

from __future__ import annotations

import pytest

from zvisiongenerator.utils.color_names import COLOR_NAMES, color_name, parse_hex


class TestParseHex:
    @pytest.mark.parametrize("text", ["#1b3a4b", "1B3A4B", " #1B3A4B "])
    def test_accepts_six_digits_with_or_without_hash(self, text: str) -> None:
        assert parse_hex(text) == "#1B3A4B"

    @pytest.mark.parametrize("text", ["#123", "teal", "#GGGGGG", ""])
    def test_rejects_anything_else(self, text: str) -> None:
        with pytest.raises(ValueError, match="not a colour"):
            parse_hex(text)


class TestColorName:
    def test_every_named_swatch_names_itself(self) -> None:
        assert all(color_name(value) == name for name, value in COLOR_NAMES.items())

    @pytest.mark.parametrize(
        ("hex_color", "name"),
        [("#FF0000", "red"), ("#000000", "black"), ("#FFFFFF", "white"), ("#0A1F44", "navy blue"), ("#CB6D51", "terracotta"), ("#FFA500", "amber"), ("#2A9D8F", "teal"), ("#1B3A4B", "petrol blue")],
    )
    def test_nearby_colours_get_the_fitting_name(self, hex_color: str, name: str) -> None:
        assert color_name(hex_color) == name

    @pytest.mark.parametrize(("hex_color", "name"), [("#0000FF", "electric blue"), ("#39FF14", "neon green"), ("#FF00FF", "fuchsia"), ("#FF6600", "bright orange"), ("#00FFFF", "bright cyan")])
    def test_vivid_colours_keep_a_vivid_name(self, hex_color: str, name: str) -> None:
        assert color_name(hex_color) == name
