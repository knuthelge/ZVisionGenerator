"""Tests for the typed reads of optional JSON body fields."""

from __future__ import annotations

import pytest

from zvisiongenerator.web.json_fields import json_field


class TestJsonField:
    @pytest.mark.parametrize(("key", "expected", "value"), [("name", str, "a"), ("count", int, 3), ("ratio", (int, float), 1.5), ("flag", bool, False)])
    def test_a_value_of_the_expected_type_is_returned(self, key, expected, value):
        assert json_field({key: value}, key, expected, "a value") == value

    @pytest.mark.parametrize("payload", [{}, {"key": None}])
    def test_an_absent_or_null_field_is_none(self, payload):
        assert json_field(payload, "key", int, "an integer") is None

    @pytest.mark.parametrize(("expected", "value"), [(str, 5), (int, "5"), (int, 1.5), (bool, 1), ((int, float), "1")])
    def test_a_value_of_another_type_is_refused_by_name(self, expected, value):
        with pytest.raises(ValueError, match="key must be a thing."):
            json_field({"key": value}, "key", expected, "a thing")

    @pytest.mark.parametrize("expected", [int, (int, float)])
    def test_a_boolean_is_not_a_number(self, expected):
        with pytest.raises(ValueError, match="must be"):
            json_field({"key": True}, "key", expected, "a number")

    def test_a_value_under_the_minimum_is_refused(self):
        assert json_field({"key": 1}, "key", int, "a positive integer", minimum=1) == 1
        with pytest.raises(ValueError, match="key must be a positive integer."):
            json_field({"key": 0}, "key", int, "a positive integer", minimum=1)

    def test_the_name_in_the_message_can_differ_from_the_key(self):
        with pytest.raises(ValueError, match=r"context\.key must be a string\."):
            json_field({"key": 1}, "key", str, "a string", name="context.key")
