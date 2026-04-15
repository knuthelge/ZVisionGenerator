"""Tests for zvisiongenerator.utils.prompt_compose — flatten_value and resolve_snippets."""

from __future__ import annotations

import pytest

from zvisiongenerator.utils.prompt_compose import flatten_value, resolve_snippets


# ── flatten_value ───────────────────────────────────────────────────────────


class TestFlattenValue:
    def test_string_passthrough(self):
        assert flatten_value("hello world") == "hello world"

    def test_string_strips_whitespace(self):
        assert flatten_value("  padded  ") == "padded"

    def test_none_returns_empty(self):
        assert flatten_value(None) == ""

    def test_bool_true(self):
        assert flatten_value(True) == "true"

    def test_bool_false(self):
        assert flatten_value(False) == "false"

    def test_int(self):
        assert flatten_value(42) == "42"

    def test_float(self):
        assert flatten_value(3.14) == "3.14"

    def test_dict_key_value_pairs(self):
        result = flatten_value({"style": "anime", "quality": "high"})
        assert "style: anime" in result
        assert "quality: high" in result

    def test_dict_with_empty_value(self):
        result = flatten_value({"mood": None})
        # When value flattens to "", should just show key
        assert "mood" in result

    def test_list_joins_items(self):
        result = flatten_value(["red", "blue", "green"])
        assert "red" in result
        assert "blue" in result
        assert "green" in result

    def test_list_filters_empty(self):
        result = flatten_value(["hello", None, "world"])
        # None flattens to "" and is filtered
        assert result == "hello. world"

    def test_unknown_type_uses_str(self):
        assert flatten_value(object) != ""


# ── resolve_snippets ────────────────────────────────────────────────────────


class TestResolveSnippets:
    def test_standalone_reference(self):
        snippets = {"color": "vivid red"}
        result = resolve_snippets("$color", snippets)
        assert result == "vivid red"

    def test_inline_reference(self):
        snippets = {"style": "watercolor"}
        result = resolve_snippets("a $style painting", snippets)
        assert result == "a watercolor painting"

    def test_multiple_inline_references(self):
        snippets = {"adj": "dark", "noun": "forest"}
        result = resolve_snippets("a $adj $noun", snippets)
        assert result == "a dark forest"

    def test_undefined_snippet_raises(self):
        with pytest.raises(ValueError, match="Undefined snippet"):
            resolve_snippets("$missing", {})

    def test_undefined_inline_snippet_raises(self):
        with pytest.raises(ValueError, match="Undefined snippet"):
            resolve_snippets("a $missing photo", {})

    def test_circular_reference_raises(self):
        snippets = {"a": "$b", "b": "$a"}
        with pytest.raises(ValueError, match="Circular snippet reference"):
            resolve_snippets("$a", snippets)

    def test_nested_snippets(self):
        snippets = {"inner": "deep blue", "outer": "$inner sky"}
        result = resolve_snippets("$outer", snippets)
        assert result == "deep blue sky"

    def test_standalone_returns_non_string_type(self):
        snippets = {"nums": [1, 2, 3]}
        result = resolve_snippets("$nums", snippets)
        assert result == [1, 2, 3]

    def test_dict_values_resolved(self):
        snippets = {"color": "red"}
        result = resolve_snippets({"fg": "$color"}, snippets)
        assert result == {"fg": "red"}

    def test_list_values_resolved(self):
        snippets = {"x": "hello"}
        result = resolve_snippets(["$x", "world"], snippets)
        assert result == ["hello", "world"]

    def test_none_passthrough(self):
        assert resolve_snippets(None, {}) is None

    def test_int_passthrough(self):
        assert resolve_snippets(42, {}) == 42

    def test_bool_passthrough(self):
        assert resolve_snippets(True, {}) is True
