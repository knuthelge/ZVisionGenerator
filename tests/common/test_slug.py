"""Tests for the file-name slug."""

from __future__ import annotations

import pytest

from zvisiongenerator.utils.slug import is_slug, slugify


@pytest.mark.parametrize(
    ("text", "slug"),
    [("Soft Look", "soft-look"), ("  Crème brûlée!  ", "creme-brulee"), ("A  B--C", "a-b-c"), ("x" * 100, "x" * 64), ("花子", ""), ("", ""), ("---", "")],
)
def test_slugify(text, slug):
    assert slugify(text) == slug


def test_a_long_name_never_ends_in_a_dash():
    assert slugify("a" * 63 + " b") == "a" * 63


@pytest.mark.parametrize(("text", "expected"), [("anna", True), ("anna-lund-2", True), ("", False), ("Anna", False), ("a--b", False), ("-a", False), ("a/b", False), ("anna.wav", False)])
def test_is_slug_accepts_only_lowercase_groups_joined_by_single_dashes(text, expected):
    assert is_slug(text) is expected
