"""Tests for the round-trip YAML helpers shared by the prompt builder and the config writer."""

from __future__ import annotations

import io

import pytest
from ruamel.yaml.comments import CommentedMap
import yaml

from zvisiongenerator.utils.yaml_round_trip import (
    attach_following,
    delete_key,
    load_round_trip,
    round_trip_yaml,
    set_key,
    split_comment_lines,
    yaml11_safe,
)


def _dump(document: object) -> str:
    buffer = io.StringIO()
    round_trip_yaml().dump(document, buffer)
    return buffer.getvalue()


@pytest.mark.parametrize(
    ("value", "first_column", "key_column", "own", "following"),
    [
        pytest.param("# note\n", 8, 0, "# note\n", None, id="end-of-line comment is the key's own"),
        pytest.param("\n", 4, 0, None, None, id="nothing but the line break"),
        pytest.param("\n  # indented\n", 2, 0, "\n  # indented\n", None, id="indented comment is own at column 0"),
        pytest.param("\n\n# Heading\n", 0, 0, None, "\n# Heading\n", id="blank line and heading follow"),
        pytest.param("# note\n    # deeper\n  # same\n# above\n\n", 8, 2, "# note\n    # deeper\n", "  # same\n# above\n\n", id="same column and above follow"),
        pytest.param("\n  # next key\n", 2, 2, None, "  # next key\n", id="a comment at the key's column follows"),
    ],
)
def test_split_comment_lines_separates_the_items_own_lines_from_the_following_ones(value, first_column, key_column, own, following):
    """Lines deeper than the key's column are its own; blank lines and lines at or above it follow."""
    assert split_comment_lines(value, first_column, key_column) == (own, following)


def test_split_comment_lines_defaults_to_column_zero():
    """The default key column keeps only indented comments, as the prompt builder does."""
    assert split_comment_lines("# eol\n  # kept\n# dropped\n\n", 6) == ("# eol\n  # kept\n", "# dropped\n\n")


def test_load_round_trip_raises_value_error_for_invalid_yaml_and_duplicate_keys():
    """A file that cannot be written back faithfully is refused with a ValueError."""
    with pytest.raises(ValueError, match="not valid YAML"):
        load_round_trip("a: [unclosed\n")
    with pytest.raises(ValueError, match="not valid YAML"):
        load_round_trip("a: 1\na: 2\n")


@pytest.mark.parametrize(("text", "quoted"), [("1.10", True), ("off", True), ("null", True), ("2024-01-02", True), ("plain text", False), ("/a/b", False)])
def test_yaml11_safe_quotes_strings_pyyaml_would_read_as_another_type(text, quoted):
    """A plain scalar PyYAML would resolve to a bool, number, date or null is written quoted."""
    document = {"key": yaml11_safe(text)}
    assert yaml.safe_load(_dump(document)) == {"key": text}
    assert (f'"{text}"' in _dump(document)) is quoted


SECTIONS = """\
# header
first:  # about first
  a: 1  # one
  b: 2

# Second heading
second:
  c: 3
  # about d
  d: 4

# tail
"""


def test_set_key_changes_an_existing_key_in_place_with_its_comment():
    """An existing key keeps its position and its end-of-line comment."""
    document = load_round_trip(SECTIONS)

    set_key(document["first"], "a", 10)

    assert _dump(document) == SECTIONS.replace("a: 1  # one", "a: 10 # one")


def test_set_key_appends_a_new_key_above_the_comments_that_followed_the_last_one():
    """A new key goes last; the blank line and heading after the old last key move below it."""
    document = load_round_trip(SECTIONS)

    set_key(document["first"], "e", 5)

    assert _dump(document) == SECTIONS.replace("  b: 2\n\n# Second heading", "  b: 2\n  e: 5\n\n# Second heading")


def test_set_key_appends_below_a_nested_last_value_and_keeps_the_trailing_comments_last():
    """When the old last key holds a mapping, the lines after its last nested key move too."""
    document = load_round_trip(SECTIONS)

    set_key(document, "third", CommentedMap(x=1))

    assert _dump(document).endswith("  # about d\n  d: 4\nthird:\n  x: 1\n\n# tail\n")


def test_set_key_into_an_empty_flow_mapping_writes_block_style():
    """``{}`` becomes a block mapping once it holds a key."""
    document = load_round_trip("ui: {}\n")

    set_key(document["ui"], "a", 1)

    assert _dump(document) == "ui:\n  a: 1\n"


def test_set_key_twice_in_one_run_keeps_the_following_lines_last():
    """A key added after another new key still ends above the trailing comments."""
    document = load_round_trip(SECTIONS)

    set_key(document["first"], "e", 5)
    set_key(document["first"], "f", 6)

    assert _dump(document).count("# Second heading") == 1
    assert _dump(document).index("f: 6") < _dump(document).index("# Second heading")


def test_delete_key_drops_the_keys_own_comments_and_returns_the_following_lines():
    """The end-of-line comment goes with the key; a blank line and heading after it are handed back."""
    document = load_round_trip(SECTIONS)

    following = delete_key(document["first"], "b")

    assert following == "\n# Second heading\n"
    assert "b: 2" not in _dump(document)

    following = delete_key(document["first"], "a")
    assert following is None
    assert "# one" not in _dump(document)


def test_delete_key_returns_the_lines_after_a_whole_section_and_attach_following_restores_them():
    """Removing a block mapping drops its nested comments deeper than its column and returns the rest."""
    document = load_round_trip(SECTIONS)

    following = delete_key(document, "second")
    assert following == "\n# tail\n"
    assert "about d" not in _dump(document)

    attach_following(document, "first", following)
    assert _dump(document) == "# header\nfirst:  # about first\n  a: 1  # one\n  b: 2\n\n# Second heading\n\n# tail\n"


def test_attach_following_after_a_key_and_at_the_key():
    """Lines go below a key's whole value, or directly after ``key:`` for a mapping that lost its first key."""
    document = load_round_trip("a:\n  b: 1\nc: 2\n")

    attach_following(document, "a", "# below a\n")
    attach_following(document, "c", "# below c\n")

    assert _dump(document) == "a:\n  b: 1\n# below a\nc: 2\n# below c\n"

    document = load_round_trip("a:\n  b: 1\n")
    attach_following(document, "a", "  # top of a\n", at_key=True)
    assert _dump(document) == "a:\n  # top of a\n  b: 1\n"
