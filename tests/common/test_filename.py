"""Tests for output file naming."""

from __future__ import annotations

from datetime import datetime

import pytest

from zvisiongenerator.utils.filename import edit_output_stem, generate_filename, unique_output_path

_NOW = datetime(2026, 10, 4, 14, 3, 22)


def test_set_name_and_timestamp():
    assert generate_filename("portrait", now=_NOW) == "portrait_2026-10-04_14-03-22"


def test_timestamp_only_without_set_name():
    assert generate_filename(None, now=_NOW) == "2026-10-04_14-03-22"


def test_set_name_is_sanitized():
    assert generate_filename("a/b:c*d", now=_NOW) == "a_b_c_d_2026-10-04_14-03-22"


def test_set_name_of_only_dots_is_dropped():
    assert generate_filename(" . ", now=_NOW) == "2026-10-04_14-03-22"


def test_unique_output_path_returns_free_name(tmp_path):
    assert unique_output_path(tmp_path, "portrait", ".png") == tmp_path / "portrait.png"


def test_unique_output_path_adds_counter_when_taken(tmp_path):
    (tmp_path / "portrait.png").write_bytes(b"")
    (tmp_path / "portrait_2.png").write_bytes(b"")

    assert unique_output_path(tmp_path, "portrait", ".png") == tmp_path / "portrait_3.png"


def test_unique_output_path_ignores_other_suffixes(tmp_path):
    (tmp_path / "clip.png").write_bytes(b"")

    assert unique_output_path(tmp_path, "clip", ".mp4") == tmp_path / "clip.mp4"


@pytest.mark.parametrize(("source", "stem"), [("photo.png", "photo_edit"), ("photo_edit.png", "photo_edit"), ("photo_edit_2.png", "photo_edit"), ("/pics/photo_editor.png", "photo_editor_edit")])
def test_an_edit_is_named_after_its_source_without_stacking_a_second_edit_suffix(source, stem):
    assert edit_output_stem(source) == stem
