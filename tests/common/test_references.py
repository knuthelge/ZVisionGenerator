"""Tests for parsing and checking reference images."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from tests._images import SIDEWAYS_UPRIGHT_SIZE, write_sideways_photo
from zvisiongenerator.utils.references import (
    MAX_REFERENCE_PIXELS,
    check_reference_count,
    check_reference_files,
    load_reference_image,
    parse_reference_paths,
)


def test_parse_splits_on_commas_and_drops_blanks():
    assert parse_reference_paths("a.png, b.jpg,,") == ["a.png", "b.jpg"]
    assert parse_reference_paths("") == []


def test_parse_expands_the_home_directory(monkeypatch):
    monkeypatch.setenv("HOME", "/home/me")
    monkeypatch.setenv("USERPROFILE", "/home/me")

    assert parse_reference_paths("~/a.png") == [str(Path("/home/me/a.png"))]


@pytest.mark.parametrize(("count", "cap"), [(0, 0), (4, 4)])
def test_counts_within_the_cap_pass(count, cap):
    check_reference_count("flux2_klein", count, cap)


@pytest.mark.parametrize(("count", "cap"), [(5, 4), (1, 0)])
def test_more_than_the_cap_is_refused(count, cap):
    with pytest.raises(ValueError):
        check_reference_count("flux2_klein", count, cap)


def test_files_must_exist_and_be_images(tmp_path):
    good = tmp_path / "a.PNG"
    good.write_bytes(b"x")
    text = tmp_path / "notes.txt"
    text.write_bytes(b"x")
    pretend = tmp_path / "gone.png"

    check_reference_files([str(good)])
    for bad in (pretend, text):
        with pytest.raises(ValueError):
            check_reference_files([str(bad)])
    # One bad file refuses the whole list, wherever it sits.
    with pytest.raises(ValueError):
        check_reference_files([str(good), str(text)])


def test_a_photo_is_loaded_upright_by_its_exif_orientation(tmp_path):
    loaded = load_reference_image(write_sideways_photo(tmp_path / "phone.jpg"))

    assert loaded.mode == "RGB"
    assert loaded.size == SIDEWAYS_UPRIGHT_SIZE


def test_a_large_reference_is_scaled_down_to_one_megapixel_keeping_its_aspect(tmp_path):
    path = tmp_path / "big.png"
    Image.new("RGB", (3000, 1500)).save(path)

    loaded = load_reference_image(str(path))

    width, height = loaded.size
    assert width * height <= MAX_REFERENCE_PIXELS
    assert width * height > 0.99 * MAX_REFERENCE_PIXELS
    assert abs(width / height - 2.0) < 0.01


def test_loading_refuses_a_missing_or_unreadable_file(tmp_path):
    broken = tmp_path / "broken.png"
    broken.write_bytes(b"not an image")

    with pytest.raises(FileNotFoundError):
        load_reference_image(str(tmp_path / "gone.png"))
    with pytest.raises(ValueError):
        load_reference_image(str(broken))
