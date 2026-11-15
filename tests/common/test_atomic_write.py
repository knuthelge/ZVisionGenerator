"""Tests for the atomic file writers."""

from __future__ import annotations

import builtins
import os

import pytest

from zvisiongenerator.utils.atomic_write import write_bytes_atomic, write_new_file, write_text_atomic


def test_text_write_replaces_the_file_with_utf8_and_leaves_no_temp_file(tmp_path):
    path = tmp_path / "notes.txt"

    write_text_atomic(path, "old")
    write_text_atomic(path, "blåbær\n")

    assert path.read_bytes() == "blåbær\n".replace("\n", os.linesep).encode()
    assert [item.name for item in tmp_path.iterdir()] == ["notes.txt"]


def test_text_write_uses_the_platform_line_separator(tmp_path, monkeypatch):
    monkeypatch.setattr(os, "linesep", "\r\n")
    path = tmp_path / "notes.txt"

    write_text_atomic(path, "one\ntwo\n")

    assert path.read_bytes() == b"one\r\ntwo\r\n"


def test_temp_file_is_opened_in_binary_mode_where_the_platform_has_the_flag(tmp_path, monkeypatch):
    monkeypatch.setattr(os, "O_BINARY", 0x8000, raising=False)
    real_open = os.open
    seen_flags: list[int] = []

    def recording_open(path, flags, *args, **kwargs):
        seen_flags.append(flags)
        return real_open(path, flags & ~0x8000, *args, **kwargs)

    monkeypatch.setattr(os, "open", recording_open)

    write_bytes_atomic(tmp_path / "image.png", b"data")
    write_text_atomic(tmp_path / "notes.txt", "one\n")

    assert len(seen_flags) == 2
    assert all(flags & os.O_BINARY for flags in seen_flags)


def test_a_new_file_is_written_where_asked(tmp_path):
    path = write_new_file(tmp_path / "a.png", b"data", stem="a")

    assert path == tmp_path / "a.png"
    assert path.read_bytes() == b"data"


def test_a_new_file_moves_to_the_next_name_when_the_name_exists(tmp_path):
    (tmp_path / "a.png").write_bytes(b"old")
    (tmp_path / "a_2.png").write_bytes(b"old")

    path = write_new_file(tmp_path / "a.png", b"new", stem="a")

    assert path == tmp_path / "a_3.png"
    assert path.read_bytes() == b"new"
    assert (tmp_path / "a.png").read_bytes() == b"old"


def test_a_name_taken_after_it_was_chosen_moves_to_a_counter_name(tmp_path, monkeypatch):
    real_open = builtins.open
    calls = []

    def racing_open(file, mode="r", *args, **kwargs):
        calls.append(file)
        if len(calls) == 1:
            real_open(file, "wb").close()  # another job took the name between the choice and the create
            raise FileExistsError(file)
        return real_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", racing_open)

    path = write_new_file(tmp_path / "a.png", b"data", stem="a")

    assert path == tmp_path / "a_2.png"
    assert path.read_bytes() == b"data"


def test_a_failed_write_removes_the_partial_file_and_raises(tmp_path, monkeypatch):
    real_open = builtins.open

    class _Failing:
        def __init__(self, handle):
            self._handle = handle

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self._handle.close()

        def write(self, data):
            raise OSError("disk full")

    monkeypatch.setattr(builtins, "open", lambda file, mode="r", *a, **k: _Failing(real_open(file, mode, *a, **k)))

    with pytest.raises(OSError, match="disk full"):
        write_new_file(tmp_path / "a.png", b"data", stem="a")

    assert list(tmp_path.iterdir()) == []
