"""Tests for the atomic file writers."""

from __future__ import annotations

import os

from zvisiongenerator.utils.atomic_write import write_bytes_atomic, write_text_atomic


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
