"""Tests for the LoRA family index: new or changed files are detected on the first lookup and cached."""

from __future__ import annotations

from contextlib import closing
import json
import os
import sqlite3
import struct
from pathlib import Path

import pytest

from zvisiongenerator.utils.safetensors_stream import read_header_and_metadata
from zvisiongenerator.web import lora_index
from zvisiongenerator.web.lora_index import LoraFamilyInfo, lookup

KLEIN = LoraFamilyInfo("flux2_klein", "4b")
UNKNOWN = LoraFamilyInfo(None, None)


def _write_lora(path: Path, *, extra: int = 0) -> Path:
    """Write a header-only safetensors file; *extra* changes its size so its stat key differs."""
    header = {"layer.lora_A.weight": {"dtype": "F16", "shape": [4, 8], "data_offsets": [0, 0]}, "__metadata__": {"ss_base_model_version": "klein_4b"}}
    payload = json.dumps(header).encode("utf-8") + b" " * extra
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<Q", len(payload)) + payload)
    return path


@pytest.fixture
def loras_dir(tmp_path: Path) -> Path:
    return tmp_path / "loras"


@pytest.fixture
def reads(monkeypatch) -> list[str]:
    """Record the stem of every LoRA file whose header is read."""
    seen: list[str] = []

    def read(path: Path):
        seen.append(path.stem)
        return read_header_and_metadata(path)

    monkeypatch.setattr(lora_index, "read_header_and_metadata", read)
    return seen


def test_a_cold_lookup_reads_each_new_file_once_and_stores_it(loras_dir, reads):
    _write_lora(loras_dir / "a.safetensors")
    _write_lora(loras_dir / "b.safetensors")

    result = lookup(loras_dir, ["a", "b"])

    assert result == {"a": KLEIN, "b": KLEIN}
    assert sorted(reads) == ["a", "b"]


def test_a_warm_lookup_reads_no_file(loras_dir, reads):
    _write_lora(loras_dir / "a.safetensors")
    lookup(loras_dir, ["a"])
    reads.clear()

    assert lookup(loras_dir, ["a"]) == {"a": KLEIN}
    assert reads == []


def test_a_changed_file_is_read_again(loras_dir, reads):
    path = _write_lora(loras_dir / "a.safetensors")
    _write_lora(loras_dir / "b.safetensors")
    lookup(loras_dir, ["a", "b"])
    reads.clear()

    _write_lora(path, extra=5)
    os.utime(path, ns=(1, 1))

    assert lookup(loras_dir, ["a", "b"]) == {"a": KLEIN, "b": KLEIN}
    assert reads == ["a"]


def test_a_file_that_is_not_safetensors_is_stored_as_unknown_and_not_read_again(loras_dir, reads):
    bad = loras_dir / "bad.safetensors"
    bad.parent.mkdir(parents=True)
    bad.write_bytes(b"not safetensors")

    assert lookup(loras_dir, ["bad"]) == {"bad": UNKNOWN}
    assert lookup(loras_dir, ["bad"]) == {"bad": UNKNOWN}
    assert reads == ["bad"]


def test_an_unreadable_file_is_unknown_and_leaves_the_others_alone(loras_dir, monkeypatch):
    _write_lora(loras_dir / "good.safetensors")
    _write_lora(loras_dir / "locked.safetensors")
    real = lora_index.read_header_and_metadata

    def read(path: Path):
        if path.stem == "locked":
            raise PermissionError("denied")
        return real(path)

    monkeypatch.setattr(lora_index, "read_header_and_metadata", read)

    assert lookup(loras_dir, ["good", "locked"]) == {"good": KLEIN, "locked": UNKNOWN}


def test_a_name_without_a_file_is_unknown(loras_dir, reads):
    assert lookup(loras_dir, ["missing"]) == {"missing": UNKNOWN}
    assert reads == []


def test_a_database_error_gives_unknown_for_all(loras_dir, monkeypatch):
    _write_lora(loras_dir / "a.safetensors")

    def broken(*_args, **_kwargs):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(lora_index.app_db, "connect", broken)

    with pytest.warns(UserWarning):
        assert lookup(loras_dir, ["a"]) == {"a": UNKNOWN}


def _rows(root: Path) -> list[str]:
    with closing(lora_index.app_db.connect(components=(lora_index._COMPONENT,))) as conn:
        return sorted(name for (name,) in conn.execute(f"SELECT name FROM {lora_index._TABLE} WHERE root = ?", (str(root.resolve()),)))


def test_a_removed_lora_loses_its_row(loras_dir, reads):
    _write_lora(loras_dir / "a.safetensors")
    _write_lora(loras_dir / "b.safetensors")
    lookup(loras_dir, ["a", "b"])

    (loras_dir / "b.safetensors").unlink()
    lookup(loras_dir, ["a"])

    assert _rows(loras_dir) == ["a"]


def test_an_empty_name_list_keeps_the_cached_rows(loras_dir, reads):
    _write_lora(loras_dir / "a.safetensors")
    lookup(loras_dir, ["a"])

    lookup(loras_dir, [])

    assert _rows(loras_dir) == ["a"]
