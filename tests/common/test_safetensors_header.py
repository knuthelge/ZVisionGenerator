"""Tests for the bounded safetensors header reader (no torch needed)."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from zvisiongenerator.utils import safetensors_stream
from zvisiongenerator.utils.safetensors_stream import read_header_and_metadata


def _write_safetensors(path: Path, metadata: dict[str, str]) -> Path:
    """Write a header-only safetensors file."""
    header = {"a.weight": {"dtype": "BF16", "shape": [4, 8], "data_offsets": [0, 0]}, "__metadata__": metadata}
    payload = json.dumps(header).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(payload)) + payload)
    return path


def test_round_trips_a_header_only_file(tmp_path):
    path = _write_safetensors(tmp_path / "w.safetensors", {"format": "pt"})

    header, metadata = read_header_and_metadata(path)

    assert header == {"a.weight": {"dtype": "BF16", "shape": [4, 8], "data_offsets": [0, 0]}}
    assert metadata == {"format": "pt"}


def test_refuses_a_length_above_the_cap(tmp_path, monkeypatch):
    monkeypatch.setattr(safetensors_stream, "MAX_HEADER_BYTES", 16)
    path = tmp_path / "big.safetensors"
    path.write_bytes(struct.pack("<Q", 17) + b" " * 64)

    with pytest.raises(ValueError):
        read_header_and_metadata(path)


def test_refuses_a_length_above_the_file_size(tmp_path):
    path = tmp_path / "short.safetensors"
    path.write_bytes(struct.pack("<Q", 1000) + b"{}")

    with pytest.raises(ValueError):
        read_header_and_metadata(path)
