"""Shared helpers for the common tests: header-only safetensors files."""

from __future__ import annotations

import json
import struct
from pathlib import Path

__all__ = ["write_safetensors"]


def write_safetensors(path: Path, tensors: dict[str, tuple[str, list[int]]]) -> Path:
    """Write a header-only safetensors file (tensor name to dtype and shape); the memory estimates never read tensor data."""
    header: dict[str, object] = {name: {"dtype": dtype, "shape": shape, "data_offsets": [0, 0]} for name, (dtype, shape) in tensors.items()}
    header["__metadata__"] = {"format": "pt"}
    payload = json.dumps(header).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<Q", len(payload)) + payload)
    return path
