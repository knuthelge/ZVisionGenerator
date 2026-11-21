"""Tests for the per-operating-system CUDA offset."""

from __future__ import annotations

import importlib
import sys

import pytest

from zvisiongenerator.utils import memory_offsets
from zvisiongenerator.utils.model_memory import GIB


def test_the_offsets_are_the_measured_ones() -> None:
    assert memory_offsets.CUDA_OFFSET_LINUX_BYTES == round(0.40 * GIB)
    assert memory_offsets.CUDA_OFFSET_WINDOWS_BYTES == round(1.00 * GIB)


@pytest.mark.parametrize(("platform", "expected"), [("win32", "CUDA_OFFSET_WINDOWS_BYTES"), ("linux", "CUDA_OFFSET_LINUX_BYTES"), ("darwin", "CUDA_OFFSET_LINUX_BYTES")])
def test_the_offset_the_engine_adds_is_the_running_systems(monkeypatch: pytest.MonkeyPatch, platform: str, expected: str) -> None:
    monkeypatch.setattr(sys, "platform", platform)
    try:
        reloaded = importlib.reload(memory_offsets)
        assert reloaded.CUDA_OFFSET_BYTES == getattr(reloaded, expected)
    finally:
        monkeypatch.undo()
        importlib.reload(memory_offsets)
