"""Pixel-alignment helpers for image and video resolution."""

from __future__ import annotations


def round_to_alignment(value: int, alignment: int = 16) -> int:
    """Round *value* up to the nearest multiple of *alignment*."""
    return (value + alignment - 1) // alignment * alignment
