"""Tests for the transparent-background bench: the model-free keys, the difference matte, the hybrid and the survey plan."""

from __future__ import annotations

import random
from collections import Counter

import numpy as np
import pytest

from bench.transparent.alpha import KEYS, chroma_key, difference_matte, hybrid_alpha, ink_key, key_channels, key_conflict, screen_key
from bench.transparent.survey import _balanced_pairs
from bench.transparent.tilt import centre_mask, parse_tilt


def _scene(background: tuple[float, float, float], subject: tuple[float, float, float], *, size: int = 64) -> np.ndarray:
    """Return a square of *background* with a centred block of *subject*."""
    image = np.empty((size, size, 3), dtype=np.float32)
    image[:] = background
    image[size // 4 : 3 * size // 4, size // 4 : 3 * size // 4] = subject
    return image


@pytest.mark.parametrize(("key", "colour"), [("green", (0.0, 1.0, 0.0)), ("magenta", (1.0, 0.0, 1.0)), ("blue", (0.0, 0.0, 1.0)), ("cyan", (0.0, 1.0, 1.0))])
def test_chroma_key_reads_the_key_from_the_border(key: str, colour: tuple[float, float, float]) -> None:
    image = _scene(colour, (0.8, 0.5, 0.3))
    alpha, foreground = chroma_key(image)
    assert key_channels(image) == KEYS[key]
    assert alpha[0, 0] == pytest.approx(0.0)
    assert alpha[32, 32] == pytest.approx(1.0)
    np.testing.assert_allclose(foreground[32, 32], (0.8, 0.5, 0.3), atol=1e-6)


def test_key_channels_rejects_a_grey_border() -> None:
    with pytest.raises(ValueError, match="not a key colour"):
        key_channels(_scene((0.8, 0.8, 0.8), (0.1, 0.1, 0.1)))


def test_key_conflict_flags_a_subject_in_the_key_colour() -> None:
    parrot = _scene((0.8, 0.8, 0.8), (0.2, 0.9, 0.1))
    mask = (parrot[..., 0] < 0.5).astype(np.float32)
    assert key_conflict(parrot, mask, KEYS["green"]) == pytest.approx(1.0)
    assert key_conflict(parrot, mask, KEYS["magenta"]) == pytest.approx(0.0)


def test_ink_key_keeps_ink_opaque_and_paper_clear() -> None:
    alpha, foreground = ink_key(_scene((1.0, 1.0, 1.0), (0.0, 0.0, 0.0)))
    assert alpha[0, 0] == pytest.approx(0.0)
    assert alpha[32, 32] == pytest.approx(1.0)
    np.testing.assert_allclose(foreground[32, 32], (0.0, 0.0, 0.0), atol=1e-6)


def test_screen_key_turns_grey_smoke_half_transparent_white() -> None:
    alpha, foreground = screen_key(_scene((0.0, 0.0, 0.0), (0.5, 0.5, 0.5)))
    assert alpha[0, 0] == pytest.approx(0.0)
    assert alpha[32, 32] == pytest.approx(0.5)
    np.testing.assert_allclose(foreground[32, 32], (1.0, 1.0, 1.0), atol=1e-6)


def test_difference_matte_recovers_a_half_transparent_subject() -> None:
    colour, opacity = np.array([0.6, 0.2, 0.4]), 0.5
    on_white = _scene((1.0, 1.0, 1.0), tuple(opacity * colour + (1 - opacity)))
    on_black = _scene((0.0, 0.0, 0.0), tuple(opacity * colour))
    alpha, foreground, residual = difference_matte(on_white, on_black)
    assert alpha[32, 32] == pytest.approx(opacity, abs=1e-5)
    np.testing.assert_allclose(foreground[32, 32], colour, atol=1e-4)
    assert residual.max() == pytest.approx(0.0, abs=1e-5)


def test_difference_matte_reports_a_redrawn_subject_as_residual() -> None:
    on_white = _scene((1.0, 1.0, 1.0), (0.6, 0.6, 0.6))
    on_black = _scene((0.0, 0.0, 0.0), (0.1, 0.5, 0.1))
    _, _, residual = difference_matte(on_white, on_black)
    assert residual[32, 32] > 0.1
    assert residual[0, 0] == pytest.approx(0.0)


def test_hybrid_alpha_falls_back_to_the_core_where_the_pair_disagrees() -> None:
    core, pair = np.array([1.0, 1.0]), np.array([0.3, 0.3])
    mixed = hybrid_alpha(core, pair, np.array([0.0, 0.5]), tolerance=0.03)
    np.testing.assert_allclose(mixed, [0.3, 1.0])


def test_balanced_pairs_cover_every_combination_equally() -> None:
    pairs = _balanced_pairs(["a", "b", "c", "d"], 60, random.Random(1))
    counts = Counter(tuple(sorted(pair)) for pair in pairs)
    assert len(counts) == 6
    assert set(counts.values()) == {10}


def test_centre_mask_is_one_in_the_middle_and_falls_off_to_the_corners() -> None:
    mask = centre_mask(33, 33)
    assert mask[16, 16] == pytest.approx(1.0)
    assert mask[0, 0] < 0.1


@pytest.mark.parametrize(("value", "expected"), [("white:1.0", ("white", 1.0)), ("black:0.05", ("black", 0.05)), ("white", ("white", 1.0))])
def test_parse_tilt(value: str, expected: tuple[str, float]) -> None:
    assert parse_tilt(value) == expected


def test_parse_tilt_rejects_other_colours() -> None:
    with pytest.raises(ValueError, match="white or black"):
        parse_tilt("green:1.0")
