"""Tests for the fade-out of a clip that reached the length limit."""

from __future__ import annotations

import numpy as np
import pytest

from zvisiongenerator.utils.audio_fade import fade_out


def _ones(frames: int, channels: int = 2) -> np.ndarray:
    return np.ones((channels, frames), dtype=np.float32)


def test_the_last_frame_is_silent_and_the_start_is_untouched():
    faded = fade_out(_ones(100), 10, 2.0)

    assert faded[:, :80].min() == 1.0
    assert faded[:, -1].tolist() == [0.0, 0.0]
    assert faded.dtype == np.float32


def test_the_fade_falls_steadily_over_the_last_seconds():
    faded = fade_out(_ones(100, channels=1), 10, 2.0)[0]

    assert faded[80] == 1.0
    assert np.all(np.diff(faded[80:]) < 0)


def test_the_input_is_left_untouched():
    samples = _ones(50)

    fade_out(samples, 10, 2.0)

    assert samples.min() == 1.0


def test_a_fade_longer_than_the_clip_fades_all_of_it():
    faded = fade_out(_ones(20), 10, 5.0)

    assert faded[0, 0] == 1.0
    assert faded[0, -1] == 0.0


def test_no_fade_returns_an_equal_copy():
    samples = _ones(20)

    faded = fade_out(samples, 10, 0)

    assert np.array_equal(faded, samples)
    assert faded is not samples


@pytest.mark.parametrize(("rate", "seconds"), [(0, 1.0), (10, -1.0)])
def test_a_bad_rate_or_fade_is_refused(rate, seconds):
    with pytest.raises(ValueError):
        fade_out(_ones(10), rate, seconds)


def test_samples_that_are_not_channels_by_frames_are_refused():
    with pytest.raises(ValueError, match="channels, frames"):
        fade_out(np.ones(10, dtype=np.float32), 10, 1.0)
