"""Tests for the pure waveform peaks, the peak limiter, and reading peaks from a file."""

from __future__ import annotations

import numpy as np
import pytest

from tests._media import REAL
from tests.audio._helpers import sine

from zvisiongenerator.utils.audio_level import limit_peak, peak_of
from zvisiongenerator.utils.audio_peaks import PEAK_COUNT, read_waveform_peaks, waveform_peaks
from zvisiongenerator.utils.audio_save import save_flac


def _db(level: float) -> float:
    return 20 * np.log10(level)


class TestWaveformPeaks:
    def test_a_sine_has_a_full_height_peak_in_every_stretch(self):
        peaks = waveform_peaks(sine(2.0, amplitude=0.3))

        assert len(peaks) == PEAK_COUNT
        assert min(peaks) >= 250 and max(peaks) == 255

    def test_silence_is_all_zero(self):
        assert waveform_peaks(np.zeros((2, 48000), dtype=np.float32)) == [0] * PEAK_COUNT

    def test_the_loudest_stretch_is_255_and_the_rest_follow_it(self):
        samples = np.zeros((1, 1000), dtype=np.float32)
        samples[0, 0:10] = 0.5  # the first stretch
        samples[0, 990:1000] = 0.25  # the last stretch

        peaks = waveform_peaks(samples, count=10)

        assert peaks[0] == 255 and peaks[-1] == 128 and peaks[1:-1] == [0] * 8

    def test_the_loudest_channel_counts(self):
        samples = np.zeros((2, 100), dtype=np.float32)
        samples[1, 50:] = -0.5

        assert waveform_peaks(samples, count=2) == [0, 255]

    @REAL
    def test_peaks_are_read_from_a_file(self, tmp_path):
        path = save_flac(sine(1.0, rate=44100, amplitude=0.4), 44100, tmp_path / "tone.flac", {"kind": "sound"})

        peaks = read_waveform_peaks(path)

        assert len(peaks) == PEAK_COUNT and min(peaks) >= 240 and max(peaks) == 255

    @REAL
    def test_an_unreadable_file_raises(self, tmp_path):
        (tmp_path / "bad.flac").write_bytes(b"not audio")

        with pytest.raises(RuntimeError, match="could not decode"):
            read_waveform_peaks(tmp_path / "bad.flac")


class TestLimitPeak:
    def test_a_clip_6_point_3_db_over_full_scale_comes_down_to_minus_1_dbfs(self):
        loud = sine(0.5, amplitude=10 ** (6.3 / 20))

        limited = limit_peak(loud)

        assert _db(peak_of(limited)) == pytest.approx(-1.0, abs=0.01)

    def test_the_shape_is_kept(self):
        loud = sine(0.1, amplitude=4.0)

        limited = limit_peak(loud)

        ratio = limited[0, 100] / loud[0, 100]
        np.testing.assert_allclose(limited, loud * ratio, rtol=1e-5, atol=1e-6)

    def test_another_ceiling_is_honoured(self):
        assert _db(peak_of(limit_peak(sine(0.1, amplitude=2.0), ceiling_db=-6.0))) == pytest.approx(-6.0, abs=0.01)
