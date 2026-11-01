"""Tests for the audio track rows of the macOS MiniMax H3 backend: the pure packing step runs without MLX."""

from __future__ import annotations

import numpy as np
import pytest

from zvisiongenerator.backends.mac.h3_staged import pack_track_latents

CHANNELS = 4


def _mean(frames: int) -> np.ndarray:
    """Return a posterior mean ``(2, CHANNELS, frames)`` whose value names its stereo channel, latent channel and frame."""
    stereo, channel, frame = np.meshgrid(np.arange(2), np.arange(CHANNELS), np.arange(frames), indexing="ij")
    return (stereo * 1000 + channel * 100 + frame).astype(np.float32)


def _statistics() -> tuple[list[float], list[float]]:
    return [0.5 * channel for channel in range(CHANNELS)], [1.0 + channel for channel in range(CHANNELS)]


class TestPackTrackLatents:
    def test_the_rows_are_channel_major_with_the_latents_in_time_order(self):
        mean, std = _statistics()
        rows = pack_track_latents(_mean(6), mean, std, 6)

        assert (rows.dtype, rows.shape) == (np.float32, (12, CHANNELS))
        assert rows.flags.c_contiguous
        for stereo in range(2):
            for frame in range(6):
                expected = [(stereo * 1000 + channel * 100 + frame - mean[channel]) / std[channel] for channel in range(CHANNELS)]
                np.testing.assert_allclose(rows[stereo * 6 + frame], expected, rtol=1e-6)

    def test_the_mean_is_normalised_the_way_the_decode_undoes_it(self):
        mean, std = _statistics()
        rows = pack_track_latents(_mean(3), mean, std, 3)

        # The pipeline decodes with ``rows * std + mean``.
        restored = rows * np.asarray(std, dtype=np.float32) + np.asarray(mean, dtype=np.float32)
        np.testing.assert_allclose(restored.reshape(2, 3, CHANNELS).transpose(0, 2, 1), _mean(3), rtol=1e-6)

    def test_a_shorter_track_is_padded_with_each_channels_last_latent(self):
        rows = pack_track_latents(_mean(3), *_statistics(), 5).reshape(2, 5, CHANNELS)

        for stereo in range(2):
            np.testing.assert_array_equal(rows[stereo, 3], rows[stereo, 2])
            np.testing.assert_array_equal(rows[stereo, 4], rows[stereo, 2])
        assert not np.array_equal(rows[0, 2], rows[1, 2])

    def test_a_longer_track_is_trimmed_to_the_clip(self):
        long = pack_track_latents(_mean(8), *_statistics(), 8).reshape(2, 8, CHANNELS)
        rows = pack_track_latents(_mean(8), *_statistics(), 5).reshape(2, 5, CHANNELS)

        np.testing.assert_array_equal(rows, long[:, :5])

    @pytest.mark.parametrize(
        ("mean", "latents", "message"),
        [
            (np.zeros((1, CHANNELS, 4), dtype=np.float32), 4, "shaped"),
            (np.zeros((2, CHANNELS + 1, 4), dtype=np.float32), 4, "statistics"),
        ],
    )
    def test_a_posterior_or_length_that_does_not_fit_is_refused(self, mean, latents, message):
        with pytest.raises(ValueError, match=message):
            pack_track_latents(mean, *_statistics(), latents)
