"""Tests for reading an audio track and the window sound of an extension."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tests._factories import make_extend_source
from tests._media import REAL, ffmpeg

from zvisiongenerator.core.video_types import AudioTrack
from zvisiongenerator.utils.audio_track import (
    TRACK_UNREADABLE,
    continue_track,
    extension_audio,
    probe_track_seconds,
    read_track,
    require_readable_track,
    window_waveform,
)
from zvisiongenerator.utils.video_extend import AUDIO_SAMPLE_RATE
from zvisiongenerator.utils.video_splice import write_pcm16_wav


def _steps(path: Path, levels: list[float], *, rate: int = 16000, channels: int = 1) -> Path:
    """Write a WAV whose every second holds one constant level, so a stretch of it says where it was cut from."""
    audio = np.repeat(np.array(levels, dtype=np.float32), rate)
    write_pcm16_wav(np.tile(audio, (channels, 1)), path, rate)
    return path


def _mean(audio: np.ndarray, begin: float, end: float, rate: int = AUDIO_SAMPLE_RATE) -> float:
    """Return the mean of the first channel between two moments, away from the edges the resampler smooths."""
    return float(audio[0, round(begin * rate) : round(end * rate)].mean())


@REAL
class TestReadTrack:
    def test_reads_stereo_float32_of_the_asked_length(self, tmp_path):
        track = _steps(tmp_path / "t.wav", [0.2, 0.4, 0.6])

        audio = read_track(AudioTrack(str(track), 0.0), AUDIO_SAMPLE_RATE, 1.5)

        assert audio.dtype == np.float32
        assert audio.shape == (2, 72000)

    def test_a_mono_track_is_doubled_into_both_channels(self, tmp_path):
        track = _steps(tmp_path / "t.wav", [0.5])

        audio = read_track(AudioTrack(str(track), 0.0), 16000, 0.5)

        assert np.array_equal(audio[0], audio[1])
        assert abs(float(audio[0, 2000:6000].mean()) - 0.5) < 0.01

    def test_starts_where_asked(self, tmp_path):
        track = _steps(tmp_path / "t.wav", [0.2, 0.4, 0.6])

        audio = read_track(AudioTrack(str(track), 1.0), AUDIO_SAMPLE_RATE, 1.0)

        assert abs(_mean(audio, 0.2, 0.8) - 0.4) < 0.01

    def test_pads_with_silence_past_the_end_and_after_a_start_beyond_it(self, tmp_path):
        track = _steps(tmp_path / "t.wav", [0.4, 0.4])

        audio = read_track(AudioTrack(str(track), 1.5), AUDIO_SAMPLE_RATE, 2.0)

        assert audio.shape == (2, 96000)
        assert abs(_mean(audio, 0.1, 0.4) - 0.4) < 0.01
        assert not audio[:, round(0.6 * AUDIO_SAMPLE_RATE) :].any()

        beyond = read_track(AudioTrack(str(track), 5.0), AUDIO_SAMPLE_RATE, 0.5)

        assert beyond.shape == (2, 24000)
        assert not beyond.any()

    def test_resamples_to_the_asked_rate(self, tmp_path):
        track = _steps(tmp_path / "t.wav", [0.3, 0.3], rate=44100)

        assert read_track(AudioTrack(str(track), 0.0), 32000, 1.0).shape == (2, 32000)

    @pytest.mark.parametrize("kind", ["missing", "not-audio"])
    def test_a_file_that_cannot_be_read_says_so(self, tmp_path, kind):
        path = tmp_path / "notes.txt"
        if kind == "not-audio":
            path.write_text("hello")

        with pytest.raises(ValueError, match=TRACK_UNREADABLE):
            read_track(AudioTrack(str(path), 0.0), AUDIO_SAMPLE_RATE, 1.0)


@REAL
class TestProbeTrackSeconds:
    def test_returns_the_length_of_the_audio(self, tmp_path):
        track = _steps(tmp_path / "t.wav", [0.2] * 4)

        assert probe_track_seconds(str(track)) == pytest.approx(4.0, abs=0.01)

    @pytest.mark.parametrize("kind", ["missing", "not-audio", "video-without-sound"])
    def test_a_file_that_cannot_be_read_says_so(self, tmp_path, kind):
        path = tmp_path / "file.bin"
        if kind == "not-audio":
            path.write_text("hello")
        if kind == "video-without-sound":
            ffmpeg("-f", "lavfi", "-i", "color=c=red:s=64x64:d=1:r=24", "-pix_fmt", "yuv420p", str(tmp_path / "v.mp4"))
            path = tmp_path / "v.mp4"

        with pytest.raises(ValueError, match=TRACK_UNREADABLE):
            probe_track_seconds(str(path))


class TestRequireReadableTrack:
    @REAL
    def test_a_file_that_cannot_be_read_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match=TRACK_UNREADABLE):
            require_readable_track(AudioTrack(str(tmp_path / "missing.mp3"), 0.0))


class TestContinueTrack:
    def test_a_source_that_followed_the_same_file_continues_from_where_its_sound_ended(self, tmp_path):
        song = tmp_path / "song.mp3"
        source = make_extend_source(audio_track_path=str(song), audio_track_end=94.0)

        assert continue_track(AudioTrack(str(song), 0.0), source) == AudioTrack(str(song), 94.0)

    def test_the_same_file_is_found_through_a_different_spelling(self, tmp_path):
        (tmp_path / "music").mkdir()
        source = make_extend_source(audio_track_path=str(tmp_path / "music" / ".." / "song.mp3"), audio_track_end=12.5)

        assert continue_track(AudioTrack(str(tmp_path / "song.mp3"), 0.0), source).start == 12.5

    @pytest.mark.parametrize(("recorded_path", "recorded_end"), [(None, None), ("/other/song.mp3", 94.0), ("/music/song.mp3", None)])
    def test_another_file_or_no_record_keeps_the_track_as_given(self, recorded_path, recorded_end):
        source = make_extend_source(audio_track_path=recorded_path, audio_track_end=recorded_end)
        track = AudioTrack("/music/song.mp3", 3.0)

        assert continue_track(track, source) == track


@REAL
class TestWindowWaveform:
    def test_a_silent_source_gives_silence_then_the_track_from_start_at(self, tmp_path):
        track = _steps(tmp_path / "t.wav", [0.2, 0.4, 0.6])
        source = make_extend_source(path=tmp_path / "silent.mp4", has_audio=False, frame_count=48)

        window = window_waveform(source, AudioTrack(str(track), 1.0), carry_frames=25, extend_frames=24)

        carried = round(25 * AUDIO_SAMPLE_RATE / 24)
        assert window.shape == (2, carried + AUDIO_SAMPLE_RATE)
        assert not window[:, :carried].any()
        assert abs(_mean(window[:, carried:], 0.1, 0.9) - 0.4) < 0.01

    def test_a_source_with_sound_gives_its_last_frames_sound_then_the_track(self, tmp_path):
        track = _steps(tmp_path / "t.wav", [0.2, 0.4, 0.6])
        clip = tmp_path / "clip.mp4"
        ffmpeg(
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=64x64:d=2:r=24",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000:duration=2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            str(clip),
        )
        source = make_extend_source(path=clip, has_audio=True, frame_count=48)

        window = window_waveform(source, AudioTrack(str(track), 0.0), carry_frames=25, extend_frames=24)

        carried = round(25 * AUDIO_SAMPLE_RATE / 24)
        assert window[:, :carried].std() > 0.05  # the sine, whose amplitude is 0.125
        assert abs(_mean(window[:, carried:], 0.1, 0.9) - 0.2) < 0.01


@REAL
class TestExtensionAudio:
    def test_the_new_part_of_a_silent_source_is_the_track_span(self, tmp_path):
        track = _steps(tmp_path / "t.wav", [0.2, 0.4, 0.6])
        source = make_extend_source(path=tmp_path / "silent.mp4", has_audio=False, frame_count=48)

        audio, lead = extension_audio(source, AudioTrack(str(track), 1.0), carry_frames=25, extend_frames=24)

        assert lead == round(0.25 * AUDIO_SAMPLE_RATE)
        assert audio.shape == (2, lead + AUDIO_SAMPLE_RATE)
        assert not audio[:, :lead].any()
        assert abs(_mean(audio[:, lead:], 0.1, 0.9) - 0.4) < 0.01

    def test_a_window_that_carries_less_than_the_lead_has_a_shorter_lead(self, tmp_path):
        track = _steps(tmp_path / "t.wav", [0.4, 0.4])
        source = make_extend_source(path=tmp_path / "silent.mp4", has_audio=False, frame_count=48)

        audio, lead = extension_audio(source, AudioTrack(str(track), 0.0), carry_frames=2, extend_frames=24)

        assert lead == round(2 * AUDIO_SAMPLE_RATE / 24)
        assert audio.shape[1] == lead + AUDIO_SAMPLE_RATE

    def test_a_track_that_ends_early_still_fills_the_whole_segment(self, tmp_path):
        track = _steps(tmp_path / "t.wav", [0.4])
        source = make_extend_source(path=tmp_path / "silent.mp4", has_audio=False, frame_count=48)

        audio, lead = extension_audio(source, AudioTrack(str(track), 0.5), carry_frames=25, extend_frames=48)

        assert audio.shape == (2, lead + 2 * AUDIO_SAMPLE_RATE)
        assert abs(_mean(audio[:, lead:], 0.1, 0.4) - 0.4) < 0.01
        assert not audio[:, lead + round(0.6 * AUDIO_SAMPLE_RATE) :].any()
