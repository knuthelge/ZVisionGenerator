"""Tests for reading an audio track and the window sound of an extension."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from tests._factories import make_extend_source
from tests._fake_media import completed, f32le

from zvisiongenerator.core.video_types import AudioTrack
from zvisiongenerator.utils import audio_track
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


def _arg(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


def _stereo(level: float, samples: int) -> np.ndarray:
    return np.full((2, samples), level, dtype=np.float32)


def _decoding(monkeypatch, audio: np.ndarray, *, channels: int = 2, returncode: int = 0, stderr: bytes = b"") -> MagicMock:
    """Replace ffprobe (it reports *channels*) and ffmpeg (it prints *audio*), and return the ``subprocess.run`` mock."""
    monkeypatch.setattr(audio_track, "ffprobe_json", lambda *_a, **_k: {"streams": [{"codec_type": "audio", "channels": channels}]})
    run = MagicMock(return_value=completed(f32le(audio), stderr=stderr, returncode=returncode))
    monkeypatch.setattr(audio_track.subprocess, "run", run)
    return run


class TestReadTrack:
    def test_returns_stereo_float32_of_the_asked_length(self, monkeypatch):
        _decoding(monkeypatch, _stereo(0.2, 72000))

        audio = read_track(AudioTrack("t.wav", 0.0), AUDIO_SAMPLE_RATE, 1.5)

        assert audio.dtype == np.float32
        assert audio.shape == (2, 72000)

    def test_asks_ffmpeg_for_the_stretch_at_the_rate_from_the_first_audio_stream(self, monkeypatch):
        run = _decoding(monkeypatch, _stereo(0.2, 16000))

        read_track(AudioTrack("t.wav", 1.25), 16000, 1.0)

        command = run.call_args.args[0]
        assert command[0] == "ffmpeg"
        assert command.index("-ss") < command.index("-i")
        assert (_arg(command, "-ss"), _arg(command, "-t")) == ("1.250000", "1.000000")
        assert (_arg(command, "-map"), _arg(command, "-ar"), _arg(command, "-f")) == ("0:a:0", "16000", "f32le")
        assert _arg(command, "-i") == "t.wav"
        assert "-vn" in command

    def test_a_negative_start_reads_from_the_beginning(self, monkeypatch):
        run = _decoding(monkeypatch, _stereo(0.2, 100))

        read_track(AudioTrack("t.wav", -3.0), 100, 1.0)

        assert _arg(run.call_args.args[0], "-ss") == "0.000000"

    def test_a_mono_track_is_copied_to_both_channels_instead_of_ffmpegs_upmix(self, monkeypatch):
        run = _decoding(monkeypatch, _stereo(0.5, 100), channels=1)

        read_track(AudioTrack("t.wav", 0.0), 100, 1.0)

        command = run.call_args.args[0]
        assert _arg(command, "-af") == "pan=stereo|c0=c0|c1=c0"
        assert "-ac" not in command

    def test_a_stereo_track_is_left_to_ffmpeg(self, monkeypatch):
        run = _decoding(monkeypatch, _stereo(0.5, 100), channels=2)

        read_track(AudioTrack("t.wav", 0.0), 100, 1.0)

        command = run.call_args.args[0]
        assert _arg(command, "-ac") == "2"
        assert "-af" not in command

    def test_a_file_ffprobe_cannot_read_is_left_to_ffmpeg_to_report(self, monkeypatch):
        run = _decoding(monkeypatch, _stereo(0.5, 100))
        monkeypatch.setattr(audio_track, "ffprobe_json", MagicMock(side_effect=RuntimeError("ffprobe could not read")))

        read_track(AudioTrack("t.wav", 0.0), 100, 1.0)

        assert _arg(run.call_args.args[0], "-ac") == "2"

    def test_pads_with_silence_when_the_file_ends_early(self, monkeypatch):
        _decoding(monkeypatch, _stereo(0.4, 12000))

        audio = read_track(AudioTrack("t.wav", 1.5), AUDIO_SAMPLE_RATE, 2.0)

        assert audio.shape == (2, 96000)
        assert (audio[:, :12000] == 0.4).all()
        assert not audio[:, 12000:].any()

    def test_a_start_beyond_the_end_gives_all_silence(self, monkeypatch):
        _decoding(monkeypatch, np.zeros((2, 0), dtype=np.float32))

        beyond = read_track(AudioTrack("t.wav", 5.0), AUDIO_SAMPLE_RATE, 0.5)

        assert beyond.shape == (2, 24000)
        assert not beyond.any()

    def test_extra_samples_are_cut_and_a_partial_trailing_sample_is_dropped(self, monkeypatch):
        run = _decoding(monkeypatch, np.arange(2 * 150, dtype=np.float32).reshape(150, 2).T)
        run.return_value = completed(run.return_value.stdout + b"\x00\x00\x00", returncode=0)

        audio = read_track(AudioTrack("t.wav", 0.0), 100, 1.0)

        assert audio.shape == (2, 100)
        assert audio[0, 5] == 10 and audio[1, 5] == 11  # channels stay interleaved as written

    def test_asking_for_no_time_does_not_start_ffmpeg(self, monkeypatch):
        run = _decoding(monkeypatch, _stereo(0.4, 100))

        assert read_track(AudioTrack("t.wav", 0.0), AUDIO_SAMPLE_RATE, 0.0).shape == (2, 0)
        run.assert_not_called()

    def test_a_file_ffmpeg_cannot_decode_says_so_and_carries_its_message(self, monkeypatch):
        _decoding(monkeypatch, np.zeros((2, 0), dtype=np.float32), returncode=1, stderr=b"Invalid data found when processing input")

        with pytest.raises(ValueError, match=TRACK_UNREADABLE) as caught:
            read_track(AudioTrack("notes.txt", 0.0), AUDIO_SAMPLE_RATE, 1.0)

        assert "Invalid data found" in str(caught.value.__cause__)


def _probing(monkeypatch, data: dict | Exception, *, decodes: bool = True) -> MagicMock:
    """Replace ffprobe with *data* (or its failure) and the decode check with *decodes*; return the decode check."""
    if isinstance(data, Exception):
        monkeypatch.setattr(audio_track, "ffprobe_json", MagicMock(side_effect=data))
    else:
        monkeypatch.setattr(audio_track, "ffprobe_json", MagicMock(return_value=data))
    check = MagicMock(return_value=decodes)
    monkeypatch.setattr(audio_track, "audio_decodes", check)
    return check


class TestProbeTrackSeconds:
    def test_returns_the_length_of_the_audio_stream(self, monkeypatch):
        _probing(monkeypatch, {"streams": [{"codec_type": "video"}, {"codec_type": "audio", "duration": "4.000000"}], "format": {"duration": "9.0"}})

        assert probe_track_seconds("t.wav") == pytest.approx(4.0)

    def test_falls_back_to_the_containers_length(self, monkeypatch):
        _probing(monkeypatch, {"streams": [{"codec_type": "audio", "duration": "N/A"}], "format": {"duration": "3.5"}})

        assert probe_track_seconds("t.mka") == pytest.approx(3.5)

    def test_checks_that_the_first_second_decodes(self, monkeypatch):
        check = _probing(monkeypatch, {"streams": [{"codec_type": "audio", "duration": "4.0"}]})

        probe_track_seconds("t.wav")

        assert check.call_args.args == ("t.wav",)
        assert check.call_args.kwargs == {"start": 0.0, "duration": 1.0}

    def test_a_shorter_track_is_checked_for_its_whole_length(self, monkeypatch):
        check = _probing(monkeypatch, {"streams": [{"codec_type": "audio", "duration": "0.4"}]})

        probe_track_seconds("t.wav")

        assert check.call_args.kwargs["duration"] == pytest.approx(0.4)

    @pytest.mark.parametrize(
        "case",
        [
            RuntimeError("ffprobe could not read"),
            {"streams": [{"codec_type": "video"}]},
            {"streams": []},
            {},
            {"streams": [{"codec_type": "audio"}], "format": {}},
            {"streams": [{"codec_type": "audio", "duration": "0"}], "format": {"duration": "nan"}},
        ],
        ids=["unreadable", "no-audio-stream", "no-streams", "empty", "no-length", "zero-length"],
    )
    def test_a_file_that_cannot_be_read_says_so(self, monkeypatch, case):
        _probing(monkeypatch, case)

        with pytest.raises(ValueError, match=TRACK_UNREADABLE):
            probe_track_seconds("file.bin")

    def test_audio_that_does_not_decode_says_so(self, monkeypatch):
        _probing(monkeypatch, {"streams": [{"codec_type": "audio", "duration": "4.0"}]}, decodes=False)

        with pytest.raises(ValueError, match=TRACK_UNREADABLE):
            probe_track_seconds("t.wav")


class TestRequireReadableTrack:
    def test_a_file_that_cannot_be_read_is_refused(self, monkeypatch):
        _probing(monkeypatch, RuntimeError("ffprobe could not read"))

        with pytest.raises(ValueError, match=TRACK_UNREADABLE):
            require_readable_track(AudioTrack("missing.mp3", 0.0))

    def test_a_readable_file_passes(self, monkeypatch):
        _probing(monkeypatch, {"streams": [{"codec_type": "audio", "duration": "4.0"}]})

        require_readable_track(AudioTrack("song.mp3", 0.0))


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


def _levels(monkeypatch) -> list[tuple[AudioTrack, int, float]]:
    """Replace ``read_track`` with a reader whose every sample is the level of its file, and return its calls."""
    calls: list[tuple[AudioTrack, int, float]] = []
    levels = {"track.wav": 0.4, "clip.mp4": 0.9}

    def read(track: AudioTrack, rate: int, seconds: float) -> np.ndarray:
        calls.append((track, rate, seconds))
        return _stereo(levels[track.path.rsplit("/", 1)[-1]], round(seconds * rate))

    monkeypatch.setattr(audio_track, "read_track", read)
    return calls


class TestWindowWaveform:
    def test_a_silent_source_gives_silence_then_the_track_from_start_at(self, monkeypatch):
        calls = _levels(monkeypatch)
        source = make_extend_source(path="/v/silent.mp4", has_audio=False, frame_count=48)

        window = window_waveform(source, AudioTrack("/m/track.wav", 1.0), carry_frames=25, extend_frames=24)

        carried = round(25 * AUDIO_SAMPLE_RATE / 24)
        assert window.shape == (2, carried + AUDIO_SAMPLE_RATE)
        assert not window[:, :carried].any()
        assert (window[:, carried:] == 0.4).all()
        assert calls == [(AudioTrack("/m/track.wav", 1.0), AUDIO_SAMPLE_RATE, pytest.approx(1.0))]

    def test_a_source_with_sound_gives_the_sound_under_its_last_frames_then_the_track(self, monkeypatch):
        calls = _levels(monkeypatch)
        source = make_extend_source(path="/v/clip.mp4", has_audio=True, frame_count=48, fps=24.0, video_start=0.5)

        window = window_waveform(source, AudioTrack("/m/track.wav", 0.0), carry_frames=24, extend_frames=24)

        assert window.shape == (2, 2 * AUDIO_SAMPLE_RATE)
        assert (window[:, :AUDIO_SAMPLE_RATE] == 0.9).all()
        assert (window[:, AUDIO_SAMPLE_RATE:] == 0.4).all()
        carried, new = calls
        # The carried sound starts at the first carried frame, counted from the start of the file, not of the video.
        assert carried == (AudioTrack("/v/clip.mp4", pytest.approx(0.5 + 24 / 24)), AUDIO_SAMPLE_RATE, pytest.approx(1.0))
        assert new[0] == AudioTrack("/m/track.wav", 0.0)

    def test_carrying_more_frames_than_the_source_has_starts_at_its_first_frame(self, monkeypatch):
        calls = _levels(monkeypatch)
        source = make_extend_source(path="/v/clip.mp4", has_audio=True, frame_count=10, fps=24.0, video_start=0.0)

        window_waveform(source, AudioTrack("/m/track.wav", 0.0), carry_frames=25, extend_frames=24)

        assert calls[0][0].start == 0.0

    def test_the_rate_is_passed_on(self, monkeypatch):
        calls = _levels(monkeypatch)
        source = make_extend_source(path="/v/clip.mp4", has_audio=True, frame_count=48)

        window = window_waveform(source, AudioTrack("/m/track.wav", 0.0), carry_frames=24, extend_frames=24, rate=16000)

        assert window.shape == (2, 32000)
        assert {rate for _, rate, _ in calls} == {16000}


class TestExtensionAudio:
    def test_the_new_part_of_a_silent_source_is_the_track_span(self, monkeypatch):
        _levels(monkeypatch)
        source = make_extend_source(path="/v/silent.mp4", has_audio=False, frame_count=48)

        audio, lead = extension_audio(source, AudioTrack("/m/track.wav", 1.0), carry_frames=25, extend_frames=24)

        assert lead == round(0.25 * AUDIO_SAMPLE_RATE)
        assert audio.shape == (2, lead + AUDIO_SAMPLE_RATE)
        assert not audio[:, :lead].any()
        assert (audio[:, lead:] == 0.4).all()

    def test_the_lead_is_the_end_of_the_sources_own_sound(self, monkeypatch):
        _levels(monkeypatch)
        source = make_extend_source(path="/v/clip.mp4", has_audio=True, frame_count=48, video_start=0.0)

        audio, lead = extension_audio(source, AudioTrack("/m/track.wav", 0.0), carry_frames=25, extend_frames=24)

        assert (audio[:, :lead] == 0.9).all()
        assert (audio[:, lead:] == 0.4).all()

    def test_a_window_that_carries_less_than_the_lead_has_a_shorter_lead(self, monkeypatch):
        _levels(monkeypatch)
        source = make_extend_source(path="/v/silent.mp4", has_audio=False, frame_count=48)

        audio, lead = extension_audio(source, AudioTrack("/m/track.wav", 0.0), carry_frames=2, extend_frames=24)

        assert lead == round(2 * AUDIO_SAMPLE_RATE / 24)
        assert audio.shape[1] == lead + AUDIO_SAMPLE_RATE

    def test_a_longer_segment_is_as_long_as_its_new_frames(self, monkeypatch):
        _levels(monkeypatch)
        source = make_extend_source(path="/v/silent.mp4", has_audio=False, frame_count=48)

        audio, lead = extension_audio(source, AudioTrack("/m/track.wav", 0.5), carry_frames=25, extend_frames=48)

        assert audio.shape == (2, lead + 2 * AUDIO_SAMPLE_RATE)
